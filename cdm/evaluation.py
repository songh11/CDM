# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import threading
import time
import logging
from collections import defaultdict
from contextlib import contextmanager
from functools import partial

import numpy as np
import os, json
from PIL import Image
import torch
import torch.distributed as dist
from torch.amp import autocast as torch_autocast

from cdm.pipeline.inference import (
    InferenceConfig,
    use_adapter,
    generate_images_simple,
    compute_seed_from_prompts,
    make_generator_from_prompts,
    unwrap_transformer,
)

from cdm.utils.common import (
    is_main_process,
    gather_tensor_to_all,
)

tqdm_module = __import__('tqdm', fromlist=['tqdm'])
tqdm = partial(tqdm_module.tqdm, dynamic_ncols=True)

logger = logging.getLogger(__name__)

# Background thread pool for async image saving (avoids blocking training and NCCL timeouts)
_image_save_executor = None
_image_save_lock = threading.Lock()

def _resolve_eval_image_save_dir(config, exp_logger):
    """Resolve the root directory for saving evaluation images.

    - If ``config.eval.image_save_root`` is a non-empty string, images are stored
      under ``<image_save_root>/<run_name>/eval_images`` so that multiple runs
      sharing the same root directory stay isolated.
    - Otherwise, fall back to the legacy behavior that places ``eval_images``
      next to the tensorboard log directory of the current experiment.
    """
    image_save_root = getattr(getattr(config, 'eval', config), 'image_save_root', '') or ''
    if image_save_root:
        run_name = getattr(config, 'run_name', '') or 'default_run'
        return os.path.join(image_save_root, run_name, "eval_images")
    return os.path.join(os.path.dirname(exp_logger.log_dir), "eval_images")

def _get_image_save_executor():
    """Lazily create a single-thread executor for background image saving."""
    global _image_save_executor
    if _image_save_executor is None:
        with _image_save_lock:
            if _image_save_executor is None:
                from concurrent.futures import ThreadPoolExecutor
                _image_save_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="img_save")
    return _image_save_executor

def shutdown_image_save_executor():
    """Wait for pending image saves to finish and shut down the executor.

    Should be called before process group destruction to avoid NCCL timeout
    errors caused by background threads still running during cleanup.
    """
    global _image_save_executor
    if _image_save_executor is not None:
        _image_save_executor.shutdown(wait=True)
        _image_save_executor = None

def _save_images_async(images, prompts, image_meta, rewards, global_step, save_dir, prefix):
    """Submit image saving to a background thread to avoid blocking training.

    All tensor data is cloned to CPU before submission to ensure no GPU references
    are held by the background thread.
    """
    images_cpu = images.cpu().clone() if images is not None else None
    prompts_copy = list(prompts) if prompts else []
    meta_copy = [dict(m) for m in image_meta] if image_meta else []
    rewards_copy = {k: v.copy() for k, v in rewards.items()} if rewards else None

    executor = _get_image_save_executor()
    future = executor.submit(
        _save_images_locally, images_cpu, prompts_copy, meta_copy,
        rewards_copy, global_step, save_dir, prefix,
    )
    future.add_done_callback(
        lambda f: logger.error(f"Background image save failed: {f.exception()}")
        if f.exception() else None
    )

# Global cache for teacher results (always used for teacher caching)
_teacher_cache = {
    'rewards': None,
    'images': None,
    'prompts': None,
    'image_meta': None,
    'fake_viz_images': None,
    'fake_viz_prompts': None,
    'fake_viz_image_meta': None,
}

def _save_images_locally(images, prompts, image_meta, rewards, global_step, save_dir, prefix):
    """Save all evaluation images to local disk with metadata.

    Args:
        images: Tensor of shape (N, C, H, W) with values in [0, 1], or None.
        prompts: List of prompt strings for each image.
        image_meta: List of dicts with 'seed' and 'rank' per image.
        rewards: Dict of {reward_name: np.array} or None.
        global_step: Current training step.
        save_dir: Base directory for saving (e.g. experiment_dir/eval_images).
        prefix: Label prefix such as "student", "teacher", "fake_teacher", "real_teacher".
    """
    if images is None or len(images) == 0:
        return

    step_dir = os.path.join(save_dir, f"step_{global_step}", prefix)
    os.makedirs(step_dir, exist_ok=True)

    metadata_entries = []
    for idx in range(len(images)):
        img_tensor = images[idx].cpu().float()
        img_np = img_tensor.numpy()
        if img_np.shape[0] in (1, 3, 4):
            img_np = np.transpose(img_np, (1, 2, 0))
        if img_np.max() <= 1.0 and img_np.dtype in (np.float32, np.float64, np.float16):
            img_np = (img_np * 255).astype(np.uint8)
        elif img_np.dtype != np.uint8:
            img_np = img_np.astype(np.uint8)

        # Squeeze single-channel to 2D for grayscale
        if img_np.ndim == 3 and img_np.shape[2] == 1:
            img_np = img_np.squeeze(2)

        meta = image_meta[idx] if image_meta and idx < len(image_meta) else {}
        seed = meta.get('seed', 'unknown')
        rank = meta.get('rank', 'unknown')
        filename = f"{prefix}_{idx:04d}_seed{seed}_rank{rank}.png"

        pil_image = Image.fromarray(img_np)
        pil_image.save(os.path.join(step_dir, filename))

        entry = {
            'index': idx,
            'filename': filename,
            'prompt': prompts[idx] if prompts and idx < len(prompts) else '',
            'seed': seed,
            'rank': rank,
        }
        if rewards is not None:
            for reward_name, reward_values in rewards.items():
                if idx < len(reward_values):
                    entry[f'reward_{reward_name}'] = float(reward_values[idx])
        metadata_entries.append(entry)

    metadata_path = os.path.join(step_dir, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata_entries, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved {len(images)} {prefix} images to {step_dir}")


def _select_indices(num_available, num_samples, random=False):
    """Select indices for image logging. Returns sorted random indices or sequential indices."""
    if random and num_available > num_samples:
        return sorted(np.random.choice(num_available, size=num_samples, replace=False).tolist())
    return list(range(num_samples))

def _gather_image_metadata(local_batch_seeds, world_size):
    """Gather per-image metadata from all ranks, aligned with gather_tensor_to_all output order.

    gather_tensor_to_all concatenates by rank: [rank0_data, rank1_data, ...] per batch.
    After torch.cat across batches: [batch0_rank0, batch0_rank1, batch1_rank0, batch1_rank1, ...]

    Args:
        local_batch_seeds: List of (prompts_list, seed) tuples from this rank, one per batch.
        world_size: Number of GPUs.

    Returns:
        all_prompts: Flat list of prompts aligned with gathered images.
        image_meta: List of dicts with 'seed', 'rank' per image, aligned with gathered images.
        all_rank_seeds: List of (rank, batch_idx, prompts, seed) for reproduction_info logging.
    """
    if world_size > 1:
        per_rank_seeds = [None] * world_size
        dist.all_gather_object(per_rank_seeds, local_batch_seeds)
    else:
        per_rank_seeds = [local_batch_seeds]

    num_batches = len(per_rank_seeds[0])
    all_prompts = []
    image_meta = []
    all_rank_seeds = []

    for rank_id, rank_seeds in enumerate(per_rank_seeds):
        for batch_idx, (prompts, seed) in enumerate(rank_seeds):
            all_rank_seeds.append((rank_id, batch_idx, prompts, seed))

    for batch_idx in range(num_batches):
        for rank_id in range(world_size):
            prompts, seed = per_rank_seeds[rank_id][batch_idx]
            for prompt in prompts:
                all_prompts.append(prompt)
                image_meta.append({'seed': seed, 'rank': rank_id})

    return all_prompts, image_meta, all_rank_seeds

def _format_reproduction_info(title, all_rank_seeds, extra_lines=None):
    """Format reproduction info grouped by rank."""
    lines = [title]
    if extra_lines:
        lines.extend(extra_lines)
    ranks = sorted(set(r for r, _, _, _ in all_rank_seeds))
    for rank_id in ranks:
        lines.append(f"Rank {rank_id}:")
        for r, batch_idx, prompts, seed in all_rank_seeds:
            if r == rank_id:
                lines.append(f"  Batch {batch_idx}: seed={seed}, prompts={prompts}")
    return "\n".join(lines)


def _generate_teacher_images_simple(pipeline, prompts, inf_cfg, transformer_ddp=None,
                                    guidance_scale=None, device=None, **kwargs):
    """Generate images using the real teacher model (base model without adapters).

    Passes prompt strings directly to the pipeline and uses a deterministic
    generator seeded by the prompt hash for reproducible noise.

    Extra ``**kwargs`` (e.g. ``image`` for LongCat Edit) are forwarded to
    ``generate_images_simple``.

    For LoRA mode: ``transformer_ddp`` is unwrapped to access ``disable_adapter``,
    which disables all adapters to recover the original base model weights.

    For full fine-tuning mode: ``pipeline.real_teacher`` (a frozen copy of the
    original base model created before training) is temporarily swapped into
    ``pipeline.transformer`` so that the pipeline generates with the untrained
    teacher weights.
    """
    actual_guidance_scale = guidance_scale if guidance_scale is not None else inf_cfg.teacher_guidance_scale
    unwrapped = unwrap_transformer(transformer_ddp)
    if unwrapped is not None and hasattr(unwrapped, 'disable_adapter'):
        # LoRA mode: disable all adapters to recover the original base model
        with unwrapped.disable_adapter():
            images = generate_images_simple(
                pipeline, prompts, inf_cfg,
                inf_cfg.teacher_num_steps, actual_guidance_scale,
                device=device, **kwargs,
            )
    elif hasattr(pipeline, 'real_teacher') and pipeline.real_teacher is not None:
        # Full fine-tuning mode: swap in the frozen real teacher model
        original_transformer = pipeline.transformer
        try:
            pipeline.transformer = pipeline.real_teacher
            images = generate_images_simple(
                pipeline, prompts, inf_cfg,
                inf_cfg.teacher_num_steps, actual_guidance_scale,
                device=device, **kwargs,
            )
        finally:
            pipeline.transformer = original_transformer
    else:
        logger.warning("No real teacher available (no disable_adapter and no pipeline.real_teacher). "
                       "Using current model as teacher — results may be incorrect.")
        images = generate_images_simple(
            pipeline, prompts, inf_cfg,
            inf_cfg.teacher_num_steps, actual_guidance_scale,
            device=device, **kwargs,
        )

    return images


def _load_reward_models_to_device(reward_fn, device):
    """Load reward scorer models to the specified device for evaluation.

    Updates both the module device placement and the scorer's internal
    ``self.device`` attribute so that input tensors are sent to the correct device.
    """
    from cdm.rewards.rewards import get_scorer_models
    scorer_models = get_scorer_models(reward_fn)
    for score_name, scorer in scorer_models:
        scorer.to(device)
        if hasattr(scorer, 'device'):
            scorer.device = device
    if scorer_models:
        torch.cuda.empty_cache()
    return scorer_models


def _offload_reward_models_to_cpu(scorer_models):
    """Offload reward scorer models back to CPU after evaluation to free GPU memory."""
    for score_name, scorer in scorer_models:
        scorer.cpu()
        if hasattr(scorer, 'device'):
            scorer.device = torch.device('cpu')
    if scorer_models:
        torch.cuda.empty_cache()


def eval_distillation_fn(
    pipeline, test_dataloader, text_encoders, tokenizers, config, device, rank, world_size,
    global_step, reward_fn, executor, mixed_precision_dtype, ema, transformer_trainable_parameters,
    exp_logger, transformer_ddp, model_adapter=None, accelerator=None,
):
    """Evaluate distillation gap between student and teacher models.

    - Uses deterministic generator seeded by prompt hash
    - Teacher results are cached after first run and reused
    - Pipeline handles text encoding and latent preparation internally
    """
    global _teacher_cache

    # Load reward models to GPU for evaluation (offloaded to CPU during training)
    reward_scorer_models = _load_reward_models_to_device(reward_fn, device)

    if config.train.student.ema and ema is not None:
        ema.copy_ema_to(transformer_trainable_parameters, store_temp=True)

    pipeline.transformer.eval()

    inf_cfg = InferenceConfig.from_config(config)

    all_student_rewards = defaultdict(list)
    all_teacher_rewards = defaultdict(list)
    all_student_images = []
    all_teacher_images = []
    local_batch_seeds = []  # Track (prompts, seed) per batch on this rank

    # Teacher only needs to be generated once (cached after first run)
    need_teacher = config.eval.generate_teacher and _teacher_cache['rewards'] is None


    for batch in tqdm(test_dataloader, desc="Distillation Eval: ", disable=not is_main_process(rank), position=0):
        prompts, prompt_metadata = batch

        batch_seed = compute_seed_from_prompts(prompts)
        local_batch_seeds.append((list(prompts), batch_seed))

        with torch_autocast(device_type="cuda", enabled=(config.mixed_precision in ["fp16", "bf16"]), dtype=mixed_precision_dtype):
            with torch.no_grad():
                student_use_lora = getattr(getattr(config.train, 'student', None), 'use_lora', False)
                unwrapped = unwrap_transformer(transformer_ddp)

                with use_adapter(unwrapped, "default") if student_use_lora else contextmanager(lambda: (yield))():
                    # Note: pipeline.transformer is now the DDP/FSDP wrapped model
                    # (patched with .config/.dtype). Forward calls go through the
                    # distributed wrapper. Use torch.no_grad() instead of
                    # torch.inference_mode() for FSDP compatibility.
                    student_images = generate_images_simple(
                        pipeline, prompts, inf_cfg,
                        inf_cfg.student_num_steps, inf_cfg.student_guidance_scale,
                        device=device, sigmas=inf_cfg.custom_sigmas,
                    )

                if need_teacher:
                    teacher_images = _generate_teacher_images_simple(
                        pipeline, prompts, inf_cfg, transformer_ddp,
                        device=device,
                    )

        # Compute rewards
        # Disable autocast to prevent BFloat16 issues in reward model (consistent with losses.py).
        # Use synchronous execution to avoid FSDP CUDA context conflicts in worker threads.
        with torch.amp.autocast(device_type='cuda', enabled=False):
            student_rewards, _ = reward_fn(student_images, prompts, prompt_metadata, only_strict=False)
            if need_teacher:
                teacher_rewards, _ = reward_fn(teacher_images, prompts, prompt_metadata, only_strict=False)

        for key, value in student_rewards.items():
            gathered = gather_tensor_to_all(torch.as_tensor(value, device=device).float(), world_size)
            all_student_rewards[key].append(gathered.numpy())

        gathered_student = gather_tensor_to_all(student_images.cpu().to(device), world_size)
        all_student_images.append(gathered_student.cpu())

        if need_teacher:
            for key, value in teacher_rewards.items():
                gathered = gather_tensor_to_all(torch.as_tensor(value, device=device).float(), world_size)
                all_teacher_rewards[key].append(gathered.numpy())
            gathered_teacher = gather_tensor_to_all(teacher_images.cpu().to(device), world_size)
            all_teacher_images.append(gathered_teacher.cpu())

    # Gather metadata from all ranks, aligned with gather_tensor_to_all output order
    all_prompts, image_meta, all_rank_seeds = _gather_image_metadata(local_batch_seeds, world_size)

    # Concatenate all student images
    final_student_images = torch.cat(all_student_images, dim=0) if all_student_images else None

    # Cache teacher results after first generation
    if need_teacher and all_teacher_images:
        final_teacher_rewards = {k: np.concatenate(v) for k, v in all_teacher_rewards.items()}
        final_teacher_images = torch.cat(all_teacher_images, dim=0)
        _teacher_cache['rewards'] = final_teacher_rewards
        _teacher_cache['images'] = final_teacher_images
        _teacher_cache['prompts'] = all_prompts
        _teacher_cache['image_meta'] = image_meta
        if is_main_process(rank):
            logger.info("Teacher rewards and images cached.")

    # Use cached teacher results
    teacher_rewards = _teacher_cache['rewards']
    teacher_images_for_log = _teacher_cache['images']
    teacher_image_meta = _teacher_cache.get('image_meta', image_meta)

    if is_main_process(rank) and exp_logger is not None:
        final_student_rewards = {k: np.concatenate(v) for k, v in all_student_rewards.items()}

        # Log rewards
        for key, value in final_student_rewards.items():
            exp_logger.add_scalar(f"eval_distillation/student_reward_{key}", np.mean(value[value != -10]), global_step)

        if teacher_rewards is not None:
            for key, value in teacher_rewards.items():
                exp_logger.add_scalar(f"eval_distillation/teacher_reward_{key}", np.mean(value[value != -10]), global_step)
            for key in final_student_rewards:
                if key in teacher_rewards:
                    gap = np.mean(teacher_rewards[key][teacher_rewards[key] != -10]) - np.mean(final_student_rewards[key][final_student_rewards[key] != -10])
                    exp_logger.add_scalar(f"eval_distillation/reward_gap_{key}", gap, global_step)

        # Save all images locally (async to avoid NCCL timeout)
        if getattr(config.eval, 'save_images', False):
            eval_save_dir = _resolve_eval_image_save_dir(config, exp_logger)
            _save_images_async(final_student_images, all_prompts, image_meta,
                               final_student_rewards, global_step, eval_save_dir, "student")
            if teacher_images_for_log is not None:
                teacher_prompts_for_save = _teacher_cache['prompts'] if _teacher_cache['prompts'] else all_prompts
                _save_images_async(teacher_images_for_log, teacher_prompts_for_save, teacher_image_meta,
                                   teacher_rewards, global_step, eval_save_dir, "teacher")

        # Log images
        _log_distillation_images(exp_logger, final_student_images, teacher_images_for_log, all_prompts,
                                 final_student_rewards, teacher_rewards, config, global_step,
                                 image_meta=image_meta, teacher_image_meta=teacher_image_meta)

        # Build and log summary
        summary = [f"[Eval Distillation Step {global_step}]"]
        for key, value in final_student_rewards.items():
            summary.append(f"student_{key}={np.mean(value[value != -10]):.4f}")
        if teacher_rewards is not None:
            for key, value in teacher_rewards.items():
                summary.append(f"teacher_{key}={np.mean(value[value != -10]):.4f}")
        logger.info(" ".join(summary))

        # Log reproduction info grouped by rank
        repro_text = _format_reproduction_info(
            f"[Eval Distillation Reproduction Info - Step {global_step}]",
            all_rank_seeds,
            extra_lines=[
                f"student_num_steps={inf_cfg.student_num_steps}, student_guidance_scale={inf_cfg.student_guidance_scale}",
                f"teacher_num_steps={inf_cfg.teacher_num_steps}, teacher_guidance_scale={inf_cfg.teacher_guidance_scale}",
            ],
        )
        exp_logger.add_text("eval_distillation/reproduction_info", repro_text, global_step)

    if config.train.student.ema and ema is not None:
        ema.copy_temp_to(transformer_trainable_parameters)

    # Offload reward models back to CPU to free GPU memory during training
    _offload_reward_models_to_cpu(reward_scorer_models)

    if world_size > 1:
        dist.barrier()


def _meta_caption(idx, meta_list, label):
    """Build caption string with rank and seed info from image metadata."""
    if meta_list and idx < len(meta_list):
        meta = meta_list[idx]
        return f"{label} | rank={meta['rank']} | seed={meta['seed']}"
    return label

def _log_distillation_images(exp_logger, student_images, teacher_images, student_prompts, student_rewards, teacher_rewards, config, global_step, image_meta=None, teacher_image_meta=None):
    """Log distillation comparison images."""
    global _teacher_cache

    if student_images is None:
        return

    teacher_prompts = _teacher_cache['prompts'] if _teacher_cache['prompts'] else student_prompts

    num_log = getattr(config.eval, 'num_log_images', 8)
    num_student = len(student_images)
    num_teacher = len(teacher_images) if teacher_images is not None else 0
    num_available = min(num_student, num_teacher) if teacher_images is not None else num_student
    num_samples = min(num_log, num_available)
    log_image_resolution = getattr(config, 'log_image_resolution', 512)
    first_key = list(student_rewards.keys())[0] if student_rewards else None

    for tag_suffix, use_random in [("_fixed", False), ("_random", True)]:
        indices = _select_indices(num_available, num_samples, random=use_random)

        images, prompt_list, captions, rewards = [], [], [], []

        for idx in indices:
            images.append(student_images[idx].cpu().float())
            prompt_list.append(student_prompts[idx] if idx < len(student_prompts) else "")
            captions.append(_meta_caption(idx, image_meta, f"Student ({config.eval.student_num_steps} steps)"))
            rewards.append(float(student_rewards[first_key][idx]) if first_key and idx < len(student_rewards[first_key]) else 0.0)

        if teacher_images is not None:
            for idx in indices:
                images.append(teacher_images[idx].cpu().float())
                prompt_list.append(teacher_prompts[idx] if idx < len(teacher_prompts) else "")
                captions.append(_meta_caption(idx, teacher_image_meta, f"Teacher ({config.eval.teacher_num_steps} steps)"))
                rewards.append(float(teacher_rewards[first_key][idx]) if teacher_rewards and first_key and idx < len(teacher_rewards[first_key]) else 0.0)

        exp_logger.add_image_table(f"distillation_comparison{tag_suffix}", images, global_step, prompts=prompt_list, captions=captions, rewards=rewards, resize_to=log_image_resolution)


def eval_fake_teacher_viz_fn(
    pipeline, test_dataloader, text_encoders, tokenizers, config, device, rank, world_size,
    global_step, epoch, mixed_precision_dtype, exp_logger, accelerator, transformer_ddp=None,
    model_adapter=None,
):
    """Generate visualization images from Fake Teacher and Real Teacher models.

    - Uses deterministic generator seeded by prompt hash
    - Real teacher results are cached after first run and reused
    - Fake teacher is generated real-time (parameters change during training)
    - Pipeline handles text encoding and latent preparation internally
    """
    global _teacher_cache

    ft_viz_config = getattr(config.eval, 'fake_teacher_viz', None)
    fake_teacher_config = getattr(config.train, 'fake_teacher', None)
    if ft_viz_config is None or fake_teacher_config is None:
        logger.warning("Fake teacher config not found")
        return

    inf_cfg = InferenceConfig.from_config(config)

    # Configuration
    ft_num_images_total = getattr(ft_viz_config, 'num_images', 32)
    ft_num_log_images = getattr(ft_viz_config, 'num_log_images', 8)
    real_teacher_steps = getattr(config.eval, 'teacher_num_steps', 28)

    ft_use_lora = getattr(fake_teacher_config, 'use_lora', True)

    ft_num_images_per_gpu = max(1, ft_num_images_total // world_size)

    if is_main_process(rank):
        logger.info(f"Generating Fake/Real Teacher visualization at epoch {epoch} "
                   f"(total={ft_num_images_total}, per_gpu={ft_num_images_per_gpu})...")

    need_real_teacher = _teacher_cache['fake_viz_images'] is None

    local_ft_images = []
    local_real_teacher_images = []
    local_batch_seeds = []  # Track (prompts, seed) per batch on this rank
    generated_count = 0

    # Set eval mode on the wrapped model (not unwrapped) so that FSDP
    # correctly transitions to eval mode. For DDP this is equivalent.
    pipeline.transformer.eval()
    if hasattr(pipeline, 'fake_teacher') and pipeline.fake_teacher is not None and pipeline.fake_teacher is not pipeline.transformer:
        pipeline.fake_teacher.eval()

    unwrapped_transformer = unwrap_transformer(transformer_ddp, accelerator)
    original_transformer = pipeline.transformer

    student_use_lora = getattr(getattr(config.train, 'student', None), 'use_lora', False)

    # Iterate dataloader directly (same batch composition as distillation eval)
    for batch in tqdm(test_dataloader, desc="Fake Teacher Viz: ", disable=not is_main_process(rank), position=0):
        batch_prompts, batch_metadata = batch

        if generated_count >= ft_num_images_per_gpu:
            break

        with torch_autocast(device_type="cuda", enabled=(config.mixed_precision in ["fp16", "bf16"]), dtype=mixed_precision_dtype):
            with torch.no_grad():
                if ft_use_lora and student_use_lora:
                    with use_adapter(unwrapped_transformer, "fake_teacher"):
                        batch_ft_images = generate_images_simple(
                            pipeline, batch_prompts, inf_cfg,
                            inf_cfg.fake_teacher_num_steps, inf_cfg.fake_teacher_guidance_scale,
                            device=device,
                        )
                else:
                    if not hasattr(pipeline, 'fake_teacher') or pipeline.fake_teacher is None:
                        logger.warning("Fake teacher model not found")
                        return
                    # pipeline.fake_teacher is already the DDP/FSDP wrapped model
                    # (patched with .config/.dtype). Swap it in directly.
                    # Use torch.no_grad() instead of torch.inference_mode()
                    # for FSDP compatibility.
                    try:
                        pipeline.transformer = pipeline.fake_teacher
                        batch_ft_images = generate_images_simple(
                            pipeline, batch_prompts, inf_cfg,
                            inf_cfg.fake_teacher_num_steps, inf_cfg.fake_teacher_guidance_scale,
                            device=device,
                        )
                    finally:
                        pipeline.transformer = original_transformer

        local_ft_images.append(batch_ft_images.cpu())
        batch_seed = compute_seed_from_prompts(batch_prompts)
        local_batch_seeds.append((list(batch_prompts), batch_seed))
        generated_count += len(batch_prompts)

        # Generate real teacher images only if not cached
        if need_real_teacher:
            with torch_autocast(device_type="cuda", enabled=(config.mixed_precision in ["fp16", "bf16"]), dtype=mixed_precision_dtype):
                with torch.no_grad():
                    batch_real_teacher_images = _generate_teacher_images_simple(
                        pipeline, batch_prompts, inf_cfg, transformer_ddp,
                        guidance_scale=inf_cfg.real_teacher_viz_guidance_scale,
                        device=device,
                    )
            local_real_teacher_images.append(batch_real_teacher_images.cpu())

    if generated_count == 0:
        logger.warning("No data for fake teacher visualization")
        return

    # Gather metadata from all ranks, aligned with gather_tensor_to_all output order
    all_viz_prompts, image_meta, all_rank_seeds = _gather_image_metadata(local_batch_seeds, world_size)

    # Concatenate local results
    local_ft_images_tensor = torch.cat(local_ft_images, dim=0) if local_ft_images else torch.zeros(0, 3, config.height, config.width)

    # Gather from all GPUs
    gathered_ft = gather_tensor_to_all(local_ft_images_tensor.to(device), world_size)
    final_ft_images = gathered_ft.cpu()

    # Cache real teacher images after first generation
    if need_real_teacher and local_real_teacher_images:
        local_real_teacher_tensor = torch.cat(local_real_teacher_images, dim=0)
        gathered_real = gather_tensor_to_all(local_real_teacher_tensor.to(device), world_size)
        final_real_teacher_images = gathered_real.cpu()
        _teacher_cache['fake_viz_images'] = final_real_teacher_images
        _teacher_cache['fake_viz_prompts'] = all_viz_prompts
        _teacher_cache['fake_viz_image_meta'] = image_meta
        if is_main_process(rank):
            logger.info(f"Real teacher images cached (total: {len(final_real_teacher_images)}).")

    real_teacher_images = _teacher_cache['fake_viz_images']
    cached_prompts = _teacher_cache['fake_viz_prompts'] if _teacher_cache['fake_viz_prompts'] else all_viz_prompts
    real_teacher_meta = _teacher_cache.get('fake_viz_image_meta', image_meta)

    # Save all images locally on main process (async to avoid NCCL timeout)
    if is_main_process(rank) and final_ft_images is not None and len(final_ft_images) > 0 and exp_logger is not None and getattr(config.eval, 'save_images', False):
        eval_save_dir = _resolve_eval_image_save_dir(config, exp_logger)
        _save_images_async(final_ft_images, all_viz_prompts, image_meta,
                           None, global_step, eval_save_dir, "fake_teacher")
        if real_teacher_images is not None:
            _save_images_async(real_teacher_images, cached_prompts, real_teacher_meta,
                               None, global_step, eval_save_dir, "real_teacher")

    # Log images on main process
    if is_main_process(rank) and exp_logger is not None and final_ft_images is not None and len(final_ft_images) > 0:
        num_available = min(len(final_ft_images), len(real_teacher_images)) if real_teacher_images is not None else len(final_ft_images)
        num_log = min(ft_num_log_images, num_available)
        log_image_resolution = getattr(config, 'log_image_resolution', 512)

        for tag_suffix, use_random in [("_fixed", False), ("_random", True)]:
            indices = _select_indices(num_available, num_log, random=use_random)

            img_list, prompt_list, caption_list = [], [], []

            for idx in indices:
                img_list.append(final_ft_images[idx].cpu().float())
                prompt_list.append(all_viz_prompts[idx] if idx < len(all_viz_prompts) else "")
                caption_list.append(_meta_caption(idx, image_meta, f"Fake Teacher ({inf_cfg.fake_teacher_num_steps} steps, cfg={inf_cfg.fake_teacher_guidance_scale})"))

            if real_teacher_images is not None:
                for idx in indices:
                    if idx < len(real_teacher_images):
                        img_list.append(real_teacher_images[idx].cpu().float())
                        prompt_list.append(cached_prompts[idx] if idx < len(cached_prompts) else "")
                        caption_list.append(_meta_caption(idx, real_teacher_meta, f"Real Teacher ({real_teacher_steps} steps, cfg={inf_cfg.real_teacher_viz_guidance_scale})"))

            exp_logger.add_image_table(f"fake_teacher_images{tag_suffix}", img_list, global_step, prompts=prompt_list, captions=caption_list, rewards=None, resize_to=log_image_resolution)

        # Log reproduction info grouped by rank
        repro_text = _format_reproduction_info(
            f"[Fake Teacher Viz Reproduction Info - Step {global_step}]",
            all_rank_seeds,
            extra_lines=[
                f"fake_teacher_num_steps={inf_cfg.fake_teacher_num_steps}, fake_teacher_guidance_scale={inf_cfg.fake_teacher_guidance_scale}",
                f"real_teacher_num_steps={real_teacher_steps}, real_teacher_guidance_scale={inf_cfg.real_teacher_viz_guidance_scale}",
            ],
        )
        exp_logger.add_text("eval_fake_teacher_viz/reproduction_info", repro_text, global_step)

        logger.info(f"Logged {num_log} Fake + {num_log if real_teacher_images is not None else 0} Real Teacher images "
                   f"(total generated: {len(final_ft_images)}, target: {ft_num_images_total})")

    del local_ft_images
    # torch.cuda.empty_cache()

    if world_size > 1:
        dist.barrier()
