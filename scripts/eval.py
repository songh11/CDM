

"""Unified evaluation script for diffusion model checkpoints.

Evaluates a single model pipeline against multiple metrics in two phases:
  1. Generation: produce images for all required datasets
  2. Evaluation: compute all metrics on the generated images

"""

import os

import argparse
import gc
import hashlib
import json
import logging
import re
from collections import defaultdict
from datetime import timedelta
from functools import partial

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from cdm.models.model_adapters import create_model_adapter
from cdm.pipeline.inference import (
    InferenceConfig,
    generate_images_simple,
    make_diversity_generators,
    make_generator_from_prompts,
)
from cdm.rewards.rewards import multi_score
from cdm.data.datasets import TextPromptDataset, GenevalPromptDataset, DPGPromptDataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
logging.getLogger("openai").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)

import transformers
transformers.logging.set_verbosity_error()
import diffusers
diffusers.logging.set_verbosity_error()

tqdm_module = __import__("tqdm", fromlist=["tqdm"])
tqdm = partial(tqdm_module.tqdm, dynamic_ncols=True)

# ==================== Metric Registry ====================

METRIC_REGISTRY = {
    "imagereward": {
        "dataset_type": "text",
        "default_dataset": "dataset/pickscore",
        "scorer_key": "imagereward",
    },
    "clipscore": {
        "dataset_type": "text",
        "default_dataset": "dataset/pickscore",
        "scorer_key": "clipscore",
    },
    "pickscore": {
        "dataset_type": "text",
        "default_dataset": "dataset/pickscore",
        "scorer_key": "pickscore",
    },
    "hpsv2": {
        "dataset_type": "text",
        "default_dataset": "dataset/pickscore",
        "scorer_key": "hpsv2",
    },
    "hpsv3": {
        "dataset_type": "text",
        "default_dataset": "dataset/pickscore",
        "scorer_key": "hpsv3",
    },
    "aesthetic": {
        "dataset_type": "text",
        "default_dataset": "dataset/pickscore",
        "scorer_key": "aesthetic",
    },
    "ocr": {
        "dataset_type": "text",
        "default_dataset": "dataset/ocr",
        "scorer_key": "ocr",
    },
    "fid": {
        "dataset_type": "text",
        "default_dataset": "dataset/coco2014val_10k",
        "scorer_key": "fid",
        "post_hoc": True,
    },
    "dpgbench": {
        "dataset_type": "dpgbench",
        "default_dataset": "dataset/dpgbench",
        "scorer_key": "dpgbench",
        "post_hoc": True,
    },
    "intra_lpips": {
        "dataset_type": "text",
        "default_dataset": "dataset/pickscore",
        "scorer_key": "intra_lpips",
        "post_hoc": True,
        "diversity": True,
    },
    "vendi": {
        "dataset_type": "text",
        "default_dataset": "dataset/pickscore",
        "scorer_key": "vendi",
        "post_hoc": True,
        "diversity": True,
    },
    "llm_judge": {
        "dataset_type": "text",
        "default_dataset": "dataset/pickscore",
        "scorer_key": "llm_judge",
        "post_hoc": True,
        "pairwise": True,
    },
}

# ==================== Dataset Helpers ====================

def resolve_datasets(eval_metrics, default_dataset, dataset_overrides, project_root):
    """Group metrics by their resolved dataset path and type."""
    dataset_groups = defaultdict(list)

    for metric_name in eval_metrics:
        if metric_name not in METRIC_REGISTRY:
            raise ValueError(f"Unknown metric '{metric_name}'. Available: {list(METRIC_REGISTRY.keys())}")

        meta = METRIC_REGISTRY[metric_name]
        is_post_hoc = meta.get("post_hoc", False)

        if metric_name in dataset_overrides:
            dataset_path = dataset_overrides[metric_name]
        elif default_dataset and meta["dataset_type"] == "text" and not is_post_hoc:
            dataset_path = default_dataset
        else:
            dataset_path = meta["default_dataset"]

        if not os.path.isabs(dataset_path):
            dataset_path = os.path.join(project_root, dataset_path)

        dataset_groups[(dataset_path, meta["dataset_type"])].append(metric_name)

    return dict(dataset_groups)


def create_dataset(dataset_path, dataset_type, max_samples=-1):
    """Create a dataset instance based on type."""
    if dataset_type == "geneval":
        return GenevalPromptDataset(dataset_path, split="test", max_samples=max_samples)
    if dataset_type == "dpgbench":
        return DPGPromptDataset(dataset_path, split="test", max_samples=max_samples)
    return TextPromptDataset(dataset_path, split="test", max_samples=max_samples)

# ==================== Pipeline Loading ====================

def load_pipeline(model_path, model_adapter, device, mixed_precision_dtype):
    """Load a diffusers pipeline from a directory and move components to device."""
    logger.info(f"Loading pipeline from: {model_path}")
    pipeline_kwargs = {}
    if mixed_precision_dtype is not None:
        pipeline_kwargs["torch_dtype"] = mixed_precision_dtype

    pipeline = model_adapter.load_pipeline(model_path, **pipeline_kwargs)
    pipeline.safety_checker = None

    text_encoder_dtype = mixed_precision_dtype or torch.float32
    pipeline.transformer.to(device, dtype=mixed_precision_dtype or torch.float32)
    pipeline.vae.to(device, dtype=torch.float32)
    for encoder in model_adapter.get_text_encoders(pipeline):
        if encoder is not None:
            encoder.to(device, dtype=text_encoder_dtype)

    pipeline.transformer.eval()
    pipeline.transformer.requires_grad_(False)
    return pipeline


def unload_pipeline(pipeline):
    """Move all pipeline components to CPU and free GPU memory."""
    for attr_name in ["transformer", "vae", "text_encoder", "text_encoder_2", "text_encoder_3"]:
        component = getattr(pipeline, attr_name, None)
        if component is not None:
            component.to("cpu")
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

# ==================== Image Export Helpers ====================

_FILENAME_UNSAFE_RE = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_MAX_FILENAME_LENGTH = 150


def _sanitize_prompt_for_filename(prompt, max_length=_MAX_FILENAME_LENGTH):
    """Convert a prompt string into a safe, human-readable filename stem."""
    sanitized = _FILENAME_UNSAFE_RE.sub("_", prompt)
    sanitized = sanitized.strip().replace(" ", "_")
    sanitized = re.sub(r"_+", "_", sanitized)
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length]
    return sanitized


def _save_tensor_as_png(image_tensor, save_path):
    """Save a single (3, H, W) float tensor in [0, 1] as a PNG file."""
    pixel_array = (image_tensor.clamp(0, 1) * 255).byte().permute(1, 2, 0).cpu().numpy()
    Image.fromarray(pixel_array).save(save_path)


def export_images_from_batch(images_tensor, prompts, batch_offset, image_dir):
    """Export a batch of image tensors as PNG files for visualization.

    File naming: ``{global_index:06d}_{sanitized_prompt}.png``
    """
    os.makedirs(image_dir, exist_ok=True)
    for idx_in_batch, (image, prompt) in enumerate(zip(images_tensor, prompts)):
        global_index = batch_offset + idx_in_batch
        safe_name = _sanitize_prompt_for_filename(prompt)
        filename = f"{global_index:06d}_{safe_name}.png"
        _save_tensor_as_png(image, os.path.join(image_dir, filename))

# ==================== Generation Batch I/O ====================

def save_generation_batch(images_tensor, prompts, metadata, batch_offset, generation_dir, save_images=False, image_save_dir=None):
    """Save a batch of generated images and metadata to disk as a .pt file.

    When *save_images* is True, also exports each image as a PNG file.
    Images are written directly to *image_save_dir* (the final output
    location) so that no post-hoc copy is needed.
    """
    batch_path = os.path.join(generation_dir, f"batch_{batch_offset:06d}.pt")
    torch.save({
        "images": images_tensor.cpu().float(),
        "prompts": prompts,
        "metadata": metadata,
        "batch_offset": batch_offset,
    }, batch_path)

    if save_images:
        target_dir = image_save_dir or os.path.join(generation_dir, "images")
        export_images_from_batch(images_tensor, prompts, batch_offset, target_dir)


def load_all_generation_batches(generation_dir):
    """Load all batch .pt files from a directory, sorted by offset."""
    batch_files = sorted(
        f for f in os.listdir(generation_dir) if f.startswith("batch_") and f.endswith(".pt")
    )
    all_images, all_prompts, all_metadata = [], [], []
    for batch_file in batch_files:
        data = torch.load(os.path.join(generation_dir, batch_file), map_location="cpu", weights_only=False)
        all_images.append(data["images"])
        all_prompts.extend(data["prompts"])
        all_metadata.extend(data["metadata"])

    combined_images = torch.cat(all_images, dim=0) if all_images else torch.empty(0)
    return combined_images, all_prompts, all_metadata


def gather_all_generations(generation_dir):
    """Load all generation batches from a single directory."""
    return load_all_generation_batches(generation_dir)

# ==================== Phase 1: Generation ====================

def _has_cached_batches(generation_dir, pattern="batch_"):
    """Check whether a generation directory contains cached .pt files."""
    if not os.path.isdir(generation_dir):
        return False
    return any(f.startswith(pattern) and f.endswith(".pt") for f in os.listdir(generation_dir))

def _has_cached_diversity(generation_dir):
    """Check whether a diversity generation directory contains cached .pt files."""
    if not os.path.isdir(generation_dir):
        return False
    return any(f.startswith("prompt_") and f.endswith(".pt") for f in os.listdir(generation_dir))

def generate_for_dataset_group(
    pipeline, inf_cfg, num_steps, guidance_scale,
    dataset_path, dataset_type, device, rank, world_size,
    mixed_precision_dtype, eval_batch_size, generation_dir,
    max_eval_samples=-1, save_images=False, image_save_dir=None,
    use_cache=False,
):
    """Generate images for a single dataset group and save batch .pt files."""
    if use_cache and _has_cached_batches(generation_dir):
        logger.info(f"  Cache hit: reusing existing generations in {generation_dir}")
        return 0

    dataset = create_dataset(dataset_path, dataset_type, max_samples=max_eval_samples)
    logger.info(f"  Dataset: {dataset_path} ({len(dataset)} samples, type={dataset_type})")

    sampler = (
        DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)
        if world_size > 1 else None
    )
    dataloader = DataLoader(
        dataset, batch_size=eval_batch_size, sampler=sampler,
        collate_fn=dataset.collate_fn, shuffle=False,
    )

    os.makedirs(generation_dir, exist_ok=True)
    sample_offset = 0

    for batch in tqdm(dataloader, desc="  Generating", disable=rank != 0, position=1, leave=False):
        prompts, metadata = batch

        with torch.amp.autocast("cuda", enabled=(mixed_precision_dtype is not None), dtype=mixed_precision_dtype):
            with torch.inference_mode():
                images = generate_images_simple(
                    pipeline=pipeline, prompts=prompts, inf_cfg=inf_cfg,
                    num_steps=num_steps, guidance_scale=guidance_scale, device=device,
                    generator=make_generator_from_prompts(prompts, device),
                    sigmas=inf_cfg.custom_sigmas,
                )

        global_offset = sample_offset + rank * len(dataset)
        save_generation_batch(
            images, prompts, metadata, global_offset, generation_dir,
            save_images=save_images, image_save_dir=image_save_dir,
        )
        sample_offset += len(prompts)
        del images

    return sample_offset


def generate_diversity_for_dataset(
    pipeline, inf_cfg, num_steps, guidance_scale,
    dataset_path, dataset_type, device, rank, world_size,
    mixed_precision_dtype, generation_dir,
    num_variations=4, num_prompts=100, max_eval_samples=-1,
    save_images=False, image_save_dir=None,
    use_cache=False, eval_batch_size=None,
):
    """Generate multiple images per prompt for diversity metrics (intra_lpips, vendi).

    Saves per-prompt .pt files containing all variations.
    """
    if use_cache and _has_cached_diversity(generation_dir):
        logger.info(f"  Cache hit: reusing existing diversity generations in {generation_dir}")
        return 0

    dataset = create_dataset(dataset_path, dataset_type, max_samples=max_eval_samples)
    total_available = len(dataset)
    actual_num_prompts = min(num_prompts, total_available)

    all_prompt_indices = np.linspace(0, total_available - 1, actual_num_prompts, dtype=int).tolist()
    rank_prompt_indices = all_prompt_indices[rank::world_size]

    if not rank_prompt_indices:
        return 0

    os.makedirs(generation_dir, exist_ok=True)

    for prompt_idx in tqdm(rank_prompt_indices, desc="  Diversity gen", disable=rank != 0, position=1, leave=False):
        sample = dataset[prompt_idx]
        prompt_text = sample["prompt"]
        prompt_hash = hashlib.md5(prompt_text.encode()).hexdigest()[:16]
        generators = make_diversity_generators(prompt_text, num_variations, device)
        batch_size = eval_batch_size or num_variations
        variation_batches = []

        for var_start in range(0, num_variations, batch_size):
            var_end = min(var_start + batch_size, num_variations)
            batch_prompts = [prompt_text] * (var_end - var_start)
            batch_generators = generators[var_start:var_end]

            with torch.amp.autocast("cuda", enabled=(mixed_precision_dtype is not None), dtype=mixed_precision_dtype):
                with torch.inference_mode():
                    batch_images = generate_images_simple(
                        pipeline=pipeline, prompts=batch_prompts, inf_cfg=inf_cfg,
                        num_steps=num_steps, guidance_scale=guidance_scale,
                        device=device, generator=batch_generators,
                    )
            variation_batches.append(batch_images.cpu())
            del batch_images

        diversity_images = torch.cat(variation_batches, dim=0) if len(variation_batches) > 1 else variation_batches[0]

        torch.save({
            "images": diversity_images.cpu().float(),
            "prompt": prompt_text,
            "prompt_hash": prompt_hash,
            "sample_idx": prompt_idx,
        }, os.path.join(generation_dir, f"prompt_{prompt_hash}.pt"))

        if save_images:
            target_dir = image_save_dir or os.path.join(generation_dir, "images")
            os.makedirs(target_dir, exist_ok=True)
            safe_name = _sanitize_prompt_for_filename(prompt_text)
            for var_idx in range(diversity_images.size(0)):
                filename = f"{prompt_hash}_{safe_name}_var{var_idx:02d}.png"
                _save_tensor_as_png(diversity_images[var_idx], os.path.join(target_dir, filename))

        del diversity_images

    return len(rank_prompt_indices)


def run_generation(
    pipeline, inf_cfg, num_steps, guidance_scale,
    dataset_groups, args, device, rank, world_size, mixed_precision_dtype,
):
    """Phase 1: Generate images for all dataset groups.

    Returns:
        generation_registry: dict of (dataset_path, dataset_type) -> generation_base_dir
        diversity_generation_dir: str or None, base dir for diversity generations
    """
    logger.info(f"\n{'='*60}")
    logger.info("PHASE 1: IMAGE GENERATION")
    logger.info(f"{'='*60}")

    generation_registry = {}

    for (dataset_path, dataset_type), metric_names in dataset_groups.items():
        logger.info(f"\n  Generating for dataset: {dataset_path} (metrics: {metric_names})")
        dataset_name = os.path.basename(dataset_path.rstrip("/"))
        gen_dir = os.path.join(args.output_dir, "_generations", dataset_name)

        image_save_dir = os.path.join(args.output_dir, "saved_images", dataset_name) if args.save_images else None

        generate_for_dataset_group(
            pipeline=pipeline, inf_cfg=inf_cfg,
            num_steps=num_steps, guidance_scale=guidance_scale,
            dataset_path=dataset_path, dataset_type=dataset_type,
            device=device, rank=rank, world_size=world_size,
            mixed_precision_dtype=mixed_precision_dtype,
            eval_batch_size=args.eval_batch_size,
            generation_dir=gen_dir,
            max_eval_samples=args.max_eval_samples,
            save_images=args.save_images,
            image_save_dir=image_save_dir,
            use_cache=args.use_cache,
        )
        generation_registry[(dataset_path, dataset_type)] = gen_dir

    # Diversity generation for diversity metrics
    diversity_metrics = [m for m in args.eval_metrics if METRIC_REGISTRY[m].get("diversity", False)]
    diversity_generation_dir = None

    if diversity_metrics:
        logger.info(f"\n  Generating diversity data for metrics: {diversity_metrics}")

        # Find a suitable text dataset for diversity
        diversity_dataset_path, diversity_dataset_type = None, None
        for (dataset_path, dataset_type), metric_names in dataset_groups.items():
            if any(m in diversity_metrics for m in metric_names) and dataset_type == "text":
                diversity_dataset_path, diversity_dataset_type = dataset_path, dataset_type
                break
        if diversity_dataset_path is None:
            for (dataset_path, dataset_type) in dataset_groups:
                if dataset_type == "text":
                    diversity_dataset_path, diversity_dataset_type = dataset_path, dataset_type
                    break

        if diversity_dataset_path is not None:
            diversity_generation_dir = os.path.join(args.output_dir, "_diversity_generations")
            diversity_image_save_dir = os.path.join(args.output_dir, "saved_images", "diversity") if args.save_images else None

            generate_diversity_for_dataset(
                pipeline=pipeline, inf_cfg=inf_cfg,
                num_steps=num_steps, guidance_scale=guidance_scale,
                dataset_path=diversity_dataset_path, dataset_type=diversity_dataset_type,
                device=device, rank=rank, world_size=world_size,
                mixed_precision_dtype=mixed_precision_dtype,
                generation_dir=diversity_generation_dir,
                num_variations=args.diversity_num_variations,
                num_prompts=args.diversity_num_prompts,
                max_eval_samples=args.max_eval_samples,
                save_images=args.save_images,
                image_save_dir=diversity_image_save_dir,
                use_cache=args.use_cache,
                eval_batch_size=args.eval_batch_size,
            )
        else:
            logger.warning("No suitable text dataset found for diversity metrics. Skipping.")

    if world_size > 1:
        dist.barrier()

    return generation_registry, diversity_generation_dir

# ==================== Phase 2: Evaluation ====================

def _to_list(values):
    """Convert scores to a plain list of floats."""
    if isinstance(values, torch.Tensor):
        return values.detach().cpu().tolist()
    if isinstance(values, np.ndarray):
        return values.tolist()
    if isinstance(values, list):
        return [float(v) if not isinstance(v, (int, float)) else v for v in values]
    return list(values)


def _unload_scorer(scoring_fn):
    """Unload a single-metric scoring function and free GPU memory.
    """
    del scoring_fn
    gc.collect()


def evaluate_standard_metrics(
    generation_registry, dataset_groups, device, rank, world_size, scoring_batch_size,
):
    """Compute standard per-image metrics (ImageReward, CLIPScore, etc.).

    Memory-optimized: loads one scorer at a time, scores all images, then
    unloads it before loading the next scorer. This keeps GPU memory usage
    at ~1 scorer model instead of all scorers simultaneously.

    Returns:
        all_scores: dict of metric_name -> average score
        per_sample_results: list of per-sample dicts
    """
    standard_metrics = [
        m for m in sum(dataset_groups.values(), [])
        if not METRIC_REGISTRY[m].get("post_hoc", False)
    ]
    if not standard_metrics:
        return {}, []

    logger.info(f"\n--- Standard Metrics (sequential loading): {standard_metrics} ---")

    all_scores = {}
    per_sample_results = []

    for (dataset_path, dataset_type), metric_names in dataset_groups.items():
        current_metrics = [m for m in metric_names if not METRIC_REGISTRY[m].get("post_hoc", False)]
        if not current_metrics:
            continue

        gen_base_dir = generation_registry.get((dataset_path, dataset_type))
        if gen_base_dir is None:
            continue

        all_images, all_prompts, all_metadata = load_all_generation_batches(gen_base_dir)
        total_count = len(all_prompts)

        if total_count == 0:
            continue

        # Partition data across ranks
        indices = list(range(rank, total_count, world_size)) if world_size > 1 else list(range(total_count))
        local_images = all_images[indices]
        local_prompts = [all_prompts[i] for i in indices]
        local_metadata = [all_metadata[i] for i in indices]
        del all_images, all_prompts, all_metadata
        local_count = len(local_prompts)

        if local_count == 0:
            continue

        # Score one metric at a time to minimize GPU memory usage
        local_scores = defaultdict(list)

        for metric_name in current_metrics:
            scorer_key = METRIC_REGISTRY[metric_name]["scorer_key"]
            logger.info(f"  Loading scorer: {scorer_key}")

            scoring_fn = multi_score(device, {scorer_key: 1.0})

            for batch_start in tqdm(
                range(0, local_count, scoring_batch_size),
                desc=f"  Scoring [{scorer_key}]", disable=rank != 0, position=1, leave=False,
            ):
                batch_end = min(batch_start + scoring_batch_size, local_count)
                batch_images = local_images[batch_start:batch_end].to(device)
                batch_prompts = local_prompts[batch_start:batch_end]
                batch_metadata = local_metadata[batch_start:batch_end]

                score_details, _ = scoring_fn(
                    batch_images, batch_prompts, batch_metadata,
                    only_strict=(scorer_key != "geneval"),
                )

                if scorer_key == "geneval":
                    for key in score_details:
                        if key != "avg":
                            local_scores[f"geneval_{key}"].extend(_to_list(score_details[key]))
                elif scorer_key in score_details:
                    local_scores[metric_name].extend(_to_list(score_details[scorer_key]))

                del batch_images

            logger.info(f"  Unloading scorer: {scorer_key}")
            _unload_scorer(scoring_fn)

        # Build per-sample results for this rank
        local_results = []
        for i in range(local_count):
            result_item = {"sample_id": i, "prompt": local_prompts[i], "scores": {}}
            for score_name, score_values in local_scores.items():
                if i < len(score_values):
                    result_item["scores"][score_name] = score_values[i]
            local_results.append(result_item)

        del local_images, local_metadata

        # Gather across ranks
        if world_size > 1:
            gathered_results = [None] * world_size
            dist.all_gather_object(gathered_results, local_results)

            if rank == 0:
                merged_results = []
                global_id = 0
                for rank_results in gathered_results:
                    for item in rank_results:
                        item["sample_id"] = global_id
                        merged_results.append(item)
                        global_id += 1

                merged_scores = defaultdict(list)
                for item in merged_results:
                    for name, val in item["scores"].items():
                        merged_scores[name].append(val)

                for name, values in merged_scores.items():
                    filtered = [v for v in values if v != -10.0]
                    if filtered:
                        all_scores[name] = float(np.mean(filtered))

                per_sample_results.extend(merged_results)
        else:
            for name, values in local_scores.items():
                filtered = [v for v in values if v != -10.0]
                if filtered:
                    all_scores[name] = float(np.mean(filtered))
            per_sample_results.extend(local_results)

    return all_scores, per_sample_results


def evaluate_fid(generation_registry, dataset_groups, args, device, rank, world_size, project_root):
    """Compute FID score against reference images."""
    if rank != 0:
        return {}

    logger.info("\n--- FID Evaluation ---")
    from cdm.rewards.fid_scorer import FIDScorer

    fid_scorer = FIDScorer(device=str(device), batch_size=args.scoring_batch_size)

    # Resolve FID dataset path
    fid_dataset_path = None
    for (dataset_path, dataset_type), metric_names in dataset_groups.items():
        if "fid" in metric_names:
            fid_dataset_path = dataset_path
            break
    if fid_dataset_path is None:
        fid_dataset_path = os.path.join(project_root, METRIC_REGISTRY["fid"]["default_dataset"])

    fid_reference_dir = os.path.join(fid_dataset_path, "images")
    fid_reference_stats = args.fid_reference_stats or os.path.join(fid_dataset_path, "inception_stats.npz")
    has_reference_dir = os.path.isdir(fid_reference_dir)
    has_reference_stats = os.path.isfile(fid_reference_stats)

    if not has_reference_stats and not has_reference_dir:
        logger.warning(f"FID reference not found at {fid_dataset_path}. Skipping FID.")
        del fid_scorer
        return {}

    gen_base_dir = generation_registry.get((fid_dataset_path, "text"))
    if gen_base_dir is None:
        logger.warning("No generated images for FID dataset. Skipping FID.")
        del fid_scorer
        return {}

    try:
        fid_value = fid_scorer.compute_fid_from_pt_files(
            batch_pt_dirs=[gen_base_dir],
            reference_dir=fid_reference_dir if has_reference_dir else None,
            reference_stats_path=fid_reference_stats if has_reference_stats else None,
        )
        logger.info(f"  FID: {fid_value:.4f}")
        return {"fid": fid_value}
    except Exception as error:
        logger.error(f"  FID computation failed: {error}")
        return {}
    finally:
        del fid_scorer
        gc.collect()


def evaluate_dpgbench(generation_registry, dataset_groups, args, device, rank, world_size, project_root):
    """Compute DPG-Bench scores."""
    if rank != 0:
        return {}

    logger.info("\n--- DPG-Bench Evaluation ---")
    from cdm.rewards.dpg_scorer import DPGScorer

    dpg_dataset_path = None
    for (dataset_path, dataset_type), metric_names in dataset_groups.items():
        if "dpgbench" in metric_names:
            dpg_dataset_path = dataset_path
            break
    if dpg_dataset_path is None:
        dpg_dataset_path = os.path.join(project_root, METRIC_REGISTRY["dpgbench"]["default_dataset"])

    dpg_csv_path = os.path.join(dpg_dataset_path, "dpg_bench.csv")
    if not os.path.isfile(dpg_csv_path):
        logger.warning(f"DPG-Bench CSV not found at {dpg_csv_path}. Skipping.")
        return {}

    gen_base_dir = generation_registry.get((dpg_dataset_path, "dpgbench"))
    if gen_base_dir is None:
        logger.warning("No generated images for DPG-Bench dataset. Skipping.")
        return {}

    dpg_scorer = DPGScorer(csv_path=dpg_csv_path, device=str(device))

    try:
        combined_images, _, combined_metadata = gather_all_generations(gen_base_dir)
        if len(combined_images) == 0:
            logger.warning("No images found for DPG-Bench. Skipping.")
            return {}

        item_ids = [
            meta.get("item_id", "") if isinstance(meta, dict) else ""
            for meta in combined_metadata
        ]
        if not any(item_ids):
            logger.warning("No item_ids in metadata. DPG-Bench requires DPGPromptDataset. Skipping.")
            return {}

        dpg_results = dpg_scorer.evaluate(images=combined_images, item_ids=item_ids)
        scores = {"dpgbench": dpg_results["overall_score"]}

        logger.info(f"  DPG-Bench overall: {scores['dpgbench']:.2f}")

        # Save detailed results
        detail_path = os.path.join(args.output_dir, "dpgbench_results.json")
        with open(detail_path, "w") as f:
            json.dump(dpg_results, f, indent=2, default=str)

        return scores
    except Exception as error:
        logger.error(f"  DPG-Bench evaluation failed: {error}")
        return {}
    finally:
        dpg_scorer.unload()
        del dpg_scorer, combined_images
        gc.collect()


def evaluate_diversity(diversity_generation_dir, args, device, rank, world_size):
    """Compute diversity metrics (Intra-LPIPS, Vendi Score)."""
    if rank != 0 or diversity_generation_dir is None:
        return {}

    diversity_metrics = [m for m in args.eval_metrics if METRIC_REGISTRY[m].get("diversity", False)]
    if not diversity_metrics:
        return {}

    logger.info(f"\n--- Diversity Metrics: {diversity_metrics} ---")
    from cdm.rewards.diversity_scorer import DiversityScorer

    diversity_scorer = DiversityScorer(device=str(device))
    images_by_prompt = _load_diversity_generations(diversity_generation_dir)

    if not images_by_prompt:
        logger.warning("No diversity data found. Skipping.")
        diversity_scorer.unload()
        del diversity_scorer
        return {}

    logger.info(f"  Loaded {len(images_by_prompt)} prompt groups")
    scores = {}

    if "intra_lpips" in diversity_metrics:
        try:
            value = diversity_scorer.compute_intra_lpips(images_by_prompt, batch_size=args.scoring_batch_size)
            scores["intra_lpips"] = value
            logger.info(f"  Intra-LPIPS: {value:.4f}")
        except Exception as error:
            logger.error(f"  Intra-LPIPS failed: {error}")

    if "vendi" in diversity_metrics:
        try:
            value = diversity_scorer.compute_vendi_score(images_by_prompt, batch_size=args.scoring_batch_size)
            scores["vendi"] = value
            logger.info(f"  Vendi Score: {value:.4f}")
        except Exception as error:
            logger.error(f"  Vendi Score failed: {error}")

    diversity_scorer.unload()
    del diversity_scorer, images_by_prompt
    gc.collect()

    return scores


def _load_diversity_generations(diversity_dir):
    """Load all per-prompt diversity .pt files from the generation directory."""
    images_by_prompt = {}
    if not os.path.isdir(diversity_dir):
        return images_by_prompt
    for fname in os.listdir(diversity_dir):
        if fname.startswith("prompt_") and fname.endswith(".pt"):
            data = torch.load(os.path.join(diversity_dir, fname), map_location="cpu", weights_only=False)
            images_by_prompt[data["prompt_hash"]] = data["images"]
    return images_by_prompt


def evaluate_llm_judge(
    generation_registry, dataset_groups, args,
    device, rank, world_size, project_root,
    model_adapter, mixed_precision_dtype,
):
    """Compute LLM Judge pairwise evaluation against a baseline model.

    If --baseline_output_dir is provided, loads baseline generations from there.
    Otherwise, generates baseline images using --baseline_path.
    """
    if rank != 0:
        return {}

    if not args.baseline_path and not args.baseline_output_dir:
        logger.error("LLM Judge requires --baseline_path or --baseline_output_dir. Skipping.")
        return {}

    logger.info(f"\n--- LLM Judge Pairwise Evaluation ---")
    from cdm.rewards.llm_judge_scorer import LLMJudgeScorer

    # Resolve the dataset used for llm_judge
    llm_judge_dataset_path, llm_judge_dataset_type = None, None
    for (dataset_path, dataset_type), metric_names in dataset_groups.items():
        if "llm_judge" in metric_names:
            llm_judge_dataset_path, llm_judge_dataset_type = dataset_path, dataset_type
            break
    if llm_judge_dataset_path is None:
        llm_judge_dataset_path = os.path.join(project_root, METRIC_REGISTRY["llm_judge"]["default_dataset"])
        llm_judge_dataset_type = "text"

    # Load model images
    gen_base_dir = generation_registry.get((llm_judge_dataset_path, llm_judge_dataset_type))
    if gen_base_dir is None:
        logger.warning("No generated images for LLM Judge dataset. Skipping.")
        return {}

    model_images, model_prompts, _ = gather_all_generations(gen_base_dir)
    if len(model_images) == 0:
        logger.warning("No model images found. Skipping LLM Judge.")
        return {}

    # Load or generate baseline images
    if args.baseline_output_dir:
        logger.info(f"  Loading baseline images from: {args.baseline_output_dir}")
        baseline_gen_dir = _find_baseline_generation_dir(
            args.baseline_output_dir, llm_judge_dataset_path
        )
        if baseline_gen_dir is None:
            logger.error("Could not find baseline generation data. Skipping LLM Judge.")
            del model_images
            return {}
        baseline_images, baseline_prompts, _ = gather_all_generations(baseline_gen_dir)
    else:
        logger.info(f"  Generating baseline images from: {args.baseline_path}")
        baseline_images, baseline_prompts = _generate_baseline_images(
            args, llm_judge_dataset_path, llm_judge_dataset_type,
            model_adapter, device, mixed_precision_dtype,
        )

    if len(baseline_images) == 0:
        logger.warning("No baseline images available. Skipping LLM Judge.")
        del model_images
        return {}

    # Align by prompt
    prompt_to_baseline_idx = {}
    for idx, prompt in enumerate(baseline_prompts):
        if prompt not in prompt_to_baseline_idx:
            prompt_to_baseline_idx[prompt] = idx

    matched_model_indices, matched_baseline_indices, matched_prompts = [], [], []
    for idx, prompt in enumerate(model_prompts):
        if prompt in prompt_to_baseline_idx:
            matched_model_indices.append(idx)
            matched_baseline_indices.append(prompt_to_baseline_idx[prompt])
            matched_prompts.append(prompt)

    if not matched_prompts:
        logger.warning("No matching prompts between model and baseline. Skipping LLM Judge.")
        del model_images, baseline_images
        return {}

    # Subsample if needed
    max_samples = args.llm_judge_max_samples
    if max_samples > 0 and len(matched_prompts) > max_samples:
        subsample = np.linspace(0, len(matched_prompts) - 1, max_samples, dtype=int).tolist()
        matched_model_indices = [matched_model_indices[i] for i in subsample]
        matched_baseline_indices = [matched_baseline_indices[i] for i in subsample]
        matched_prompts = [matched_prompts[i] for i in subsample]

    final_model_images = model_images[matched_model_indices]
    final_baseline_images = baseline_images[matched_baseline_indices]

    judge = LLMJudgeScorer(
        api_base_url=args.llm_judge_api_base,
        api_key=args.llm_judge_api_key,
        model_name=args.llm_judge_model,
        max_concurrent=args.llm_judge_max_concurrent,
        image_max_size=args.llm_judge_image_max_size,
    )

    try:
        judge_results = judge.evaluate(
            images_a=final_model_images,
            images_b=final_baseline_images,
            prompts=matched_prompts,
            role_a="model",
            role_b="baseline",
        )

        scores = {
            "llm_judge_win_rate_model": judge_results["win_rate_a"],
            "llm_judge_win_rate_baseline": judge_results["win_rate_b"],
            "llm_judge_tie_rate": judge_results["tie_rate"],
            "llm_judge_consistency_rate": judge_results["consistency_rate"],
            "llm_judge_total_comparisons": judge_results["total_comparisons"],
        }

        # Save detailed results
        judge_output_dir = os.path.join(args.output_dir, "llm_judge")
        os.makedirs(judge_output_dir, exist_ok=True)

        summary_results = {k: v for k, v in judge_results.items() if k != "per_sample"}
        with open(os.path.join(judge_output_dir, "llm_judge_summary.json"), "w") as f:
            json.dump(summary_results, f, indent=2, default=str)

        with open(os.path.join(judge_output_dir, "llm_judge_details.jsonl"), "w") as f:
            for sample_result in judge_results["per_sample"]:
                f.write(json.dumps(sample_result, default=str) + "\n")

        logger.info(f"  Win rate (model): {scores['llm_judge_win_rate_model']:.2%}")
        logger.info(f"  Win rate (baseline): {scores['llm_judge_win_rate_baseline']:.2%}")
        logger.info(f"  Tie rate: {scores['llm_judge_tie_rate']:.2%}")

        return scores
    except Exception as error:
        logger.error(f"  LLM Judge failed: {error}")
        return {}
    finally:
        judge.unload()
        del judge, model_images, baseline_images, final_model_images, final_baseline_images
        gc.collect()


def _find_baseline_generation_dir(baseline_output_dir, dataset_path):
    """Find the generation directory for a dataset within a previous eval output."""
    dataset_name = os.path.basename(dataset_path.rstrip("/"))
    candidate = os.path.join(baseline_output_dir, "_generations", dataset_name)
    if os.path.isdir(candidate):
        return candidate
    return None


def _generate_baseline_images(
    args, dataset_path, dataset_type, model_adapter, device, mixed_precision_dtype,
):
    """Generate baseline images for LLM Judge comparison."""
    baseline_gen_dir = os.path.join(args.output_dir, "_baseline_generations")

    if args.use_cache and _has_cached_batches(baseline_gen_dir):
        logger.info(f"  Cache hit: reusing existing baseline generations in {baseline_gen_dir}")
        images, prompts, _ = load_all_generation_batches(baseline_gen_dir)
        return images, prompts

    pipeline = load_pipeline(args.baseline_path, model_adapter, device, mixed_precision_dtype)
    pipeline.set_progress_bar_config(position=2, disable=True, leave=False, desc="Timestep", dynamic_ncols=True)

    generate_for_dataset_group(
        pipeline=pipeline,
        inf_cfg=InferenceConfig(
            height=args.height, width=args.width,
            model_type=_detect_model_type(args.baseline_path),
        ),
        num_steps=args.baseline_num_steps,
        guidance_scale=args.baseline_guidance_scale,
        dataset_path=dataset_path, dataset_type=dataset_type,
        device=device, rank=0, world_size=1,
        mixed_precision_dtype=mixed_precision_dtype,
        eval_batch_size=args.eval_batch_size,
        generation_dir=baseline_gen_dir,
        max_eval_samples=args.max_eval_samples,
    )

    unload_pipeline(pipeline)
    del pipeline
    gc.collect()
    torch.cuda.empty_cache()

    images, prompts, _ = load_all_generation_batches(baseline_gen_dir)
    return images, prompts


def _detect_model_type(model_path):
    """Detect model type from pipeline config if available."""
    config_path = os.path.join(model_path, "model_index.json")
    if os.path.isfile(config_path):
        with open(config_path) as f:
            config = json.load(f)
        class_name = config.get("_class_name", "").lower()
        if "longcat" in class_name:
            return "longcat"
    return "sd3"


def run_evaluation(
    generation_registry, diversity_generation_dir, dataset_groups,
    args, device, rank, world_size, project_root,
    model_adapter, mixed_precision_dtype,
):
    """Phase 2: Compute all metrics on generated images.

    Returns:
        final_scores: dict of metric_name -> score
    """
    logger.info(f"\n{'='*60}")
    logger.info("PHASE 2: EVALUATION")
    logger.info(f"{'='*60}")

    final_scores = {}
    per_sample_results = []

    # Standard per-image metrics
    standard_scores, standard_results = evaluate_standard_metrics(
        generation_registry, dataset_groups, device, rank, world_size, args.scoring_batch_size,
    )
    if rank == 0:
        final_scores.update(standard_scores)
        per_sample_results.extend(standard_results)

    # Post-hoc metrics
    post_hoc_metrics = [m for m in args.eval_metrics if METRIC_REGISTRY[m].get("post_hoc", False)]

    if "fid" in post_hoc_metrics:
        fid_scores = evaluate_fid(
            generation_registry, dataset_groups, args, device, rank, world_size, project_root,
        )
        final_scores.update(fid_scores)

    if "dpgbench" in post_hoc_metrics:
        dpg_scores = evaluate_dpgbench(
            generation_registry, dataset_groups, args, device, rank, world_size, project_root,
        )
        final_scores.update(dpg_scores)

    # Diversity metrics
    diversity_scores = evaluate_diversity(
        diversity_generation_dir, args, device, rank, world_size,
    )
    final_scores.update(diversity_scores)

    # LLM Judge
    if "llm_judge" in post_hoc_metrics:
        llm_judge_scores = evaluate_llm_judge(
            generation_registry, dataset_groups, args,
            device, rank, world_size, project_root,
            model_adapter, mixed_precision_dtype,
        )
        final_scores.update(llm_judge_scores)

    # Save results
    if rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)

        if per_sample_results:
            results_path = os.path.join(args.output_dir, "evaluation_results.jsonl")
            with open(results_path, "w") as f:
                for item in per_sample_results:
                    f.write(json.dumps(item, default=str) + "\n")

        scores_path = os.path.join(args.output_dir, "average_scores.json")
        with open(scores_path, "w") as f:
            json.dump(final_scores, f, indent=4)

        _print_scores(final_scores)

    return final_scores

# ==================== Output Helpers ====================

def _print_scores(scores):
    """Print formatted scores."""
    logger.info(f"\n{'='*60}")
    logger.info("EVALUATION RESULTS")
    logger.info(f"{'='*60}")
    for name, score in sorted(scores.items()):
        if isinstance(score, float):
            logger.info(f"  {name:<30}: {score:.4f}")
        else:
            logger.info(f"  {name:<30}: {score}")

# ==================== Main ====================

def _save_generation_metadata(output_dir, dataset_groups, args):
    """Save metadata needed to reconstruct generation_registry in evaluate-only phase."""
    metadata = {
        "dataset_groups": {
            f"{path}|{dtype}": metrics
            for (path, dtype), metrics in dataset_groups.items()
        },
        "eval_metrics": args.eval_metrics,
        "diversity_num_variations": args.diversity_num_variations,
        "diversity_num_prompts": args.diversity_num_prompts,
    }
    metadata_path = os.path.join(output_dir, "_generation_metadata.json")
    os.makedirs(output_dir, exist_ok=True)
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)


def _load_generation_metadata(output_dir):
    """Load metadata saved during generation phase."""
    metadata_path = os.path.join(output_dir, "_generation_metadata.json")
    if not os.path.isfile(metadata_path):
        return None
    with open(metadata_path) as f:
        return json.load(f)


def _rebuild_generation_registry(output_dir, dataset_groups):
    """Rebuild generation_registry from cached generation directories."""
    generation_registry = {}
    gen_base = os.path.join(output_dir, "_generations")

    for (dataset_path, dataset_type), metric_names in dataset_groups.items():
        dataset_name = os.path.basename(dataset_path.rstrip("/"))
        gen_dir = os.path.join(gen_base, dataset_name)
        if _has_cached_batches(gen_dir):
            generation_registry[(dataset_path, dataset_type)] = gen_dir

    return generation_registry


def main(args):
    """Orchestrate the evaluation pipeline with phase control."""
    # Validate arguments
    if args.phase in ("all", "generate") and not args.model_path:
        raise ValueError("--model_path is required for 'generate' and 'all' phases.")

    # Distributed setup
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    # Only initialize process group when world_size > 1 or when running in a distributed environment
    if world_size > 1 or os.environ.get("MASTER_ADDR") is not None:
        dist.init_process_group("nccl", rank=rank, world_size=world_size, timeout=timedelta(seconds=14400))
    
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    mixed_precision_dtype = None
    if args.mixed_precision == "fp16":
        mixed_precision_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        mixed_precision_dtype = torch.bfloat16

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # scripts/eval.py -> repo root is two levels up.
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    run_generate = args.phase in ("all", "generate")
    run_evaluate = args.phase in ("all", "evaluate")

    # Resolve datasets per metric
    dataset_overrides = {}
    if args.dataset_overrides:
        for override in args.dataset_overrides:
            parts = override.split("=", 1)
            if len(parts) == 2:
                dataset_overrides[parts[0]] = parts[1]

    dataset_groups = resolve_datasets(args.eval_metrics, args.default_dataset, dataset_overrides, project_root)

    if rank == 0:
        if args.model_path:
            logger.info(f"Model: {args.model_path}")
        logger.info(f"Phase: {args.phase}")
        logger.info(f"Metrics: {args.eval_metrics}")
        logger.info(f"Output: {args.output_dir}")

    # ==================== Generation Phase ====================
    generation_registry = {}
    diversity_generation_dir = None

    if run_generate:
        model_adapter = create_model_adapter(args.base_model or args.model_path)

        # Read custom_sigmas from student_pipeline_meta.json if available.
        # This allows the eval to automatically use the same denoising schedule
        # that was configured during training, without any extra CLI flags.
        custom_sigmas = None
        deterministic = True  # default: ODE (deterministic) sampling
        meta_path = os.path.join(args.model_path, "student_pipeline_meta.json")
        if os.path.isfile(meta_path):
            with open(meta_path) as meta_file:
                pipeline_meta = json.load(meta_file)
            custom_sigmas = pipeline_meta.get("custom_sigmas", None)
            if custom_sigmas is not None:
                if args.ignore_custom_sigmas:
                    logger.info(
                        f"Ignoring custom_sigmas from student_pipeline_meta.json (len="
                        f"{len(custom_sigmas)}) due to --ignore_custom_sigmas; using uniform "
                        f"schedule with num_steps={args.num_steps}."
                    )
                    custom_sigmas = None
                else:
                    # custom_sigmas (when passed to the pipeline as `sigmas`) overrides
                    # `num_inference_steps`, so a mismatch silently makes --num_steps a no-op.
                    # Refuse to run instead of producing misleading results.
                    if len(custom_sigmas) != args.num_steps:
                        raise ValueError(
                            f"--num_steps={args.num_steps} does not match the length of "
                            f"custom_sigmas ({len(custom_sigmas)}) loaded from "
                            f"{meta_path}. The custom_sigmas would silently override "
                            f"--num_steps. Either set --num_steps={len(custom_sigmas)} to "
                            f"reproduce the training schedule, or pass --ignore_custom_sigmas "
                            f"to fall back to a uniform schedule with the requested num_steps."
                        )
                    logger.info(f"Loaded custom_sigmas from student_pipeline_meta.json: {custom_sigmas}")
            meta_deterministic = pipeline_meta.get("deterministic", None)
            if meta_deterministic is not None:
                deterministic = meta_deterministic
                logger.info(f"Loaded deterministic from student_pipeline_meta.json: {deterministic}")

        inf_cfg = InferenceConfig(
            height=args.height,
            width=args.width,
            model_type=_detect_model_type(args.model_path),
            custom_sigmas=custom_sigmas,
            deterministic=deterministic,
        )

        # Check cache
        all_cached = False
        if args.use_cache:
            all_cached = True
            for (dataset_path, dataset_type), _ in dataset_groups.items():
                dataset_name = os.path.basename(dataset_path.rstrip("/"))
                gen_dir = os.path.join(args.output_dir, "_generations", dataset_name)
                if not _has_cached_batches(gen_dir):
                    all_cached = False
                    break
            if all_cached:
                diversity_metrics = [m for m in args.eval_metrics if METRIC_REGISTRY[m].get("diversity", False)]
                if diversity_metrics:
                    diversity_dir = os.path.join(args.output_dir, "_diversity_generations")
                    if not _has_cached_diversity(diversity_dir):
                        all_cached = False

        if all_cached:
            logger.info("All generation caches found. Skipping model loading.")
            generation_registry, diversity_generation_dir = run_generation(
                pipeline=None, inf_cfg=inf_cfg,
                num_steps=args.num_steps, guidance_scale=args.guidance_scale,
                dataset_groups=dataset_groups, args=args,
                device=device, rank=rank, world_size=world_size,
                mixed_precision_dtype=mixed_precision_dtype,
            )
        else:
            pipeline = load_pipeline(args.model_path, model_adapter, device, mixed_precision_dtype)
            pipeline.set_progress_bar_config(position=2, disable=rank != 0, leave=False, desc="Timestep", dynamic_ncols=True)

            generation_registry, diversity_generation_dir = run_generation(
                pipeline=pipeline, inf_cfg=inf_cfg,
                num_steps=args.num_steps, guidance_scale=args.guidance_scale,
                dataset_groups=dataset_groups, args=args,
                device=device, rank=rank, world_size=world_size,
                mixed_precision_dtype=mixed_precision_dtype,
            )

            unload_pipeline(pipeline)
            del pipeline
            gc.collect()
            torch.cuda.empty_cache()

        # Save metadata for evaluate-only phase
        if rank == 0:
            _save_generation_metadata(args.output_dir, dataset_groups, args)

        if args.phase == "generate":
            if rank == 0:
                logger.info(f"\n{'='*60}")
                logger.info("GENERATION PHASE COMPLETE")
                logger.info(f"{'='*60}")
                logger.info(f"Cached generations saved to: {args.output_dir}")
                if args.save_images:
                    logger.info(f"Saved images to: {os.path.join(args.output_dir, 'saved_images')}")
            if world_size > 1:
                dist.barrier()
                dist.destroy_process_group()
            return

    # ==================== Evaluation Phase ====================
    if run_evaluate:
        model_adapter = create_model_adapter(args.base_model or "sd3")

        # Rebuild registry from cache if evaluate-only
        if args.phase == "evaluate":
            metadata = _load_generation_metadata(args.output_dir)
            if metadata is None:
                raise RuntimeError(
                    f"No generation metadata found at {args.output_dir}. "
                    "Run with --phase generate first."
                )
            generation_registry = _rebuild_generation_registry(args.output_dir, dataset_groups)
            if not generation_registry:
                raise RuntimeError(
                    f"No cached generations found in {args.output_dir}/_generations/. "
                    "Run with --phase generate first."
                )

            diversity_metrics = [m for m in args.eval_metrics if METRIC_REGISTRY[m].get("diversity", False)]
            if diversity_metrics:
                diversity_generation_dir = os.path.join(args.output_dir, "_diversity_generations")
                if not _has_cached_diversity(diversity_generation_dir):
                    diversity_generation_dir = None

        final_scores = run_evaluation(
            generation_registry=generation_registry,
            diversity_generation_dir=diversity_generation_dir,
            dataset_groups=dataset_groups, args=args,
            device=device, rank=rank, world_size=world_size,
            project_root=project_root,
            model_adapter=model_adapter,
            mixed_precision_dtype=mixed_precision_dtype,
        )

        if rank == 0:
            if args.save_images:
                logger.info(f"Saved visualization images to: {os.path.join(args.output_dir, 'saved_images')}")
            logger.info(f"Generation cache preserved in: {args.output_dir}")

    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a single diffusion model pipeline against multiple metrics.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Phase control
    parser.add_argument(
        "--phase", type=str, choices=["all", "generate", "evaluate"], default="all",
        help="Which phase to run: 'generate' for image generation only, 'evaluate' for metrics only, 'all' for both.",
    )

    # Model
    parser.add_argument(
        "--model_path", type=str, default=None,
        help="Path to the diffusers pipeline directory. Required for 'generate' and 'all' phases.",
    )
    parser.add_argument(
        "--base_model", type=str, default="sd3",
        help="Base model identifier for selecting the model adapter. If not set, inferred from --model_path.",
    )
    parser.add_argument("--height", type=int, default=1024, help="Image height for generation.")
    parser.add_argument("--width", type=int, default=1024, help="Image width for generation.")
    parser.add_argument("--num_steps", type=int, default=4, help="Number of denoising steps.")
    parser.add_argument("--guidance_scale", type=float, default=1.0, help="Classifier-free guidance scale.")
    parser.add_argument(
        "--ignore_custom_sigmas", action="store_true", default=False,
        help="Ignore custom_sigmas from student_pipeline_meta.json and use the uniform schedule "
             "derived from --num_steps. Useful for sweeping inference steps that differ from the "
             "training-time custom_sigmas length.",
    )

    # Metrics
    parser.add_argument(
        "--eval_metrics", type=str, nargs="+", default=["imagereward", "clipscore"],
        choices=list(METRIC_REGISTRY.keys()),
        help="Evaluation metrics to compute.",
    )

    # Dataset
    parser.add_argument("--default_dataset", type=str, default=None, help="Default dataset path for text-type metrics.")
    parser.add_argument("--dataset_overrides", type=str, nargs="*", default=None, help="Per-metric dataset overrides: metric=path.")
    parser.add_argument("--max_eval_samples", type=int, default=-1, help="Max evaluation samples per dataset. -1 for all.")

    # Output
    parser.add_argument("--output_dir", type=str, default="./logs/eval", help="Directory to save evaluation results.")
    parser.add_argument(
        "--save_images", action="store_true", default=False,
        help="Export generated images as PNG files for visualization (saved to output_dir/saved_images/).",
    )
    parser.add_argument(
        "--use_cache", action="store_true", default=False,
        help="Reuse cached .pt generation files from a previous run instead of regenerating images.",
    )

    # Batch sizes
    parser.add_argument("--eval_batch_size", type=int, default=1, help="Batch size per GPU for image generation.")
    parser.add_argument("--scoring_batch_size", type=int, default=1, help="Batch size for scoring phase.")
    parser.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"], help="Mixed precision mode.")

    # FID
    parser.add_argument("--fid_reference_stats", type=str, default=None, help="Path to precomputed Inception stats (.npz) for FID.")

    # Diversity metrics
    parser.add_argument("--diversity_num_variations", type=int, default=16, help="Number of noise variations per prompt for diversity metrics.")
    parser.add_argument("--diversity_num_prompts", type=int, default=128, help="Number of prompts to sample for diversity metrics.")

    # LLM Judge (pairwise comparison with baseline)
    parser.add_argument(
        "--baseline_path", type=str, default=None,
        help="Path to baseline model pipeline for LLM Judge pairwise comparison. Typically the teacher model.",
    )
    parser.add_argument(
        "--baseline_output_dir", type=str, default=None,
        help="Path to a previous eval output_dir for the baseline model. Reuses its generated images instead of regenerating.",
    )
    parser.add_argument("--baseline_num_steps", type=int, default=50, help="Denoising steps for baseline model generation.")
    parser.add_argument("--baseline_guidance_scale", type=float, default=4.5, help="Guidance scale for baseline model generation.")
    parser.add_argument(
        "--llm_judge_api_base", type=str,
        default="https://idealab.alibaba-inc.com/api/openai/v1/chat/completions",
        help="OpenAI-compatible API base URL for LLM Judge.",
    )
    parser.add_argument("--llm_judge_api_key", type=str, default=None, help="API key for LLM Judge.")
    parser.add_argument("--llm_judge_model", type=str, default="qwen-vl-max", help="Model name for LLM Judge API calls.")
    parser.add_argument("--llm_judge_max_samples", type=int, default=256, help="Max prompt pairs for LLM Judge. -1 for all.")
    parser.add_argument("--llm_judge_max_concurrent", type=int, default=8, help="Max concurrent API requests for LLM Judge.")
    parser.add_argument("--llm_judge_image_max_size", type=int, default=1024, help="Max image dimension for LLM Judge.")

    args = parser.parse_args()
    main(args)