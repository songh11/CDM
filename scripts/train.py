from collections import defaultdict
import os
import json
import datetime
from concurrent import futures
import time
import math
from functools import partial
from absl import app, flags
from PIL import Image

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from ml_collections import config_flags
import tqdm
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed as accelerate_set_seed, DistributedType
from accelerate.logging import get_logger

import cdm.rewards.rewards
from cdm.models.ema import EMAModuleWrapper
from cdm.utils.logger import create_logger, log_fake_teacher_metrics
from cdm.models.model_adapters import create_model_adapter, wrap_longcat_transformer_for_fsdp
from cdm.pipeline.pipeline import pipeline_infer
from cdm.utils.common import (
    set_seed, _repeat_embeds, sample_sigmas, enable_full_determinism,
    get_mixed_precision_dtype, create_autocast_context,
)
from cdm.data.datasets import DATASET_REGISTRY, DistributedKRepeatSampler
from cdm.evaluation import eval_distillation_fn, eval_fake_teacher_viz_fn, shutdown_image_save_executor
from cdm.training import (
    compute_cfg_loss,
    compute_fake_teacher_diffusion_loss, compute_dm_loss,
    compute_batch_mean_var_kl_loss,
    ModelMode, ModelComponents, get_model_mode, create_frozen_model,
    LORA_TARGET_MODULES, initialize_models, is_student_lora, is_ft_lora, has_fake_teacher,
    ema_merge_ft_with_student,
    ModelManager, patch_model_for_pipeline,
    save_fsdp_full_checkpoint, compute_grad_stats, create_lr_lambda, build_fsdp_plugin_if_needed,
)

tqdm = partial(tqdm.tqdm, dynamic_ncols=True)

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "config/config.py", "Training configuration.")
flags.DEFINE_string("unique_id", "-1", "Unique experiment ID passed from submit.py. '-1' means not assigned.")

logger = get_logger(__name__)


def extract_traj(data_list, indices_cpu):
    """Extract per-sample data from a trajectory list using per-sample indices."""
    return torch.stack([data_list[idx][i] for i, idx in enumerate(indices_cpu)])


def make_indices(shared_cpu, batch_size, num_steps, device):
    """Return per-sample trajectory indices as (tensor, cpu_list).
    """
    if shared_cpu is not None:
        idx_cpu = shared_cpu
        idx_tensor = torch.tensor(idx_cpu, device=device, dtype=torch.long)
    else:
        idx_tensor = torch.randint(0, num_steps, (batch_size,), device=device)
        idx_cpu = idx_tensor.cpu().tolist()
    return idx_tensor, idx_cpu


def main(_):
    config = FLAGS.config
    assert config.sample.guidance_scale == 1.0

    mixed_precision = config.mixed_precision or "no"
     
    # Unified experiment directory: logs and checkpoints live together
    experiment_dir = os.path.join(config.output_dir, config.run_name)
    log_dir = os.path.join(experiment_dir, "tensorboard")
    checkpoint_dir = os.path.join(experiment_dir, "checkpoints")
    
    # Build FSDP plugin dynamically based on model type (only when using FSDP backend)
    fsdp_plugin = build_fsdp_plugin_if_needed(config)
    
    accelerator = Accelerator(
        mixed_precision=mixed_precision,
        gradient_accumulation_steps=config.train.gradient_accumulation_steps,
        project_config=ProjectConfiguration(project_dir=experiment_dir, logging_dir=log_dir),
        fsdp_plugin=fsdp_plugin,
    )
    
    device = accelerator.device
    rank, world_size = accelerator.process_index, accelerator.num_processes
    
    logger.info(f"Mixed precision: {mixed_precision}", main_process_only=True)

    # Logger Init
    exp_logger = None
    if accelerator.is_main_process:
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)
        config_dict = config.to_dict()
        config_dict.update({"world_size": world_size})
        
        # Save config snapshot for reproducibility
        config_save_path = os.path.join(experiment_dir, "config.json")
        with open(config_save_path, "w") as f:
            json.dump(config_dict, f, indent=2, default=str)
        logger.info(f"Saved config to {config_save_path}")
        
        exp_logger = create_logger(
            log_dir=log_dir, run_name=config.run_name, config=config_dict,
            logger_type=config.logger_type,
            project=config.wandb_project,
            entity=config.wandb_entity, api_key=config.wandb_api_key,
        )
        
    # ==================== Seed & Determinism ====================
    if config.deterministic:
        # Full deterministic mode: reproducible across runs (slower due to deterministic algorithms)
        # Each rank uses seed+rank so different GPUs produce different noise, but the same
        # rank always produces the same sequence across runs.
        enable_full_determinism(config.seed + rank)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        logger.info(f"🔒 Deterministic mode ON: seed={config.seed}, rank_seed={config.seed + rank}, TF32=disabled", main_process_only=True)
    else:
        # Performance mode: set seed for basic reproducibility, allow non-deterministic algorithms
        accelerate_set_seed(config.seed + rank, deterministic=False)
        set_seed(config.seed, rank=rank, deterministic=False)
        if config.allow_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        logger.info(f"⚡ Performance mode: seed={config.seed}, rank_seed={config.seed + rank}, TF32={config.allow_tf32}", main_process_only=True)

    mixed_precision_dtype = get_mixed_precision_dtype(config.mixed_precision)
    frozen_model_dtype = mixed_precision_dtype or torch.float32
    text_encoder_dtype = mixed_precision_dtype or torch.float32
    train_config = config.train
    student_config = train_config.student
    # Fake teacher node may be absent in config (FT feature can be turned off entirely).
    fake_teacher_config = getattr(train_config, 'fake_teacher', None)
    fake_teacher_enabled = fake_teacher_config is not None and fake_teacher_config.enabled
    # if not fake_teacher_enabled:
    #     raise ValueError("Fake teacher must be enabled. Set train.fake_teacher.enabled=True in config.")
    student_use_lora = student_config.use_lora

    # ==================== Create Model Adapter ====================
    base_model_type = config.base_model
    model_adapter = create_model_adapter(base_model_type, config)
    model_type_str = model_adapter.get_model_type()  # e.g. "sd3" or "longcat"
    logger.info(f"Model adapter: {base_model_type} (pipeline model_type={model_type_str})", main_process_only=True)

    # ==================== Load Pipeline ====================
    pipeline_kwargs = {}
    if config.attn_implementation == "flash_attention_2":
        pipeline_kwargs["torch_dtype"] = mixed_precision_dtype or torch.bfloat16
    pipeline = model_adapter.load_pipeline(config.pretrained.model, **pipeline_kwargs)

    if hasattr(pipeline, 'safety_checker'):
        pipeline.safety_checker = None
    pipeline.set_progress_bar_config(position=1, disable=not accelerator.is_main_process, leave=False, desc="Timestep", dynamic_ncols=True)
    
    # Freeze non-trainable components
    model_adapter.freeze_non_trainable(pipeline)
    pipeline.transformer.requires_grad_(not student_use_lora)
    
    text_encoders = model_adapter.get_text_encoders(pipeline)
    tokenizers = model_adapter.get_tokenizers(pipeline)

    # Move models to device
    pipeline.vae.to(device, dtype=torch.float32)
    for enc in text_encoders:
        enc.to(device, dtype=text_encoder_dtype)
    transformer = pipeline.transformer.to(device)

    # Memory optimizations
    if train_config.enable_xformers:
        try:
            pipeline.transformer.enable_xformers_memory_efficient_attention()
            pipeline.vae.enable_xformers_memory_efficient_attention()
            logger.info("😊 [Memory优化] xformers enabled", main_process_only=True)
        except Exception as e:
            logger.warning(f"Failed to enable xformers: {e}")
    if train_config.vae_slicing:
        pipeline.vae.enable_slicing()
        logger.info("😊 [Memory优化] VAE slicing enabled", main_process_only=True)

    # ==================== Model Initialization ====================
    # Guard: gradient_accumulation_steps > 1 is not supported with fake teacher training.
    # FT and Student share the same accelerator accumulation counter, causing sync timing conflicts.
    if train_config.gradient_accumulation_steps > 1:
        raise ValueError(
            f"gradient_accumulation_steps={train_config.gradient_accumulation_steps} is not supported "
            f"with fake teacher training. FT and Student share the same accelerator accumulation "
            f"counter, which causes gradient sync timing conflicts. Please set gradient_accumulation_steps=1."
        )

    # Determine model mode and initialize all components.
    # When FT is disabled (fake_teacher_config is None), default ft_use_lora to True
    # (value is irrelevant since FT branch will be skipped, but get_model_mode requires a bool).
    ft_use_lora = fake_teacher_config.use_lora if fake_teacher_config is not None else True
    
    model_mode = get_model_mode(fake_teacher_enabled, ft_use_lora, student_use_lora)
    logger.info(f"Model mode: {model_mode.name}", main_process_only=True)

    # Guard: LoRA mode is not supported with FSDP
    is_fsdp = accelerator.distributed_type == accelerator.distributed_type.FSDP
    if is_fsdp and (student_use_lora or ft_use_lora):
        raise ValueError(
            f"LoRA mode is not supported with FSDP (student_use_lora={student_use_lora}, "
            f"ft_use_lora={ft_use_lora}). Please use DDP for LoRA training, or switch to "
            f"full fine-tuning mode for FSDP."
        )
    
    model_components = initialize_models(
        pipeline=pipeline, device=device, mode=model_mode,
        student_config=student_config, fake_teacher_config=fake_teacher_config,
        frozen_model_dtype=frozen_model_dtype,
        lora_target_modules=model_adapter.get_lora_target_modules(
            include_ff=student_config.lora_include_ff
        ),
    )
    
    transformer = model_components.transformer
    transformer_trainable_parameters = model_components.transformer_trainable_params
    fake_teacher = model_components.fake_teacher
    fake_teacher_trainable_parameters = model_components.fake_teacher_trainable_params
    
    # Log model setup info
    logger.info(f"Fake Teacher initialized: mode={'LoRA' if is_ft_lora(model_mode) else 'Full'}", main_process_only=True)
    
    # Gradient checkpointing for student
    if student_config.gradient_checkpointing:
        base_model = transformer.base_model.model if is_student_lora(model_mode) else transformer
        if hasattr(base_model, 'enable_gradient_checkpointing'):
            base_model.enable_gradient_checkpointing()
            logger.info("😊 [Memory优化] Student gradient checkpointing enabled", main_process_only=True)
    
    # Gradient checkpointing for fake teacher (only when FT is enabled)
    if fake_teacher_enabled and fake_teacher_config.gradient_checkpointing \
            and fake_teacher is not None and fake_teacher is not transformer:
        ft_base_model = fake_teacher.base_model.model if hasattr(fake_teacher, 'base_model') else fake_teacher
        if hasattr(ft_base_model, 'enable_gradient_checkpointing'):
            ft_base_model.enable_gradient_checkpointing()
            logger.info("😊 [Memory优化] Fake Teacher gradient checkpointing enabled", main_process_only=True)

    # ==================== Training Statistics ====================
    train_batch_size = train_config.batch_size
    num_batches_per_epoch = config.sample.num_batches_per_epoch
    grad_accum = train_config.gradient_accumulation_steps
    
    # Student training steps
    student_update_ratio = student_config.update_ratio
    student_batches_per_epoch = num_batches_per_epoch
    student_steps_per_epoch = student_batches_per_epoch // grad_accum
    # Calculate total steps across all epochs, then divide by update_ratio
    # This correctly handles cases where steps_per_epoch < update_ratio
    student_total_raw_steps = student_steps_per_epoch * config.num_epochs
    total_steps = math.ceil(student_total_raw_steps / student_update_ratio) if student_update_ratio > 0 else 0
    student_effective_steps_per_epoch = total_steps / config.num_epochs if config.num_epochs > 0 else 0  # For logging only
    
    # Fake Teacher training steps (Fake Teacher trains every batch, no update_ratio)
    ft_batches_per_epoch = num_batches_per_epoch
    ft_steps_per_epoch = ft_batches_per_epoch // grad_accum
    ft_total_steps = ft_steps_per_epoch * config.num_epochs

    # ==================== Student Optimizer & LR Scheduler ====================
    adam_base_kwargs = dict(betas=(train_config.adam_beta1, train_config.adam_beta2),
                            weight_decay=train_config.adam_weight_decay, eps=train_config.adam_epsilon)
    
    student_lr = student_config.learning_rate if is_student_lora(model_mode) else student_config.full_learning_rate
    student_adam_kwargs = dict(lr=student_lr, **adam_base_kwargs)
    if student_config.use_8bit_adam:
        try:
            import bitsandbytes as bnb
            optimizer = bnb.optim.AdamW8bit(transformer_trainable_parameters, **student_adam_kwargs)
            logger.info("😊 [Memory优化] Student using 8-bit AdamW", main_process_only=True)
        except ImportError:
            optimizer = torch.optim.AdamW(transformer_trainable_parameters, **student_adam_kwargs)
    else:
        optimizer = torch.optim.AdamW(transformer_trainable_parameters, **student_adam_kwargs)

    student_warmup_steps = student_config.warmup_steps
    if student_warmup_steps == 0 and student_config.warmup_ratio > 0:
        student_warmup_steps = int(total_steps * student_config.warmup_ratio)
    student_lr_scheduler_type = student_config.lr_scheduler_type
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, create_lr_lambda(student_warmup_steps, total_steps, student_lr_scheduler_type))
    logger.info(f"Student: lr={student_lr}, scheduler={student_lr_scheduler_type}, warmup={student_warmup_steps}, total_steps={total_steps}", main_process_only=True)

    # ==================== Fake Teacher Optimizer & LR Scheduler ====================
    fake_teacher_optimizer, fake_teacher_lr_scheduler = None, None
    ft_step = 0
    ft_lr = None
    ft_lr_scheduler_type = None
    ft_warmup_steps = 0
    if fake_teacher_enabled:
        ft_lr = fake_teacher_config.learning_rate if is_ft_lora(model_mode) else fake_teacher_config.full_learning_rate
        ft_adam_kwargs = dict(lr=ft_lr, **adam_base_kwargs)

        if fake_teacher_config.use_8bit_adam:
            try:
                import bitsandbytes as bnb
                fake_teacher_optimizer = bnb.optim.AdamW8bit(fake_teacher_trainable_parameters, **ft_adam_kwargs)
                logger.info("😊 [Memory优化] Fake Teacher using 8-bit AdamW", main_process_only=True)
            except ImportError:
                fake_teacher_optimizer = torch.optim.AdamW(fake_teacher_trainable_parameters, **ft_adam_kwargs)
        else:
            fake_teacher_optimizer = torch.optim.AdamW(fake_teacher_trainable_parameters, **ft_adam_kwargs)

        ft_warmup_steps = fake_teacher_config.warmup_steps
        if ft_warmup_steps == 0 and fake_teacher_config.warmup_ratio > 0:
            ft_warmup_steps = int(ft_total_steps * fake_teacher_config.warmup_ratio)
        ft_lr_scheduler_type = fake_teacher_config.lr_scheduler_type
        fake_teacher_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(fake_teacher_optimizer, create_lr_lambda(ft_warmup_steps, ft_total_steps, ft_lr_scheduler_type))
        logger.info(f"Fake Teacher: lr={ft_lr}, mode={'LoRA' if is_ft_lora(model_mode) else 'full'}, scheduler={ft_lr_scheduler_type}, warmup={ft_warmup_steps}, total_steps={ft_total_steps}", main_process_only=True)

    # ==================== Datasets and Dataloaders ====================
    max_sample_samples = config.max_sample_samples
    max_eval_samples = config.eval.max_eval_samples
    sample_dataset_path = config.sample_dataset
    eval_dataset_path = config.eval_dataset
    
    sample_prompt_fn = config.prompt_fn
    eval_prompt_fn = config.eval_prompt_fn
    
    SampleDatasetClass = DATASET_REGISTRY.get(sample_prompt_fn)
    EvalDatasetClass = DATASET_REGISTRY.get(eval_prompt_fn)
    if not SampleDatasetClass:
        raise NotImplementedError(f"Unknown prompt_fn '{sample_prompt_fn}'. Available: {list(DATASET_REGISTRY.keys())}")
    if not EvalDatasetClass:
        raise NotImplementedError(f"Unknown eval_prompt_fn '{eval_prompt_fn}'. Available: {list(DATASET_REGISTRY.keys())}")

    # Build extra kwargs for datasets that need image-related parameters.
    # image_base_path may be empty string, in which case it falls back to None.
    image_size = (config.height, config.width)
    image_base_path = config.image_base_path or None
    image_kwargs = dict(image_size=image_size, image_base_path=image_base_path)
    image_dataset_types = {"text_image_pair", "image_edit"}
    sample_dataset_kwargs = image_kwargs if sample_prompt_fn in image_dataset_types else {}
    eval_dataset_kwargs = image_kwargs if eval_prompt_fn in image_dataset_types else {}

    sample_dataset = SampleDatasetClass(sample_dataset_path, "train", max_samples=max_sample_samples, **sample_dataset_kwargs)
    eval_dataset = EvalDatasetClass(eval_dataset_path, "test", max_samples=max_eval_samples, **eval_dataset_kwargs)
    logger.info(f"Dataset sizes - sample: {len(sample_dataset)}, eval: {len(eval_dataset)}", main_process_only=True)

    sample_sampler = DistributedKRepeatSampler(
        dataset=sample_dataset, batch_size=train_batch_size,
        k=config.sample.num_image_per_prompt, num_replicas=world_size, rank=rank, seed=config.seed,
    )
    sample_dataloader = DataLoader(sample_dataset, batch_sampler=sample_sampler, num_workers=0, collate_fn=sample_dataset.collate_fn, pin_memory=True)
    eval_sampler = DistributedSampler(eval_dataset, num_replicas=world_size, rank=rank, shuffle=False) if world_size > 1 else None
    eval_dataloader = DataLoader(eval_dataset, batch_size=config.eval.eval_batch_size, sampler=eval_sampler, collate_fn=eval_dataset.collate_fn, num_workers=0, pin_memory=True)

    executor = futures.ThreadPoolExecutor(max_workers=8)

    # ==================== Prepare with Accelerator ====================
    transformer, optimizer, lr_scheduler = accelerator.prepare(transformer, optimizer, lr_scheduler)
    patch_model_for_pipeline(transformer, accelerator.unwrap_model(transformer))
    pipeline.transformer = transformer
    transformer_trainable_parameters = [p for p in transformer.parameters() if p.requires_grad]

    # Prepare fake teacher with Accelerator.
    # For FT_LORA_STUDENT_LORA mode, FT shares the same transformer (handled by adapter switching).
    if fake_teacher_optimizer is not None and not model_mode == ModelMode.FT_LORA_STUDENT_LORA:
        fake_teacher, fake_teacher_optimizer, fake_teacher_lr_scheduler = accelerator.prepare(
            fake_teacher, fake_teacher_optimizer, fake_teacher_lr_scheduler
        )
        patch_model_for_pipeline(fake_teacher, accelerator.unwrap_model(fake_teacher))
        pipeline.fake_teacher = fake_teacher
        fake_teacher_trainable_parameters = [p for p in fake_teacher.parameters() if p.requires_grad]
        logger.info("😊 Fake Teacher prepared with Accelerator", main_process_only=True)

    # Prepare real teacher with Accelerator (FSDP compatibility).
    real_teacher = pipeline.real_teacher  # May be None for LoRA mode

    # ==================== LongCat ids cast proxy ====================
    if config.base_model == "longcat":
        pipeline.transformer = wrap_longcat_transformer_for_fsdp(pipeline.transformer)
        if hasattr(pipeline, "fake_teacher") and pipeline.fake_teacher is not None:
            pipeline.fake_teacher = wrap_longcat_transformer_for_fsdp(pipeline.fake_teacher)
        if real_teacher is not None:
            real_teacher = wrap_longcat_transformer_for_fsdp(real_teacher)
            pipeline.real_teacher = real_teacher
        logger.info("🐈 LongCat ids cast proxy applied to transformer/fake_teacher/real_teacher",
                    main_process_only=True)

    # ==================== Prompt Embeddings ====================
    # Pre-compute negative embeddings AFTER accelerator.prepare() so that
    neg_prompt_embed, neg_auxiliary_text_embed = model_adapter.compute_text_embeddings([""], text_encoders, tokenizers, max_sequence_length=model_adapter.get_max_sequence_length(), device=device)
    neg_prompt_embeds = neg_prompt_embed.repeat(train_batch_size, 1, 1)
    neg_auxiliary_text_embeds = _repeat_embeds(neg_auxiliary_text_embed, train_batch_size)
    # Pre-compute negative model kwargs for CFG forward calls
    neg_model_kwargs = model_adapter.prepare_forward_kwargs(neg_prompt_embeds, neg_auxiliary_text_embeds)

    # ==================== Initialize Model Manager ====================
    # Update model_components with prepared (accelerator-wrapped) models.
    model_components = ModelComponents(
        transformer=transformer,
        transformer_trainable_params=transformer_trainable_parameters,
        fake_teacher=fake_teacher,
        fake_teacher_trainable_params=fake_teacher_trainable_parameters,
        real_teacher=real_teacher,
        mode=model_mode
    )
    
    model_manager = ModelManager(
        pipeline=pipeline,
        mode=model_mode,
        accelerator=accelerator,
        device=device,
        model_components=model_components,
        model_adapter=model_adapter,
    )
    
    logger.info("😊 ModelManager initialized", main_process_only=True)

    logger.info(f"***** Running training *****", main_process_only=True)
    
    logger.info(f"***** Training Config *****", main_process_only=True)
    logger.info(f"  General: world_size={world_size}, epochs={config.num_epochs}, grad_accum={grad_accum}", main_process_only=True)
    logger.info(f"  Train: per_gpu_bs={train_batch_size}, effective_bs={train_batch_size * world_size * grad_accum}, batches/epoch={num_batches_per_epoch}", main_process_only=True)
    logger.info(f"  Eval: per_gpu_bs={config.eval.eval_batch_size}, effective_bs={config.eval.eval_batch_size * world_size}, freq={config.eval_freq} epochs", main_process_only=True)
    logger.info(f"  Student: mode={'LoRA' if is_student_lora(model_mode) else 'Full'}, steps/epoch={student_steps_per_epoch}, update_ratio={student_update_ratio}, effective_steps/epoch={student_effective_steps_per_epoch:.2f}, total_steps={total_steps}, lr={student_lr}", main_process_only=True)
    if fake_teacher_enabled:
        logger.info(f"  FakeTeacher: mode={'LoRA' if is_ft_lora(model_mode) else 'Full'}, steps/epoch={ft_steps_per_epoch}, total_steps={ft_total_steps}", main_process_only=True)
        logger.info(f"  FakeTeacher: lr={ft_lr}, lr_scheduler={ft_lr_scheduler_type}, warmup={ft_warmup_steps}", main_process_only=True)
    else:
        logger.info("  FakeTeacher: disabled", main_process_only=True)
    logger.info(f"  Output: {log_dir}", main_process_only=True)

    eval_reward_config = config.eval.reward_fn or config.train.reward_fn
    eval_reward_fn = getattr(cdm.rewards.rewards, "multi_score")(device, eval_reward_config)

    # Offload reward models to CPU during training to save GPU memory.
    # They will be loaded back to GPU on-demand during eval phases.
    from cdm.rewards.rewards import get_scorer_models
    reward_scorer_models = get_scorer_models(eval_reward_fn)
    for score_name, scorer in reward_scorer_models:
        scorer.cpu()
        if hasattr(scorer, 'device'):
            scorer.device = torch.device('cpu')
        logger.info(f"Offloaded reward model '{score_name}' to CPU to save GPU memory", main_process_only=True)
    if reward_scorer_models:
        torch.cuda.empty_cache()

    # Resume from checkpoint
    first_epoch, student_step = 0, 0
    if config.resume_from:
        accelerator.load_state(config.resume_from)
        try:
            student_step = int(os.path.basename(config.resume_from).split("-")[-1])
        except ValueError:
            student_step = 0
        logger.info(f"Resumed from {config.resume_from}, student_step={student_step}", main_process_only=True)

    ema_decay = student_config.ema_decay
    ema_device = device  # Keep EMA on GPU for faster updates (no cross-device copy)
    if config.train.student.ema:
        # FSDP2: EMA operates on local shards via _get_data() in EMAModuleWrapper
        # DDP/single-GPU: operates on full parameters directly
        ema = EMAModuleWrapper(transformer_trainable_parameters, decay=ema_decay, update_step_interval=1, device=ema_device)
        logger.info(f"😊 EMA parameters stored on GPU (uses ~{sum(p.numel() * p.element_size() for p in transformer_trainable_parameters) / 1024**3:.1f} GB GPU memory)", main_process_only=True)
    else:
        ema = None

    # Load EMA state from checkpoint if available
    if config.resume_from and ema is not None:
        ema_path = os.path.join(config.resume_from, "ema_state.pt")
        if os.path.exists(ema_path):
            ema.load_state_dict(torch.load(ema_path, map_location=device))
            logger.info("Loaded EMA state from checkpoint", main_process_only=True)
        else:
            logger.warning("No EMA state found in checkpoint, EMA initialized from current model", main_process_only=True)

    sample_iter = iter(sample_dataloader)
    optimizer.zero_grad()

    # Note: old model EMA update is now performed after each student step in the training loop

    train_start_time = time.time()

    # ==================== Config node aliases (for concise access in the loop) ====================
    sample_config = config.sample
    loss_config = config.train.loss
    cfg_loss = loss_config.cfg
    ddm_loss = loss_config.ddm
    cdm_loss = loss_config.cdm
    # FT alias is only valid when fake_teacher is enabled.
    ft = fake_teacher_config if fake_teacher_enabled else None

    # Global sigma truncation bounds
    sigma_min = train_config.sigma_min
    sigma_max = train_config.sigma_max

    if fake_teacher_enabled and ft.ema_from_student:
        logger.info(f"FT←Student EMA merge: enabled, decay={ft.ema_from_student_decay}, interval={ft.ema_from_student_interval}", main_process_only=True)
    
# Custom sigmas for denoising schedule (used by both training sampling and eval).
    # Set config.custom_sigmas to None to use the default schedule from num_inference_steps.
    custom_sigmas = list(config.custom_sigmas) if config.custom_sigmas is not None else None
    if custom_sigmas is not None:
        logger.info(f"Custom sigmas: {custom_sigmas}", main_process_only=True)
    
    # Random sigma schedule: randomize sigmas each epoch for ODE discretization generalization
    if sample_config.random_sigma_schedule:
        logger.info(f"Random sigma schedule: enabled, steps=[{sample_config.random_sigma_num_steps_min}, {sample_config.random_sigma_num_steps_max}], "
                     f"sigma_range=[{sample_config.random_sigma_min}, {sample_config.random_sigma_max}]", main_process_only=True)
    
    if cdm_loss.teacher_x0_base == "student_pred":
        assert cdm_loss.teacher_xt_mode != "share_student", (
            "cdm.teacher_x0_base='student_pred' is incompatible with cdm.teacher_xt_mode='share_student' "
            "(share_student forces teacher to reuse student's xt, leaving no room for a re-noised x0_pred). "
            "Use 'fresh_shared_t' or 'fresh_independent' instead."
        )
    if cdm_loss.teacher_xt_mode == "share_student":
        assert cdm_loss.teacher_x0_base == cdm_loss.student_x0_base, (
            f"cdm.teacher_xt_mode='share_student' requires teacher_x0_base ({cdm_loss.teacher_x0_base!r}) "
            f"to equal student_x0_base ({cdm_loss.student_x0_base!r})."
        )
    logger.info(f"CDM xt mode: student={cdm_loss.student_xt_mode}, teacher={cdm_loss.teacher_xt_mode}, "
                f"student_x0_base={cdm_loss.student_x0_base}, teacher_x0_base={cdm_loss.teacher_x0_base}",
                main_process_only=True)

    logger.info(f"Loss: cfg={cfg_loss.enabled}(w={cfg_loss.weight}), ddm={ddm_loss.enabled}(w={ddm_loss.weight}), cdm={cdm_loss.enabled}(w={cdm_loss.weight})", main_process_only=True)


    epoch_pbar = tqdm(range(first_epoch, config.num_epochs), desc="Training Progress", 
                       position=0, disable=not accelerator.is_main_process)
    
    for epoch in epoch_pbar:
        epoch_pbar.set_description(f"Epoch {epoch}/{config.num_epochs - 1}")
        
        # --- Merged Training Loop Body ---
        torch.cuda.reset_peak_memory_stats(device)
        
        unwrapped_transformer = accelerator.unwrap_model(transformer)
        model_dtype = next(unwrapped_transformer.parameters()).dtype
        
        # ========== Evaluation Phase ==========
        should_eval = epoch % config.eval_freq == 0
        if config.eval.skip_first_eval and epoch == 0:
            should_eval = False
        
        if should_eval:
            if config.eval.distillation_enabled:
                eval_distillation_fn(pipeline, eval_dataloader, text_encoders, tokenizers, config,
                    device, rank, world_size, student_step, eval_reward_fn, executor,
                    mixed_precision_dtype, ema, transformer_trainable_parameters, exp_logger, transformer_ddp=transformer, model_adapter=model_adapter, accelerator=accelerator)
            # ========== Fake Teacher Visualization ==========
            ft_viz_config = config.eval.fake_teacher_viz
            ft_viz_enabled = ft_viz_config is not None and ft_viz_config.enabled
            should_viz_fake_teacher = ft_viz_enabled
            
            if should_viz_fake_teacher:
                eval_fake_teacher_viz_fn(
                    pipeline=pipeline, test_dataloader=eval_dataloader, text_encoders=text_encoders, tokenizers=tokenizers,
                    config=config, device=device, rank=rank, world_size=world_size, global_step=student_step, epoch=epoch,
                    mixed_precision_dtype=mixed_precision_dtype, exp_logger=exp_logger, accelerator=accelerator,
                    transformer_ddp=transformer, model_adapter=model_adapter,
                )

        # ========== Checkpoint Saving ==========
        if epoch % config.save_freq == 0 and epoch > 0:
            checkpoint_save_path = os.path.join(checkpoint_dir, f"checkpoint-{student_step}")
            if accelerator.is_main_process:
                os.makedirs(checkpoint_save_path, exist_ok=True)
            accelerator.wait_for_everyone()
            accelerator.save_state(checkpoint_save_path)
            if ema is not None and accelerator.is_main_process:
                torch.save(ema.state_dict(), os.path.join(checkpoint_save_path, "ema_state.pt"))
            # FSDP: save full (unsharded) state dicts for prepare_student_pipeline compatibility
            save_fsdp_full_checkpoint(accelerator, transformer, ema, checkpoint_save_path, logger)
            if accelerator.is_main_process:
                logger.info(f"Saved checkpoint to {checkpoint_save_path}")

        if accelerator.is_main_process and exp_logger is not None:
            exp_logger.add_scalar("global/epoch", epoch, student_step)

        # ========== Unified Sample-Train Loop ==========
        # Each batch: sample → FT train → Student train (no collect-then-split)
        ft_model_for_training = transformer if model_mode == ModelMode.FT_LORA_STUDENT_LORA else fake_teacher
        gradient_update_times = 0
        info_accumulated = defaultdict(list)

        batch_pbar = tqdm(range(num_batches_per_epoch), desc=f"Epoch {epoch}",
                          position=0, disable=not accelerator.is_main_process)

        for batch_idx in batch_pbar:
            # ---- 1. Sampling ----
            unwrapped_transformer.eval()
            if hasattr(sample_sampler, "set_epoch") and isinstance(sample_sampler, DistributedKRepeatSampler):
                sample_sampler.set_epoch(epoch * num_batches_per_epoch + batch_idx)

            prompts, prompt_metadata = next(sample_iter)

            num_steps = config.sample.num_steps
            model_manager.prepare_for_sampling(sample_config.sampling_adapter)
            current_batch_size = len(prompts)

            with create_autocast_context(accelerator), torch.no_grad():
                pipeline_kwargs = dict(
                    prompt=prompts,
                    num_inference_steps=num_steps, guidance_scale=config.sample.guidance_scale,
                    height=config.height, width=config.width,
                    deterministic=config.sample.deterministic,
                )

                # Random sigma schedule: generate new random sigmas each epoch.
                if sample_config.random_sigma_schedule:
                    import random as _random
                    _len_rng = _random.Random((epoch, batch_idx))
                    rand_num_steps = _len_rng.randint(sample_config.random_sigma_num_steps_min, sample_config.random_sigma_num_steps_max)
                    inner_sigmas = sorted(
                        [_random.uniform(sample_config.random_sigma_min, sample_config.random_sigma_max) for _ in range(rand_num_steps - 1)],
                        reverse=True,
                    )
                    epoch_sigmas = [1.0] + inner_sigmas
                    pipeline_kwargs["sigmas"] = epoch_sigmas
                    pipeline_kwargs["num_inference_steps"] = rand_num_steps
                    num_steps = rand_num_steps
                # Custom sigmas override the default schedule from num_inference_steps
                elif custom_sigmas is not None:
                    pipeline_kwargs["sigmas"] = custom_sigmas

                # LongCat: pass enable_prompt_rewrite for training sampling
                if config.base_model == "longcat":
                    pipeline_kwargs["enable_prompt_rewrite"] = config.eval.enable_prompt_rewrite

                # backward simulation
                (_, all_xt, all_x0, all_v, prompt_embeds, auxiliary_text_embeds,
                 latent_image_ids) = pipeline_infer(pipeline, return_images=False, **pipeline_kwargs)

            # Pre-generate shared trajectory indices for this batch when enabled.
            if train_config.share_latent_indices:
                shared_latent_indices_cpu = torch.randint(
                    0, num_steps, (current_batch_size,), device=device).cpu().tolist()
            else:
                shared_latent_indices_cpu = None

            # ---- 2. Fake Teacher Training ----
            model_manager.prepare_for_ft_training()

            # Extract FT target x0 from trajectory based on ft.x0_source config
            # FT samples its own trajectory indices independently (only when needed)
            if ft.x0_source == "x0_final":
                ft_batch_x0 = all_xt[-1].to(model_dtype)
            else:
                _, ft_indices_cpu = make_indices(
                    shared_latent_indices_cpu, current_batch_size, num_steps, device)
                if ft.x0_source == "x0_next":
                    ft_batch_x0 = extract_traj(all_x0, [idx + 1 for idx in ft_indices_cpu]).to(model_dtype)
                else:  # x0_current
                    ft_batch_x0 = extract_traj(all_x0, ft_indices_cpu).to(model_dtype)
            ft_batch_prompt_embeds = prompt_embeds.to(model_dtype)
            ft_batch_auxiliary_embeds = auxiliary_text_embeds if auxiliary_text_embeds is not None else None
            ft_current_batch_size = ft_batch_x0.shape[0]

            # Note: latent_image_ids is global (per-pipeline-call), independent of trajectory indices.
            ft_model_kwargs = model_adapter.prepare_forward_kwargs(
                ft_batch_prompt_embeds, ft_batch_auxiliary_embeds, img_ids=latent_image_ids)

            ft_sigma = sample_sigmas(
                ft_current_batch_size, ft.sigma_sampling_method, device,
                sigma_min=sigma_min, sigma_max=sigma_max,
                logit_mean=ft.logit_mean, logit_std=ft.logit_std)
            ft_sigma_expanded = ft_sigma.view(-1, *([1] * (len(ft_batch_x0.shape) - 1)))
            ft_timestep = (ft_sigma * 1000.0).float()

            ft_noise = torch.randn_like(ft_batch_x0)
            ft_xt = (1 - ft_sigma_expanded) * ft_batch_x0.detach() + ft_sigma_expanded * ft_noise

            ft_xt = ft_xt.to(model_dtype)
            with accelerator.accumulate(ft_model_for_training):
                with create_autocast_context(accelerator):
                    ft_prediction = model_manager.fake_teacher_forward(
                        ft_xt, ft_timestep, requires_grad=True, **ft_model_kwargs)
                    ft_diffusion_loss, ft_diffusion_info = compute_fake_teacher_diffusion_loss(
                        fake_teacher_prediction=ft_prediction, x0=ft_batch_x0.detach(), noise=ft_noise, t_expanded=ft_sigma_expanded)
                    scaled_loss = loss_config.diffusion.weight * ft_diffusion_loss

                accelerator.backward(scaled_loss)

            if accelerator.sync_gradients:
                if fake_teacher_optimizer:
                    ft_grad_stats = compute_grad_stats(parameters=fake_teacher_trainable_parameters)
                    if accelerator.distributed_type == DistributedType.FSDP:
                        torch.nn.utils.clip_grad_norm_(fake_teacher_trainable_parameters, config.train.max_grad_norm)
                    else:
                        accelerator.clip_grad_norm_(fake_teacher_trainable_parameters, config.train.max_grad_norm)
                    fake_teacher_optimizer.step()
                    if fake_teacher_lr_scheduler is not None:
                        fake_teacher_lr_scheduler.step()
                    fake_teacher_optimizer.zero_grad()

                    # FT ← Student EMA merge: blend student params into FT after gradient update
                    if ft.ema_from_student and ft_step % ft.ema_from_student_interval == 0:
                        ema_merge_ft_with_student(
                            ft_params=fake_teacher_trainable_parameters,
                            student_params=transformer_trainable_parameters,
                            decay=ft.ema_from_student_decay,
                        )

                ft_step += 1

                if accelerator.is_main_process and exp_logger is not None:
                    current_ft_lr = fake_teacher_lr_scheduler.get_last_lr()[0] if fake_teacher_lr_scheduler is not None else ft_lr
                    log_fake_teacher_metrics(exp_logger, ft_step, ft_diffusion_loss, ft_sigma,
                                             ft_diffusion_info, ft_grad_stats, current_ft_lr, ft_current_batch_size)

                del ft_prediction, ft_xt, ft_noise

            model_manager.finish_ft_training()
            batch_pbar.set_postfix({"ft_loss": f"{ft_diffusion_loss.item():.4f}", "σ": f"{ft_sigma.mean().item():.3f}", "ft_step": ft_step})

            # ---- 3. Student Training ----
            should_train_student = (student_update_ratio == 1) or (ft_step > 0 and ft_step % student_update_ratio == 0)
            if not should_train_student:
                continue

            model_manager.prepare_for_student_training()

            batch_prompt_embeds = prompt_embeds.to(model_dtype)
            batch_auxiliary_embeds = auxiliary_text_embeds if auxiliary_text_embeds is not None else None
            batch_model_kwargs = model_adapter.prepare_forward_kwargs(
                batch_prompt_embeds, batch_auxiliary_embeds, img_ids=latent_image_ids)

            # Shared with CDM (used as x0_final base when cdm_*_x0_base = "x0_final"). Cheap, no indexing.
            batch_x0_final = all_xt[-1].to(model_dtype)

            current_batch_size = len(batch_prompt_embeds)
            # Build negative model kwargs for CFG
            batch_neg_model_kwargs = dict(neg_model_kwargs)
            # Inject latent_image_ids into neg kwargs (img_ids is the same for cond/uncond)
            if latent_image_ids is not None:
                batch_neg_model_kwargs["img_ids"] = latent_image_ids

            with accelerator.accumulate(transformer):
                loss_terms = {}
                zero_tensor = torch.tensor(0.0, device=device)
                total_loss = torch.tensor(0.0, device=device)

                # ========== Loss 1: CFG on_trajectory ==========
                if cfg_loss.enabled and cfg_loss.weight != 0:
                    with create_autocast_context(accelerator):
                        # Sample trajectory indices for CFG (shared across branches when enabled)
                        cfg_latent_indices, cfg_latent_indices_cpu = make_indices(
                            shared_latent_indices_cpu, current_batch_size, num_steps, device)
                        cfg_batch_xt = extract_traj(all_xt, cfg_latent_indices_cpu).float()
                        cfg_batch_stop_timesteps = pipeline.scheduler.timesteps[cfg_latent_indices.cpu()].to(device).float()
                        cfg_stop_sigma_base = (cfg_batch_stop_timesteps / 1000.0).float()
                        cfg_stop_sigma_base_expanded = cfg_stop_sigma_base.view(-1, *([1] * (len(cfg_batch_xt.shape) - 1)))

                        cfg_student_sigma = cfg_stop_sigma_base
                        cfg_student_xt = cfg_batch_xt
                        cfg_student_sigma_expanded = cfg_stop_sigma_base_expanded
                        cfg_student_timesteps = cfg_batch_stop_timesteps
                        
                        cfg_student_prediction = model_manager.student_forward(
                            cfg_student_xt.to(model_dtype), cfg_student_timesteps, 
                            **batch_model_kwargs)
                        
                        cfg_x0_pred = cfg_student_xt - cfg_student_sigma_expanded * cfg_student_prediction
                        
                        # Teacher re-noise base: always use student's online x0 prediction.
                        cfg_x0_for_teacher = cfg_x0_pred
                        # Teacher sigma upper bound: always the student's stop sigma.
                        cfg_teacher_upper_bound = cfg_student_sigma
                        
                        cfg_teacher_sigma = sample_sigmas(
                            current_batch_size, cfg_loss.teacher_sigma_method, device,
                            sigma_min=sigma_min, sigma_max=sigma_max,
                            logit_mean=cfg_loss.teacher_sigma_logit_mean, logit_std=cfg_loss.teacher_sigma_logit_std,
                            upper_bound=cfg_teacher_upper_bound)
                        cfg_teacher_sigma_expanded = cfg_teacher_sigma.view(-1, *([1] * (len(cfg_x0_for_teacher.shape) - 1)))
                        cfg_teacher_timesteps = (cfg_teacher_sigma * 1000.0).float()
                        cfg_teacher_noise = torch.randn_like(cfg_x0_for_teacher)
                        cfg_teacher_xt = (1 - cfg_teacher_sigma_expanded) * cfg_x0_for_teacher.detach() + cfg_teacher_sigma_expanded * cfg_teacher_noise
                        
                        cfg_uncond_pred, cfg_cond_pred = model_manager.teacher_forward_cfg_batched(
                            cfg_teacher_xt.to(model_dtype), cfg_teacher_timesteps,
                            batch_model_kwargs, batch_neg_model_kwargs)
                        
                        cfg_loss_val, cfg_loss_info = compute_cfg_loss(
                            student_prediction=cfg_student_prediction,
                            real_teacher_cond_prediction=cfg_cond_pred,
                            real_teacher_uncond_prediction=cfg_uncond_pred,
                            xt=cfg_student_xt, xt_teacher=cfg_teacher_xt,
                            student_sigma_expanded=cfg_student_sigma_expanded,
                            teacher_sigma_expanded=cfg_teacher_sigma_expanded,
                            guidance_scale=cfg_loss.guidance_scale,
                            return_per_sample=False,
                            norm_clip_min=cfg_loss.norm_clip_min)
                        
                        cfg_loss_weighted = cfg_loss.weight * cfg_loss_val
                        total_loss = total_loss + cfg_loss_weighted
                        loss_terms.update({f"cfg/{k}": v for k, v in cfg_loss_info.items()})
                        loss_terms.update({"loss/cfg_raw": cfg_loss_val.detach(), "loss/cfg": cfg_loss_weighted.detach()})

                else:
                    loss_terms["loss/cfg"] = zero_tensor

                # ========== Loss 2: DM on_trajectory ==========
                if ddm_loss.enabled and ddm_loss.weight != 0:
                    with create_autocast_context(accelerator):
                        # Sample trajectory indices for DDM (shared across branches when enabled)
                        ddm_latent_indices, ddm_latent_indices_cpu = make_indices(
                            shared_latent_indices_cpu, current_batch_size, num_steps, device)
                        # Extract DDM trajectory data on-demand
                        ddm_batch_xt = extract_traj(all_xt, ddm_latent_indices_cpu)
                        ddm_batch_stop_timesteps = pipeline.scheduler.timesteps[ddm_latent_indices.cpu()].to(device)
                        ddm_student_timesteps = ddm_batch_stop_timesteps.float()
                        ddm_stop_sigma = (ddm_student_timesteps / 1000.0).float()
                        
                        ddm_student_sigma = ddm_stop_sigma
                        ddm_student_xt = ddm_batch_xt.float().to(model_dtype)
                        ddm_student_sigma_expanded = ddm_stop_sigma.view(-1, *([1] * (len(ddm_student_xt.shape) - 1)))
                        
                        ddm_student_prediction = model_manager.student_forward(
                            ddm_student_xt, ddm_student_timesteps, **batch_model_kwargs)
                        
                        ddm_x0_pred = ddm_student_xt - ddm_student_sigma_expanded * ddm_student_prediction
                        
                        # Teacher re-noise base: always use student's online x0 prediction.
                        ddm_x0_for_teacher = ddm_x0_pred
                        # Teacher sigma upper bound: always the student's stop sigma.
                        ddm_teacher_upper_bound = ddm_student_sigma
                        
                        ddm_teacher_sigma = sample_sigmas(
                            current_batch_size, ddm_loss.teacher_sigma_method, device,
                            sigma_min=sigma_min, sigma_max=sigma_max,
                            logit_mean=ddm_loss.teacher_sigma_logit_mean, logit_std=ddm_loss.teacher_sigma_logit_std,
                            upper_bound=ddm_teacher_upper_bound)
                        ddm_teacher_sigma_expanded = ddm_teacher_sigma.view(-1, *([1] * (len(ddm_x0_for_teacher.shape) - 1)))
                        ddm_teacher_timesteps = (ddm_teacher_sigma * 1000.0).float()
                        ddm_teacher_noise = torch.randn_like(ddm_x0_for_teacher)
                        ddm_teacher_xt = (1 - ddm_teacher_sigma_expanded) * ddm_x0_for_teacher.detach() + ddm_teacher_sigma_expanded * ddm_teacher_noise
                        
                        ddm_real_teacher_prediction = model_manager.teacher_forward(
                            ddm_teacher_xt.to(model_dtype), ddm_teacher_timesteps,
                            **batch_model_kwargs)
                        ddm_fake_teacher_prediction = model_manager.fake_teacher_forward(
                            ddm_teacher_xt.to(model_dtype), ddm_teacher_timesteps,
                            requires_grad=False, **batch_model_kwargs)
                        
                        # Get CFG gradient direction for alignment analysis (if CFG loss was computed)
                        ddm_cfg_grad_direction = cfg_loss_info.get("cfg_grad_direction", None) if cfg_loss.enabled and cfg_loss.weight != 0 else None
                        
                        ddm_loss_val, ddm_loss_info = compute_dm_loss(
                            student_prediction=ddm_student_prediction,
                            real_teacher_prediction=ddm_real_teacher_prediction,
                            fake_teacher_prediction=ddm_fake_teacher_prediction,
                            xt=ddm_student_xt, xt_teacher=ddm_teacher_xt,
                            student_sigma_expanded=ddm_student_sigma_expanded,
                            teacher_sigma_expanded=ddm_teacher_sigma_expanded,
                            return_per_sample=False,
                            cfg_grad_direction=ddm_cfg_grad_direction,
                            norm_clip_min=ddm_loss.norm_clip_min)
                        
                        ddm_loss_weighted = ddm_loss.weight * ddm_loss_val
                        total_loss = total_loss + ddm_loss_weighted
                        loss_terms.update({f"ddm/{k}": v for k, v in ddm_loss_info.items()})
                        loss_terms.update({"loss/ddm_raw": ddm_loss_val.detach(), "loss/ddm": ddm_loss_weighted.detach()})

                else:
                    loss_terms["loss/ddm"] = zero_tensor

                # ========== Loss 3: DM fresh_noise ==========
                if cdm_loss.enabled and cdm_loss.weight != 0:
                    with create_autocast_context(accelerator):
                        # ---------- Step A: trajectory indices (if needed) and student x0_base tensor ----------
                        if cdm_loss.student_xt_mode in ("trajectory_eps", "trajectory_eps_last", "trajectory_xt"):
                            if cdm_loss.student_xt_mode == "trajectory_eps_last":
                                # Always pick the last trajectory step (closest to x0) instead of random sampling.
                                # This branch is intentionally NOT shared (it is deterministic by design).
                                cdm_traj_indices_cpu = [num_steps - 1] * current_batch_size
                            else:
                                _, cdm_traj_indices_cpu = make_indices(
                                    shared_latent_indices_cpu, current_batch_size, num_steps, device)
                            if cdm_loss.student_x0_base == "x0_next":
                                cdm_student_x0_tensor = extract_traj(
                                    all_x0, [i + 1 for i in cdm_traj_indices_cpu]).to(model_dtype)
                            else:  # x0_final
                                cdm_student_x0_tensor = all_xt[-1].to(model_dtype)
                        else:
                            # "fresh" mode: independently sample x0_next if needed, else use x0_final
                            if cdm_loss.student_x0_base == "x0_next":
                                _, cdm_x0_next_indices_cpu = make_indices(
                                    shared_latent_indices_cpu, current_batch_size, num_steps, device)
                                cdm_student_x0_tensor = extract_traj(
                                    all_x0, [i + 1 for i in cdm_x0_next_indices_cpu]).to(model_dtype)
                            else:
                                cdm_student_x0_tensor = batch_x0_final

                        # ---------- Step B: build student (sigma, xt) by mode ----------
                        if cdm_loss.student_xt_mode == "trajectory_xt":
                            # Directly reuse the on-trajectory point: (traj_xt, σ_traj)
                            cdm_student_sigma = torch.tensor(
                                [pipeline.scheduler.sigmas[i].item() for i in cdm_traj_indices_cpu],
                                device=device, dtype=torch.float32)
                            cdm_student_xt = extract_traj(all_xt, cdm_traj_indices_cpu)
                        elif cdm_loss.student_xt_mode in ("trajectory_eps", "trajectory_eps_last"):
                            # Reuse on-trajectory noise direction (v); independently sampled σ_s with v-extrapolation:
                            # xt = traj_xt + (σ_s - σ_traj) * traj_v
                            # For "trajectory_eps_last", traj index is fixed to the last step (closest to x0).
                            cdm_traj_xt = extract_traj(all_xt, cdm_traj_indices_cpu)
                            cdm_traj_v = extract_traj(all_v, cdm_traj_indices_cpu)
                            cdm_traj_stop_sigma = torch.tensor(
                                [pipeline.scheduler.sigmas[i].item() for i in cdm_traj_indices_cpu],
                                device=device, dtype=torch.float32)
                            cdm_student_sigma = sample_sigmas(
                                current_batch_size, cdm_loss.student_sigma_method, device,
                                sigma_min=sigma_min, sigma_max=sigma_max,
                                logit_mean=cdm_loss.student_sigma_logit_mean,
                                logit_std=cdm_loss.student_sigma_logit_std,
                                upper_bound=cdm_traj_stop_sigma)
                            sigma_diff = (cdm_student_sigma - cdm_traj_stop_sigma).view(
                                -1, *([1] * (cdm_traj_xt.ndim - 1)))
                            cdm_student_xt = cdm_traj_xt + sigma_diff * cdm_traj_v
                        else:  # "fresh"
                            # Off-trajectory: xt = (1 - σ_s) * x0_base + σ_s * ε_fresh
                            cdm_student_sigma = sample_sigmas(
                                current_batch_size, cdm_loss.student_sigma_method, device,
                                sigma_min=sigma_min, sigma_max=sigma_max,
                                logit_mean=cdm_loss.student_sigma_logit_mean,
                                logit_std=cdm_loss.student_sigma_logit_std)
                            sigma_exp = cdm_student_sigma.view(-1, *([1] * (cdm_student_x0_tensor.ndim - 1)))
                            cdm_noise = torch.randn_like(cdm_student_x0_tensor)
                            cdm_student_xt = (1 - sigma_exp) * cdm_student_x0_tensor.detach() + sigma_exp * cdm_noise

                        # ---------- Step C: derive expanded sigma & timesteps ----------
                        cdm_student_sigma_expanded = cdm_student_sigma.view(
                            -1, *([1] * (cdm_student_xt.ndim - 1)))
                        cdm_student_timesteps = (cdm_student_sigma * 1000.0).float()

                        # ---------- Step E: student forward (moved earlier so teacher can consume student's x0_pred) ----------
                        cdm_student_prediction = model_manager.student_forward(
                            cdm_student_xt.to(model_dtype), cdm_student_timesteps,
                            **batch_model_kwargs)

                        # ---------- Step A2: teacher x0_base tensor (deferred until student_forward) ----------
                        # When teacher_xt_mode == "share_student", teacher must reuse student's x0 (no decoupling possible;
                        # already validated above that teacher_x0_base == student_x0_base in this case).
                        if cdm_loss.teacher_xt_mode == "share_student":
                            cdm_teacher_x0_tensor = cdm_student_x0_tensor
                        elif cdm_loss.teacher_x0_base == "student_pred":
                            # Use student's online x0 prediction as the teacher's x0 base.
                            # Detach to prevent gradients flowing back through the teacher's noised input.
                            cdm_teacher_x0_tensor = (
                                cdm_student_xt - cdm_student_sigma_expanded * cdm_student_prediction
                            ).detach().to(model_dtype)
                        elif cdm_loss.teacher_x0_base == "x0_next":
                            # Sample trajectory indices for teacher's x0_next (shared across branches when enabled)
                            _, teacher_traj_indices_cpu = make_indices(
                                shared_latent_indices_cpu, current_batch_size, num_steps, device)
                            cdm_teacher_x0_tensor = extract_traj(
                                all_x0, [i + 1 for i in teacher_traj_indices_cpu]).to(model_dtype)
                        else:  # x0_final
                            cdm_teacher_x0_tensor = all_xt[-1].to(model_dtype)

                        # ---------- Step D: build teacher (sigma, xt) by mode ----------
                        if cdm_loss.teacher_xt_mode == "share_student":
                            # Teacher sees exactly the same (xt, t) as student
                            cdm_teacher_sigma = cdm_student_sigma
                            cdm_teacher_sigma_expanded = cdm_student_sigma_expanded
                            cdm_teacher_timesteps = cdm_student_timesteps
                            cdm_teacher_xt = cdm_student_xt
                        elif cdm_loss.teacher_xt_mode == "fresh_shared_t":
                            # Same t as student, but xt rebuilt from x0_base + fresh noise (decouples noise)
                            cdm_teacher_sigma = cdm_student_sigma
                            cdm_teacher_sigma_expanded = cdm_student_sigma_expanded
                            cdm_teacher_timesteps = cdm_student_timesteps
                            cdm_teacher_noise = torch.randn_like(cdm_teacher_x0_tensor)
                            cdm_teacher_xt = ((1 - cdm_teacher_sigma_expanded) * cdm_teacher_x0_tensor.detach()
                                              + cdm_teacher_sigma_expanded * cdm_teacher_noise)
                        else:  # "fresh_independent"
                            # Independently sampled σ_t and ε_fresh (strongest decoupling)
                            cdm_teacher_sigma = sample_sigmas(
                                current_batch_size, cdm_loss.teacher_sigma_method, device,
                                sigma_min=sigma_min, sigma_max=sigma_max,
                                logit_mean=cdm_loss.teacher_sigma_logit_mean,
                                logit_std=cdm_loss.teacher_sigma_logit_std)
                            cdm_teacher_sigma_expanded = cdm_teacher_sigma.view(
                                -1, *([1] * (cdm_teacher_x0_tensor.ndim - 1)))
                            cdm_teacher_timesteps = (cdm_teacher_sigma * 1000.0).float()
                            cdm_teacher_noise = torch.randn_like(cdm_teacher_x0_tensor)
                            cdm_teacher_xt = ((1 - cdm_teacher_sigma_expanded) * cdm_teacher_x0_tensor.detach()
                                              + cdm_teacher_sigma_expanded * cdm_teacher_noise)

                        # DM gradient mode: use real - fake teacher gradient (DMD2 style)
                        cdm_real_teacher_prediction = model_manager.teacher_forward(
                            cdm_teacher_xt.to(model_dtype), cdm_teacher_timesteps,
                            **batch_model_kwargs)
                        cdm_fake_teacher_prediction = model_manager.fake_teacher_forward(
                            cdm_teacher_xt.to(model_dtype), cdm_teacher_timesteps,
                            requires_grad=False, **batch_model_kwargs)

                        # Get CFG gradient direction for alignment analysis (if CFG loss was computed)
                        cdm_cfg_grad_direction = cfg_loss_info.get("cfg_grad_direction", None) if cfg_loss.enabled and cfg_loss.weight != 0 else None

                        cdm_loss_val, cdm_loss_info = compute_dm_loss(
                            student_prediction=cdm_student_prediction,
                            real_teacher_prediction=cdm_real_teacher_prediction,
                            fake_teacher_prediction=cdm_fake_teacher_prediction,
                            xt=cdm_student_xt, xt_teacher=cdm_teacher_xt,
                            student_sigma_expanded=cdm_student_sigma_expanded,
                            teacher_sigma_expanded=cdm_teacher_sigma_expanded,
                            return_per_sample=False,
                            cfg_grad_direction=cdm_cfg_grad_direction,
                            norm_clip_min=cdm_loss.norm_clip_min)

                        cdm_loss_weighted = cdm_loss.weight * cdm_loss_val
                        total_loss = total_loss + cdm_loss_weighted
                        loss_terms.update({f"cdm/{k}": v for k, v in cdm_loss_info.items()})
                        loss_terms.update({"loss/cdm_raw": cdm_loss_val.detach(), "loss/cdm": cdm_loss_weighted.detach()})
                        loss_terms["cdm/student_sigma_mean"] = cdm_student_sigma.mean().detach()
                else:
                    loss_terms["loss/cdm"] = zero_tensor

                # ========== Unified backward ==========
                accelerator.backward(total_loss)

                loss_terms.update({
                    "weights/cfg": torch.tensor(cfg_loss.weight, device=device),
                    "weights/ddm": torch.tensor(ddm_loss.weight, device=device),
                    "weights/cdm": torch.tensor(cdm_loss.weight, device=device),
                    "loss/total": total_loss.detach(),
                })

                for k, v in loss_terms.items():
                    info_accumulated[k].append(v)

                pbar_info = {"ft_step": ft_step, "stu_step": student_step}
                for loss_key in ["loss/total", "loss/cfg", "loss/ddm", "loss/cdm"]:
                    if loss_key in loss_terms:
                        short_key = loss_key.split("/")[-1]
                        pbar_info[short_key] = f"{loss_terms[loss_key].item():.4f}"
                batch_pbar.set_postfix(pbar_info)

                del loss_terms
                if accelerator.sync_gradients:
                    student_grad_stats = compute_grad_stats(parameters=transformer_trainable_parameters)
                    if accelerator.distributed_type == DistributedType.FSDP:
                        torch.nn.utils.clip_grad_norm_(transformer_trainable_parameters, config.train.max_grad_norm)
                    else:
                        accelerator.clip_grad_norm_(transformer.parameters(), config.train.max_grad_norm)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    gradient_update_times += 1

                    log_info = {k: torch.mean(torch.stack(v_list)).item() for k, v_list in info_accumulated.items()}
                    info_tensor = torch.tensor([log_info[k] for k in sorted(log_info.keys())], device=device)
                    info_tensor = accelerator.reduce(info_tensor, reduction="mean")
                    reduced_log_info = {k: info_tensor[i].item() for i, k in enumerate(sorted(log_info.keys()))}
                    
                    if accelerator.is_main_process and exp_logger is not None:
                        exp_logger.add_scalar("student/step", student_step, student_step)
                        exp_logger.add_scalar("student/learning_rate", lr_scheduler.get_last_lr()[0], student_step)
                        exp_logger.add_scalar("student/batch_size", train_batch_size, student_step)
                        exp_logger.add_scalar("student/gradient_update_times", gradient_update_times, student_step)
                        exp_logger.add_scalar("student/update_ratio", student_update_ratio, student_step)
                        exp_logger.add_scalar("student/grad_norm", student_grad_stats.get("grad_norm", 0.0), student_step)
                        for k, v in reduced_log_info.items():
                            exp_logger.add_scalar(k, v, student_step)
                        exp_logger.add_scalar("global/peak_gpu_memory_gb", torch.cuda.max_memory_allocated(device) / (1024 ** 3), student_step)

                    student_step += 1
                    info_accumulated = defaultdict(list)

                if config.train.student.ema and ema is not None and accelerator.sync_gradients:
                    # FSDP2: EMA operates on local shards via _get_data() in EMAModuleWrapper
                    ema.step(transformer_trainable_parameters, student_step)

        # End of epoch: cleanup to prevent memory fragmentation accumulation
        # torch.cuda.empty_cache()

    # Final checkpoint save (skip if already saved by the periodic checkpoint logic)
    final_checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint-{student_step}")
    already_saved = os.path.exists(final_checkpoint_path)
    if not already_saved:
        if accelerator.is_main_process:
            os.makedirs(final_checkpoint_path, exist_ok=True)
        accelerator.wait_for_everyone()
        accelerator.save_state(final_checkpoint_path)
        if ema is not None and accelerator.is_main_process:
            torch.save(ema.state_dict(), os.path.join(final_checkpoint_path, "ema_state.pt"))
        # FSDP: save full (unsharded) state dicts for prepare_student_pipeline compatibility
        save_fsdp_full_checkpoint(accelerator, transformer, ema, final_checkpoint_path, logger)
        if accelerator.is_main_process:
            logger.info(f"Saved final checkpoint to {final_checkpoint_path}")
    else:
        if accelerator.is_main_process:
            logger.info(f"Final checkpoint already exists at {final_checkpoint_path}, skipping duplicate save")

    if accelerator.is_main_process:
        logger.info(f"Training completed in {time.time() - train_start_time:.2f} seconds")
        # Write marker file so submit.py can locate the experiment directory
        marker_path = os.path.join(config.output_dir, f".experiment_dir")
        with open(marker_path, "w") as f:
            f.write(experiment_dir)
        logger.info(f"Wrote experiment marker to {marker_path}")

    accelerator.wait_for_everyone()
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
    shutdown_image_save_executor()

if __name__ == "__main__":
    app.run(main)