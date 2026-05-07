# Training module for distillation
#
# This package contains all training-related utilities:
#   - losses.py: Loss functions (CFG, DM, KL, diffusion, reward weight)
#   - model_utils.py: Model management (ModelMode, ModelManager, LoRA/full-FT initialization)

from cdm.training.losses import (
    compute_cfg_loss,
    compute_batch_mean_var_kl_loss,
    compute_fake_teacher_diffusion_loss,
    compute_dm_loss,
)

from cdm.training.model_utils import (
    ModelMode,
    ModelComponents,
    ModelManager,
    get_model_mode,
    get_attr,
    create_frozen_model,
    patch_model_for_pipeline,
    initialize_models,
    is_student_lora,
    is_ft_lora,
    has_fake_teacher,
    ema_merge_ft_with_student,
    LORA_TARGET_MODULES,
    save_fsdp_full_checkpoint,
    compute_grad_stats,
    create_lr_lambda,
    build_fsdp_plugin_if_needed,
)

__all__ = [
    # Loss functions
    "compute_cfg_loss",
    "compute_batch_mean_var_kl_loss",
    "compute_fake_teacher_diffusion_loss",
    "compute_dm_loss",
    # Model utilities
    "ModelMode",
    "ModelComponents",
    "get_attr",
    "create_frozen_model",
    "patch_model_for_pipeline",
    "initialize_models",
    "is_student_lora",
    "is_ft_lora",
    "has_fake_teacher",
    "ema_merge_ft_with_student",
    "LORA_TARGET_MODULES",
    # Checkpoint & training utilities
    "save_fsdp_full_checkpoint",
    "compute_grad_stats",
    "create_lr_lambda",
    "build_fsdp_plugin_if_needed",
]
