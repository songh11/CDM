"""
Model utilities for LoRA, full fine-tuning, and accelerator preparation.
This module handles model loading, LoRA configuration, and accelerator setup.
"""

import copy
import math
import os
from dataclasses import dataclass
from enum import Enum, auto

import torch
import safetensors.torch
from accelerate.utils import DistributedType
from peft import LoraConfig, get_peft_model, PeftModel


# ==================== Model Configuration ====================
class ModelMode(Enum):
    """2 supported combinations of fake teacher and student training modes."""
    FT_LORA_STUDENT_LORA = auto()    # Shared transformer + 2 adapters (default, fake_teacher)
    FT_FULL_STUDENT_FULL = auto()    # Both full fine-tuning, separate models


@dataclass
class ModelComponents:
    """Container for all model components and their trainable parameters."""
    transformer: torch.nn.Module
    transformer_trainable_params: list
    fake_teacher: torch.nn.Module = None
    fake_teacher_trainable_params: list = None
    real_teacher: torch.nn.Module = None
    mode: ModelMode = None


def get_model_mode(fake_teacher_enabled: bool, ft_use_lora: bool, student_use_lora: bool) -> ModelMode:
    """Determine the model configuration mode based on settings."""
    # if not fake_teacher_enabled:
    #     raise ValueError("Fake teacher must be enabled. Set fake_teacher.enabled=True in config.")
    if ft_use_lora and student_use_lora:
        return ModelMode.FT_LORA_STUDENT_LORA
    if not ft_use_lora and not student_use_lora:
        return ModelMode.FT_FULL_STUDENT_FULL
    raise ValueError(
        f"Unsupported mode: ft_use_lora={ft_use_lora}, student_use_lora={student_use_lora}. "
        f"Only lora+lora or full+full is supported."
    )


# ==================== Helper Functions ====================
def get_attr(obj, name, default=None):
    """Safe getattr with default value."""
    return getattr(obj, name, default) if obj else default


def create_frozen_model(base_model, device, dtype):
    """Create a frozen copy of a model."""
    model = copy.deepcopy(base_model).to(device, dtype=dtype)
    model.requires_grad_(False)
    model.eval()
    return model

def patch_model_for_pipeline(wrapped_model, unwrapped_model):
    """Patch a DDP/FSDP wrapped model so it can be used directly by diffusers pipeline.

    Diffusers pipelines access ``self.transformer.config`` (and sometimes ``.dtype``)
    during ``__call__``.  DDP's ``__getattr__`` proxies these automatically, but
    FSDP does not.  This helper copies the necessary attributes from the
    unwrapped (inner) model onto the wrapper so that the pipeline works
    transparently with any distributed backend.

    This must be called **after** ``accelerator.prepare()`` and before
    assigning the model to ``pipeline.transformer``.

    Args:
        wrapped_model: The DDP/FSDP wrapped model returned by ``accelerator.prepare()``.
        unwrapped_model: The inner model obtained via ``accelerator.unwrap_model()``.
    """
    for attr in ("config", "dtype", "in_channels"):
        if not hasattr(wrapped_model, attr) and hasattr(unwrapped_model, attr):
            setattr(wrapped_model, attr, getattr(unwrapped_model, attr))

    # Proxy CacheMixin methods needed by LongCat pipeline's denoising loop.
    # DDP/FSDP wrappers only proxy parameter/buffer/submodule access, not
    # regular methods like cache_context().  Binding them here allows the
    # pipeline to call e.g. `self.transformer.cache_context("cond")` even
    # when the transformer is wrapped.
    _cache_mixin_attrs = (
        "cache_context", "_reset_stateful_cache", "is_cache_enabled",
        "enable_cache", "disable_cache", "_cache_config",
    )
    for attr in _cache_mixin_attrs:
        if not hasattr(wrapped_model, attr) and hasattr(unwrapped_model, attr):
            setattr(wrapped_model, attr, getattr(unwrapped_model, attr))


# Default LoRA target modules (SD3). Can be overridden by passing lora_target_modules
# to initialize_models() for different model architectures (e.g., FLUX).
LORA_TARGET_MODULES = [
    "attn.add_k_proj", "attn.add_q_proj", "attn.add_v_proj", "attn.to_add_out",
    "attn.to_k", "attn.to_out.0", "attn.to_q", "attn.to_v",
]


def initialize_models(
    pipeline, device, mode: ModelMode, student_config, fake_teacher_config,
    frozen_model_dtype,
    lora_target_modules: list = None,
) -> ModelComponents:
    """
    Initialize all model components based on the configuration mode.
    
    Args:
        lora_target_modules: List of module names to apply LoRA to.
            If None, defaults to LORA_TARGET_MODULES (SD3).
    
    Returns a ModelComponents dataclass containing all models and their trainable parameters.
    """
    target_modules = lora_target_modules or LORA_TARGET_MODULES
    transformer = pipeline.transformer.to(device)
    
    # Get LoRA configs
    student_lora_config = LoraConfig(
        r=student_config.lora_rank, lora_alpha=student_config.lora_alpha,
        init_lora_weights="gaussian", target_modules=target_modules
    ) if get_attr(student_config, 'use_lora', False) else None
    
    ft_lora_config = None
    if fake_teacher_config and get_attr(fake_teacher_config, 'use_lora', True):
        ft_lora_config = LoraConfig(
            r=get_attr(fake_teacher_config, 'lora_rank', 32),
            lora_alpha=get_attr(fake_teacher_config, 'lora_alpha', 64),
            init_lora_weights="gaussian", target_modules=target_modules
        )
    
    # Initialize based on mode
    if mode == ModelMode.FT_LORA_STUDENT_LORA:
        # Shared transformer with 2 adapters: default (student), fake_teacher
        transformer = (PeftModel.from_pretrained(transformer, student_config.lora_path) 
                      if get_attr(student_config, 'lora_path') else get_peft_model(transformer, student_lora_config))
        transformer.add_adapter("fake_teacher", ft_lora_config)
        transformer.set_adapter("default")
        
        transformer_trainable_params = list(filter(lambda p: p.requires_grad, transformer.parameters()))
        transformer.set_adapter("fake_teacher")
        fake_teacher_trainable_params = list(filter(lambda p: p.requires_grad, transformer.parameters()))
        transformer.set_adapter("default")
        
        pipeline.transformer = transformer
        pipeline.fake_teacher = transformer  # Same model, different adapter
        pipeline.real_teacher = None  # LoRA mode: use disable_adapter() for real teacher
        
        return ModelComponents(
            transformer=transformer,
            transformer_trainable_params=transformer_trainable_params,
            fake_teacher=transformer,
            fake_teacher_trainable_params=fake_teacher_trainable_params,
            mode=mode
        )
    
    elif mode == ModelMode.FT_FULL_STUDENT_FULL:
        # Both full fine-tuning, all separate models
        # Create frozen real teacher BEFORE enabling gradients on student
        pipeline.real_teacher = create_frozen_model(transformer, device, frozen_model_dtype)
        
        transformer.requires_grad_(True)
        transformer_trainable_params = list(transformer.parameters())
        
        # Create fake_teacher as separate model (same dtype as student for training consistency)
        fake_teacher = copy.deepcopy(transformer).to(device)
        fake_teacher.requires_grad_(True)
        fake_teacher.train()
        fake_teacher_trainable_params = list(fake_teacher.parameters())
        pipeline.fake_teacher = fake_teacher
        
        return ModelComponents(
            transformer=transformer,
            transformer_trainable_params=transformer_trainable_params,
            fake_teacher=fake_teacher,
            fake_teacher_trainable_params=fake_teacher_trainable_params,
            mode=mode
        )
    
    else:
        raise ValueError(f"Unsupported model mode: {mode}")


def is_student_lora(mode: ModelMode) -> bool:
    """Check if student uses LoRA in this mode."""
    return mode == ModelMode.FT_LORA_STUDENT_LORA

def is_ft_lora(mode: ModelMode) -> bool:
    """Check if fake teacher uses LoRA in this mode."""
    return mode == ModelMode.FT_LORA_STUDENT_LORA

def has_fake_teacher(mode: ModelMode) -> bool:
    """Check if fake teacher is enabled in this mode. Always True for supported modes."""
    return True

@torch.no_grad()
def ema_merge_ft_with_student(
    ft_params: list,
    student_params: list,
    decay: float = 0.999,
) -> None:
    """
    EMA merge: theta_ft = decay * theta_ft + (1 - decay) * theta_student.

    Merges student parameters into fake teacher parameters in-place.
    Works for both LoRA mode (merging LoRA adapter weights) and
    Full fine-tuning mode (merging all model parameters).

    Args:
        ft_params: List of fake teacher trainable parameters.
        student_params: List of student trainable parameters (must match ft_params in length and shapes).
        decay: EMA decay factor. Higher values retain more of the FT's own parameters.

    Raises:
        ValueError: If parameter counts or shapes mismatch.
    """
    if len(ft_params) != len(student_params):
        raise ValueError(
            f"Parameter count mismatch: FT has {len(ft_params)} params, "
            f"Student has {len(student_params)} params. "
            f"For LoRA mode, ensure both adapters use the same lora_rank and target_modules."
        )

    one_minus_decay = 1.0 - decay
    for ft_param, student_param in zip(ft_params, student_params):
        if ft_param.shape != student_param.shape:
            raise ValueError(
                f"Parameter shape mismatch: FT {ft_param.shape} vs Student {student_param.shape}. "
                f"For LoRA mode, lora_rank must be identical for EMA merge."
            )
        ft_param.data.mul_(decay).add_(
            student_param.data.to(device=ft_param.device, dtype=ft_param.dtype),
            alpha=one_minus_decay,
        )


# ==================== Model Manager ====================
class ModelManager:
    """
    Unified model manager that abstracts away LoRA/Full fine-tuning differences.
    
    This class provides a consistent API for model forward passes regardless of
    whether the model uses LoRA adapters or full fine-tuning.
    """
    
    def __init__(
        self,
        pipeline,
        mode: ModelMode,
        accelerator,
        device,
        model_components: ModelComponents,
        model_adapter=None,
    ):
        """
        Initialize the ModelManager.
        
        Args:
            pipeline: The diffusion pipeline containing models
            mode: The current ModelMode
            accelerator: Hugging Face Accelerator instance
            device: Target device for computation
            model_components: ModelComponents from initialize_models()
            model_adapter: Optional ModelAdapter instance for architecture-specific forward passes.
                          If None, uses default SD3-style forward.
        """
        self.pipeline = pipeline
        self.mode = mode
        self.accelerator = accelerator
        self.device = device
        self.components = model_components
        self.model_adapter = model_adapter
        
        # Cache the unwrapped transformer for adapter operations
        self._unwrapped_transformer = None
    
    @property
    def transformer(self):
        """Get the main transformer (may be wrapped by accelerator)."""
        return self.components.transformer
    
    @property
    def transformer_trainable_params(self):
        """Get trainable parameters for the student model."""
        return self.components.transformer_trainable_params
    
    @property
    def fake_teacher_trainable_params(self):
        """Get trainable parameters for the fake teacher."""
        return self.components.fake_teacher_trainable_params
    
    def get_unwrapped_transformer(self):
        """Get the unwrapped transformer for direct operations."""
        if self._unwrapped_transformer is None:
            self._unwrapped_transformer = self.accelerator.unwrap_model(self.components.transformer)
        return self._unwrapped_transformer
    
    def get_model_dtype(self):
        """Get the dtype of the main transformer."""
        return next(self.get_unwrapped_transformer().parameters()).dtype
    
    # ==================== Forward Methods ====================
    
    def student_forward(self, hidden_states, timestep, **model_kwargs):
        """
        Student model forward pass (with gradients).
        Automatically handles adapter switching for LoRA modes.
        
        Args:
            hidden_states: Noisy latent tensor.
            timestep: Timestep tensor.
            **model_kwargs: Model-specific kwargs from ``ModelAdapter.prepare_forward_kwargs()``.
        """
        self._activate_student()
        return self._model_forward(self.components.transformer, hidden_states, timestep, **model_kwargs)
    
    def teacher_forward(self, hidden_states, timestep, **model_kwargs):
        """
        Real teacher forward pass (no gradients).
        Uses base model (disabled adapters for LoRA, or frozen real_teacher for full fine-tuning).
        
        Args:
            hidden_states: Noisy latent tensor.
            timestep: Timestep tensor.
            **model_kwargs: Model-specific kwargs from ``ModelAdapter.prepare_forward_kwargs()``.
        """
        with torch.no_grad():
            if is_student_lora(self.mode):
                unwrapped = self.get_unwrapped_transformer()
                with unwrapped.disable_adapter():
                    result = self._model_forward(
                        self.components.transformer, hidden_states, timestep, **model_kwargs
                    )
                self._activate_student()
                return result.detach()
            else:
                real_teacher = self._get_real_teacher_model()
                teacher_dtype = next(self._unwrap_model(real_teacher).parameters()).dtype
                result = self._model_forward(
                    real_teacher,
                    hidden_states.to(teacher_dtype), timestep,
                    **model_kwargs
                ).detach()
                return result
    
    def fake_teacher_forward(self, hidden_states, timestep, requires_grad=False, **model_kwargs):
        """
        Fake teacher forward pass.
        
        Args:
            hidden_states: Noisy latent tensor.
            timestep: Timestep tensor.
            requires_grad: If True, enables gradients (for FT training phase).
            **model_kwargs: Model-specific kwargs from ``ModelAdapter.prepare_forward_kwargs()``.
        """
        if not has_fake_teacher(self.mode):
            raise RuntimeError("Fake teacher is not enabled in this mode")
        
        if self.mode == ModelMode.FT_LORA_STUDENT_LORA:
            unwrapped = self.get_unwrapped_transformer()
            unwrapped.set_adapter("fake_teacher")
            
            if requires_grad:
                result = self._model_forward(
                    self.components.transformer, hidden_states, timestep, **model_kwargs
                )
            else:
                with torch.no_grad():
                    result = self._model_forward(
                        self.components.transformer, hidden_states, timestep, **model_kwargs
                    ).detach()
                self._activate_student()
            
            return result
        else:
            ft_model = self._get_fake_teacher_model()
            ft_dtype = next(self._unwrap_model(ft_model).parameters()).dtype
            
            if requires_grad:
                return self._model_forward(
                    ft_model, hidden_states.to(ft_dtype), timestep, **model_kwargs
                )
            else:
                with torch.no_grad():
                    return self._model_forward(
                        ft_model, hidden_states.to(ft_dtype), timestep, **model_kwargs
                    ).detach()
    
    def fake_teacher_forward_cfg_batched(self, xt, timestep, cond_kwargs, uncond_kwargs, requires_grad=False, **extra_kwargs):
        """
        Batched fake teacher forward for CFG.
        
        Args:
            xt: Noisy latent tensor.
            timestep: Timestep tensor.
            cond_kwargs: Conditional model kwargs from ``prepare_forward_kwargs()``.
            uncond_kwargs: Unconditional model kwargs from ``prepare_forward_kwargs()``.
            requires_grad: If True, enables gradients (for FT training phase).
        
        Returns: (uncond_prediction, cond_prediction)
        """
        if not has_fake_teacher(self.mode):
            raise RuntimeError("Fake teacher is not enabled in this mode")
        
        if self.mode == ModelMode.FT_LORA_STUDENT_LORA:
            unwrapped = self.get_unwrapped_transformer()
            unwrapped.set_adapter("fake_teacher")
            
            if requires_grad:
                uncond_pred, cond_pred = self._cfg_batched_forward(
                    self.components.transformer, xt, timestep,
                    cond_kwargs, uncond_kwargs, **extra_kwargs
                )
                # Do NOT call _activate_student() here when requires_grad=True.
                # See fake_teacher_forward() for detailed explanation.
            else:
                with torch.no_grad():
                    uncond_pred, cond_pred = self._cfg_batched_forward(
                        self.components.transformer, xt, timestep,
                        cond_kwargs, uncond_kwargs, **extra_kwargs
                    )
                    uncond_pred, cond_pred = uncond_pred.detach(), cond_pred.detach()
                self._activate_student()
            
            return uncond_pred, cond_pred
        else:
            ft_model = self._get_fake_teacher_model()
            ft_dtype = next(self._unwrap_model(ft_model).parameters()).dtype
            
            if requires_grad:
                uncond_pred, cond_pred = self._cfg_batched_forward(
                    ft_model, xt.to(ft_dtype), timestep,
                    cond_kwargs, uncond_kwargs, **extra_kwargs
                )
            else:
                with torch.no_grad():
                    uncond_pred, cond_pred = self._cfg_batched_forward(
                        ft_model, xt.to(ft_dtype), timestep,
                        cond_kwargs, uncond_kwargs, **extra_kwargs
                    )
                    uncond_pred, cond_pred = uncond_pred.detach(), cond_pred.detach()
            
            return uncond_pred, cond_pred
    
    # ==================== Batched Forward Methods ====================
    
    def teacher_forward_cfg_batched(self, xt, timestep, cond_kwargs, uncond_kwargs, **extra_kwargs):
        """
        Batched teacher forward for CFG (computes both cond and uncond in one pass).
        
        Args:
            xt: Noisy latent tensor.
            timestep: Timestep tensor.
            cond_kwargs: Conditional model kwargs from ``prepare_forward_kwargs()``.
            uncond_kwargs: Unconditional model kwargs from ``prepare_forward_kwargs()``.
        
        Returns: (uncond_prediction, cond_prediction)
        """
        with torch.no_grad():
            if is_student_lora(self.mode):
                unwrapped = self.get_unwrapped_transformer()
                with unwrapped.disable_adapter():
                    uncond_pred, cond_pred = self._cfg_batched_forward(
                        self.components.transformer, xt, timestep,
                        cond_kwargs, uncond_kwargs, **extra_kwargs
                    )
                self._activate_student()
            else:
                real_teacher = self._get_real_teacher_model()
                teacher_dtype = next(self._unwrap_model(real_teacher).parameters()).dtype
                uncond_pred, cond_pred = self._cfg_batched_forward(
                    real_teacher,
                    xt.to(teacher_dtype), timestep,
                    cond_kwargs,
                    uncond_kwargs,
                    **extra_kwargs
                )
        
        return uncond_pred.detach(), cond_pred.detach()
    
    # ==================== Training Mode Management ====================
    
    def prepare_for_sampling(self, adapter_name: str = "default"):
        """Prepare model for sampling phase.

        Calls eval() on the wrapped model (not unwrapped) so that both
        DDP and FSDP wrappers correctly transition to eval mode.
        For DDP this is equivalent; for FSDP it is required because
        FSDP checks its own ``training`` flag to decide reshard behaviour.
        """
        self.components.transformer.eval()
        if is_student_lora(self.mode):
            self.get_unwrapped_transformer().set_adapter(adapter_name)
    
    def prepare_for_student_training(self):
        """Prepare model for student training phase."""
        self._activate_student()
        self.components.transformer.train()
    
    def prepare_for_ft_training(self):
        """Prepare model for fake teacher training phase."""
        if not has_fake_teacher(self.mode):
            return
        
        if self.mode == ModelMode.FT_LORA_STUDENT_LORA:
            unwrapped = self.get_unwrapped_transformer()
            unwrapped.set_adapter("fake_teacher")
            self.components.transformer.train()
        else:
            ft_model = self._get_fake_teacher_model()
            ft_model.train()
    
    def finish_ft_training(self):
        """Finish fake teacher training phase and restore state.

        Calls eval() on the wrapped model (not unwrapped) so that FSDP
        correctly transitions to eval mode.
        """
        if not has_fake_teacher(self.mode):
            return
        
        if self.mode == ModelMode.FT_LORA_STUDENT_LORA:
            self._activate_student()
        else:
            ft_model = self._get_fake_teacher_model()
            ft_model.eval()  # Already the wrapped model from _get_fake_teacher_model()
    
    # ==================== Internal Methods ====================
    
    def _model_forward(self, model, hidden_states, timestep, **model_kwargs):
        """
        Unified model forward pass. Delegates to model_adapter if available.
        
        Args:
            model: The transformer model to call.
            hidden_states: Noisy latent tensor.
            timestep: Timestep tensor.
            **model_kwargs: Model-specific kwargs (e.g. encoder_hidden_states,
                pooled_projections for SD3; encoder_hidden_states, txt_ids for Flux2).
        """
        if self.model_adapter is not None:
            return self.model_adapter.model_forward(
                model, hidden_states, timestep, **model_kwargs
            )
        # Default SD3-style fallback
        return model(
            hidden_states=hidden_states,
            timestep=timestep,
            **model_kwargs,
            return_dict=False
        )[0]
    
    def _cfg_batched_forward(self, model, xt, timestep, cond_kwargs, uncond_kwargs, **extra_kwargs):
        """
        Internal CFG batched forward. Delegates to model_adapter if available,
        otherwise uses standard cond/uncond concatenation (SD3 style).
        
        Args:
            model: The transformer model.
            xt: Noisy latent tensor.
            timestep: Timestep tensor.
            cond_kwargs: Conditional model kwargs from ``prepare_forward_kwargs()``.
            uncond_kwargs: Unconditional model kwargs from ``prepare_forward_kwargs()``.
        
        Returns: (uncond_prediction, cond_prediction)
        """
        if self.model_adapter is not None:
            return self.model_adapter.model_forward_cfg_batched(
                model, xt, timestep,
                cond_kwargs, uncond_kwargs, **extra_kwargs
            )
        
        return uncond_pred, cond_pred
    
    def _activate_student(self):
        """Activate student adapter (for LoRA modes)."""
        if is_student_lora(self.mode):
            self.get_unwrapped_transformer().set_adapter("default")
    
    def _unwrap_model(self, model):
        """Unwrap a model that may be wrapped by DDP or Accelerator.
        
        Handles both DDP-wrapped (has .module) and Accelerator-wrapped models.
        """
        if hasattr(model, 'module'):
            return model.module
        return self.accelerator.unwrap_model(model)
    
    def _get_fake_teacher_model(self):
        """Get the fake teacher model (separate model, not adapter).
        
        Returns the model from components (which may be wrapped by Accelerator)
        rather than pipeline.fake_teacher (which is always unwrapped).
        This ensures forward passes go through the distributed wrapper so gradients
        are properly synchronized.
        """
        return self.components.fake_teacher
    
    def _get_real_teacher_model(self):
        """Get the real teacher model (frozen, no gradients).
        
        Returns the model from components if available (which may be wrapped by
        Accelerator for FSDP compatibility), otherwise falls back to
        pipeline.real_teacher for backward compatibility.
        """
        if self.components.real_teacher is not None:
            return self.components.real_teacher
        return self.pipeline.real_teacher


# ==================== Checkpoint & Training Utilities ====================

def save_fsdp_full_checkpoint(accelerator, transformer, ema, checkpoint_save_path, logger):
    """Save full (unsharded) state dicts for FSDP checkpoints.

    FSDP shards model parameters across ranks, so ``accelerator.save_state()``
    produces per-rank shard files that cannot be loaded on a single GPU.
    This helper additionally saves:
      - ``transformer_full_state_dict.safetensors``: the complete transformer
        weights gathered from all ranks (always saved).
      - ``ema_full_state.pt``: the complete EMA weights as a full state_dict
        gathered from all ranks (saved when EMA is enabled).

    For non-FSDP runs this function is a no-op.

    Strategy for EMA: temporarily swap EMA parameters into the FSDP-wrapped
    model, call ``accelerator.get_state_dict()`` to all-gather them into a
    complete state_dict, then restore the original training weights. This
    reuses FSDP's own all-gather machinery and avoids manual DTensor handling.
    """
    if accelerator.distributed_type != DistributedType.FSDP:
        return

    trainable_params = [p for p in transformer.parameters() if p.requires_grad]

    save_dtype = torch.bfloat16

    # --- Save full transformer state_dict ---
    full_state_dict = accelerator.get_state_dict(transformer)
    if accelerator.is_main_process and full_state_dict is not None:
        bf16_state_dict = {k: v.to(save_dtype) for k, v in full_state_dict.items()}
        save_path = os.path.join(checkpoint_save_path, "transformer_full_state_dict.safetensors")
        safetensors.torch.save_file(bf16_state_dict, save_path)
        logger.info(f"  Saved full transformer state_dict to {save_path} (dtype={save_dtype})")
        del bf16_state_dict
    del full_state_dict

    # --- Save full EMA state_dict ---
    if ema is not None:
        ema.copy_ema_to(trainable_params, store_temp=True, grad=False)
        ema_full_state_dict = accelerator.get_state_dict(transformer)
        ema.copy_temp_to(trainable_params)

        if accelerator.is_main_process and ema_full_state_dict is not None:
            bf16_ema_state_dict = {k: v.to(save_dtype) for k, v in ema_full_state_dict.items()}
            ema_save_path = os.path.join(checkpoint_save_path, "ema_full_state.pt")
            torch.save({"decay": ema.decay, "full_state_dict": bf16_ema_state_dict}, ema_save_path)
            logger.info(f"  Saved full EMA state_dict to {ema_save_path} (dtype={save_dtype})")
            del bf16_ema_state_dict
        del ema_full_state_dict


def compute_grad_stats(parameters=None):
    """Compute gradient norm from model parameters.

    Uses incremental squared-norm accumulation to avoid allocating a single
    giant tensor that concatenates all gradients (which can OOM on large models).
    """
    if parameters is not None:
        total_norm_sq = 0.0
        for p in parameters:
            if p.grad is not None:
                total_norm_sq += p.grad.detach().float().norm().item() ** 2
        if total_norm_sq == 0.0:
            return {}
        return {"grad_norm": total_norm_sq ** 0.5}
    return {}


def create_lr_lambda(warmup_steps, total_steps, scheduler_type):
    """
    Factory function to create LR lambda for torch.optim.lr_scheduler.LambdaLR.

    Args:
        warmup_steps: Number of warmup steps
        total_steps: Total number of training steps
        scheduler_type: One of "constant", "constant_with_warmup", "linear", "cosine"

    Returns:
        A lambda function that takes step and returns LR multiplier
    """
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        if scheduler_type in ("constant", "constant_with_warmup"):
            return 1.0
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        if scheduler_type == "linear":
            return max(0.0, 1.0 - progress)
        if scheduler_type == "cosine":
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
        return 1.0
    return lr_lambda


def build_fsdp_plugin_if_needed(config):
    """Build FSDP plugin with model-specific transformer wrap classes.

    Reads the accelerate launch config from environment variables to detect
    whether FSDP is being used.  When it is, creates a
    ``FullyShardedDataParallelPlugin`` with the correct
    ``transformer_cls_names_to_wrap`` for the current model, so the static
    YAML does not need to hard-code any model-specific class names.

    Returns ``None`` when the backend is not FSDP.
    """
    from cdm.models.model_adapters import create_model_adapter

    use_fsdp = os.environ.get("ACCELERATE_USE_FSDP", "false").lower() == "true"
    if not use_fsdp:
        return None

    from accelerate.utils import FullyShardedDataParallelPlugin

    base_model_type = getattr(config, "base_model", "sd3")
    model_adapter = create_model_adapter(base_model_type, config)
    wrap_cls_names = model_adapter.get_fsdp_wrap_cls_names()

    fsdp_plugin = FullyShardedDataParallelPlugin(
        auto_wrap_policy="transformer_based_wrap",
        transformer_cls_names_to_wrap=wrap_cls_names,
    )
    print(
        f"[FSDP] auto-wrap policy: transformer_based_wrap, "
        f"cls_to_wrap={wrap_cls_names} (model={base_model_type})"
    )
    return fsdp_plugin
    
