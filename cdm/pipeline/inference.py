"""
Shared inference layer for image generation across training evaluation and standalone evaluation.

This module provides unified functions for:
  - Adapter switching (use_adapter)
  - Unified inference config (InferenceConfig)
  - Simple image generation via pipeline (generate_images_simple)
  - Role-based simple generation (generate_for_role_simple) — training-time eval
  - Pipeline-based generation (generate_for_pipeline) — standalone eval (no ModelMode needed)
  - Deterministic generator utilities (make_generator_from_prompts, make_diversity_generators)

Both `flow_grpo/evaluation.py` (training-time eval) and `scripts/evaluation_full.py`
(standalone eval) should import from this module to avoid duplicated inference logic.
"""

import hashlib
import logging
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Optional

import torch

from cdm.training.model_utils import ModelMode, is_student_lora

logger = logging.getLogger(__name__)


# ==================== Inference Configuration ====================

@dataclass
class InferenceConfig:
    """Unified inference configuration that works with both ml_collections and DictConfig.

    Construct via `InferenceConfig.from_config(config)` to read from any config object,
    or pass values directly for testing / overrides.
    """
    height: int = 1024
    width: int = 1024
    noise_level: float = 1.0
    deterministic: bool = True
    solver: str = "flow_simple"

    student_num_steps: int = 4
    student_guidance_scale: float = 1.0

    teacher_num_steps: int = 50
    teacher_guidance_scale: float = 4.5

    fake_teacher_num_steps: int = 50
    fake_teacher_guidance_scale: float = 1.0
    real_teacher_viz_guidance_scale: float = 1.0  # Real teacher guidance scale for fake teacher viz comparison

    custom_sigmas: Optional[list] = None  # Custom sigmas for denoising schedule (overrides uniform schedule)

    model_type: str = "sd3"  # "sd3" or "longcat"
    enable_prompt_rewrite: bool = False  # LongCat: whether to enable prompt rewrite during generation

    @classmethod
    def from_config(cls, config) -> "InferenceConfig":
        """Build from a training config object (ml_collections or DictConfig).

        Reads from `config.eval.*` for most fields.
        Uses safe getattr so both config flavors work.
        """
        def _get(obj, name, default=None):
            if obj is None:
                return default
            val = getattr(obj, name, None)
            return val if val is not None else default

        eval_cfg = getattr(config, "eval", None)
        ft_viz = _get(eval_cfg, "fake_teacher_viz")

        # Determine model_type from base_model config
        base_model = _get(config, "base_model", "sd3")
        model_type_map = {"sd3": "sd3", "sd3_hypersd": "sd3", "sd3_tdm": "sd3", "longcat": "longcat"}
        model_type = model_type_map.get(base_model, "sd3")

        return cls(
            height=_get(config, "height", 1024),
            width=_get(config, "width", 1024),
            deterministic=_get(eval_cfg, "deterministic", True),
            student_num_steps=_get(eval_cfg, "student_num_steps", 4),
            student_guidance_scale=_get(eval_cfg, "student_guidance_scale", 1.0),
            teacher_num_steps=_get(eval_cfg, "teacher_num_steps", 28),
            teacher_guidance_scale=_get(eval_cfg, "teacher_guidance_scale", 4.5),
            fake_teacher_num_steps=_get(ft_viz, "num_steps", 28),
            fake_teacher_guidance_scale=1.0,
            real_teacher_viz_guidance_scale=1.0,
            custom_sigmas=_get(config, "custom_sigmas"),
            model_type=model_type,
            enable_prompt_rewrite=_get(eval_cfg, "enable_prompt_rewrite", False),
        )


# ==================== Adapter Switching ====================

@contextmanager
def use_adapter(transformer, adapter_name):
    """Temporarily switch LoRA adapter and restore on exit.

    If the transformer has no adapter support (e.g. full fine-tuning), this is a no-op.
    """
    if transformer is None or not hasattr(transformer, "set_adapter"):
        yield
        return

    previous = getattr(transformer, "active_adapter", "default")
    if isinstance(previous, list):
        previous = previous[0] if previous else "default"
    try:
        transformer.set_adapter(adapter_name)
        yield
    finally:
        transformer.set_adapter(previous)



# ==================== Generator Utilities ====================

def compute_seed_from_prompts(prompts):
    """Compute a deterministic seed from a list of prompt strings.

    Uses the MD5 hash of the joined prompts to produce a reproducible
    32-bit integer seed.  This is the same logic used by
    ``make_generator_from_prompts`` and can be called independently when
    only the seed value is needed (e.g. for logging reproduction info).

    Args:
        prompts: List of prompt strings to derive the seed from.

    Returns:
        An integer seed in ``[0, 2**32)``.
    """
    prompts_str = "||".join(prompts)
    return int(hashlib.md5(prompts_str.encode()).hexdigest(), 16) % (2**32)

def make_generator_from_prompts(prompts, device):
    """Create a deterministic torch.Generator seeded by the MD5 hash of prompts.

    The same list of prompt strings always produces the same seed, ensuring
    reproducible latent noise when passed to ``pipeline(..., generator=gen)``.

    Args:
        prompts: List of prompt strings to derive the seed from.
        device: Device for the generator (should match the pipeline device).

    Returns:
        A ``torch.Generator`` with a deterministic seed.
    """
    seed = compute_seed_from_prompts(prompts)
    return torch.Generator(device=device).manual_seed(seed)


def make_diversity_generators(prompt, num_variations, device):
    """Create a list of deterministic generators for diversity evaluation.

    Each generator is seeded by the prompt's MD5 hash plus a variation-specific
    offset, producing deterministic but diverse noise for the same prompt.

    Args:
        prompt: Single prompt string to derive the base seed from.
        num_variations: Number of different generators to create.
        device: Device for the generators.

    Returns:
        List of ``torch.Generator``, one per variation.
    """
    base_seed = int(hashlib.md5(prompt.encode()).hexdigest(), 16) % (2**32)
    generators = []
    for variation_idx in range(num_variations):
        seed = (base_seed + variation_idx * 7919) % (2**32)
        generators.append(torch.Generator(device=device).manual_seed(seed))
    return generators


def generate_images_simple(
    pipeline, prompts, inf_cfg: InferenceConfig, num_steps: int,
    guidance_scale: float, device=None, generator=None, sigmas=None, **kwargs,
):
    """Generate images by passing prompt strings directly to the pipeline.

    Unlike ``generate_images`` which requires pre-computed text embeddings and
    latent noise, this function lets the pipeline handle text encoding and
    latent preparation internally.  A deterministic ``torch.Generator`` seeded
    by the prompt hash is used to ensure reproducible results when no external
    generator is provided.

    Args:
        pipeline: A diffusers pipeline (SD3, Flux, etc.).
        prompts: List of prompt strings.
        inf_cfg: Inference configuration (height, width, model_type, …).
        num_steps: Number of denoising steps.
        guidance_scale: Classifier-free guidance scale.
        device: Device for the generator. If None, uses the transformer device.
        generator: Optional external ``torch.Generator``. If None, a
            deterministic generator seeded by the prompt hash is created.
        sigmas: Optional custom sigmas list/tensor for the denoising schedule.
            When provided, overrides the default uniform schedule derived from
            num_steps. For example, ``[1.0, 0.75, 0.5, 0.25]`` for 4-step
            distillation with custom noise levels.
        **kwargs: Additional keyword arguments forwarded to the pipeline
            (e.g. ``image`` for LongCat Edit source images).

    Returns:
        Image tensor of shape ``(B, 3, H, W)`` in ``[0, 1]``.
    """
    if device is None:
        device = next(pipeline.transformer.parameters()).device

    if generator is None:
        generator = make_generator_from_prompts(prompts, device)

    # Defensive copy: avoid pipelines that may in-place modify the prompt list
    # inside _encode_prompt corrupting the caller's data for subsequent use.
    prompts = list(prompts)

    pipeline_kwargs = dict(
        prompt=prompts,
        num_inference_steps=num_steps,
        guidance_scale=guidance_scale,
        output_type="pt",
        height=inf_cfg.height,
        width=inf_cfg.width,
        generator=generator,
    )

    # Custom sigmas override the default schedule from num_inference_steps
    if sigmas is not None:
        pipeline_kwargs["sigmas"] = sigmas

    # LongCat pipeline does not support negative_prompt.
    if inf_cfg.model_type not in ("longcat",):
        pipeline_kwargs["negative_prompt"] = [""] * len(prompts)

    # LongCat: pass enable_prompt_rewrite from config.
    if inf_cfg.model_type == "longcat":
        pipeline_kwargs["enable_prompt_rewrite"] = inf_cfg.enable_prompt_rewrite

    # Merge caller-provided kwargs.
    pipeline_kwargs.update(kwargs)

    # When deterministic=False, enable SDE (stochastic) sampling by temporarily
    # setting the scheduler's stochastic_sampling config.  This mirrors the
    # behaviour of TrainingSchedulerWrapper used during training.
    use_stochastic = not inf_cfg.deterministic
    original_stochastic = getattr(pipeline.scheduler.config, "stochastic_sampling", False)
    if use_stochastic:
        pipeline.scheduler.config.stochastic_sampling = True

    try:
        result = pipeline(**pipeline_kwargs)
    finally:
        # Restore original scheduler config to avoid side effects
        pipeline.scheduler.config.stochastic_sampling = original_stochastic

    return result.images

def unwrap_transformer(transformer_ddp, accelerator=None):
    """Unwrap a DDP / accelerator-wrapped transformer to the raw module."""
    if transformer_ddp is None:
        return None
    if accelerator is not None and hasattr(accelerator, "unwrap_model"):
        return accelerator.unwrap_model(transformer_ddp)
    if hasattr(transformer_ddp, "module"):
        return transformer_ddp.module
    return transformer_ddp
