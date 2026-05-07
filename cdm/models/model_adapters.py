"""
Model Adapters for multi-model distillation support.

This module provides a unified abstraction layer that encapsulates the differences
between different diffusion model architectures (SD3, FLUX, etc.), enabling the
training pipeline to work with any supported model without architecture-specific
code in the training loop.

Each adapter handles:
  - Pipeline loading
  - Text encoder / tokenizer management
  - Text embedding computation
  - Transformer forward pass signature differences
  - LoRA target module specification
  - Latent decoding

Note: Sampling-related logic (prompt encoding for pipeline, latent preparation,
timestep scheduling, velocity prediction, output decoding/formatting) has been
removed. The native diffusers pipeline handles all sampling internally, with
TrainingSchedulerWrapper collecting intermediates. See pipeline_with_logprob.py.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

import torch

logger = logging.getLogger(__name__)

# ==================== LoRA Target Modules ====================

SD3_LORA_TARGET_MODULES = [
    "attn.add_k_proj", "attn.add_q_proj", "attn.add_v_proj", "attn.to_add_out",
    "attn.to_k", "attn.to_out.0", "attn.to_q", "attn.to_v",
]

SD3_LORA_FF_MODULES = [
    "ff.net.0.proj", "ff.net.2",
    "ff_context.net.0.proj", "ff_context.net.2",
]

LONGCAT_LORA_TARGET_MODULES = [
    "attn.to_k", "attn.to_q", "attn.to_v", "attn.to_out.0",
    "attn.add_k_proj", "attn.add_q_proj", "attn.add_v_proj", "attn.to_add_out",
]

LONGCAT_LORA_FF_MODULES = [
    "ff.net.0.proj", "ff.net.2",
    "ff_context.net.0.proj", "ff_context.net.2",
]

# ==================== Abstract Base Adapter ====================

class ModelAdapter(ABC):
    """Abstract base class for model-specific adapters.

    Adapters encapsulate architecture-specific differences for **training**:
      - Pipeline / model loading
      - Text encoder access
      - Text embedding computation
      - Transformer forward pass signatures
      - LoRA target modules
      - Latent decoding (for reward computation)

    Sampling is handled entirely by the native diffusers pipeline + scheduler
    wrapper (see ``pipeline_with_logprob.py``).
    """

    def __init__(self, config):
        self.config = config

    @abstractmethod
    def get_model_type(self) -> str:
        """Return the model type string (e.g. 'sd3' or 'longcat')."""

    @abstractmethod
    def load_pipeline(self, pretrained_path: str, **kwargs):
        """Load and return the diffusion pipeline."""

    @abstractmethod
    def get_text_encoders(self, pipeline) -> List:
        """Return list of text encoder models from the pipeline."""

    @abstractmethod
    def get_tokenizers(self, pipeline) -> List:
        """Return list of tokenizers from the pipeline."""

    @abstractmethod
    def freeze_non_trainable(self, pipeline):
        """Freeze all non-trainable components (VAE, text encoders)."""

    @abstractmethod
    def compute_text_embeddings(
        self, prompts, text_encoders, tokenizers, max_sequence_length, device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute text embeddings for prompts.

        Returns:
            (prompt_embeds, auxiliary_text_embeds) — the second element is
            model-specific: pooled projections for SD3, text position IDs
            for LongCat.
        """

    @abstractmethod
    def compute_negative_embeddings(
        self, text_encoders, tokenizers, batch_size, device
    ) -> Dict[str, torch.Tensor]:
        """
        Compute negative (unconditional) embeddings.

        Returns:
            Dict with keys: 'prompt_embeds', 'auxiliary_text_embeds'.
        """

    @abstractmethod
    def get_lora_target_modules(self) -> List[str]:
        """Return the list of LoRA target module names for this model."""

    def get_fsdp_wrap_cls_names(self) -> List[str]:
        """Return transformer block class names for FSDP auto-wrap policy.

        Each adapter returns the block class names that should be individually
        wrapped by FSDP (typically the repeated transformer blocks).  The
        training script reads these **before** ``Accelerator`` is created so
        that the FSDP plugin can be configured dynamically.

        Override in subclasses; the default raises ``NotImplementedError``.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement get_fsdp_wrap_cls_names()"
        )

    @abstractmethod
    def prepare_forward_kwargs(
        self, prompt_embeds: torch.Tensor, auxiliary_text_embeds: torch.Tensor,
        img_ids: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Convert generic text embeddings into model-specific forward kwargs.

        This is the key abstraction for multi-model support: the training loop
        only deals with (prompt_embeds, auxiliary_text_embeds) from
        ``compute_text_embeddings``, and each adapter translates them into the
        kwargs that its transformer actually needs.

        Args:
            prompt_embeds: Text encoder hidden states (all models use this).
            auxiliary_text_embeds: Second output of ``compute_text_embeddings``.
                For SD3 this is pooled projections; for LongCat this is text_ids.
            img_ids: Latent image position IDs captured from pipeline sampling.
                Required for LongCat (shape ``(B, seq_len, 3)``), ignored by SD3.

        Returns:
            Dict to be unpacked as ``**model_kwargs`` in ModelManager forward calls.
            Always contains ``encoder_hidden_states``; other keys are model-specific.
        """

    @abstractmethod
    def model_forward(self, model, hidden_states, timestep, **model_kwargs) -> torch.Tensor:
        """
        Unified transformer forward pass.

        Args:
            model: The transformer model.
            hidden_states: Noisy latent tensor.
            timestep: Timestep tensor.
            **model_kwargs: Model-specific kwargs produced by ``prepare_forward_kwargs()``.
                For SD3: encoder_hidden_states, pooled_projections.
                For LongCat: encoder_hidden_states, txt_ids.

        """

    @abstractmethod
    def model_forward_cfg_batched(
        self, model, xt, timestep,
        cond_kwargs, uncond_kwargs,
        **extra_kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Batched forward for CFG: computes both cond and uncond predictions.

        Args:
            model: The transformer model.
            xt: Noisy latent tensor.
            timestep: Timestep tensor.
            cond_kwargs: Conditional model kwargs from ``prepare_forward_kwargs()``.
            uncond_kwargs: Unconditional model kwargs from ``prepare_forward_kwargs()``.

        Returns:
            (uncond_prediction, cond_prediction)
        """

    @abstractmethod
    def decode_latents(self, latents, vae, image_processor=None, output_type="pt", latent_ids=None):
        """Decode latent tensors to images."""

    @abstractmethod
    def encode_images(self, images: torch.Tensor, vae) -> torch.Tensor:
        """Encode pixel-space images to latent space.
        
        Args:
            images: Pixel-space images, shape (B, 3, H, W), normalized to [-1, 1].
            vae: VAE model.
        
        Returns:
            Latent tensor in the same space as the training latents.
        """

    # ---- Shared utility methods ----

    def get_default_height(self) -> int:
        return 1024

    def get_default_width(self) -> int:
        return 1024

    def get_default_num_inference_steps(self) -> int:
        return 50

    def get_default_guidance_scale(self) -> float:
        return 1.0

    def get_max_sequence_length(self) -> int:
        """Return the default max_sequence_length for the text encoder.

        SD3 uses 256 (matching the official StableDiffusion3Pipeline default),
        while LongCat uses 512.
        """
        raise NotImplementedError

# ==================== SD3 Adapter ====================
class SD3Adapter(ModelAdapter):
    """Adapter for Stable Diffusion 3 models."""

    def get_model_type(self) -> str:
        return "sd3"

    def load_pipeline(self, pretrained_path: str, **kwargs):
        from diffusers import StableDiffusion3Pipeline
        pipeline = StableDiffusion3Pipeline.from_pretrained(pretrained_path, **kwargs)
        self.pipeline = pipeline
        return pipeline

    def get_text_encoders(self, pipeline) -> List:
        return [pipeline.text_encoder, pipeline.text_encoder_2, pipeline.text_encoder_3]

    def get_tokenizers(self, pipeline) -> List:
        return [pipeline.tokenizer, pipeline.tokenizer_2, pipeline.tokenizer_3]

    def freeze_non_trainable(self, pipeline):
        for model in [pipeline.vae, pipeline.text_encoder, pipeline.text_encoder_2, pipeline.text_encoder_3]:
            model.requires_grad_(False)

    def compute_text_embeddings(self, prompts, text_encoders, tokenizers, max_sequence_length, device):
        with torch.no_grad():
            prompt_embeds, _, pooled_prompt_embeds, _ = self.pipeline.encode_prompt(
                prompt=prompts,
                prompt_2=prompts,
                prompt_3=prompts,
                device=device,
                do_classifier_free_guidance=False,
                max_sequence_length=max_sequence_length,
            )
        return prompt_embeds, pooled_prompt_embeds  # auxiliary = pooled projections

    def compute_negative_embeddings(self, text_encoders, tokenizers, batch_size, device):
        prompt_embeds, auxiliary_text_embeds = self.compute_text_embeddings(
            [""] * batch_size, text_encoders, tokenizers, max_sequence_length=self.get_max_sequence_length(), device=device
        )
        return {
            "prompt_embeds": prompt_embeds,
            "auxiliary_text_embeds": auxiliary_text_embeds,
        }

    def get_max_sequence_length(self) -> int:
        return 256

    def prepare_forward_kwargs(
        self, prompt_embeds: torch.Tensor, auxiliary_text_embeds: torch.Tensor,
        img_ids: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:
        """SD3-specific: auxiliary_text_embeds carries pooled projections. img_ids is ignored."""
        return {
            "encoder_hidden_states": prompt_embeds,
            "pooled_projections": auxiliary_text_embeds,
        }

    def get_lora_target_modules(self, include_ff=False) -> List[str]:
        modules = list(SD3_LORA_TARGET_MODULES)
        if include_ff:
            modules.extend(SD3_LORA_FF_MODULES)
        return modules

    def get_fsdp_wrap_cls_names(self) -> List[str]:
        return ["JointTransformerBlock"]

    def model_forward(self, model, hidden_states, timestep, **model_kwargs):
        return model(
            hidden_states=hidden_states,
            timestep=timestep,
            **model_kwargs,
            return_dict=False,
        )[0]

    def model_forward_cfg_batched(
        self, model, xt, timestep,
        cond_kwargs, uncond_kwargs,
        **extra_kwargs
    ):
        batch_size = xt.shape[0]
        xt_batched = torch.cat([xt, xt], dim=0)
        t_batched = torch.cat([timestep, timestep], dim=0)
        # Merge cond/uncond kwargs by concatenating tensors along batch dim
        merged_kwargs = {}
        for key in cond_kwargs:
            cond_val = cond_kwargs[key]
            uncond_val = uncond_kwargs.get(key)
            if isinstance(cond_val, torch.Tensor) and uncond_val is not None:
                merged_kwargs[key] = torch.cat([uncond_val[:batch_size], cond_val], dim=0)
            else:
                merged_kwargs[key] = cond_val

        preds_batched = self.model_forward(model, xt_batched, t_batched, **merged_kwargs)
        uncond_pred, cond_pred = preds_batched.chunk(2, dim=0)
        return uncond_pred, cond_pred

    def decode_latents(self, latents, vae, image_processor=None, output_type="pt", latent_ids=None):
        """
        Decode SD3 latents to images.

        SD3 VAE uses scaling_factor and shift_factor for latent denormalization.
        """
        scaling_factor = vae.config.scaling_factor
        shift_factor = getattr(vae.config, 'shift_factor', 0.0)
        scaled_latents = (latents / scaling_factor) + shift_factor
        scaled_latents = scaled_latents.to(dtype=vae.dtype)
        decoded = vae.decode(scaled_latents, return_dict=False)[0]

        if image_processor is not None:
            images = image_processor.postprocess(decoded, output_type=output_type)
        else:
            images = (decoded / 2 + 0.5).clamp(0, 1).float()

        return images

    def encode_images(self, images, vae):
        """Encode images to SD3 latent space: apply VAE encode then normalize."""
        images = images.to(dtype=vae.dtype)
        latent_dist = vae.encode(images).latent_dist
        latents = latent_dist.sample()
        scaling_factor = vae.config.scaling_factor
        shift_factor = getattr(vae.config, 'shift_factor', 0.0)
        return (latents - shift_factor) * scaling_factor

# ==================== LongCat Helpers ====================

def _patch_longcat_rope_dtype(transformer):
    """Cast LongCat RoPE cos/sin to the transformer compute dtype.

    LongCat's ``LongCatImagePosEmbed`` always returns fp32 freqs (uses fp64
    internally) regardless of the compute dtype. This breaks FSDP2 +
    non-reentrant gradient checkpointing: FSDP2's saved-tensor pack downcasts
    the forward intermediates to bf16, while recompute keeps them fp32,
    triggering ``CheckpointError: saved/recomputed dtype mismatch``.

    Casting freqs to ``transformer.dtype`` aligns both paths and matches the
    SD3/Flux convention (RoPE in bf16, numerically validated in practice).
    Applied via a forward hook so ``transformer.to(dtype)`` stays self-consistent.
    """
    def _cast_hook(_module, _inputs, output):
        cos, sin = output
        dt = transformer.dtype
        # print(f"LongCat RoPE dtype: {cos.dtype} -> {dt}")
        return cos.to(dt), sin.to(dt)

    transformer.pos_embed.register_forward_hook(_cast_hook)


class _LongCatIdsCastTransformerProxy:
    """Transparent callable proxy that converts ``txt_ids``/``img_ids`` to int64
    *before* invoking the wrapped (FSDP/DDP) transformer.
    """

    def __init__(self, wrapped):
        # Use object.__setattr__ to avoid recursing into our own __setattr__.
        object.__setattr__(self, "_wrapped", wrapped)

    @staticmethod
    def _cast_kwargs(kwargs):
        for key in ("txt_ids", "img_ids"):
            v = kwargs.get(key)
            if isinstance(v, torch.Tensor) and v.is_floating_point():
                kwargs[key] = v.to(torch.int64)
        return kwargs

    def __call__(self, *args, **kwargs):
        from accelerate.utils.operations import convert_to_fp32

        kwargs = self._cast_kwargs(kwargs)

        device_type = "cuda" if torch.cuda.is_available() else "cpu"
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            out = self._wrapped(*args, **kwargs)
        return convert_to_fp32(out)

    # ----- Transparent attribute proxying -----
    def __getattr__(self, name):
        # __getattr__ is only invoked when normal lookup fails (e.g. not in
        # self.__dict__), so proxying via _wrapped is safe and avoids loops.
        return getattr(object.__getattribute__(self, "_wrapped"), name)

    def __setattr__(self, name, value):
        setattr(object.__getattribute__(self, "_wrapped"), name, value)

    def __delattr__(self, name):
        delattr(object.__getattribute__(self, "_wrapped"), name)

    def __repr__(self):
        return f"_LongCatIdsCastTransformerProxy({object.__getattribute__(self, '_wrapped')!r})"

def wrap_longcat_transformer_for_fsdp(module):
    """Wrap a (possibly FSDP/DDP-wrapped) LongCat transformer with the ids cast proxy.

    Idempotent: re-wrapping an already-wrapped module returns it unchanged.
    Safe to call on ``None`` (returns None), so callers don't need to guard.
    """
    if module is None:
        return None
    if isinstance(module, _LongCatIdsCastTransformerProxy):
        return module
    return _LongCatIdsCastTransformerProxy(module)

def _longcat_prepare_pos_ids(modality_id=0, pos_type="text", start=(0, 0), num_token=None, height=None, width=None):
    """
    Generate 3D position IDs for LongCat transformer (modality_id, row, col).

    Mirrors the ``prepare_pos_ids`` function from the LongCat pipeline.

    Args:
        modality_id: 0 for text, 1 for image.
        pos_type: "text" or "image".
        start: (row_start, col_start) offset.
        num_token: Number of tokens (required for "text" type).
        height: Spatial height in patches (required for "image" type).
        width: Spatial width in patches (required for "image" type).

    Returns:
        Tensor of shape (num_token, 3) or (height*width, 3).
    """
    if pos_type == "text":
        pos_ids = torch.zeros(num_token, 3)
        pos_ids[..., 0] = modality_id
        pos_ids[..., 1] = torch.arange(num_token) + start[0]
        pos_ids[..., 2] = torch.arange(num_token) + start[1]
    elif pos_type == "image":
        pos_ids = torch.zeros(height, width, 3)
        pos_ids[..., 0] = modality_id
        pos_ids[..., 1] = pos_ids[..., 1] + torch.arange(height)[:, None] + start[0]
        pos_ids[..., 2] = pos_ids[..., 2] + torch.arange(width)[None, :] + start[1]
        pos_ids = pos_ids.reshape(height * width, 3)
    else:
        raise ValueError(f'Unknown pos_type "{pos_type}", only "text" or "image" are supported.')
    return pos_ids

# ==================== LongCat Adapter ====================

class LongCatAdapter(ModelAdapter):
    """
    Adapter for MeiTuan LongCat-Image models.

    Key characteristics:
      - Uses LongCatImagePipeline with Qwen2.5-VL text encoder
      - Transformer forward requires txt_ids, img_ids
      - CFG via two separate forwards (cond vs uncond)
      - Single Qwen2.5-VL text encoder with custom tokenization and prompt template
      - Packed latents (2x2 patchification)
    """

    def get_model_type(self) -> str:
        return "longcat"

    def load_pipeline(self, pretrained_path: str, **kwargs):
        from diffusers import LongCatImagePipeline
        pipeline = LongCatImagePipeline.from_pretrained(pretrained_path, **kwargs)
        self.pipeline = pipeline
        _patch_longcat_rope_dtype(pipeline.transformer)
        return pipeline

    def get_text_encoders(self, pipeline) -> List:
        return [pipeline.text_encoder]

    def get_tokenizers(self, pipeline) -> List:
        return [pipeline.tokenizer]

    def freeze_non_trainable(self, pipeline):
        for model in [pipeline.vae, pipeline.text_encoder]:
            model.requires_grad_(False)

    def compute_text_embeddings(self, prompts, text_encoders, tokenizers, max_sequence_length, device):
        """
        Compute text embeddings for LongCat using the pipeline's internal encoding.

        LongCat uses a custom tokenization flow with prefix/suffix prompt templates
        and special quotation-aware splitting. We delegate to the pipeline's
        ``_encode_prompt`` to ensure exact consistency.

        Returns:
            (prompt_embeds, text_ids) where text_ids has shape (B, seq_len, 3).
            Note: The LongCat transformer internally expects unbatched txt_ids
            (seq_len, 3), but we return batched format here for consistency with
            the training script's generic batch handling (``_repeat_embeds``,
            batch slicing). The ``model_forward`` method strips the batch dim
            before passing to the transformer.
        """
        with torch.no_grad():
            prompt_embeds = self.pipeline._encode_prompt(
                prompts if isinstance(prompts, list) else [prompts]
            )
            batch_size = prompt_embeds.shape[0]
            text_ids = _longcat_prepare_pos_ids(
                modality_id=0, pos_type="text",
                start=(0, 0), num_token=prompt_embeds.shape[1],
            ).to(device)
            # Expand to batched format (B, seq_len, 3) for consistency with
            # the training script's generic batch handling
            text_ids = text_ids.unsqueeze(0).expand(batch_size, -1, -1)

        return prompt_embeds.to(device), text_ids

    def compute_negative_embeddings(self, text_encoders, tokenizers, batch_size, device):
        prompt_embeds, auxiliary_text_embeds = self.compute_text_embeddings(
            [""] * batch_size, text_encoders, tokenizers,
            max_sequence_length=self.get_max_sequence_length(), device=device,
        )
        return {
            "prompt_embeds": prompt_embeds,
            "auxiliary_text_embeds": auxiliary_text_embeds,
        }

    def get_max_sequence_length(self) -> int:
        return 512

    def get_lora_target_modules(self, include_ff=False) -> List[str]:
        modules = list(LONGCAT_LORA_TARGET_MODULES)
        if include_ff:
            modules.extend(LONGCAT_LORA_FF_MODULES)
        return modules

    def get_fsdp_wrap_cls_names(self) -> List[str]:
        return ["LongCatImageTransformerBlock", "LongCatImageSingleTransformerBlock"]

    def prepare_forward_kwargs(
        self, prompt_embeds: torch.Tensor, auxiliary_text_embeds: torch.Tensor,
        img_ids: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:
        """
        LongCat-specific: auxiliary_text_embeds carries text_ids (3D position encodings),
        img_ids carries latent image position IDs captured from pipeline sampling.
        """
        kwargs = {
            "encoder_hidden_states": prompt_embeds,
            "txt_ids": auxiliary_text_embeds,
        }
        if img_ids is not None:
            kwargs["img_ids"] = img_ids
        return kwargs

    def model_forward(self, model, hidden_states, timestep, **model_kwargs):
        """
        LongCat transformer forward pass.

        The LongCat transformer expects timestep in [0, 1] range (it internally
        multiplies by 1000). The training framework passes timestep in [0, 1000],
        so this method handles the conversion.

        Note: txt_ids and img_ids are stored in batched format (B, seq_len, 3)
        for compatibility with the training script's generic batch handling,
        but the LongCat transformer expects unbatched (seq_len, 3). We strip
        the batch dimension here since position IDs are identical across the batch.
        """
        encoder_hidden_states = model_kwargs["encoder_hidden_states"]
        txt_ids = model_kwargs["txt_ids"]
        img_ids = model_kwargs["img_ids"]

        # Strip batch dimension: (B, seq_len, 3) -> (seq_len, 3)
        if txt_ids.dim() == 3:
            txt_ids = txt_ids[0]
        if img_ids.dim() == 3:
            img_ids = img_ids[0]

        normalized_timestep = timestep / 1000.0

        return model(
            hidden_states=hidden_states,
            timestep=normalized_timestep,
            guidance=None,
            encoder_hidden_states=encoder_hidden_states,
            txt_ids=txt_ids.to(dtype=torch.int64),
            img_ids=img_ids.to(dtype=torch.int64),
            return_dict=False,
        )[0]

    def model_forward_cfg_batched(
        self, model, xt, timestep,
        cond_kwargs, uncond_kwargs,
        **extra_kwargs
    ):
        """
        LongCat CFG batched forward.

        CFG is done via two separate forwards with different text embeddings
        (cond vs uncond), consistent with the pipeline's denoising loop.
        """
        uncond_pred = self.model_forward(model, xt, timestep, **uncond_kwargs)
        cond_pred = self.model_forward(model, xt, timestep, **cond_kwargs)
        return uncond_pred, cond_pred

    def decode_latents(self, latents, vae, image_processor=None, output_type="pt", latent_ids=None):
        """
        Decode LongCat packed latents to images.

        Expects packed latents (B, seq_len, C*4) and latent_ids (B, seq_len, 3).
        Pipeline: unpack → denormalize → VAE decode → postprocess.
        """
        from diffusers import LongCatImagePipeline

        ids = latent_ids[0] if latent_ids.dim() == 3 else latent_ids
        vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
        patch_height = int(ids[:, 1].max().item() - ids[:, 1].min().item()) + 1
        patch_width = int(ids[:, 2].max().item() - ids[:, 2].min().item()) + 1
        height = patch_height * vae_scale_factor * 2
        width = patch_width * vae_scale_factor * 2

        latents = LongCatImagePipeline._unpack_latents(latents, height, width, vae_scale_factor)

        scaling_factor = vae.config.scaling_factor
        shift_factor = getattr(vae.config, 'shift_factor', 0.0)
        latents = (latents / scaling_factor) + shift_factor
        latents = latents.to(dtype=vae.dtype)
        decoded = vae.decode(latents, return_dict=False)[0]

        if image_processor is not None:
            images = image_processor.postprocess(decoded, output_type=output_type)
        else:
            images = (decoded / 2 + 0.5).clamp(0, 1).float()

        return images

    def encode_images(self, images, vae):
        """Encode images to LongCat latent space: VAE encode → normalize → pack."""
        from diffusers import LongCatImagePipeline
        from diffusers.pipelines.longcat_image.pipeline_longcat_image_edit import retrieve_latents

        images = images.to(dtype=vae.dtype)
        latents = retrieve_latents(vae.encode(images))
        scaling_factor = vae.config.scaling_factor
        shift_factor = getattr(vae.config, 'shift_factor', 0.0)
        latents = (latents - shift_factor) * scaling_factor

        batch_size, num_channels, height, width = latents.shape
        return LongCatImagePipeline._pack_latents(
            latents, batch_size, num_channels, height, width)
    def get_default_num_inference_steps(self) -> int:
        return 50

    def get_default_guidance_scale(self) -> float:
        return 4.5

# ==================== SD3 HyperSD Adapter ====================

class SD3HyperSDAdapter(SD3Adapter):
    """Adapter for SD3 with Hyper-SD LoRA for accelerated inference.

    Inherits all behavior from SD3Adapter. The only difference is that
    ``load_pipeline`` additionally loads and fuses the Hyper-SD LoRA weights
    into the transformer, enabling fewer-step generation.
    """

    HYPER_SD_REPO = "ByteDance/Hyper-SD"
    HYPER_SD_CKPT = "Hyper-SD3-8steps-CFG-lora.safetensors"
    HYPER_SD_LORA_SCALE = 0.125

    def get_model_type(self) -> str:
        return "sd3"

    def load_pipeline(self, pretrained_path: str, **kwargs):
        from huggingface_hub import hf_hub_download

        pipeline = super().load_pipeline(pretrained_path, **kwargs)

        ckpt_path = hf_hub_download(self.HYPER_SD_REPO, self.HYPER_SD_CKPT)
        pipeline.load_lora_weights(ckpt_path)
        pipeline.fuse_lora(lora_scale=self.HYPER_SD_LORA_SCALE)
        logger.info(
            f"Loaded and fused Hyper-SD LoRA: {self.HYPER_SD_CKPT} "
            f"(scale={self.HYPER_SD_LORA_SCALE})"
        )

        return pipeline

    def get_default_num_inference_steps(self) -> int:
        return 8

    def get_default_guidance_scale(self) -> float:
        return 5.0

# ==================== SD3 TDM Adapter ====================

class SD3TDMAdapter(SD3Adapter):
    """Adapter for SD3 with TDM LoRA, Tiny VAE, and DPM-Solver scheduler.

    Inherits all behavior from SD3Adapter. ``load_pipeline`` additionally:
      1. Loads TDM LoRA weights (without fusing)
      2. Replaces VAE with AutoencoderTiny (shift_factor=0)
      3. Replaces scheduler with DPMSolverMultistepScheduler (flow_shift=6)
    """

    TDM_LORA_REPO = "Luo-Yihong/TDM_sd3_lora"
    TDM_LORA_SCALE = 0.125
    TINY_VAE_REPO = "madebyollin/taesd3"
    SCHEDULER_REPO = "Efficient-Large-Model/Sana_1600M_1024px_BF16_diffusers"
    FLOW_SHIFT = 6

    def get_model_type(self) -> str:
        return "sd3"

    def load_pipeline(self, pretrained_path: str, **kwargs):
        from diffusers import AutoencoderTiny, DPMSolverMultistepScheduler

        pipeline = super().load_pipeline(pretrained_path, **kwargs)

        # LoRA
        pipeline.load_lora_weights(self.TDM_LORA_REPO, adapter_name="tdm")
        pipeline.set_adapters(["tdm"], [self.TDM_LORA_SCALE])

        # Tiny VAE
        vae_dtype = kwargs.get("torch_dtype", torch.float16)
        pipeline.vae = AutoencoderTiny.from_pretrained(self.TINY_VAE_REPO, torch_dtype=vae_dtype)
        pipeline.vae.config.shift_factor = 0.0

        # DPM-Solver scheduler with flow_shift
        pipeline.scheduler = DPMSolverMultistepScheduler.from_pretrained(
            self.SCHEDULER_REPO, subfolder="scheduler"
        )
        pipeline.scheduler.config["flow_shift"] = self.FLOW_SHIFT
        pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)

        logger.info(
            f"Loaded TDM: LoRA={self.TDM_LORA_REPO} (scale={self.TDM_LORA_SCALE}), "
            f"VAE={self.TINY_VAE_REPO}, flow_shift={self.FLOW_SHIFT}"
        )
        return pipeline

    def get_default_num_inference_steps(self) -> int:
        return 4

    def get_default_guidance_scale(self) -> float:
        return 1.0

# ==================== SD3 Flash Adapter ====================

class SD3FlashAdapter(SD3Adapter):
    """Adapter for SD3 with Flash LoRA and FlashFlowMatch scheduler.

    Uses PeftModel to load the flash-sd3 LoRA and replaces the scheduler
    with FlashFlowMatchEulerDiscreteScheduler. text_encoder_3 / tokenizer_3
    are disabled to match the original flash-sd3 setup.
    """

    FLASH_LORA_REPO = "jasperai/flash-sd3"

    def get_model_type(self) -> str:
        return "sd3"

    def load_pipeline(self, pretrained_path: str, **kwargs):
        from diffusers import (
            StableDiffusion3Pipeline,
            SD3Transformer2DModel,
            FlashFlowMatchEulerDiscreteScheduler,
        )
        from peft import PeftModel as PeftModelCls

        transformer = SD3Transformer2DModel.from_pretrained(
            pretrained_path, subfolder="transformer",
            torch_dtype=kwargs.get("torch_dtype", torch.float16),
        )
        transformer = PeftModelCls.from_pretrained(transformer, self.FLASH_LORA_REPO)

        pipeline = StableDiffusion3Pipeline.from_pretrained(
            pretrained_path,
            transformer=transformer,
            text_encoder_3=None,
            tokenizer_3=None,
            **kwargs,
        )
        self.pipeline = pipeline

        pipeline.scheduler = FlashFlowMatchEulerDiscreteScheduler.from_pretrained(
            pretrained_path, subfolder="scheduler",
        )

        logger.info(f"Loaded Flash-SD3: LoRA={self.FLASH_LORA_REPO}")
        return pipeline

    def get_text_encoders(self, pipeline) -> List:
        return [pipeline.text_encoder, pipeline.text_encoder_2]

    def get_tokenizers(self, pipeline) -> List:
        return [pipeline.tokenizer, pipeline.tokenizer_2]

    def get_default_num_inference_steps(self) -> int:
        return 4

    def get_default_guidance_scale(self) -> float:
        return 0.0

# ==================== Adapter Factory ====================

_ADAPTER_REGISTRY = {
    "sd3": SD3Adapter,
    "sd3_hypersd": SD3HyperSDAdapter,
    "sd3_tdm": SD3TDMAdapter,
    "sd3_flash": SD3FlashAdapter,
    "longcat": LongCatAdapter,
}

def create_model_adapter(base_model: str, config=None) -> ModelAdapter:
    """
    Factory function to create the appropriate model adapter.

    Args:
        base_model: Model identifier (e.g., 'sd3', 'longcat')
        config: Optional config object

    Returns:
        ModelAdapter instance
    """
    adapter_cls = _ADAPTER_REGISTRY.get(base_model)
    if adapter_cls is None:
        raise ValueError(
            f"Unknown base_model '{base_model}'. "
            f"Supported models: {list(_ADAPTER_REGISTRY.keys())}"
        )
    return adapter_cls(config)
