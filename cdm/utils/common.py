# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import random
import logging
import numpy as np
import torch
import torch.distributed as dist
from PIL import Image, ImageDraw, ImageFont
import textwrap

from flash_attn import flash_attn_func
from diffusers.models.transformers.transformer_flux import FluxAttnProcessor, _get_qkv_projections
from diffusers.models.embeddings import apply_rotary_emb
from diffusers.models.attention_processor import JointAttnProcessor2_0
from typing import Optional

logger = logging.getLogger(__name__)

def _repeat_embeds(embeds, num_repeats):
    """Repeat embeddings along the batch dimension, handling 1D, 2D and 3D tensors.

    SD3 pooled_prompt_embeds are 2D (B, D), LongCat text_ids are 3D (B, L, 3).
    This helper picks the correct repeat signature automatically.

    Returns None if *embeds* is None.
    """
    if embeds is None:
        return None
    if embeds.dim() == 1:
        return embeds.repeat(num_repeats)
    elif embeds.dim() == 2:
        return embeds.repeat(num_repeats, 1)
    elif embeds.dim() == 3:
        return embeds.repeat(num_repeats, 1, 1)
    return embeds

def new_flux_attn_call(
    self,
    attn,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    image_rotary_emb: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    # 1. QKV 投影
    query, key, value, encoder_query, encoder_key, encoder_value = _get_qkv_projections(
        attn, hidden_states, encoder_hidden_states
    )
    query = query.unflatten(-1, (attn.heads, -1))
    key = key.unflatten(-1, (attn.heads, -1))
    value = value.unflatten(-1, (attn.heads, -1))

    # 2. QK Norm
    query = attn.norm_q(query)
    key = attn.norm_k(key)

    # 3. 拼接 encoder 的 QKV（如果有 cross-attention）
    if attn.added_kv_proj_dim is not None:
        encoder_query = encoder_query.unflatten(-1, (attn.heads, -1))
        encoder_key = encoder_key.unflatten(-1, (attn.heads, -1))
        encoder_value = encoder_value.unflatten(-1, (attn.heads, -1))
        encoder_query = attn.norm_added_q(encoder_query)
        encoder_key = attn.norm_added_k(encoder_key)
        query = torch.cat([encoder_query, query], dim=1)
        key = torch.cat([encoder_key, key], dim=1)
        value = torch.cat([encoder_value, value], dim=1)

    # 4. RoPE
    if image_rotary_emb is not None:
        query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
        key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)

    # 5. Flash Attention
    original_dtype = query.dtype
    hidden_states = flash_attn_func(
        query.to(torch.bfloat16),
        key.to(torch.bfloat16),
        value.to(torch.bfloat16),
        deterministic=True,
    )
    hidden_states = hidden_states.to(original_dtype)
    hidden_states = hidden_states.flatten(2, 3)

    # 6. 输出投影
    if encoder_hidden_states is not None:
        encoder_hidden_states, hidden_states = hidden_states.split_with_sizes(
            [encoder_hidden_states.shape[1], hidden_states.shape[1] - encoder_hidden_states.shape[1]], dim=1
        )
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        encoder_hidden_states = attn.to_add_out(encoder_hidden_states)
        return hidden_states, encoder_hidden_states

    return hidden_states


def new_sd3_joint_attn_call(
    self,
    attn,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor = None,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    context_input_ndim = encoder_hidden_states.ndim if encoder_hidden_states is not None else None

    # 1. QKV projections for hidden_states
    query = attn.to_q(hidden_states)
    key = attn.to_k(hidden_states)
    value = attn.to_v(hidden_states)

    query = query.unflatten(2, (attn.heads, -1))
    key = key.unflatten(2, (attn.heads, -1))
    value = value.unflatten(2, (attn.heads, -1))

    # 2. QK Norm
    if attn.norm_q is not None:
        query = attn.norm_q(query)
    if attn.norm_k is not None:
        key = attn.norm_k(key)

    # 3. Encoder QKV projections (joint attention)
    if encoder_hidden_states is not None and hasattr(attn, 'add_q_proj') and attn.add_q_proj is not None:
        encoder_query = attn.add_q_proj(encoder_hidden_states)
        encoder_key = attn.add_k_proj(encoder_hidden_states)
        encoder_value = attn.add_v_proj(encoder_hidden_states)

        encoder_query = encoder_query.unflatten(2, (attn.heads, -1))
        encoder_key = encoder_key.unflatten(2, (attn.heads, -1))
        encoder_value = encoder_value.unflatten(2, (attn.heads, -1))

        if attn.norm_added_q is not None:
            encoder_query = attn.norm_added_q(encoder_query)
        if attn.norm_added_k is not None:
            encoder_key = attn.norm_added_k(encoder_key)

        query = torch.cat([encoder_query, query], dim=1)
        key = torch.cat([encoder_key, key], dim=1)
        value = torch.cat([encoder_value, value], dim=1)

    # 4. Flash Attention (deterministic)
    original_dtype = query.dtype
    hidden_states = flash_attn_func(
        query.to(torch.bfloat16),
        key.to(torch.bfloat16),
        value.to(torch.bfloat16),
        deterministic=True,
    )
    hidden_states = hidden_states.to(original_dtype)
    hidden_states = hidden_states.flatten(2, 3)

    # 5. Output projections
    if encoder_hidden_states is not None:
        if hasattr(attn, 'add_q_proj') and attn.add_q_proj is not None:
            encoder_seq_len = encoder_hidden_states.shape[1]
            encoder_hidden_states_out, hidden_states = hidden_states.split(
                [encoder_seq_len, hidden_states.shape[1] - encoder_seq_len], dim=1
            )

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if not getattr(attn, 'context_pre_only', False):
            encoder_hidden_states_out = attn.to_add_out(encoder_hidden_states_out)
            return hidden_states, encoder_hidden_states_out
        else:
            return hidden_states, encoder_hidden_states

    hidden_states = attn.to_out[0](hidden_states)
    hidden_states = attn.to_out[1](hidden_states)
    return hidden_states


def patch_sd3_attn_processor():
    JointAttnProcessor2_0.__call__ = new_sd3_joint_attn_call
    print("JointAttnProcessor2_0 successfully patched with Flash Attention (Deterministic).")


def patch_flux_attn_processor():
    FluxAttnProcessor.__call__ = new_flux_attn_call 
    print("FluxAttnProcessor successfully patched with Flash Attention Varlen (Deterministic).")


def enable_full_determinism(seed=42, warn_only=False):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=warn_only)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    # torch.set_default_dtype(torch.float32)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['FLASH_ATTENTION_DETERMINISTIC'] = '1'
    os.environ['NCCL_DETERMINISTIC'] = '1'
    os.environ['NVTE_ALLOW_NONDETERMINISTIC_ALGO'] = '0'
    # os.environ['NCCL_ALGO'] = 'Ring'
    # os.environ['NCCL_MAX_NCHANNELS'] = '1'
    patch_sd3_attn_processor()
    patch_flux_attn_processor()


def setup_distributed(rank, lock_rank, world_size):
    """Initialize distributed training environment."""
    os.environ["MASTER_ADDR"] = os.getenv("MASTER_ADDR", "localhost")
    os.environ["MASTER_PORT"] = os.getenv("MASTER_PORT", "12355")
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(lock_rank)


def cleanup_distributed():
    """Clean up distributed training environment."""
    # 在销毁进程组之前，先清理 CUDA 缓存，避免 OOM 错误
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    dist.destroy_process_group()


def is_main_process(rank):
    """Check if current process is the main process."""
    return rank == 0


def set_seed(seed: int, rank: int = 0, deterministic: bool = False):
    """Set random seed for reproducibility.
    
    Args:
        seed: Base random seed
        rank: Process rank for distributed training
        deterministic: If True, enable fully deterministic mode (may reduce performance)
    """
    random.seed(seed + rank)
    np.random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    torch.cuda.manual_seed_all(seed + rank)
    
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        # Set CUBLAS workspace config for deterministic CuBLAS operations (CUDA >= 10.2)
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", value=":4096:8")
        
        if hasattr(torch, 'use_deterministic_algorithms'):
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except Exception:
                pass


def add_caption_to_image(image_tensor, caption):
    """Add caption text overlay to an image tensor.
    
    Args:
        image_tensor: Tensor of shape (C, H, W)
        caption: Text string to overlay on the image
        
    Returns:
        Tensor of shape (C, H, W) with caption overlay
    """
    if image_tensor.min() < 0:
        image_tensor = (image_tensor * 0.5 + 0.5).clamp(0, 1)
    
    ndarr = image_tensor.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    image = Image.fromarray(ndarr)
    draw = ImageDraw.Draw(image)
    
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
    except:
        font = ImageFont.load_default()
        
    char_width = 10
    wrap_width = max(1, image.width // char_width)
    lines = textwrap.wrap(caption, width=wrap_width)
    if len(lines) > 5:
        lines = lines[:5]
        lines[-1] += "..."
    text = "\n".join(lines)
    
    draw.text((10, 10), text, font=font, fill=(255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0))
    
    return torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0


def gather_tensor_to_all(tensor, world_size):
    """Gather tensors from all processes to all processes.
    
    Args:
        tensor: Local tensor to gather
        world_size: Number of processes
        
    Returns:
        Concatenated tensor from all processes on CPU
    """
    gathered_tensors = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered_tensors, tensor)
    return torch.cat(gathered_tensors, dim=0).cpu()

def encode_prompt(
    text_encoders,
    tokenizers,
    prompt: str,
    max_sequence_length,
    device=None,
    num_images_per_prompt: int = 1,
    text_input_ids_list=None,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt

    clip_tokenizers = tokenizers[:2]
    clip_text_encoders = text_encoders[:2]

    clip_prompt_embeds_list = []
    clip_pooled_prompt_embeds_list = []
    for i, (tokenizer, text_encoder) in enumerate(zip(clip_tokenizers, clip_text_encoders)):
        prompt_embeds, pooled_prompt_embeds = _encode_prompt_with_clip(
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            prompt=prompt,
            device=device if device is not None else text_encoder.device,
            num_images_per_prompt=num_images_per_prompt,
            text_input_ids=text_input_ids_list[i] if text_input_ids_list else None,
        )
        clip_prompt_embeds_list.append(prompt_embeds)
        clip_pooled_prompt_embeds_list.append(pooled_prompt_embeds)

    clip_prompt_embeds = torch.cat(clip_prompt_embeds_list, dim=-1)
    pooled_prompt_embeds = torch.cat(clip_pooled_prompt_embeds_list, dim=-1)

    t5_prompt_embed = _encode_prompt_with_t5(
        text_encoders[-1],
        tokenizers[-1],
        max_sequence_length,
        prompt=prompt,
        num_images_per_prompt=num_images_per_prompt,
        text_input_ids=text_input_ids_list[-1] if text_input_ids_list else None,
        device=device if device is not None else text_encoders[-1].device,
    )

    clip_prompt_embeds = torch.nn.functional.pad(
        clip_prompt_embeds, (0, t5_prompt_embed.shape[-1] - clip_prompt_embeds.shape[-1])
    )
    prompt_embeds = torch.cat([clip_prompt_embeds, t5_prompt_embed], dim=-2)

    return prompt_embeds, pooled_prompt_embeds


def compute_text_embeddings(prompt, text_encoders, tokenizers, max_sequence_length, device):
    """Compute text embeddings for prompts using SD3 text encoders.
    
    Note: This function is SD3-specific (expects 3 text encoders: CLIP×2 + T5).
    For multi-model support, use model_adapter.compute_text_embeddings() instead,
    which handles architecture-specific text encoding (e.g., FLUX uses 2 encoders).
    
    Args:
        prompt: Text prompt or list of prompts
        text_encoders: List of 3 text encoder models (CLIP, CLIP, T5)
        tokenizers: List of 3 tokenizers
        max_sequence_length: Maximum sequence length for encoding
        device: Target device for embeddings
        
    Returns:
        Tuple of (prompt_embeds, pooled_prompt_embeds)
    """
    # from flow_grpo.pipeline.train_dreambooth_lora_sd3 import encode_prompt
    
    with torch.no_grad():
        prompt_embeds, pooled_prompt_embeds = encode_prompt(text_encoders, tokenizers, prompt, max_sequence_length)
        prompt_embeds = prompt_embeds.to(device)
        pooled_prompt_embeds = pooled_prompt_embeds.to(device)
    return prompt_embeds, pooled_prompt_embeds


def sample_sigmas(batch_size, method, device, sigma_min=0.02, sigma_max=0.98, **kwargs):
    """
    Sample sigmas based on the given method.

    Always returns float32 for numerical precision.

    All methods clamp the final output to [sigma_min, sigma_max] to avoid
    numerical instability near the boundaries (e.g. division by sigma when
    sigma -> 0, or vanishing signal when sigma -> 1).

    Args:
        batch_size: Number of sigmas to sample.
        method: "uniform", "logit_normal", or "uniform_capped".
        device: Torch device.
        sigma_min: Lower bound for sigma truncation (applied to all methods).
        sigma_max: Upper bound for sigma truncation (applied to all methods).
        **kwargs: Additional args:
            - logit_mean, logit_std: For logit_normal method.
            - upper_bound: Scalar or tensor of shape (batch_size,) for "uniform_capped" method.
              Samples uniformly in [0, upper_bound], then clamped to [sigma_min, sigma_max].
    """
    if method == "uniform_capped":
        upper_bound = kwargs.get('upper_bound')
        if upper_bound is None:
            raise ValueError("method='uniform_capped' requires 'upper_bound' kwarg (scalar or tensor of shape (batch_size,))")
        upper_bound = torch.as_tensor(upper_bound, device=device, dtype=torch.float32)
        sigmas = torch.rand(batch_size, device=device, dtype=torch.float32) * upper_bound
    elif method == "logit_normal":
        u = torch.normal(mean=kwargs.get('logit_mean', 0.0), std=kwargs.get('logit_std', 1.0),
                        size=(batch_size,), device=device, dtype=torch.float32)
        sigmas = torch.nn.functional.sigmoid(u)
    else:  # uniform
        sigmas = torch.rand(batch_size, device=device, dtype=torch.float32)
    return sigmas.clamp(sigma_min, sigma_max)

def return_decay(step, decay_type):
    """Calculate decay factor based on step and decay type.
    
    Args:
        step: Current training step
        decay_type: Type of decay (0, 1, or 2)
        
    Returns:
        Decay factor between 0 and 1
    """
    if decay_type == 0:
        flat = 0
        uprate = 0.0
        uphold = 0.0
    elif decay_type == 1:
        flat = 0
        uprate = 0.001
        uphold = 0.5
    elif decay_type == 2:
        flat = 75
        uprate = 0.0075
        uphold = 0.999
    else:
        assert False

    if step < flat:
        return 0.0
    else:
        decay = (step - flat) * uprate
        return min(decay, uphold)


def calculate_zero_std_ratio(prompts, gathered_rewards):
    """Calculate the ratio of prompts with zero standard deviation in rewards.
    
    Args:
        prompts: List of prompt strings
        gathered_rewards: Dictionary containing 'avg' rewards
        
    Returns:
        Tuple of (zero_std_ratio, mean_std)
    """
    prompt_array = np.array(prompts)
    unique_prompts, inverse_indices, counts = np.unique(prompt_array, return_inverse=True, return_counts=True)
    
    # 处理 rewards 可能是 1D 或 2D 的情况
    rewards_avg = gathered_rewards["avg"]
    if rewards_avg.ndim == 1:
        # 1D 情况：直接使用
        grouped_rewards = rewards_avg[np.argsort(inverse_indices)]
    else:
        # 2D 情况：取第一个 timestep
        grouped_rewards = rewards_avg[np.argsort(inverse_indices), 0]
    
    split_indices = np.cumsum(counts)[:-1]
    reward_groups = np.split(grouped_rewards, split_indices)
    prompt_std_devs = np.array([np.std(group) for group in reward_groups])
    zero_std_count = np.count_nonzero(prompt_std_devs == 0)
    zero_std_ratio = zero_std_count / len(prompt_std_devs)
    return zero_std_ratio, prompt_std_devs.mean()


def get_mixed_precision_dtype(mp_setting):
    """Get torch dtype from mixed precision setting string.

    Args:
        mp_setting: Mixed precision setting string ("fp16", "bf16", or None).

    Returns:
        Corresponding torch dtype, or None if not recognized.
    """
    return {"fp16": torch.float16, "bf16": torch.bfloat16}.get(mp_setting)


def create_autocast_context(accelerator):
    """Create appropriate autocast context based on distributed type.

    FSDP manages mixed precision via its own MixedPrecision policy, so wrapping
    forward passes with torch.autocast is redundant and can cause dtype conflicts.
    For DDP/single-GPU, accelerator.autocast() is the standard approach.
    """
    import contextlib
    from accelerate.utils import DistributedType

    if accelerator.distributed_type == DistributedType.FSDP:
        return contextlib.nullcontext()
    return accelerator.autocast()


