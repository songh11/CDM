# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Loss functions for distillation training.

This module contains various loss computation functions used in different training strategies:
- Reward-Weighted CFG Loss: Uses reward as CFG guidance weight
- DMD CFG Loss: Distribution Matching Distillation with CFG guidance
- KL Divergence Loss: Regularization to prevent deviation from reference model
- Reward Weight: Dynamic loss weighting based on reward scores
"""

import torch
from typing import Optional, Callable, List, Any, Tuple


def compute_cfg_loss(
    student_prediction: torch.Tensor,
    real_teacher_cond_prediction: torch.Tensor,
    real_teacher_uncond_prediction: torch.Tensor,
    xt: torch.Tensor,
    xt_teacher: torch.Tensor,
    student_sigma_expanded: torch.Tensor,
    teacher_sigma_expanded: torch.Tensor,
    guidance_scale: float,
    return_per_sample: bool = False,
    norm_clip_min: float = 0.1,
) -> tuple[torch.Tensor, dict]:
    """
    Compute CFG guidance loss in x0 space.
    
    This loss distills CFG guidance from a frozen reference model into the student model.
    The student learns to predict x0 that matches the CFG-enhanced target.
    
    In Flow Matching (x0 = xt - t * v):
    - x0_student = xt - t * v_student
    - x0_cond    = xt - t * v_cond
    - x0_uncond  = xt - t * v_uncond
    - x0_target  = x0_student.detach() + alpha * (x0_cond - x0_uncond)
    
    This is mathematically equivalent to applying CFG in v space:
    - v_cfg = v_uncond + alpha * (v_cond - v_uncond)
    - The gradient pushes student towards the CFG guidance direction.
    
    The CFG update vector is per-sample normalized by the abs mean of
    (x0_cond - x0_student) to stabilize training.
    
    Args:
        student_prediction: Student model's v prediction (with gradient), shape (B, C, H, W)
        real_teacher_cond_prediction: Real Teacher's conditional v prediction (frozen), shape (B, C, H, W)
        real_teacher_uncond_prediction: Real Teacher's unconditional v prediction (frozen), shape (B, C, H, W)
        xt: Noisy latent at timestep t, shape (B, C, H, W)
        xt_teacher: Noisy latent at timestep t for teacher, shape (B, C, H, W)
        student_sigma_expanded: Timestep sigma (normalized to 0-1), shape (B, 1, 1, 1)
        teacher_sigma_expanded: Timestep sigma (normalized to 0-1) for teacher, shape (B, 1, 1, 1)
        guidance_scale: CFG guidance scale (e.g., 4.5)
        return_per_sample: If True, return per-sample loss (B,) instead of scalar mean.
        norm_clip_min: Minimum value to clamp the norm factor, preventing gradient explosion
                       when the norm factor is too small. Default: 0.1.
    
    Returns:
        loss: CFG Loss (scalar if return_per_sample=False, else shape (B,))
        info: Debug information dictionary
    """
    info = {}
       
    # Compute x0 predictions: x0 = xt - t * v (in float32 for precision)
    x0_student = xt.float() - student_sigma_expanded.float() * student_prediction.float()
    x0_cond = xt_teacher.float() - teacher_sigma_expanded.float() * real_teacher_cond_prediction.detach().float()
    x0_uncond = xt_teacher.float() - teacher_sigma_expanded.float() * real_teacher_uncond_prediction.detach().float()
    
    # Target x0: student shifted by alpha * (x0_cond - x0_uncond)
    # Equivalent to CFG in v space: v_cfg = v_uncond + alpha * (v_cond - v_uncond)
    cfg_effect_x0 = guidance_scale * (x0_cond - x0_uncond)
    
    # Per-sample normalize the update vector by abs mean of (x0_cond - x0_student)
    spatial_dims = tuple(range(1, x0_cond.ndim))
    cfg_norm_factor = (x0_cond - x0_student.detach()).abs().mean(
        dim=spatial_dims, keepdim=True
    ) + 1e-8
    cfg_norm_factor = cfg_norm_factor.clamp(min=norm_clip_min)
    cfg_effect_x0 = cfg_effect_x0 / cfg_norm_factor
    cfg_effect_x0 = torch.nan_to_num(cfg_effect_x0)
    info["cfg_norm_factor_mean"] = cfg_norm_factor.mean().detach()
    
    x0_target = x0_student.detach() + cfg_effect_x0
    
    # MSE Loss - compute per-sample first, then optionally reduce
    per_sample_loss = torch.nn.functional.mse_loss(
        x0_student.float(),
        x0_target.float(),
        reduction='none'
    ).mean(dim=tuple(range(1, x0_student.ndim)))  # (B,)
    
    if return_per_sample:
        loss = per_sample_loss
    else:
        loss = per_sample_loss.mean()
    
    # Record debug information
    info["cfg_effect_norm"] = cfg_effect_x0.abs().mean().detach()
    info["x0_cond_x0_student_norm"] = (x0_cond - x0_student.detach()).abs().mean().detach()
    info["x0_student_mean"] = x0_student.mean().detach()
    info["x0_student_var"] = x0_student.var().detach()
    # info["x0_cond_norm"] = torch.mean(x0_cond ** 2).detach()
    # info["x0_uncond_norm"] = torch.mean(x0_uncond ** 2).detach()
    # Record sigma values (first sample in batch)
    # Record sigma values (batch mean for meaningful cross-GPU averaging)
    info["student_sigma"] = student_sigma_expanded.flatten().mean().detach()
    info["teacher_sigma"] = teacher_sigma_expanded.flatten().mean().detach()
    
    # Return raw CFG gradient direction for gradient alignment analysis
    # This is the unnormalized direction: x0_cond - x0_uncond (without alpha scaling)
    info["cfg_grad_direction"] = (x0_cond - x0_uncond).detach()
    
    return loss, info

def compute_batch_mean_var_kl_loss(    
    prediction: torch.Tensor,
    target_mean: float = 0.0,
    target_std: float = 1.0,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, dict]:
    """
    Compute KL divergence loss based on batch mean and variance statistics.
    
    This loss encourages the batch statistics (mean and variance) of the prediction
    to match a target Gaussian distribution N(target_mean, target_std^2).
    
    The KL divergence from N(μ_i, σ_i²) to N(μ_target, σ_target²) is:
    L_KL = (1/B) * Σ_i [ (μ_i - μ_target)² / (2σ_target²) + σ_i² / (2σ_target²) - 1 - log(σ_i / σ_target) ]
    
    For each sample in the batch, we compute its mean and std across spatial dimensions,
    then compute the KL divergence to the target distribution.
    
    Args:
        prediction: Model prediction tensor, shape (B, C, H, W)
        target_mean: Target distribution mean (default: 0.0)
        target_std: Target distribution standard deviation (default: 1.0)
        eps: Small epsilon for numerical stability
    
    Returns:
        loss: Batch mean-variance KL divergence loss
        info: Debug information dictionary containing:
            - batch_mu_mean: Mean of per-sample means
            - batch_mu_std: Std of per-sample means
            - batch_sigma_mean: Mean of per-sample stds
            - batch_sigma_std: Std of per-sample stds
            - kl_mean_term: Mean term contribution to KL
            - kl_var_term: Variance term contribution to KL
    """
    info = {}
    
    # Flatten spatial dimensions: (B, C, H, W) -> (B, C*H*W)
    batch_size = prediction.shape[0]
    prediction_flat = prediction.view(batch_size, -1)
    
    # Compute per-sample mean and std across all dimensions (C, H, W)
    # μ_i for each sample in batch
    sample_means = prediction_flat.mean(dim=1)  # (B,)
    # σ_i for each sample in batch
    sample_stds = prediction_flat.std(dim=1) + eps  # (B,)
    
    # Target distribution parameters
    mu_target = target_mean
    sigma_target = target_std
    sigma_target_sq = sigma_target ** 2
    
    # Compute KL divergence components for each sample
    # Term 1: (μ_i - μ_target)² / (2σ_target²)
    mean_term = (sample_means - mu_target) ** 2 / (2 * sigma_target_sq)
    
    # Term 2: σ_i² / (2σ_target²)
    var_term = sample_stds ** 2 / (2 * sigma_target_sq)
    
    # Term 3: -1
    constant_term = -1.0
    
    # Term 4: -log(σ_i / σ_target)
    log_term = -torch.log(sample_stds / sigma_target)
    
    # Full KL divergence per sample
    kl_per_sample = mean_term + var_term + constant_term + log_term
    
    # Average over batch
    loss = kl_per_sample.mean()
    
    # Record debug information
    info["batch_mu_mean"] = sample_means.mean().detach()
    info["batch_mu_std"] = sample_means.std().detach()
    info["batch_sigma_mean"] = sample_stds.mean().detach()
    info["batch_sigma_std"] = sample_stds.std().detach()
    info["kl_mean_term"] = mean_term.mean().detach()
    info["kl_var_term"] = var_term.mean().detach()
    info["kl_log_term"] = log_term.mean().detach()
    info["kl_per_sample_mean"] = kl_per_sample.mean().detach()
    info["kl_per_sample_std"] = kl_per_sample.std().detach()
    
    return loss, info

def compute_fake_teacher_diffusion_loss(
    fake_teacher_prediction: torch.Tensor,
    x0: torch.Tensor,
    noise: torch.Tensor,
    t_expanded: torch.Tensor,
) -> tuple[torch.Tensor, dict]:
    """
    Compute Diffusion Loss for training Fake Teacher.
    
    The Fake Teacher learns to predict the velocity (v = noise - x0) from noisy samples
    generated using the Student's x0 predictions. This trains the Fake Teacher to model
    the Student's distribution.
    
    In Flow Matching:
    - xt = (1 - t) * x0 + t * noise
    - v_target = noise - x0
    
    Args:
        fake_teacher_prediction: Fake Teacher's v prediction, shape (B, C, H, W)
        x0: Clean latent (Student's x0 prediction, detached), shape (B, C, H, W)
        noise: Random noise used to create xt, shape (B, C, H, W)
        t_expanded: Timestep (normalized to 0-1), shape (B, 1, 1, 1)
    
    Returns:
        loss: Diffusion loss (MSE between predicted v and target v)
        info: Debug information dictionary
    """
    info = {}
    
    # Target velocity: v = noise - x0
    v_target = noise - x0
    
    # MSE Loss
    loss = torch.nn.functional.mse_loss(
        fake_teacher_prediction.float(),
        v_target.detach().float(),
        reduction='mean'
    )
    # Record debug information
    info["v_target_norm"] = v_target.abs().mean().detach()
    info["fake_teacher_pred_norm"] = fake_teacher_prediction.abs().mean().detach()
    info["prediction_error"] = (fake_teacher_prediction - v_target).abs().mean().detach()
    
    return loss, info


def compute_dm_loss(
    student_prediction: torch.Tensor,
    real_teacher_prediction: torch.Tensor,
    fake_teacher_prediction: torch.Tensor,
    xt: torch.Tensor,
    xt_teacher: torch.Tensor,
    student_sigma_expanded: torch.Tensor,
    teacher_sigma_expanded: torch.Tensor,
    return_per_sample: bool = False,
    cfg_grad_direction: Optional[torch.Tensor] = None,
    norm_clip_min: float = 0.1,
) -> tuple[torch.Tensor, dict]:
    """
    Compute Distribution Matching (DM) Loss for DMD2 in x0 space.
    
    The DM Loss uses the difference between Real Teacher and Fake Teacher x0 predictions
    to guide the Student model towards the real data distribution.
    
    Core idea from DMD2:
    - Real Teacher: Frozen pretrained model, represents real data distribution
    - Fake Teacher: Trained on Student's outputs, represents Student's distribution
    - DM Loss: Encourages Student to move towards Real Teacher's distribution
    
    In x0 space (x0 = xt - t * v):
    - x0_student      = xt - t * v_student
    - x0_real_teacher = xt - t * v_real
    - x0_fake_teacher = xt - t * v_fake
    - x0_target       = x0_student.detach() + (x0_real_teacher - x0_fake_teacher)
    
    This pushes x0_student in the direction from fake distribution to real distribution.
    
    The DM update vector is per-sample normalized by the abs mean of
    (x0_real_teacher - x0_student) to stabilize training.
    
    Args:
        student_prediction: Student model's v prediction (with gradient), shape (B, C, H, W)
        real_teacher_prediction: Real Teacher's v prediction (frozen, detached), shape (B, C, H, W)
        fake_teacher_prediction: Fake Teacher's v prediction (detached), shape (B, C, H, W)
        xt: Noisy latent at timestep t, shape (B, C, H, W)
        xt_teacher: Noisy latent at timestep t for teacher, shape (B, C, H, W)
        student_sigma_expanded: Timestep (normalized to 0-1), shape (B, 1, 1, 1)
        teacher_sigma_expanded: Timestep (normalized to 0-1) for teacher, shape (B, 1, 1, 1)
        return_per_sample: If True, return per-sample loss (B,) instead of scalar mean.
        cfg_grad_direction: Optional CFG gradient direction tensor (x0_cond - x0_uncond) from
                           compute_cfg_loss. When provided, computes cosine similarity between
                           DM gradient direction and CFG gradient direction for alignment analysis.
                           Shape (B, C, H, W), should be detached.
        norm_clip_min: Minimum value to clamp the norm factor, preventing gradient explosion
                       when the norm factor is too small. Default: 0.1.
    
    Returns:
        loss: DM Loss (scalar if return_per_sample=False, else shape (B,))
        info: Debug information dictionary
    """
    info = {}
    
    # Compute x0 predictions: x0 = xt - t * v (in float32 for precision)
    x0_student = xt.float() - student_sigma_expanded.float() * student_prediction.float()
    x0_real_teacher = xt_teacher.float() - teacher_sigma_expanded.float() * real_teacher_prediction.detach().float()
    x0_fake_teacher = xt_teacher.float() - teacher_sigma_expanded.float() * fake_teacher_prediction.detach().float()
    
    # Target x0: push student from fake distribution towards real distribution
    # dm_effect = x0_real_teacher - x0_fake_teacher = -t * (v_real - v_fake)
    dm_effect_x0 = x0_real_teacher - x0_fake_teacher
    
    # Per-sample normalize the update vector by abs mean of (x0_real_teacher - x0_student)
    spatial_dims = tuple(range(1, x0_real_teacher.ndim))
    dm_norm_factor = (x0_real_teacher - x0_student.detach()).abs().mean(
        dim=spatial_dims, keepdim=True
    ) + 1e-8
    dm_norm_factor = dm_norm_factor.clamp(min=norm_clip_min)
    dm_effect_x0 = dm_effect_x0 / dm_norm_factor
    dm_effect_x0 = torch.nan_to_num(dm_effect_x0)
    info["dm_norm_factor_mean"] = dm_norm_factor.mean().detach()
    
    x0_target = x0_student.detach() + dm_effect_x0
    
    # MSE Loss - compute per-sample first, then optionally reduce
    per_sample_loss = torch.nn.functional.mse_loss(
        x0_student.float(),
        x0_target.float(),
        reduction='none'
    ).mean(dim=tuple(range(1, x0_student.ndim)))  # (B,)
    
    # Record debug information
    info["dm_effect_norm"] = dm_effect_x0.abs().mean().detach()
    info["x0_real_x0_fake_norm"] = (x0_real_teacher - x0_fake_teacher).abs().mean().detach()
    info["x0_real_x0_student_norm"] = (x0_real_teacher - x0_student.detach()).abs().mean().detach()
    info["x0_student_mean"] = x0_student.mean().detach()
    info["x0_student_var"] = x0_student.var().detach()
    # Record sigma values (batch mean for meaningful cross-GPU averaging)
    info["student_sigma"] = student_sigma_expanded.flatten().mean().detach()
    info["teacher_sigma"] = teacher_sigma_expanded.flatten().mean().detach()
    
    # Return raw DM gradient direction for gradient alignment analysis
    dm_grad_direction = (x0_real_teacher - x0_fake_teacher).detach()
    info["dm_grad_direction"] = dm_grad_direction
    
    # Compute gradient alignment with CFG if cfg_grad_direction is provided
    cosine_sim = None
    if cfg_grad_direction is not None:
        batch_size = dm_grad_direction.shape[0]
        dm_flat = dm_grad_direction.view(batch_size, -1)  # (B, D)
        cfg_flat = cfg_grad_direction.view(batch_size, -1)  # (B, D)
        
        # Compute per-sample cosine similarity
        # cos(θ) = (A · B) / (|A| * |B|)
        cosine_sim = torch.nn.functional.cosine_similarity(dm_flat, cfg_flat, dim=1)  # (B,)
        
        # Record alignment statistics
        info["dm_cfg_cosine_sim_mean"] = cosine_sim.mean().detach()
        info["dm_cfg_cosine_sim_std"] = cosine_sim.std().detach()
        info["dm_cfg_cosine_sim_min"] = cosine_sim.min().detach()
        info["dm_cfg_cosine_sim_max"] = cosine_sim.max().detach()
    
    # Final loss computation
    if return_per_sample:
        loss = per_sample_loss
    else:
        loss = per_sample_loss.mean()
        
    return loss, info




