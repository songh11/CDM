"""
Diversity metrics scorer for evaluating generation diversity.

Computes two diversity metrics from multiple images generated per prompt:

1. **Intra-LPIPS**: Average pairwise LPIPS (Learned Perceptual Image Patch
   Similarity) distance between images generated from the same prompt with
   different noise. Higher values indicate more diverse outputs.

2. **Vendi Score**: Effective number of distinct images per prompt, computed
   as the exponential entropy of the eigenvalues of a cosine-similarity
   kernel matrix built from perceptual features. Higher values indicate
   more diverse outputs.

Both metrics operate on *groups* of images sharing the same prompt, so they
require a dedicated diversity-generation phase that produces K variations
per prompt (using different latent noise seeds).

These are distribution-level metrics (not per-image), so they do NOT
integrate into the multi_score / reward_fn framework. Instead, they run as
post-hoc metrics in evaluation_v2.py Phase 3.

Dependencies:
    pip install lpips
    pip install vendi-score   # only needed for Vendi Score

Usage:
    from flow_grpo.rewards.diversity_scorer import DiversityScorer

    scorer = DiversityScorer(device="cuda")

    # images_by_prompt: dict mapping prompt_hash -> Tensor[K, C, H, W] in [0,1]
    intra_lpips = scorer.compute_intra_lpips(images_by_prompt)
    vendi = scorer.compute_vendi_score(images_by_prompt)

    scorer.unload()
"""

import logging
from itertools import combinations

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class DiversityScorer:
    """Compute Intra-LPIPS and Vendi Score diversity metrics.

    Uses the LPIPS AlexNet backbone (~20 MB) for both perceptual distance
    computation and feature extraction, so only one lightweight model is
    loaded onto the GPU.
    """

    def __init__(self, device="cuda", lpips_net="alex"):
        self.device = device
        self.lpips_net_name = lpips_net
        self._lpips_model = None

    def _ensure_lpips_loaded(self):
        """Lazily load the LPIPS model on first use."""
        if self._lpips_model is not None:
            return
        try:
            import lpips
        except ImportError:
            raise ImportError(
                "lpips is required for diversity metrics. "
                "Install it with: pip install lpips"
            )
        logger.info(f"Loading LPIPS model (backbone={self.lpips_net_name})...")
        self._lpips_model = lpips.LPIPS(net=self.lpips_net_name, verbose=False)
        self._lpips_model.to(self.device)
        self._lpips_model.eval()
        for param in self._lpips_model.parameters():
            param.requires_grad = False

    def compute_intra_lpips(self, images_by_prompt, batch_size=64):
        """Compute average pairwise LPIPS distance within each prompt group.

        Args:
            images_by_prompt: dict mapping prompt_key (str) -> Tensor[K, C, H, W]
                where K is the number of noise variations, values in [0, 1].
            batch_size: number of image pairs to score in one forward pass.

        Returns:
            float: mean Intra-LPIPS across all prompts (higher = more diverse).
        """
        self._ensure_lpips_loaded()

        per_prompt_distances = []

        for prompt_key, images in images_by_prompt.items():
            num_variations = images.shape[0]
            if num_variations < 2:
                continue

            pair_indices = list(combinations(range(num_variations), 2))
            pair_distances = []

            for batch_start in range(0, len(pair_indices), batch_size):
                batch_pairs = pair_indices[batch_start : batch_start + batch_size]

                images_a = torch.stack([images[i] for i, _ in batch_pairs]).to(self.device)
                images_b = torch.stack([images[j] for _, j in batch_pairs]).to(self.device)

                # LPIPS expects inputs in [-1, 1]
                images_a_scaled = images_a * 2.0 - 1.0
                images_b_scaled = images_b * 2.0 - 1.0

                with torch.inference_mode():
                    distances = self._lpips_model(images_a_scaled, images_b_scaled)

                # distances shape: [batch, 1, 1, 1] or [batch] depending on version
                distances_flat = distances.reshape(-1).cpu().tolist()
                pair_distances.extend(distances_flat)

                del images_a, images_b, images_a_scaled, images_b_scaled, distances
                torch.cuda.empty_cache()

            if pair_distances:
                prompt_mean = float(np.mean(pair_distances))
                per_prompt_distances.append(prompt_mean)

        if not per_prompt_distances:
            logger.warning("No valid prompt groups for Intra-LPIPS (need >= 2 variations each).")
            return 0.0

        overall_mean = float(np.mean(per_prompt_distances))
        logger.info(
            f"Intra-LPIPS: {overall_mean:.4f} "
            f"(across {len(per_prompt_distances)} prompts)"
        )
        return overall_mean

    def _extract_features(self, images, batch_size=64):
        """Extract perceptual features from images using the LPIPS backbone.

        Uses the final layer features from the LPIPS AlexNet and performs
        global average pooling to obtain a fixed-size feature vector per image.

        Args:
            images: Tensor[N, C, H, W] in [0, 1].
            batch_size: number of images per forward pass.

        Returns:
            Tensor[N, D] of L2-normalized feature vectors on CPU.
        """
        self._ensure_lpips_loaded()

        all_features = []

        for batch_start in range(0, len(images), batch_size):
            batch_images = images[batch_start : batch_start + batch_size].to(self.device)
            # LPIPS net expects [-1, 1]
            batch_scaled = batch_images * 2.0 - 1.0

            with torch.inference_mode():
                # Access the underlying feature extractor network
                # lpips.LPIPS stores the backbone in self.net which has a .forward()
                # that returns a list of feature maps at different layers
                feature_maps = self._lpips_model.net(batch_scaled)
                # feature_maps is a list of feature maps at different layers;
                # use the last (deepest) layer for richest semantics
                last_layer_features = feature_maps[-1]
                # Global average pooling -> [B, C]
                pooled = F.adaptive_avg_pool2d(last_layer_features, 1).squeeze(-1).squeeze(-1)
                all_features.append(pooled.cpu())

            del batch_images, batch_scaled, feature_maps, last_layer_features, pooled
            torch.cuda.empty_cache()

        features = torch.cat(all_features, dim=0).float()
        # L2 normalize for cosine similarity
        features = F.normalize(features, p=2, dim=1)
        return features

    def compute_vendi_score(self, images_by_prompt, batch_size=64):
        """Compute Vendi Score for each prompt group and return the mean.

        The Vendi Score is the exponential Shannon entropy of the eigenvalues
        of a similarity kernel matrix, interpreted as the effective number of
        distinct items in the set.

        Args:
            images_by_prompt: dict mapping prompt_key (str) -> Tensor[K, C, H, W]
                where K is the number of noise variations, values in [0, 1].
            batch_size: number of images per feature-extraction forward pass.

        Returns:
            float: mean Vendi Score across all prompts (higher = more diverse).
        """
        try:
            from vendi_score import vendi
        except ImportError:
            raise ImportError(
                "vendi-score is required for Vendi Score computation. "
                "Install it with: pip install vendi-score"
            )

        self._ensure_lpips_loaded()

        per_prompt_vendi = []

        for prompt_key, images in images_by_prompt.items():
            num_variations = images.shape[0]
            if num_variations < 2:
                continue

            features = self._extract_features(images, batch_size=batch_size)

            # Cosine similarity matrix (features are already L2-normalized)
            similarity_matrix = (features @ features.T).numpy()

            # Clamp to valid range to avoid numerical issues
            similarity_matrix = np.clip(similarity_matrix, 0.0, 1.0)

            score = vendi.score_K(similarity_matrix)
            per_prompt_vendi.append(float(score))

        if not per_prompt_vendi:
            logger.warning("No valid prompt groups for Vendi Score (need >= 2 variations each).")
            return 0.0

        overall_mean = float(np.mean(per_prompt_vendi))
        logger.info(
            f"Vendi Score: {overall_mean:.4f} "
            f"(across {len(per_prompt_vendi)} prompts)"
        )
        return overall_mean

    def unload(self):
        """Release the LPIPS model from GPU memory."""
        if self._lpips_model is not None:
            self._lpips_model.cpu()
            del self._lpips_model
            self._lpips_model = None
            torch.cuda.empty_cache()
            logger.info("DiversityScorer: LPIPS model unloaded.")
