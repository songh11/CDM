"""
FID (Fréchet Inception Distance) scorer for evaluating image generation quality.

Computes FID between generated images and a reference image set (e.g., COCO-2014 Val).
Uses clean-fid library for consistent, reproducible FID computation.

FID is a distribution-level metric (not per-image), so it does NOT integrate into
the multi_score / reward_fn framework. Instead, it runs as a post-hoc metric in
evaluation_v2.py Phase 3.

Dependencies:
    pip install clean-fid

Usage:
    from flow_grpo.rewards.fid_scorer import FIDScorer

    scorer = FIDScorer(device="cuda")
    fid_value = scorer.compute_fid_from_tensors(
        generated_images=images_tensor,  # [N, C, H, W] in [0, 1]
        reference_dir="/path/to/coco_val_10k/images",
    )
"""

import os
import logging
import tempfile
import shutil
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

logger = logging.getLogger(__name__)


class FIDScorer:
    """Compute FID between generated image tensors and a reference image directory.

    Supports two modes:
      1. reference_dir: path to a folder of real images (PNG/JPG)
      2. reference_stats_path: path to a precomputed .npz file containing
         mu (mean) and sigma (covariance) of Inception features

    The scorer saves generated images to a temporary directory, computes
    Inception features via clean-fid, then cleans up.
    """

    def __init__(self, device="cuda", batch_size=64, num_workers=2):
        self.device = device
        self.batch_size = batch_size
        self.num_workers = num_workers

    def compute_fid_from_tensors(
        self,
        generated_images,
        reference_dir=None,
        reference_stats_path=None,
        cleanup=True,
    ):
        """Compute FID score between generated images and reference set.

        Args:
            generated_images: Tensor [N, C, H, W] in [0, 1] float range
            reference_dir: path to directory of reference images (PNG/JPG)
            reference_stats_path: path to precomputed .npz stats file
                (must contain 'mu' and 'sigma' keys)
            cleanup: whether to remove temporary generated image directory

        Returns:
            float: FID score (lower is better)
        """
        try:
            from cleanfid import fid as cleanfid_module
        except ImportError:
            raise ImportError(
                "clean-fid is required for FID computation. "
                "Install it with: pip install clean-fid"
            )

        if reference_dir is None and reference_stats_path is None:
            raise ValueError(
                "Must provide either reference_dir or reference_stats_path"
            )

        total_images = len(generated_images)
        logger.info(f"Computing FID for {total_images} generated images...")

        temp_gen_dir = tempfile.mkdtemp(prefix="fid_gen_")

        try:
            self._save_tensors_as_images(generated_images, temp_gen_dir)

            if reference_stats_path is not None and os.path.isfile(reference_stats_path):
                fid_score = self._compute_fid_with_precomputed_stats(
                    cleanfid_module, temp_gen_dir, reference_stats_path
                )
            elif reference_dir is not None and os.path.isdir(reference_dir):
                fid_score = self._compute_fid_with_reference_dir(
                    temp_gen_dir, reference_dir
                )
            else:
                raise FileNotFoundError(
                    f"Neither reference_dir ({reference_dir}) nor "
                    f"reference_stats_path ({reference_stats_path}) is valid."
                )

            logger.info(f"FID score: {fid_score:.4f}")
            return float(fid_score)

        finally:
            if cleanup and os.path.isdir(temp_gen_dir):
                shutil.rmtree(temp_gen_dir, ignore_errors=True)

    def _save_tensors_as_images(self, images_tensor, output_dir):
        """Save a batch of [N, C, H, W] float tensors as PNG files."""
        os.makedirs(output_dir, exist_ok=True)
        for idx in tqdm(range(len(images_tensor)), desc="Saving generated images"):
            image_array = (
                images_tensor[idx]
                .clamp(0, 1)
                .cpu()
                .float()
                .numpy()
                .transpose(1, 2, 0)
                * 255
            ).astype(np.uint8)
            pil_image = Image.fromarray(image_array)
            pil_image.save(os.path.join(output_dir, f"{idx:06d}.png"))

    def _compute_fid_with_precomputed_stats(
        self, cleanfid_module, generated_dir, stats_path
    ):
        """Compute FID using precomputed reference statistics (.npz file).

        The .npz file must contain:
          - 'mu': mean vector of Inception features (shape [2048])
          - 'sigma': covariance matrix of Inception features (shape [2048, 2048])
        """
        from cleanfid.fid import build_feature_extractor, get_folder_features, frechet_distance

        stats = np.load(stats_path)
        reference_mu = stats["mu"]
        reference_sigma = stats["sigma"]

        feat_model = build_feature_extractor(
            mode="clean", device=torch.device(self.device)
        )

        generated_features = get_folder_features(
            generated_dir,
            feat_model,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            device=torch.device(self.device),
            mode="clean",
            description="Computing generated image features",
        )

        generated_mu = np.mean(generated_features, axis=0)
        generated_sigma = np.cov(generated_features, rowvar=False)

        fid_score = frechet_distance(
            generated_mu, generated_sigma, reference_mu, reference_sigma
        )
        return fid_score

    def _compute_fid_with_reference_dir(self, generated_dir, reference_dir):
        """Compute FID by manually extracting features from both directories.

        This allows showing tqdm progress bars for both generated and reference
        feature extraction, unlike the high-level compute_fid API.
        """
        from cleanfid.fid import build_feature_extractor, get_folder_features, frechet_distance

        feat_model = build_feature_extractor(
            mode="clean", device=torch.device(self.device)
        )

        generated_features = get_folder_features(
            generated_dir,
            feat_model,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            device=torch.device(self.device),
            mode="clean",
            description="Extracting generated image features",
        )

        reference_features = get_folder_features(
            reference_dir,
            feat_model,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            device=torch.device(self.device),
            mode="clean",
            description="Extracting reference image features",
        )

        generated_mu = np.mean(generated_features, axis=0)
        generated_sigma = np.cov(generated_features, rowvar=False)
        reference_mu = np.mean(reference_features, axis=0)
        reference_sigma = np.cov(reference_features, rowvar=False)

        fid_score = frechet_distance(
            generated_mu, generated_sigma, reference_mu, reference_sigma
        )
        return fid_score

    def compute_fid_from_image_dir(
        self,
        generated_image_dir,
        reference_dir=None,
        reference_stats_path=None,
    ):
        """Compute FID score directly from a directory of generated images.

        Unlike compute_fid_from_tensors, this method does not require loading
        all images into memory as tensors first. The generated images should
        already be saved as PNG/JPG files in generated_image_dir.

        Args:
            generated_image_dir: path to directory of generated images (PNG/JPG)
            reference_dir: path to directory of reference images (PNG/JPG)
            reference_stats_path: path to precomputed .npz stats file

        Returns:
            float: FID score (lower is better)
        """
        try:
            from cleanfid import fid as cleanfid_module
        except ImportError:
            raise ImportError(
                "clean-fid is required for FID computation. "
                "Install it with: pip install clean-fid"
            )

        if reference_dir is None and reference_stats_path is None:
            raise ValueError(
                "Must provide either reference_dir or reference_stats_path"
            )

        if not os.path.isdir(generated_image_dir):
            raise FileNotFoundError(
                f"Generated image directory not found: {generated_image_dir}"
            )

        image_count = len([
            f for f in os.listdir(generated_image_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ])
        logger.info(f"Computing FID for {image_count} generated images from {generated_image_dir}...")

        if reference_stats_path is not None and os.path.isfile(reference_stats_path):
            fid_score = self._compute_fid_with_precomputed_stats(
                cleanfid_module, generated_image_dir, reference_stats_path
            )
        elif reference_dir is not None and os.path.isdir(reference_dir):
            fid_score = self._compute_fid_with_reference_dir(
                generated_image_dir, reference_dir
            )
        else:
            raise FileNotFoundError(
                f"Neither reference_dir ({reference_dir}) nor "
                f"reference_stats_path ({reference_stats_path}) is valid."
            )

        logger.info(f"FID score: {fid_score:.4f}")
        return float(fid_score)

    @staticmethod
    def save_batch_tensors_as_images(images_tensor, output_dir, start_index=0):
        """Save a batch of [N, C, H, W] float tensors as PNG files.

        Processes one image at a time to minimize memory usage.
        Returns the number of images saved.
        """
        os.makedirs(output_dir, exist_ok=True)
        count = len(images_tensor)
        for idx in range(count):
            image_array = (
                images_tensor[idx]
                .clamp(0, 1)
                .cpu()
                .float()
                .numpy()
                .transpose(1, 2, 0)
                * 255
            ).astype(np.uint8)
            pil_image = Image.fromarray(image_array)
            pil_image.save(
                os.path.join(output_dir, f"{start_index + idx:06d}.png")
            )
        return count

    def _extract_features_from_tensor_batch(self, feat_model, images_tensor):
        """Extract Inception features from a batch of image tensors.

        Args:
            feat_model: Inception feature extractor from clean-fid
            images_tensor: Tensor [N, C, H, W] in [0, 1] float range

        Returns:
            np.ndarray of shape [N, 2048] containing Inception features
        """
        # clean-fid's InceptionV3 expects input in [0, 255] range, resized to 299x299
        images = images_tensor.clone().clamp(0, 1).float()

        # Resize to 299x299 (Inception input size)
        if images.shape[2] != 299 or images.shape[3] != 299:
            images = F.interpolate(
                images, size=(299, 299), mode="bicubic", align_corners=False
            ).clamp(0, 1)

        # Scale to [0, 255] as clean-fid expects
        images = images * 255.0

        all_features = []
        for start_idx in range(0, len(images), self.batch_size):
            batch = images[start_idx:start_idx + self.batch_size].to(self.device)
            with torch.no_grad():
                features = feat_model(batch)
            # features shape: [batch_size, 2048]
            all_features.append(features.cpu().numpy())
            del batch, features

        return np.concatenate(all_features, axis=0)

    def compute_fid_from_pt_files(
        self,
        batch_pt_dirs,
        reference_dir=None,
        reference_stats_path=None,
    ):
        """Compute FID by streaming .pt files and extracting Inception features
        directly from tensors, without saving intermediate PNG files.

        Each .pt file is expected to contain a dict with key "images" mapping
        to a [N, C, H, W] float tensor in [0, 1].

        Args:
            batch_pt_dirs: list of directories containing batch_*.pt files
            reference_dir: path to directory of reference images (PNG/JPG)
            reference_stats_path: path to precomputed .npz stats file

        Returns:
            float: FID score (lower is better)
        """
        from cleanfid.fid import build_feature_extractor, get_folder_features, frechet_distance

        if reference_dir is None and reference_stats_path is None:
            raise ValueError(
                "Must provide either reference_dir or reference_stats_path"
            )

        # Collect all batch .pt files across all rank directories
        all_batch_files = []
        for rank_dir in batch_pt_dirs:
            if not os.path.isdir(rank_dir):
                continue
            batch_files = sorted(
                os.path.join(rank_dir, f)
                for f in os.listdir(rank_dir)
                if f.startswith("batch_") and f.endswith(".pt")
            )
            all_batch_files.extend(batch_files)

        if not all_batch_files:
            raise FileNotFoundError(
                f"No batch_*.pt files found in directories: {batch_pt_dirs}"
            )

        logger.info(
            f"Streaming {len(all_batch_files)} .pt files from "
            f"{len(batch_pt_dirs)} dir(s) for FID feature extraction..."
        )

        # Build Inception feature extractor
        feat_model = build_feature_extractor(
            mode="clean", device=torch.device(self.device)
        )

        # Stream through .pt files, extract features one batch at a time
        all_generated_features = []
        total_images = 0
        for batch_file in tqdm(all_batch_files, desc="Extracting Inception features from .pt files"):
            data = torch.load(batch_file, map_location="cpu")
            batch_images = data["images"]
            batch_count = len(batch_images)

            features = self._extract_features_from_tensor_batch(feat_model, batch_images)
            all_generated_features.append(features)
            total_images += batch_count

            del data, batch_images, features
            torch.cuda.empty_cache()

        logger.info(f"Extracted features from {total_images} generated images total.")

        generated_features = np.concatenate(all_generated_features, axis=0)
        del all_generated_features

        generated_mu = np.mean(generated_features, axis=0)
        generated_sigma = np.cov(generated_features, rowvar=False)
        del generated_features

        # Compute reference statistics
        if reference_stats_path is not None and os.path.isfile(reference_stats_path):
            stats = np.load(reference_stats_path)
            reference_mu = stats["mu"]
            reference_sigma = stats["sigma"]
        elif reference_dir is not None and os.path.isdir(reference_dir):
            logger.info("Extracting reference image features...")
            reference_features = get_folder_features(
                reference_dir,
                feat_model,
                num_workers=self.num_workers,
                batch_size=self.batch_size,
                device=torch.device(self.device),
                mode="clean",
                description="Extracting reference image features",
            )
            reference_mu = np.mean(reference_features, axis=0)
            reference_sigma = np.cov(reference_features, rowvar=False)
            del reference_features
        else:
            raise FileNotFoundError(
                f"Neither reference_dir ({reference_dir}) nor "
                f"reference_stats_path ({reference_stats_path}) is valid."
            )

        fid_score = frechet_distance(
            generated_mu, generated_sigma, reference_mu, reference_sigma
        )
        logger.info(f"FID score: {fid_score:.4f}")
        return float(fid_score)

    @staticmethod
    def collect_rank_dirs(gen_dir, world_size):
        """Collect all rank directories for a given generation directory.

        Args:
            gen_dir: path to rank0's generation directory
            world_size: total number of ranks

        Returns:
            list of existing rank directories
        """
        rank_dirs = [gen_dir]
        parent_gen_dir = os.path.dirname(gen_dir)
        for rank_idx in range(1, world_size):
            other_rank_dir = os.path.join(parent_gen_dir, f"rank{rank_idx}")
            if os.path.isdir(other_rank_dir):
                rank_dirs.append(other_rank_dir)
        return rank_dirs

    @staticmethod
    def precompute_reference_stats(
        reference_dir, output_path, device="cuda", batch_size=64, num_workers=4
    ):
        """Precompute and save Inception feature statistics for a reference image set.

        This only needs to be run once per reference dataset. The resulting .npz
        file can then be reused across multiple evaluation runs.

        Args:
            reference_dir: path to directory of reference images
            output_path: where to save the .npz file
            device: torch device
            batch_size: batch size for feature extraction
            num_workers: number of data loading workers
        """
        from cleanfid.fid import build_feature_extractor, get_folder_features

        logger.info(f"Precomputing reference stats from: {reference_dir}")

        feat_model = build_feature_extractor(
            mode="clean", device=torch.device(device)
        )

        features = get_folder_features(
            reference_dir,
            feat_model,
            num_workers=num_workers,
            batch_size=batch_size,
            device=torch.device(device),
            mode="clean",
            description="Computing reference image features",
        )

        mu = np.mean(features, axis=0)
        sigma = np.cov(features, rowvar=False)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        np.savez_compressed(output_path, mu=mu, sigma=sigma)
        logger.info(
            f"Saved reference stats ({len(features)} images) to: {output_path}"
        )
