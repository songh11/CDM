import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from PIL import Image
import numpy as np

from cdm.rewards.reward_ckpt_path import get_reward_cache_dir

HPSV3_REPO = "MizzenAI/HPSv3"
HPSV3_FILENAME = "HPSv3.safetensors"


class HPSv3Scorer(nn.Module):
    """
    HPSv3 (Human Preference Score v3) Scorer
    Based on Qwen2-VL VLM model for evaluating image-text alignment and quality.
    Reference: https://github.com/MizzenAI/HPSv3
    """

    def __init__(self, dtype, device, differentiable=False):
        super().__init__()
        self.dtype = dtype
        self.device = device
        self.differentiable = differentiable

        # Lazy import to avoid dependency issues
        from hpsv3 import HPSv3RewardInferencer

        # Auto-download HPSv3 weights from HuggingFace Hub.
        checkpoint_path = hf_hub_download(
            repo_id=HPSV3_REPO,
            filename=HPSV3_FILENAME,
            cache_dir=get_reward_cache_dir(),
        )

        # Initialize using official API with the resolved checkpoint path.
        self.inferencer = HPSv3RewardInferencer(
            device=device,
            checkpoint_path=checkpoint_path,
            differentiable=differentiable,
        )
        self.eval()

    def _tensor_to_pil_images(self, images):
        """Convert tensor images to PIL Images."""
        if isinstance(images, torch.Tensor):
            # Assume images are in [0, 1] range with shape (N, C, H, W)
            images_np = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images_np = images_np.transpose(0, 2, 3, 1)  # NCHW -> NHWC
            pil_images = [Image.fromarray(img) for img in images_np]
        elif isinstance(images, np.ndarray):
            if images.ndim == 4:
                # Assume NHWC format
                if images.shape[-1] == 3:
                    pil_images = [Image.fromarray(img) for img in images]
                else:
                    # NCHW format
                    images = images.transpose(0, 2, 3, 1)
                    pil_images = [Image.fromarray(img) for img in images]
            else:
                pil_images = [Image.fromarray(images)]
        elif isinstance(images, list):
            pil_images = []
            for img in images:
                if isinstance(img, Image.Image):
                    pil_images.append(img)
                elif isinstance(img, np.ndarray):
                    pil_images.append(Image.fromarray(img))
                elif isinstance(img, str):
                    pil_images.append(Image.open(img))
                else:
                    raise ValueError(f"Unsupported image type: {type(img)}")
        else:
            raise ValueError(f"Unsupported images type: {type(images)}")

        return pil_images

    @torch.no_grad()
    def __call__(self, images, prompts):
        """
        Compute HPSv3 scores for images and prompts.

        Args:
            images: Tensor of shape (N, C, H, W) with values in [0, 1],
                   or numpy array, or list of PIL Images/paths
            prompts: List of text prompts

        Returns:
            Tensor of HPSv3 scores for each image-prompt pair
        """
        pil_images = self._tensor_to_pil_images(images)

        # HPSv3 API: reward(image_paths, prompts) based on source code
        rewards = self.inferencer.reward(pil_images, prompts)

        # rewards shape: (N, 2) where [:, 0] is mu (mean) and [:, 1] is sigma
        # We use mu as the final score
        hps_scores = rewards[:, 0]

        return hps_scores.contiguous()

    def forward_differentiable(self, images, prompts):
        """
        Differentiable forward pass for training.
        Only works when differentiable=True is set during initialization.

        Args:
            images: Tensor of shape (N, C, H, W) with values in [0, 1]
            prompts: List of text prompts

        Returns:
            Tensor of HPSv3 scores (differentiable)
        """
        if not self.differentiable:
            raise RuntimeError(
                "Differentiable mode is not enabled. "
                "Initialize HPSv3Scorer with differentiable=True"
            )

        # For differentiable mode, we need to pass tensors directly
        # This requires the model to be set up for differentiable processing
        pil_images = self._tensor_to_pil_images(images)
        batch = self.inferencer.prepare_batch(pil_images, prompts)

        rewards = self.inferencer.model(return_dict=True, **batch)["logits"]
        hps_scores = rewards[:, 0]

        return hps_scores.contiguous()


def main():
    """Test HPSv3Scorer with sample images."""
    scorer = HPSv3Scorer(dtype=torch.float32, device="cuda")

    images = [
        "assets/example1.png",
        "assets/example2.png",
    ]
    prompts = [
        'cute chibi anime cartoon fox, smiling wagging tail with a small cartoon heart above sticker',
        'cute chibi anime cartoon fox, smiling wagging tail with a small cartoon heart above sticker',
    ]

    # Test with image paths
    pil_images = [Image.open(img) for img in images]
    scores = scorer(pil_images, prompts)
    print(f"HPSv3 Scores (from PIL): {scores}")

    # Test with tensor
    images_np = np.array([np.array(img) for img in pil_images])
    images_tensor = torch.tensor(images_np, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
    scores = scorer(images_tensor, prompts)
    print(f"HPSv3 Scores (from tensor): {scores}")


if __name__ == "__main__":
    main()
