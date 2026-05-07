import os
from PIL import Image
import torch
import ImageReward as RM

from cdm.rewards.reward_ckpt_path import get_reward_cache_dir


class ImageRewardScorer(torch.nn.Module):
    def __init__(self, device="cuda", dtype=torch.float32, model_path=None):
        super().__init__()
        self.device = device
        self.dtype = dtype

        # Resolve download root: explicit model_path > REWARD_CKPT_DIR > HF default cache.
        if model_path is not None:
            download_root = model_path
        else:
            cache_dir = get_reward_cache_dir()
            if cache_dir is not None:
                download_root = os.path.join(cache_dir, "ImageReward")
            else:
                hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
                download_root = os.path.join(hf_home, "ImageReward")

        self.model = (
            RM.load(
                "ImageReward-v1.0",
                device=device,
                download_root=download_root,
            )
            .eval()
            .to(dtype=dtype)
        )
        self.model.requires_grad_(False)

    @torch.no_grad()
    def __call__(self, prompts, images):
        _, rewards = self.model.inference_rank(prompts, images)
        # When there is only 1 prompt and 1 image, inference_rank returns a
        # scalar float instead of a list/matrix.  Wrap it so torch.Tensor
        # always receives a sequence.
        if not isinstance(rewards, (list, tuple)):
            rewards = [[rewards]]
        rewards = torch.diagonal(torch.Tensor(rewards).to(self.device).reshape(len(prompts), len(prompts)), 0)
        return rewards.contiguous()


# Usage example
def main():
    scorer = ImageRewardScorer(device="cuda", dtype=torch.float32)

    images = [
        "test_cases/nasa.jpg",
        "test_cases/hello world.jpg",
    ]
    pil_images = [Image.open(img) for img in images]
    prompts = [
        'An astronaut’s glove floating in zero-g with "NASA 2049" on the wrist',
        'New York Skyline with "Hello World" written with fireworks on the sky',
    ]
    print(scorer(prompts, pil_images))


if __name__ == "__main__":
    main()
