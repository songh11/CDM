from transformers import AutoProcessor, AutoModel
from PIL import Image
import torch

from cdm.rewards.reward_ckpt_path import get_reward_cache_dir

PICKSCORE_PROCESSOR_REPO = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
PICKSCORE_MODEL_REPO = "yuvalkirstain/PickScore_v1"


class PickScoreScorer(torch.nn.Module):
    def __init__(self, device="cuda", dtype=torch.float32):
        super().__init__()
        cache_dir = get_reward_cache_dir()
        self.device = device
        self.dtype = dtype
        self.processor = AutoProcessor.from_pretrained(PICKSCORE_PROCESSOR_REPO, cache_dir=cache_dir)
        self.model = AutoModel.from_pretrained(PICKSCORE_MODEL_REPO, cache_dir=cache_dir).eval().to(device)
        self.model = self.model.to(dtype=dtype)

    @torch.no_grad()
    def __call__(self, prompt, images):
        # Preprocess images
        image_inputs = self.processor(
            images=images,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        )
        image_inputs = {k: v.to(device=self.device) for k, v in image_inputs.items()}
        # Preprocess text
        text_inputs = self.processor(
            text=prompt,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        )
        text_inputs = {k: v.to(device=self.device) for k, v in text_inputs.items()}

        # Get embeddings
        image_embs = self.model.get_image_features(**image_inputs)
        if not isinstance(image_embs, torch.Tensor):
            image_embs = image_embs.pooler_output
        image_embs = image_embs / image_embs.norm(p=2, dim=-1, keepdim=True)

        text_embs = self.model.get_text_features(**text_inputs)
        if not isinstance(text_embs, torch.Tensor):
            text_embs = text_embs.pooler_output
        text_embs = text_embs / text_embs.norm(p=2, dim=-1, keepdim=True)

        # Calculate scores
        logit_scale = self.model.logit_scale.exp()
        scores = logit_scale * (text_embs @ image_embs.T)
        scores = scores.diag()
        # norm到0-1
        # scores = scores / 26
        return scores


# Usage example
def main():
    scorer = PickScoreScorer(device="cuda", dtype=torch.float32)
    images = [
        "test_cases/nasa.jpg",
    ]
    pil_images = [Image.open(img) for img in images]
    prompts = [
        'An astronaut’s glove floating in zero-g with "NASA 2049" on the wrist',
    ]
    print(scorer(prompts, pil_images))


if __name__ == "__main__":
    main()
