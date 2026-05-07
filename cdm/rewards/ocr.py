import os
import torch
import numpy as np
from Levenshtein import distance
from typing import List, Union, Optional
from PIL import Image


def _patch_numpy_sctypes():
    """Patch np.sctypes for NumPy 2.0+ compatibility with imgaug."""
    import numpy as _np
    if not hasattr(_np, "sctypes"):
        _np.sctypes = {
            "int": [_np.int8, _np.int16, _np.int32, _np.int64],
            "uint": [_np.uint8, _np.uint16, _np.uint32, _np.uint64],
            "float": [_np.float16, _np.float32, _np.float64],
            "complex": [_np.complex64, _np.complex128],
            "others": [bool, object, bytes, str, _np.void],
        }


def _torch_device_to_paddle(device) -> str:
    """Convert PyTorch device specification to PaddleOCR device string.

    Examples:
        "cuda"      -> "gpu"
        "cuda:0"    -> "gpu:0"
        "cuda:2"    -> "gpu:2"
        "cpu"       -> "cpu"
        torch.device("cuda", 1) -> "gpu:1"
    """
    if isinstance(device, torch.device):
        device_type = device.type
        device_index = device.index
    else:
        device_str = str(device)
        if ":" in device_str:
            device_type, idx = device_str.split(":")
            device_index = int(idx)
        else:
            device_type = device_str
            device_index = None

    if device_type == "cuda":
        return f"gpu:{device_index}" if device_index is not None else "gpu"
    return "cpu"


class OcrScorer:
    """OCR reward calculator that runs PaddleOCR on the same GPU as other reward models."""

    def __init__(self, device="cpu", **kwargs):
        """
        OCR reward calculator
        :param device: Device to run PaddleOCR on (e.g. "cuda:0", "cpu")
        """
        self.device = device
        self._ocr_instance = None

    def _get_ocr(self):
        """Lazily initialize PaddleOCR on first use."""
        if self._ocr_instance is not None:
            return self._ocr_instance

        os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
        _patch_numpy_sctypes()

        from paddleocr import PaddleOCR

        paddle_device = _torch_device_to_paddle(self.device)
        self._ocr_instance = PaddleOCR(
            use_textline_orientation=False,
            lang="en",
            device=paddle_device,
        )
        return self._ocr_instance

    def _ocr_single_image(self, img_array) -> str:
        """Run OCR on a single image and return recognized text."""
        ocr = self._get_ocr()
        try:
            results = ocr.predict(img_array)
            if results:
                result = results[0]
                rec_texts = result.get("rec_texts", [])
                rec_scores = result.get("rec_scores", [])
                return "".join(
                    text for text, score in zip(rec_texts, rec_scores) if score > 0
                )
            return ""
        except Exception as error:
            print(f"OCR processing failed: {error}")
            return ""

    @torch.no_grad()
    def __call__(self, images: Union[List[Image.Image], List[np.ndarray]], prompts: List[str]) -> List[float]:
        """
        Calculate OCR reward
        :param images: List of input images (PIL or numpy format)
        :param prompts: Corresponding target text list
        :return: List of reward scores
        """
        prompts_extracted = [prompt.split('"')[1] for prompt in prompts]

        assert len(images) == len(prompts_extracted), "Images and prompts must have the same length"

        img_arrays = []
        for img in images:
            if isinstance(img, Image.Image):
                img_arrays.append(np.array(img))
            else:
                img_arrays.append(img)

        recognized_texts = [self._ocr_single_image(img) for img in img_arrays]

        rewards = []
        for recognized_text, prompt in zip(recognized_texts, prompts_extracted):
            recognized_text = recognized_text.replace(" ", "").lower()
            prompt_clean = prompt.replace(" ", "").lower()

            if prompt_clean in recognized_text:
                dist = 0
            else:
                dist = distance(recognized_text, prompt_clean)

            if dist > len(prompt_clean):
                dist = len(prompt_clean)

            reward = 1 - dist / len(prompt_clean) if len(prompt_clean) > 0 else 0.0
            rewards.append(reward)

        return rewards


if __name__ == "__main__":
    example_image_path = "test_cases/hello world.jpg"
    example_image = Image.open(example_image_path)
    example_prompt = 'New York Skyline with "Hello World" written with fireworks on the sky'
    scorer = OcrScorer(device="cuda")

    reward = scorer([example_image], [example_prompt])
    print(f"OCR Reward: {reward}")