"""
LLM Judge Pairwise Comparison Scorer.

Performs pairwise image comparison using a VLM API (OpenAI-compatible)
with Chain-of-Thought reasoning and position de-biasing.

For each (prompt, image_A, image_B) triple, the judge is called twice
with swapped image positions to eliminate position bias. The final verdict
is determined by agreement between the two rounds.

Usage:
    scorer = LLMJudgeScorer(
        api_base_url="https://your-api-endpoint/v1",
        api_key="your-key",
        model_name="qwen-vl-max",
    )
    results = scorer.evaluate(
        images_a=student_images,   # Tensor[N, C, H, W]
        images_b=teacher_images,   # Tensor[N, C, H, W]
        prompts=prompts,           # list[str]
        role_a="student",
        role_b="real_teacher",
    )
    scorer.unload()
"""

import asyncio
import base64
import json
import logging
import re
from io import BytesIO

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)

JUDGE_SYSTEM_PROMPT = (
    "You are an expert image quality evaluator, specializing in distinguishing "
    "between natural photographic details and AI-generated artifacts."
)

JUDGE_USER_PROMPT_TEMPLATE = """**Task:** Compare Image A and Image B based on the given Text Prompt.

**Text Prompt:** "{prompt}"

**Evaluation Criteria (Prioritized):**
1. **Instruction Following:** Does the image accurately reflect the prompt (objects, colors, spatial relations)?
2. **Realism & Texture (CRITICAL):**
   * Look closely at high-frequency details.
   * **Penalize "Noise":** Does the image have unnatural grain, pixelation, or "fried" artifacts?
   * **Penalize "Oversaturation/Greasy":** Does the image look like plastic or have excessive contrast (reward hacking)?
   * **Favor "Natural Details":** Prefer natural skin texture, realistic lighting falloff, and clean edges.
3. **Text Rendering:** If there is text, which one is more legible and correct?

**Step-by-Step Thinking Process (CoT):**
1. Analyze Image A's adherence to the prompt and its visual quality. Note any artifacts.
2. Analyze Image B's adherence to the prompt and its visual quality. Note any artifacts.
3. Directly compare A and B on "Realism". Which one looks more like a real photo and less like a generated image with artifacts?
4. Explain your reasoning for the winner.

**Verdict:**
Output your final decision as one of the following:
* [[Winner: Image A]]
* [[Winner: Image B]]
* [[Winner: Tie]]"""


def _tensor_to_pil(image_tensor):
    """Convert a [C, H, W] float tensor in [0, 1] to a PIL Image."""
    array = (image_tensor.clamp(0, 1).cpu().float().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
    return Image.fromarray(array)


def _pil_to_base64(image, max_size=1024):
    """Convert a PIL Image to a base64-encoded data URI, resizing if needed."""
    width, height = image.size
    if max(width, height) > max_size:
        scale = max_size / max(width, height)
        image = image.resize((int(width * scale), int(height * scale)), Image.LANCZOS)

    buffered = BytesIO()
    image.save(buffered, format="PNG")
    encoded = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def _parse_verdict(response_text):
    """Extract the winner from the LLM response text.

    Returns:
        "A", "B", or "Tie"
    """
    pattern = r"\[\[Winner:\s*(Image\s*[AB]|Tie)\]\]"
    match = re.search(pattern, response_text, re.IGNORECASE)
    if match:
        winner_raw = match.group(1).strip().lower()
        if "image a" in winner_raw:
            return "A"
        if "image b" in winner_raw:
            return "B"
        return "Tie"

    # Fallback: look for less strict patterns
    text_lower = response_text.lower()
    if "winner: image a" in text_lower or "winner:image a" in text_lower:
        return "A"
    if "winner: image b" in text_lower or "winner:image b" in text_lower:
        return "B"
    if "winner: tie" in text_lower or "winner:tie" in text_lower:
        return "Tie"

    logger.warning(f"Could not parse verdict from response: {response_text[-200:]}")
    return "Tie"


def _resolve_with_debiasing(verdict_ab, verdict_ba):
    """Resolve the final winner using position de-biasing.

    Round 1 (AB order): Image A = role_a, Image B = role_b
    Round 2 (BA order): Image A = role_b, Image B = role_a

    To compare fairly, we translate both verdicts into role_a/role_b space:
      - Round 1: "A" means role_a wins, "B" means role_b wins
      - Round 2: "A" means role_b wins, "B" means role_a wins (swapped)

    Agreement logic:
      - Both say role_a wins → role_a
      - Both say role_b wins → role_b
      - Both say Tie → Tie
      - Disagreement → Tie (conservative)

    Returns:
        (final_winner, is_consistent): final_winner is "role_a" | "role_b" | "Tie"
    """
    # Translate to role-space
    round1_role_winner = verdict_ab  # "A" = role_a, "B" = role_b, "Tie" = Tie
    # In round 2, positions are swapped: "A" = role_b, "B" = role_a
    if verdict_ba == "A":
        round2_role_winner = "B"  # role_b won (was placed as Image A)
    elif verdict_ba == "B":
        round2_role_winner = "A"  # role_a won (was placed as Image B)
    else:
        round2_role_winner = "Tie"

    if round1_role_winner == round2_role_winner:
        return round1_role_winner, True

    return "Tie", False


class LLMJudgeScorer:
    """Pairwise image comparison scorer using a VLM API with CoT and position de-biasing."""

    def __init__(
        self,
        api_base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key=None,
        model_name="qwen-vl-max",
        max_concurrent=8,
        max_retries=3,
        timeout=120,
        image_max_size=1024,
    ):
        """Initialize the LLM Judge scorer.

        Args:
            api_base_url: OpenAI-compatible API base URL.
            api_key: API key. If None, reads from DASHSCOPE_API_KEY env var.
            model_name: Model name for the API.
            max_concurrent: Maximum number of concurrent API requests.
            max_retries: Number of retries per failed API call.
            timeout: Timeout in seconds per API call.
            image_max_size: Max dimension for image resizing before encoding.
        """
        import os
        from test import AsyncOpenAI

        if api_key is None:
            api_key = os.environ.get("DASHSCOPE_API_KEY", os.environ.get("OPENAI_API_KEY", ""))
        if not api_key:
            raise ValueError(
                "API key required. Set --llm_judge_api_key, or DASHSCOPE_API_KEY / OPENAI_API_KEY env var."
            )

        self.client = AsyncOpenAI(base_url=api_base_url, api_key=api_key)
        self.model_name = model_name
        self.max_concurrent = max_concurrent
        self.max_retries = max_retries
        self.timeout = timeout
        self.image_max_size = image_max_size
        self._semaphore = asyncio.Semaphore(max_concurrent)

    async def _call_api(self, prompt_text, image_a_base64, image_b_base64):
        """Make a single API call for pairwise comparison.

        Returns:
            (verdict, reasoning): verdict is "A", "B", or "Tie"
        """
        messages = [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "**Image A:**"},
                    {"type": "image_url", "image_url": {"url": image_a_base64}},
                    {"type": "text", "text": "**Image B:**"},
                    {"type": "image_url", "image_url": {"url": image_b_base64}},
                    {"type": "text", "text": prompt_text},
                ],
            },
        ]

        for attempt in range(self.max_retries):
            try:
                async with self._semaphore:
                    response = await asyncio.wait_for(
                        self.client.chat.completions.create(
                            model=self.model_name,
                            messages=messages,
                            temperature=0,
                            max_tokens=1024,
                        ),
                        timeout=self.timeout,
                    )
                reasoning = response.choices[0].message.content
                verdict = _parse_verdict(reasoning)
                return verdict, reasoning

            except asyncio.TimeoutError:
                logger.warning(f"API call timed out (attempt {attempt + 1}/{self.max_retries})")
            except Exception as error:
                logger.warning(f"API call failed (attempt {attempt + 1}/{self.max_retries}): {error}")

            if attempt < self.max_retries - 1:
                await asyncio.sleep(2 ** attempt)

        logger.error("All API retries exhausted, defaulting to Tie")
        return "Tie", "API call failed after all retries."

    async def _compare_single_pair(self, prompt, pil_image_a, pil_image_b, sample_idx):
        """Compare a single pair with position de-biasing (two API calls).

        Returns a dict with detailed comparison results.
        """
        image_a_b64 = _pil_to_base64(pil_image_a, max_size=self.image_max_size)
        image_b_b64 = _pil_to_base64(pil_image_b, max_size=self.image_max_size)
        prompt_text = JUDGE_USER_PROMPT_TEMPLATE.format(prompt=prompt)

        # Round 1: A=role_a, B=role_b (original order)
        verdict_ab, reasoning_ab = await self._call_api(prompt_text, image_a_b64, image_b_b64)

        # Round 2: A=role_b, B=role_a (swapped order for de-biasing)
        verdict_ba, reasoning_ba = await self._call_api(prompt_text, image_b_b64, image_a_b64)

        final_winner, is_consistent = _resolve_with_debiasing(verdict_ab, verdict_ba)

        return {
            "sample_idx": sample_idx,
            "prompt": prompt,
            "round1_verdict": verdict_ab,
            "round1_reasoning": reasoning_ab,
            "round2_verdict": verdict_ba,
            "round2_reasoning": reasoning_ba,
            "final_winner": final_winner,
            "consistent": is_consistent,
        }

    async def _evaluate_async(self, pil_images_a, pil_images_b, prompts):
        """Run all pairwise comparisons asynchronously."""
        tasks = [
            self._compare_single_pair(prompt, img_a, img_b, idx)
            for idx, (prompt, img_a, img_b) in enumerate(zip(prompts, pil_images_a, pil_images_b))
        ]
        results = await asyncio.gather(*tasks)
        return list(results)

    def evaluate(self, images_a, images_b, prompts, role_a="student", role_b="real_teacher"):
        """Run pairwise evaluation with position de-biasing.

        Args:
            images_a: Tensor[N, C, H, W] in [0, 1] — images from role A.
            images_b: Tensor[N, C, H, W] in [0, 1] — images from role B.
            prompts: list[str] of length N.
            role_a: Name of role A (for reporting).
            role_b: Name of role B (for reporting).

        Returns:
            dict with:
                - win_rate_a: float (role A win rate)
                - win_rate_b: float (role B win rate)
                - tie_rate: float
                - consistency_rate: float (fraction where both rounds agree)
                - total_comparisons: int
                - per_sample: list[dict] with per-sample details
        """
        num_samples = min(len(images_a), len(images_b), len(prompts))
        images_a = images_a[:num_samples]
        images_b = images_b[:num_samples]
        prompts = prompts[:num_samples]

        logger.info(
            f"LLM Judge: comparing {num_samples} pairs ({role_a} vs {role_b}), "
            f"API calls = {num_samples * 2} (position de-biasing)"
        )

        # Convert tensors to PIL images
        pil_images_a = [_tensor_to_pil(images_a[i]) for i in range(num_samples)]
        pil_images_b = [_tensor_to_pil(images_b[i]) for i in range(num_samples)]

        # Run async evaluation
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import nest_asyncio
                nest_asyncio.apply()
                per_sample = loop.run_until_complete(
                    self._evaluate_async(pil_images_a, pil_images_b, prompts)
                )
            else:
                per_sample = loop.run_until_complete(
                    self._evaluate_async(pil_images_a, pil_images_b, prompts)
                )
        except RuntimeError:
            per_sample = asyncio.run(
                self._evaluate_async(pil_images_a, pil_images_b, prompts)
            )

        # Aggregate results
        wins_a = sum(1 for r in per_sample if r["final_winner"] == "A")
        wins_b = sum(1 for r in per_sample if r["final_winner"] == "B")
        ties = sum(1 for r in per_sample if r["final_winner"] == "Tie")
        consistent_count = sum(1 for r in per_sample if r["consistent"])

        results = {
            "role_a": role_a,
            "role_b": role_b,
            "win_rate_a": wins_a / num_samples if num_samples > 0 else 0.0,
            "win_rate_b": wins_b / num_samples if num_samples > 0 else 0.0,
            "tie_rate": ties / num_samples if num_samples > 0 else 0.0,
            "consistency_rate": consistent_count / num_samples if num_samples > 0 else 0.0,
            "wins_a": wins_a,
            "wins_b": wins_b,
            "ties": ties,
            "total_comparisons": num_samples,
            "per_sample": per_sample,
        }

        logger.info(
            f"LLM Judge results: {role_a} wins {wins_a} ({results['win_rate_a']:.1%}), "
            f"{role_b} wins {wins_b} ({results['win_rate_b']:.1%}), "
            f"Tie {ties} ({results['tie_rate']:.1%}), "
            f"Consistency {results['consistency_rate']:.1%}"
        )

        return results

    def unload(self):
        """Clean up resources."""
        del self.client
        self.client = None
