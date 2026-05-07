"""
DPG-Bench (Dense Prompt Graph Benchmark) scorer for evaluating text-to-image
alignment on dense, compositional prompts.

DPG-Bench contains ~1065 prompts, each with multiple VQA questions organized
in a dependency graph. For each generated image, a VQA model (mPLUG) answers
yes/no questions. Scores are computed per-question with dependency propagation
(if a parent question is answered "no", child questions are zeroed out).

This is a post-hoc metric that does NOT integrate into multi_score / reward_fn.
It runs in evaluation_v2.py Phase 3 after generation models are unloaded.

Reference: ELLA (TencentQQGYLab/ELLA) - dpg_bench/compute_dpg_bench.py

Dependencies:
    pip install modelscope

Usage:
    from flow_grpo.rewards.dpg_scorer import DPGScorer

    scorer = DPGScorer(
        csv_path="dataset/dpgbench/dpg_bench.csv",
        device="cuda",
    )
    results = scorer.evaluate(
        images=images_tensor,  # [N, C, H, W] in [0, 1]
        item_ids=["COCOval2014000000391895", ...],  # matching CSV item_ids
    )
"""

import os
import logging
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from PIL import Image

logger = logging.getLogger(__name__)


def _load_dpg_questions(csv_path):
    """Parse the DPG-Bench CSV into a structured question dictionary.

    Returns:
        question_dict: dict mapping item_id -> {
            "prompt": str,
            "qid2tuple": {qid: tuple_str},
            "qid2dependency": {qid: [parent_qid, ...]},
            "qid2question": {qid: question_str},
        }
    """
    data = pd.read_csv(csv_path)
    question_dict = {}

    for row_idx, row in data.iterrows():
        if row_idx == 0:
            continue

        item_id = row.item_id
        prompt = row.text
        question_id = int(row.proposition_id)

        dependency_parts = str(row.dependency).split(",")
        dependency_list = [int(d.strip()) for d in dependency_parts]

        if item_id not in question_dict:
            question_dict[item_id] = {
                "prompt": prompt,
                "qid2tuple": {},
                "qid2dependency": {},
                "qid2question": {},
            }

        question_dict[item_id]["qid2tuple"][question_id] = row.tuple
        question_dict[item_id]["qid2dependency"][question_id] = dependency_list
        question_dict[item_id]["qid2question"][question_id] = (
            row.question_natural_language
        )

    logger.info(
        f"Loaded DPG-Bench questions: {len(question_dict)} prompts from {csv_path}"
    )
    return question_dict


def _tensor_to_pil(image_tensor):
    """Convert a [C, H, W] float tensor in [0, 1] to a PIL Image."""
    array = (
        image_tensor.clamp(0, 1).cpu().float().numpy().transpose(1, 2, 0) * 255
    ).astype(np.uint8)
    return Image.fromarray(array)


class DPGScorer:
    """Evaluate generated images using the DPG-Bench protocol.

    Uses mPLUG VQA model to answer yes/no questions about each image,
    with dependency-aware score propagation.
    """

    def __init__(
        self,
        csv_path,
        device="cuda",
        vqa_model_name="damo/mplug_visual-question-answering_coco_large_en",
    ):
        self.csv_path = csv_path
        self.device = device
        self.vqa_model_name = vqa_model_name
        self.question_dict = _load_dpg_questions(csv_path)
        self.vqa_model = None

    def _load_vqa_model(self):
        """Lazy-load the mPLUG VQA model."""
        if self.vqa_model is not None:
            return

        logger.info(f"Loading mPLUG VQA model: {self.vqa_model_name}")
        from modelscope.pipelines import pipeline as ms_pipeline
        from modelscope.utils.constant import Tasks

        self.vqa_pipeline = ms_pipeline(
            Tasks.visual_question_answering,
            model=self.vqa_model_name,
            device=self.device,
        )
        self.vqa_model = True
        logger.info("mPLUG VQA model loaded successfully.")

    def _ask_vqa(self, pil_image, question):
        """Ask a yes/no question about an image using the VQA model."""
        result = self.vqa_pipeline({"image": pil_image, "question": question})
        return result["text"]

    def _score_single_image(self, pil_image, item_id):
        """Score a single image against all DPG questions for its item_id.

        Returns:
            overall_score: float in [0, 1]
            category_scores: dict mapping category -> score (0 or 1)
        """
        entry = self.question_dict.get(item_id)
        if entry is None:
            logger.warning(f"Item ID '{item_id}' not found in DPG-Bench CSV.")
            return 0.0, {}

        qid2question = entry["qid2question"]
        qid2dependency = entry["qid2dependency"]
        qid2tuple = entry["qid2tuple"]

        qid2raw_scores = {}
        for question_id, question_text in qid2question.items():
            answer = self._ask_vqa(pil_image, question_text)
            qid2raw_scores[question_id] = float(answer == "yes")

        qid2final_scores = qid2raw_scores.copy()
        for question_id, parent_ids in qid2dependency.items():
            parent_failed = any(
                qid2final_scores.get(parent_id, 1.0) == 0
                for parent_id in parent_ids
                if parent_id != 0
            )
            if parent_failed:
                qid2final_scores[question_id] = 0.0

        overall_score = (
            sum(qid2final_scores.values()) / len(qid2final_scores)
            if qid2final_scores
            else 0.0
        )

        category_scores = {}
        for question_id, tuple_str in qid2tuple.items():
            category = tuple_str.split("(")[0].strip()
            if category not in category_scores:
                category_scores[category] = []
            category_scores[category].append(qid2raw_scores.get(question_id, 0.0))

        return overall_score, category_scores

    def evaluate(self, images, item_ids):
        """Evaluate a batch of generated images against DPG-Bench.

        Args:
            images: Tensor [N, C, H, W] in [0, 1], or list of PIL Images
            item_ids: list[str] of length N, matching CSV item_id column

        Returns:
            dict with keys:
                "overall_score": float (mean across all images, scaled to 0-100)
                "per_sample_scores": list[float] of length N (each in [0, 1])
                "l1_category_scores": dict mapping L1 category -> mean score (0-100)
                "l2_category_scores": dict mapping L2 category -> mean score (0-100)
        """
        self._load_vqa_model()

        per_sample_scores = []
        all_category_scores = defaultdict(list)

        total_images = len(item_ids)
        logger.info(f"Evaluating {total_images} images with DPG-Bench...")

        for idx in range(total_images):
            if isinstance(images, torch.Tensor):
                pil_image = _tensor_to_pil(images[idx])
            else:
                pil_image = images[idx]

            item_id = item_ids[idx]
            sample_score, category_scores = self._score_single_image(
                pil_image, item_id
            )
            per_sample_scores.append(sample_score)

            for category, scores_list in category_scores.items():
                all_category_scores[category].extend(scores_list)

            if (idx + 1) % 50 == 0:
                print(
                    f"  DPG-Bench progress: {idx + 1}/{total_images} "
                    f"(running avg: {np.mean(per_sample_scores) * 100:.2f})"
                )

        overall_score = float(np.mean(per_sample_scores)) * 100

        l1_category_scores = defaultdict(list)
        l2_category_scores = {}
        for category, scores_list in all_category_scores.items():
            l2_mean = float(np.mean(scores_list)) * 100
            l2_category_scores[category] = l2_mean

            l1_category = category.split("-")[0].strip()
            l1_category_scores[l1_category].extend(scores_list)

        l1_category_means = {
            cat: float(np.mean(scores)) * 100
            for cat, scores in l1_category_scores.items()
        }

        print(f"DPG-Bench overall score: {overall_score:.2f}")
        for l1_cat, score in sorted(l1_category_means.items()):
            print(f"  {l1_cat}: {score:.2f}")

        return {
            "overall_score": overall_score,
            "per_sample_scores": per_sample_scores,
            "l1_category_scores": l1_category_means,
            "l2_category_scores": l2_category_scores,
        }

    def unload(self):
        """Unload the VQA model to free GPU memory."""
        if self.vqa_model is not None:
            del self.vqa_pipeline
            self.vqa_model = None
            torch.cuda.empty_cache()
            logger.info("DPG-Bench VQA model unloaded.")

    def get_all_prompts(self):
        """Return all unique prompts and their item_ids from the CSV.

        Returns:
            list of (item_id, prompt_text) tuples
        """
        prompts = []
        for item_id, entry in self.question_dict.items():
            prompts.append((item_id, entry["prompt"]))
        return prompts
