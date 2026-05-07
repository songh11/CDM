# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import json
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, Sampler


class TextPromptDataset(Dataset):
    """Dataset for loading text prompts from a file."""
    
    def __init__(self, dataset, split="train", max_samples=-1):
        """Initialize the dataset.
        
        Args:
            dataset: Path to the dataset directory
            split: Data split ('train' or 'test')
            max_samples: Maximum number of samples to load. -1 or None means no limit.
        """
        self.file_path = os.path.join(dataset, f"{split}.txt")
        with open(self.file_path, "r") as f:
            self.prompts = [line.strip() for line in f.readlines()]
        
        if max_samples is not None and max_samples > 0 and len(self.prompts) > max_samples:
            self.prompts = self.prompts[:max_samples]

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return {"prompt": self.prompts[idx], "metadata": {}}

    @staticmethod
    def collate_fn(examples):
        """Collate function for DataLoader."""
        prompts = [example["prompt"] for example in examples]
        metadatas = [example["metadata"] for example in examples]
        return prompts, metadatas


class GenevalPromptDataset(Dataset):
    """Dataset for loading prompts with metadata from JSONL file."""
    
    def __init__(self, dataset, split="train", max_samples=-1):
        """Initialize the dataset.
        
        Args:
            dataset: Path to the dataset directory
            split: Data split ('train' or 'test')
            max_samples: Maximum number of samples to load. -1 or None means no limit.
        """
        self.file_path = os.path.join(dataset, f"{split}_metadata.jsonl")
        with open(self.file_path, "r", encoding="utf-8") as f:
            self.metadatas = [json.loads(line) for line in f]
            self.prompts = [item["prompt"] for item in self.metadatas]
        
        if max_samples is not None and max_samples > 0 and len(self.prompts) > max_samples:
            self.prompts = self.prompts[:max_samples]
            self.metadatas = self.metadatas[:max_samples]

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return {"prompt": self.prompts[idx], "metadata": self.metadatas[idx]}

    @staticmethod
    def collate_fn(examples):
        """Collate function for DataLoader."""
        prompts = [example["prompt"] for example in examples]
        metadatas = [example["metadata"] for example in examples]
        return prompts, metadatas


class DPGPromptDataset(Dataset):
    """Dataset for loading DPG-Bench prompts from a CSV file.

    DPG-Bench CSV has columns: item_id, text, keywords, proposition_id,
    dependency, category_broad, category_detailed, tuple, question_natural_language.

    Each unique item_id corresponds to one prompt. This dataset deduplicates
    by item_id and returns (prompt, metadata) where metadata includes the item_id
    needed for DPG scoring.
    """

    def __init__(self, dataset, split="test", max_samples=-1):
        """Initialize the dataset.

        Args:
            dataset: Path to the dataset directory (must contain dpg_bench.csv)
            split: Data split (unused for DPG-Bench, kept for API consistency)
            max_samples: Maximum number of samples to load. -1 or None means no limit.
        """
        import pandas as pd

        csv_path = os.path.join(dataset, "dpg_bench.csv")
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(
                f"DPG-Bench CSV not found at {csv_path}. "
                f"Download it from https://github.com/TencentQQGYLab/ELLA/blob/main/dpg_bench/dpg_bench.csv"
            )

        data = pd.read_csv(csv_path)
        seen_ids = set()
        self.prompts = []
        self.item_ids = []

        for _, row in data.iterrows():
            item_id = row.item_id
            if item_id not in seen_ids:
                seen_ids.add(item_id)
                self.prompts.append(str(row.text).strip())
                self.item_ids.append(str(item_id))

        if max_samples is not None and max_samples > 0 and len(self.prompts) > max_samples:
            self.prompts = self.prompts[:max_samples]
            self.item_ids = self.item_ids[:max_samples]

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return {
            "prompt": self.prompts[idx],
            "metadata": {"item_id": self.item_ids[idx]},
        }

    @staticmethod
    def collate_fn(examples):
        """Collate function for DataLoader."""
        prompts = [example["prompt"] for example in examples]
        metadatas = [example["metadata"] for example in examples]
        return prompts, metadatas


class DistributedKRepeatSampler(Sampler):
    """Distributed sampler that repeats each sample k times with sampling without replacement.
    
    This sampler ensures that each prompt is sampled k times across all processes,
    which is useful for computing per-prompt statistics.
    
    Key feature: Uses sampling WITHOUT replacement - each sample will be used exactly once
    before the dataset is reshuffled. This ensures:
    - All samples are seen before any repetition
    - More stable training with better data utilization
    - Faster convergence compared to sampling with replacement
    """
    
    def __init__(self, dataset, batch_size, k, num_replicas, rank, seed=0):
        """Initialize the sampler.
        
        Args:
            dataset: Dataset to sample from
            batch_size: Batch size per GPU
            k: Number of times to repeat each sample
            num_replicas: Number of processes (world size)
            rank: Current process rank
            seed: Random seed for reproducibility
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.k = k
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed

        self.total_samples = self.num_replicas * self.batch_size
        assert (
            self.total_samples % self.k == 0
        ), f"k can not div n*b, k{k}-num_replicas{num_replicas}-batch_size{batch_size}"
        self.m = self.total_samples // self.k
        self.epoch = 0
        
        # 无放回采样的状态
        self._shuffled_indices = None
        self._pointer = 0
        self._shuffle_epoch = 0

    def _get_shuffled_indices(self, generator):
        """生成打乱后的完整数据集索引"""
        dataset_size = len(self.dataset)
        return torch.randperm(dataset_size, generator=generator).tolist()

    def __iter__(self):
        while True:
            g = torch.Generator()
            dataset_size = len(self.dataset)
            
            # 初始化或重新打乱索引队列
            if self._shuffled_indices is None or self._pointer + self.m > len(self._shuffled_indices):
                # 需要重新打乱
                g.manual_seed(self.seed + self._shuffle_epoch)
                self._shuffled_indices = self._get_shuffled_indices(g)
                self._pointer = 0
                self._shuffle_epoch += 1
            
            # 从队列中取 m 个不重复的索引
            if dataset_size >= self.m:
                indices = self._shuffled_indices[self._pointer : self._pointer + self.m]
                self._pointer += self.m
            else:
                # 数据集太小，需要重复采样以填满当前 batch
                # 但仍然保持无放回的逻辑：先用完当前剩余的，再重新打乱
                indices = []
                remaining = self.m
                while remaining > 0:
                    available = len(self._shuffled_indices) - self._pointer
                    if available <= 0:
                        # 重新打乱
                        g.manual_seed(self.seed + self._shuffle_epoch)
                        self._shuffled_indices = self._get_shuffled_indices(g)
                        self._pointer = 0
                        self._shuffle_epoch += 1
                        available = len(self._shuffled_indices)
                    
                    take = min(remaining, available)
                    indices.extend(self._shuffled_indices[self._pointer : self._pointer + take])
                    self._pointer += take
                    remaining -= take
            
            # 每个 prompt 重复 k 次
            repeated_indices = [idx for idx in indices for _ in range(self.k)]

            # 打乱重复后的索引（使用当前 epoch 的种子）
            g.manual_seed(self.seed + self.epoch)
            shuffled_indices = torch.randperm(len(repeated_indices), generator=g).tolist()
            shuffled_samples = [repeated_indices[i] for i in shuffled_indices]

            # 分配给各个 GPU
            per_card_samples = []
            for i in range(self.num_replicas):
                start = i * self.batch_size
                end = start + self.batch_size
                per_card_samples.append(shuffled_samples[start:end])
            
            self.epoch += 1
            yield per_card_samples[self.rank]

    def set_epoch(self, epoch):
        """Set the epoch for deterministic shuffling.
        
        Note: In the new implementation, this mainly affects the shuffling of 
        repeated indices within each batch, not the overall sampling order.
        """
        self.epoch = epoch
    
    def get_progress(self):
        """获取当前遍历进度
        
        Returns:
            tuple: (已使用的样本数, 总样本数, 完成的轮次)
        """
        dataset_size = len(self.dataset)
        return self._pointer, dataset_size, self._shuffle_epoch - 1


class TextImagePairDataset(Dataset):
    """Dataset for loading text-image pairs from a JSON file.
    
    The JSON file should be an array of objects:
        [{"input_prompt": "...", "output_image": "relative/path/to/image.png", ...}, ...]
    
    Images are loaded lazily and returned as tensors via metadata.
    """

    def __init__(self, dataset, split="train", max_samples=-1, image_size=512, image_base_path=None):
        """Initialize the dataset.

        Args:
            dataset: Path to the dataset directory containing the JSONL file.
            split: Data split ('train' or 'test').
            max_samples: Maximum number of samples to load. -1 or None means no limit.
            image_size: Target image size for resize and crop. Can be an int (square)
                        or a tuple of (height, width) for non-square images.
            image_base_path: Base directory for resolving image paths.
                             If None, defaults to ``dataset``.
        """
        self.root_dir = dataset
        self.image_base_path = image_base_path if image_base_path is not None else dataset

        if isinstance(image_size, (list, tuple)):
            target_height, target_width = image_size
            resize_short_edge = min(target_height, target_width)
            crop_size = (target_height, target_width)
        else:
            resize_short_edge = image_size
            crop_size = image_size

        self.transform = transforms.Compose([
            transforms.Resize(resize_short_edge, interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

        json_path = os.path.join(dataset, f"{split}.json")
        with open(json_path, "r", encoding="utf-8") as f:
            self.items = json.load(f)

        if max_samples is not None and max_samples > 0 and len(self.items) > max_samples:
            self.items = self.items[:max_samples]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        image_path = os.path.join(self.image_base_path, item["output_image"])
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.transform(image)
        return {
            "prompt": item["input_prompt"],
            "metadata": {"image": image_tensor},
        }

    @staticmethod
    def collate_fn(examples):
        prompts = [e["prompt"] for e in examples]
        metadatas = [e["metadata"] for e in examples]
        return prompts, metadatas


class ImageEditDataset(Dataset):
    """Dataset for image editing tasks: provides (edit_instruction, source_image, target_image).

    The JSON file should be an array of objects::

        [{"input_prompt": "...", "input_image": ["image/src.png"],
          "output_image": "image/tgt.png"}, ...]

    Each item provides:
      - ``input_prompt``: the editing instruction (used as the text prompt)
      - ``input_image``: list of source image paths (first element is used)
      - ``output_image``: target/ground-truth image path (optional, for text_image loss)

    Source images are returned as PIL Images in metadata (not tensors) because
    the LongCatImageEditPipeline handles preprocessing internally.
    Target images are returned as tensors for training loss computation.
    """

    def __init__(self, dataset, split="train", max_samples=-1, image_size=512, image_base_path=None):
        """Initialize the dataset.

        Args:
            dataset: Path to the dataset directory containing the JSON file.
            split: Data split ('train' or 'test').
            max_samples: Maximum number of samples to load. -1 or None means no limit.
            image_size: Target image size for target image transform. Can be an int (square)
                        or a tuple of (height, width).
            image_base_path: Base directory for resolving image paths.
                             If None, defaults to ``dataset``.
        """
        self.root_dir = dataset
        self.image_base_path = image_base_path if image_base_path is not None else dataset

        json_path = os.path.join(dataset, f"{split}.json")
        with open(json_path, "r", encoding="utf-8") as f:
            self.items = json.load(f)

        if max_samples is not None and max_samples > 0 and len(self.items) > max_samples:
            self.items = self.items[:max_samples]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]

        # Load source image as PIL (pipeline handles preprocessing)
        source_image_rel = item["input_image"][0]
        source_image_path = os.path.join(self.image_base_path, source_image_rel)
        source_image = Image.open(source_image_path).convert("RGB")

        metadata = {"source_image": source_image}

        # Load target image as tensor if available (for text_image loss)
        # if "output_image" in item and item["output_image"]:
        #     target_image_path = os.path.join(self.image_base_path, item["output_image"])
        #     target_image = Image.open(target_image_path).convert("RGB")
        #     metadata["image"] = self.target_transform(target_image)

        return {
            "prompt": item["input_prompt"],
            "metadata": metadata,
        }

    @staticmethod
    def collate_fn(examples):
        prompts = [e["prompt"] for e in examples]
        metadatas = [e["metadata"] for e in examples]
        return prompts, metadatas

# ==================== Dataset Registry ====================
DATASET_REGISTRY = {
    "general_ocr": TextPromptDataset,
    "geneval": GenevalPromptDataset,
    "dpg": DPGPromptDataset,
    "text_image_pair": TextImagePairDataset,
    "image_edit": ImageEditDataset,
}
