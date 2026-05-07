"""Cache directory configuration for reward model checkpoints.

All reward models are downloaded automatically via HuggingFace Hub
(``transformers.from_pretrained`` / ``huggingface_hub.hf_hub_download``)
or the corresponding library's built-in downloader (e.g., ``open_clip``,
``ImageReward``, ``PaddleOCR``).

By default, downloads go to the standard HuggingFace cache
(``~/.cache/huggingface/hub``, overridable via the ``HF_HOME`` /
``HF_HUB_CACHE`` environment variables).

To force all reward checkpoints into a custom directory, set the
``REWARD_CKPT_DIR`` environment variable. When set, this path is passed as
``cache_dir=`` to every download call in this package.
"""

import os
from typing import Optional


def get_reward_cache_dir() -> Optional[str]:
    """Return the cache directory for reward checkpoints.

    Returns the value of the ``REWARD_CKPT_DIR`` environment variable if set,
    otherwise ``None`` (which lets HuggingFace fall back to its default cache).
    """
    cache_dir = os.environ.get("REWARD_CKPT_DIR")
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


