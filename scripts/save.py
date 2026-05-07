"""
Prepare Student Pipeline

Builds a complete diffusers-compatible pipeline directory for each checkpoint
in an experiment, so that evaluation can simply call
``Pipeline.from_pretrained(student_dir)`` without any knowledge of LoRA,
Accelerator, or training-time model structure.

For LoRA checkpoints:
    1. Load base pipeline from pretrained_path
    2. Apply LoRA adapter to transformer
    3. Load LoRA weights from checkpoint
    4. Merge LoRA into base weights (merge_and_unload)
    5. Save the full pipeline to output directory

For full fine-tuning checkpoints:
    1. Load base pipeline from pretrained_path
    2. Load transformer weights from checkpoint
    3. Save the full pipeline to output directory
"""

import os

import argparse
import glob
import json
import logging
import re
import shutil

import torch
import safetensors.torch
from peft import LoraConfig, get_peft_model, PeftModel

from cdm.models.model_adapters import create_model_adapter
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ==================== Pretrained Path Resolution ====================

def resolve_pretrained_path(path_or_repo_id):
    """Resolve a pretrained model path.

    Accepts either a local directory or a HuggingFace Hub repo id (e.g.
    ``stabilityai/stable-diffusion-3-medium-diffusers``). For repo ids, this
    function locates the model snapshot inside the local HuggingFace cache and
    returns the absolute snapshot path. The snapshot is downloaded on demand
    if it is not yet cached.

    Returning a real local directory (rather than the bare repo id) is
    important because downstream code symlinks individual sub-directories
    (text_encoder, vae, ...) into the output pipeline directory, which
    requires a concrete filesystem path.
    """
    # Local path takes precedence
    if os.path.exists(path_or_repo_id):
        return path_or_repo_id

    # Heuristic: HF repo ids look like "namespace/name", with no path separators
    # beyond a single "/". Anything else that does not exist locally is treated
    # as an error.
    looks_like_repo_id = (
        "/" in path_or_repo_id
        and not path_or_repo_id.startswith((".", "/", "~"))
        and path_or_repo_id.count("/") == 1
    )
    if not looks_like_repo_id:
        raise FileNotFoundError(
            f"Pretrained model not found at {path_or_repo_id}. "
            f"Use --pretrained_path_override to specify the correct path."
        )

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise ImportError(
            "huggingface_hub is required to resolve repo ids. "
            "Install it with `pip install huggingface_hub`."
        ) from exc

    logger.info(f"Resolving HuggingFace repo id: {path_or_repo_id}")
    try:
        # local_files_only=True first to avoid unexpected downloads
        snapshot_path = snapshot_download(
            repo_id=path_or_repo_id, local_files_only=True
        )
        logger.info(f"  Found local snapshot at: {snapshot_path}")
    except Exception:
        logger.info(
            f"  No local snapshot found in HF cache, downloading {path_or_repo_id}..."
        )
        snapshot_path = snapshot_download(repo_id=path_or_repo_id)
        logger.info(f"  Downloaded snapshot to: {snapshot_path}")

    return snapshot_path

# ==================== Config Loading ====================
def load_train_config(experiment_dir):
    """Load training config.json from experiment directory."""
    config_path = os.path.join(experiment_dir, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Training config not found at {config_path}. "
            f"Make sure --experiment_dir points to the experiment root."
        )
    with open(config_path, "r") as f:
        return json.load(f)


class DictConfig:
    """Lightweight wrapper to access nested dicts with dot notation."""

    def __init__(self, data):
        object.__setattr__(self, "_data", {})
        for key, value in data.items():
            if isinstance(value, dict):
                self._data[key] = DictConfig(value)
            else:
                self._data[key] = value

    def __getattr__(self, name):
        if name.startswith("_"):
            return object.__getattribute__(self, name)
        data = object.__getattribute__(self, "_data")
        if name in data:
            return data[name]
        return None

    def __setattr__(self, name, value):
        if name.startswith("_"):
            object.__setattr__(self, name, value)
        else:
            self._data[name] = value

    def get(self, name, default=None):
        val = getattr(self, name)
        return val if val is not None else default


# ==================== Checkpoint Discovery ====================

def discover_checkpoints(experiment_dir, checkpoint_steps=None):
    """Discover checkpoint directories under experiment_dir/checkpoints/."""
    checkpoint_root = os.path.join(experiment_dir, "checkpoints")
    if not os.path.isdir(checkpoint_root):
        raise FileNotFoundError(f"Checkpoints directory not found: {checkpoint_root}")

    checkpoints = []
    for entry in os.listdir(checkpoint_root):
        match = re.match(r"checkpoint-(\d+)$", entry)
        if match:
            step = int(match.group(1))
            full_path = os.path.join(checkpoint_root, entry)
            if os.path.isdir(full_path):
                checkpoints.append((step, full_path))

    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint-* directories found in {checkpoint_root}")

    checkpoints.sort(key=lambda x: x[0])

    if checkpoint_steps is not None:
        requested = set(checkpoint_steps)
        filtered = [(s, p) for s, p in checkpoints if s in requested]
        found_steps = {s for s, _ in filtered}
        missing = requested - found_steps
        if missing:
            available = [s for s, _ in checkpoints]
            logger.warning(f"Requested steps {sorted(missing)} not found. Available: {available}")
        checkpoints = filtered

    return checkpoints


# ==================== Weight Loading ====================

def load_checkpoint_state_dict(checkpoint_path):
    """Load model state dict from a checkpoint directory.

    Supports both DDP and FSDP checkpoints:
      - **FSDP**: Looks for ``transformer_full_state_dict.safetensors`` first.
        This file is saved by ``save_fsdp_full_checkpoint()`` during training
        and contains the complete (unsharded) transformer weights.
      - **DDP**: Falls back to the standard Accelerator ``save_state()``
        convention (``model_*/*.safetensors`` or ``pytorch_model_*/*.bin``).

    Returns:
        state_dict: dict mapping parameter names to CPU tensors.
    """
    # Priority 1: FSDP full state_dict (saved by save_fsdp_full_checkpoint)
    fsdp_full_path = os.path.join(checkpoint_path, "transformer_full_state_dict.safetensors")
    if os.path.exists(fsdp_full_path):
        state_dict = safetensors.torch.load_file(fsdp_full_path, device="cpu")
        logger.info(f"  Loaded FSDP full state_dict from {fsdp_full_path}")
        return state_dict

    # Priority 2: Standard DDP safetensors files
    safetensors_candidates = sorted(
        glob.glob(os.path.join(checkpoint_path, "model_*", "*.safetensors"))
    )
    if not safetensors_candidates:
        safetensors_candidates = sorted(
            glob.glob(os.path.join(checkpoint_path, "*.safetensors"))
        )

    # Priority 3: Standard DDP pytorch bin files
    bin_candidates = sorted(
        glob.glob(os.path.join(checkpoint_path, "pytorch_model_*", "*.bin"))
    )
    if not bin_candidates:
        bin_candidates = sorted(
            glob.glob(os.path.join(checkpoint_path, "*.bin"))
        )

    if safetensors_candidates:
        weight_file = safetensors_candidates[0]
        state_dict = safetensors.torch.load_file(weight_file, device="cpu")
        logger.info(f"  Loaded weights from {weight_file}")
        return state_dict
    elif bin_candidates:
        weight_file = bin_candidates[0]
        state_dict = torch.load(weight_file, map_location="cpu", weights_only=True)
        logger.info(f"  Loaded weights from {weight_file}")
        return state_dict
    else:
        raise FileNotFoundError(
            f"No model weight files found in {checkpoint_path}. "
            f"Expected transformer_full_state_dict.safetensors (FSDP), "
            f"or safetensors/bin files in model_*/ or pytorch_model_*/ subdirectories (DDP)."
        )


# ==================== EMA Loading ====================

def _apply_ema_weights(transformer, checkpoint_path, use_ema):
    """Apply EMA weights to the transformer if available.

    Supports two EMA formats:
      - **FSDP format** (``ema_full_state.pt``): Contains a ``full_state_dict``
        key with the complete transformer state_dict gathered from all FSDP
        shards. The weights are loaded via ``load_state_dict()``.
      - **DDP format** (``ema_state.pt``): Contains an ``ema_parameters`` key
        with a list of tensors matching the trainable parameters in order.
        The weights are copied parameter-by-parameter.

    The FSDP format is checked first. If neither file exists, the function
    returns False and the raw training weights are kept.

    Args:
        transformer: The transformer model (may have LoRA adapters applied).
        checkpoint_path: Path to the checkpoint directory.
        use_ema: Whether to attempt loading EMA weights.

    Returns:
        True if EMA weights were successfully applied, False otherwise.
    """
    if not use_ema:
        return False

    # Priority 1: FSDP full EMA state_dict
    ema_full_path = os.path.join(checkpoint_path, "ema_full_state.pt")
    if os.path.exists(ema_full_path):
        logger.info(f"  Loading FSDP full EMA state_dict from {ema_full_path}")
        ema_state = torch.load(ema_full_path, map_location="cpu")
        ema_full_state_dict = ema_state["full_state_dict"]
        transformer.load_state_dict(ema_full_state_dict, strict=False)
        logger.info("  ✅ FSDP EMA weights applied successfully")
        return True

    # Priority 2: DDP EMA parameter list
    ema_path = os.path.join(checkpoint_path, "ema_state.pt")
    if os.path.exists(ema_path):
        logger.info(f"  Loading EMA weights from {ema_path}")
        ema_state = torch.load(ema_path, map_location="cpu")
        ema_parameters = ema_state["ema_parameters"]
        trainable_params = [p for p in transformer.parameters() if p.requires_grad]
        if len(ema_parameters) == len(trainable_params):
            for param, ema_param in zip(trainable_params, ema_parameters):
                param.data.copy_(ema_param.data)
            logger.info("  ✅ EMA weights applied successfully")
            return True
        else:
            logger.warning(
                f"  EMA parameter count ({len(ema_parameters)}) != "
                f"trainable parameter count ({len(trainable_params)}), skipping EMA"
            )
            return False

    logger.info("  No EMA state found (checked ema_full_state.pt and ema_state.pt), using raw training weights")
    return False


# ==================== Pipeline Building ====================

def build_student_pipeline_dir(
    pretrained_path,
    checkpoint_path,
    output_dir,
    student_use_lora,
    lora_rank,
    lora_alpha,
    model_adapter,
    lora_target_modules,
    use_symlinks=True,
    use_ema=True,
    custom_sigmas=None,
    deterministic=None,
):
    """Build a complete pipeline directory for a student checkpoint.

    For LoRA mode:
        - Load base pipeline, apply LoRA, load checkpoint weights, merge, save transformer
    For Full-FT mode:
        - Load base pipeline, load checkpoint weights into transformer, save transformer

    Non-transformer components (scheduler, text_encoder, vae, tokenizer, etc.)
    are symlinked from the pretrained_path to save disk space.

    Args:
        pretrained_path: Path to the pretrained base model.
        checkpoint_path: Path to the training checkpoint directory.
        output_dir: Where to save the complete pipeline directory.
        student_use_lora: Whether the student was trained with LoRA.
        lora_rank: LoRA rank (only used if student_use_lora=True).
        lora_alpha: LoRA alpha (only used if student_use_lora=True).
        model_adapter: Model adapter for pipeline loading.
        lora_target_modules: List of LoRA target module names.
        use_symlinks: If True, symlink non-transformer components instead of copying.
        use_ema: If True and ema_state.pt exists, use EMA weights instead of raw training weights.
    """
    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"Loading base pipeline from: {pretrained_path}")
    pipeline = model_adapter.load_pipeline(pretrained_path)

    # Load checkpoint weights
    checkpoint_state_dict = load_checkpoint_state_dict(checkpoint_path)

    transformer = pipeline.transformer

    if student_use_lora:
        logger.info(f"  Applying LoRA (rank={lora_rank}, alpha={lora_alpha}) and loading weights...")
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            init_lora_weights="gaussian",
            target_modules=lora_target_modules,
        )
        transformer = get_peft_model(transformer, lora_config)
        transformer.load_state_dict(checkpoint_state_dict, strict=False)

        # Apply EMA weights BEFORE merge_and_unload, while LoRA params still have requires_grad=True
        ema_applied = _apply_ema_weights(transformer, checkpoint_path, use_ema)

        logger.info("  Merging LoRA weights into base model...")
        transformer = transformer.merge_and_unload()
    else:
        logger.info("  Loading full fine-tuning weights...")
        transformer.load_state_dict(checkpoint_state_dict, strict=False)

        # Apply EMA weights for full fine-tuning mode
        ema_applied = _apply_ema_weights(transformer, checkpoint_path, use_ema)

    pipeline.transformer = transformer

    pipeline.transformer = pipeline.transformer.to(dtype=torch.bfloat16)
    logger.info("  Casted transformer to bfloat16 before saving")

    # Save the complete pipeline
    logger.info(f"  Saving complete pipeline to: {output_dir}")
    pipeline.save_pretrained(output_dir)

    # If use_symlinks, replace copied non-transformer directories with symlinks
    # to save disk space. The transformer directory keeps the actual merged weights.
    if use_symlinks:
        _replace_with_symlinks(pretrained_path, output_dir, keep_dirs={"transformer"})

    # Save metadata about how this pipeline was built
    metadata = {
        "pretrained_path": pretrained_path,
        "checkpoint_path": checkpoint_path,
        "student_use_lora": student_use_lora,
        "lora_rank": lora_rank if student_use_lora else None,
        "lora_alpha": lora_alpha if student_use_lora else None,
        "model_type": model_adapter.get_model_type(),
        "ema_applied": ema_applied,
        "custom_sigmas": custom_sigmas,
        "deterministic": deterministic,
    }
    metadata_path = os.path.join(output_dir, "student_pipeline_meta.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"  Pipeline saved successfully. Metadata: {metadata_path}")

    # Cleanup
    del pipeline, transformer, checkpoint_state_dict
    torch.cuda.empty_cache()


def _replace_with_symlinks(pretrained_path, output_dir, keep_dirs=None):
    """Replace copied component directories with symlinks to the pretrained model.

    This saves significant disk space since text encoders, VAE, tokenizers, and
    scheduler are identical to the pretrained model.

    Args:
        pretrained_path: Path to the pretrained base model.
        output_dir: The saved pipeline directory.
        keep_dirs: Set of directory names to keep as actual copies (e.g. {"transformer"}).
    """
    if keep_dirs is None:
        keep_dirs = {"transformer"}

    pretrained_path = os.path.abspath(pretrained_path)

    for entry in os.listdir(output_dir):
        entry_path = os.path.join(output_dir, entry)
        if not os.path.isdir(entry_path):
            continue
        if entry in keep_dirs:
            continue

        source_path = os.path.join(pretrained_path, entry)
        if not os.path.exists(source_path):
            continue

        # Remove the copied directory and create a symlink
        shutil.rmtree(entry_path)
        os.symlink(source_path, entry_path)
        logger.info(f"    Symlinked {entry}/ -> {source_path}")


# ==================== Main ====================

def main():
    parser = argparse.ArgumentParser(
        description="Build complete pipeline directories from training checkpoints.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--experiment_dir", type=str, required=True,
        help="Path to the experiment directory (containing config.json and checkpoints/).",
    )
    parser.add_argument(
        "--checkpoint_steps", type=int, nargs="*", default=None,
        help="Specific checkpoint steps to export. If not specified, exports all.",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help=(
            "Root directory for output pipelines. Each checkpoint produces a "
            "subdirectory like output_dir/checkpoint-{step}/. "
            "Defaults to experiment_dir/pipelines/."
        ),
    )
    parser.add_argument(
        "--no_symlinks", action="store_true",
        help="Copy all components instead of symlinking non-transformer directories.",
    )
    parser.add_argument(
        "--pretrained_path_override", type=str, default=None,
        help=(
            "Override the pretrained model path from config.json. "
            "Useful when the original path is no longer accessible."
        ),
    )
    parser.add_argument(
        "--no_ema", action="store_true",
        help="Use raw training weights instead of EMA weights (EMA is used by default when available).",
    )

    args = parser.parse_args()

    # Load training config
    train_config_dict = load_train_config(args.experiment_dir)
    config = DictConfig(train_config_dict)

    # Resolve pretrained path. Accept either a local directory or a
    # HuggingFace Hub repo id (e.g. "stabilityai/stable-diffusion-3-medium-diffusers").
    # In the latter case we resolve it to the local snapshot directory inside
    # the HuggingFace cache, so that downstream symlinking still works.
    pretrained_path_raw = args.pretrained_path_override or config.pretrained.model
    pretrained_path = resolve_pretrained_path(pretrained_path_raw)

    # Resolve output directory
    output_root = args.output_dir or os.path.join(args.experiment_dir, "pipelines")

    # Get model adapter and student config
    base_model = getattr(config, "base_model", None) or "sd3"
    model_adapter = create_model_adapter(base_model, config)
    train_cfg = config.train
    student_config = train_cfg.student
    lora_target_modules = model_adapter.get_lora_target_modules(
        include_ff=getattr(student_config, 'lora_include_ff', False)
    )

    # Get student training settings
    student_use_lora = getattr(student_config, "use_lora", False)
    lora_rank = getattr(student_config, "lora_rank", 64)
    lora_alpha = getattr(student_config, "lora_alpha", 128)

    # Discover checkpoints
    checkpoints = discover_checkpoints(args.experiment_dir, args.checkpoint_steps)
    logger.info(f"Found {len(checkpoints)} checkpoint(s): {[s for s, _ in checkpoints]}")
    logger.info(f"Student mode: {'LoRA' if student_use_lora else 'Full Fine-Tuning'}")
    logger.info(f"Base model: {base_model}")
    logger.info(f"Pretrained path: {pretrained_path}")
    logger.info(f"Output root: {output_root}")

    # Build pipeline for each checkpoint
    for step, checkpoint_path in checkpoints:
        step_output_dir = os.path.join(output_root, f"checkpoint-{step}")

        if os.path.exists(step_output_dir) and os.path.isfile(
            os.path.join(step_output_dir, "student_pipeline_meta.json")
        ):
            logger.info(f"\nSkipping checkpoint-{step}: pipeline already exists at {step_output_dir}")
            continue

        logger.info(f"\n{'='*60}")
        logger.info(f"Building pipeline for checkpoint-{step}")
        logger.info(f"  Checkpoint: {checkpoint_path}")
        logger.info(f"  Output: {step_output_dir}")
        logger.info(f"{'='*60}")

        build_student_pipeline_dir(
            pretrained_path=pretrained_path,
            checkpoint_path=checkpoint_path,
            output_dir=step_output_dir,
            student_use_lora=student_use_lora,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            model_adapter=model_adapter,
            lora_target_modules=lora_target_modules,
            use_symlinks=not args.no_symlinks,
            use_ema=not args.no_ema,
            custom_sigmas=getattr(config, "custom_sigmas", None),
            deterministic=getattr(getattr(config, "eval", None), "deterministic", None),
        )

    logger.info(f"\n{'='*60}")
    logger.info("All pipelines built successfully!")
    logger.info(f"Output directory: {output_root}")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
