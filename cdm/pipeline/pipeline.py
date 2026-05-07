 

from typing import List, Optional, Union

import torch




# ==================== Embedding Capture ====================

class EmbeddingCapture:
    """
    Wraps ``pipeline.encode_prompt()`` to capture text embeddings computed
    during native pipeline sampling, so they can be reused for training
    without redundant text encoder forward passes.

    Works with both SD3 and LongCat pipelines:
      - **SD3** ``encode_prompt`` returns
        ``(prompt_embeds, neg_prompt_embeds, pooled_prompt_embeds, neg_pooled)``
        → captures ``prompt_embeds`` and ``pooled_prompt_embeds``
      - **LongCat** ``encode_prompt`` returns
        ``(prompt_embeds, text_ids)``
        → captures ``prompt_embeds`` and ``text_ids``

    The captured values are exposed as ``prompt_embeds`` and
    ``auxiliary_text_embeds``, matching the ``ModelAdapter`` convention.

    Usage::

        capture = EmbeddingCapture(pipeline)
        capture.install()
        result = pipeline(prompt="a cat", ...)
        prompt_embeds, auxiliary = capture.collect()
        capture.uninstall()
    """

    def __init__(self, pipeline):
        self._pipeline = pipeline
        self._original_encode_prompt = pipeline.encode_prompt
        self.prompt_embeds: Optional[torch.Tensor] = None
        self.auxiliary_text_embeds: Optional[torch.Tensor] = None

    def install(self):
        """Monkey-patch encode_prompt to intercept embeddings."""
        original = self._original_encode_prompt
        capture = self

        def wrapped_encode_prompt(*args, **kwargs):
            result = original(*args, **kwargs)
            if len(result) == 4:
                # SD3: (prompt_embeds, neg_prompt_embeds, pooled, neg_pooled)
                capture.prompt_embeds = result[0]
                capture.auxiliary_text_embeds = result[2]
            elif len(result) == 2:
                # LongCat: (prompt_embeds, text_ids)
                capture.prompt_embeds = result[0]
                capture.auxiliary_text_embeds = result[1]
            else:
                capture.prompt_embeds = result[0]
                capture.auxiliary_text_embeds = None
            return result

        self._pipeline.encode_prompt = wrapped_encode_prompt

    def collect(self):
        """Return captured (prompt_embeds, auxiliary_text_embeds)."""
        return self.prompt_embeds, self.auxiliary_text_embeds

    def uninstall(self):
        """Restore the original encode_prompt."""
        self._pipeline.encode_prompt = self._original_encode_prompt

# ==================== Latent ID Capture ====================

class LatentIDCapture:
    """
    Wraps ``pipeline.prepare_latents()`` to capture ``latent_image_ids``
    generated during native pipeline sampling.

    For **LongCat**, ``prepare_latents`` returns ``(latents, latent_ids)``
    where ``latent_ids`` contains 3D position coordinates that the transformer
    uses for RoPE. These IDs depend on the exact spatial dimensions of the
    latent (which in turn depend on the image height/width), so they must come
    from the pipeline rather than being re-derived from ``seq_len`` (which
    loses the H vs W distinction for non-square images).

    For **SD3**, ``prepare_latents`` returns only ``latents`` (not a tuple),
    so ``latent_image_ids`` will be ``None``.

    Usage::

        capture = LatentIDCapture(pipeline)
        capture.install()
        result = pipeline(prompt="a cat", ...)
        latent_image_ids = capture.collect()
        capture.uninstall()
    """

    def __init__(self, pipeline):
        self._pipeline = pipeline
        self._original_prepare_latents = pipeline.prepare_latents
        self.latent_image_ids: Optional[torch.Tensor] = None

    def install(self):
        """Monkey-patch prepare_latents to intercept latent_image_ids."""
        original = self._original_prepare_latents
        capture = self

        def wrapped_prepare_latents(*args, **kwargs):
            result = original(*args, **kwargs)
            if isinstance(result, tuple) and len(result) == 2:
                # LongCat T2I: returns (latents, latent_ids)
                capture.latent_image_ids = result[1]
            else:
                capture.latent_image_ids = None
            return result

        self._pipeline.prepare_latents = wrapped_prepare_latents

    def collect(self):
        """Return captured latent_image_ids."""
        return self.latent_image_ids

    def uninstall(self):
        """Restore the original prepare_latents."""
        self._pipeline.prepare_latents = self._original_prepare_latents

# ==================== Scheduler Wrapper ====================
class TrainingSchedulerWrapper:
    """
    Wraps a diffusers scheduler to collect intermediate ``all_xt`` and
    ``all_x0`` during native pipeline sampling.

    Usage::

        wrapper = TrainingSchedulerWrapper(pipeline.scheduler, stochastic=True)
        wrapper.install(initial_latents)
        result = pipeline(...)          # native __call__
        all_xt, all_x0 = wrapper.collect()
        wrapper.uninstall()

    How it works:
      - Monkey-patches ``scheduler.step`` so that each call also computes
        ``pred_x0 = sample - sigma * model_output`` (flow-matching formula)
        and appends both ``pred_x0`` and the next-step latent to internal lists.
      - The pipeline itself is completely unmodified.
    """

    def __init__(self, scheduler, stochastic: bool = False):
        self._scheduler = scheduler
        self._original_step = scheduler.step
        self._stochastic = stochastic
        self._all_xt: List[torch.Tensor] = []
        self._all_x0: List[torch.Tensor] = []
        self._all_v: List[torch.Tensor] = []

    def install(self, initial_latents: Optional[torch.Tensor] = None):
        """Install the wrapper: patch scheduler.step and reset collections."""
        self._all_xt = [initial_latents] if initial_latents is not None else []
        self._all_x0 = [initial_latents] if initial_latents is not None else []
        self._all_v = []
        self._scheduler.config.stochastic_sampling = self._stochastic
        self._scheduler.step = self._wrapped_step

    def _wrapped_step(self, model_output, timestep, sample, **kwargs):
        """Replacement for scheduler.step that also collects intermediates."""
        # On the first call, if no initial_latents were provided to install(),
        # capture the current sample as the initial latent (pure noise).
        # This ensures all_xt and all_x0 have length num_steps + 1.
        if not self._all_xt:
            self._all_xt.append(sample.clone())
            self._all_x0.append(sample.clone())

        # Use the scheduler's own _step_index which is set inside
        # the original step() via _init_step_index(). Since we call
        # _original_step below, we need to initialize it ourselves first
        # if it hasn't been set yet.
        step_index = self._scheduler._step_index
        if step_index is None:
            self._scheduler._init_step_index(timestep)
            step_index = self._scheduler._step_index
        sigma = self._scheduler.sigmas[step_index]
        dtype = sample.dtype

        # Flow-matching x0 prediction: x0 = xt - sigma * v
        pred_x0 = (sample.float() - sigma * model_output.float()).to(dtype)
        self._all_x0.append(pred_x0)
        self._all_v.append(model_output.clone())

        # Delegate to the original scheduler step
        result = self._original_step(model_output, timestep, sample, **kwargs)
        next_latents = result[0]
        self._all_xt.append(next_latents)

        return result

    def collect(self):
        """Return the collected (all_xt, all_x0, all_v) lists.
        
        Returns:
            all_xt: List of latent tensors at each step (length = num_steps + 1).
            all_x0: List of predicted x0 tensors at each step (length = num_steps + 1).
            all_v: List of model-predicted velocity (v) at each step (length = num_steps).
        """
        return self._all_xt, self._all_x0, self._all_v

    def uninstall(self):
        """Restore the original scheduler.step."""
        self._scheduler.step = self._original_step


# ==================== Pipeline Sampling ====================

@torch.no_grad()
def pipeline_infer(
    pipeline,
    deterministic: bool = False,
    return_images: bool = True,
    **kwargs,
):
    """
    Sample from a diffusers pipeline while collecting intermediate latents
    and text embeddings.

    This is a thin wrapper around the native pipeline ``__call__``. It installs
    a :class:`TrainingSchedulerWrapper` to collect ``all_xt`` (latent at each
    step) and ``all_x0`` (predicted clean latent at each step), and an
    :class:`EmbeddingCapture` to intercept the text embeddings computed by
    ``encode_prompt``, then calls the pipeline normally.

    Works with **any** diffusers pipeline (SD3, LongCat, etc.) without
    model-specific adapter methods for sampling.

    The caller can pass either raw ``prompt`` strings (letting the pipeline
    compute embeddings internally) or pre-computed ``prompt_embeds``. When
    raw prompts are used, the captured embeddings can be reused for training,
    eliminating redundant text encoder forward passes.

    Args:
        pipeline: A diffusers pipeline instance (e.g. StableDiffusion3Pipeline,
            LongCatImagePipeline).
        deterministic: If False, use SDE (stochastic) sampling.
        return_images: If False, skip image decoding (returns None for images).
        **kwargs: All arguments forwarded to ``pipeline.__call__()``
            (prompt, prompt_embeds, num_inference_steps, guidance_scale, etc.).

    Returns:
        Tuple of (images, all_xt, all_x0, all_v, prompt_embeds, auxiliary_text_embeds,
                  latent_image_ids):
          - images: Decoded images (or None if return_images=False)
          - all_xt: List of latent tensors at each step (length = num_steps + 1)
          - all_x0: List of predicted x0 tensors at each step (length = num_steps + 1)
          - all_v: List of model-predicted velocity at each step (length = num_steps)
          - prompt_embeds: Text encoder hidden states captured from encode_prompt
          - auxiliary_text_embeds: Model-specific auxiliary embeddings
            (pooled_prompt_embeds for SD3, text_ids for LongCat)
          - latent_image_ids: Position IDs for latent patches (LongCat only,
            None for SD3). Shape (B, seq_len, 3).
    """
    # Defensive copy: avoid pipelines that may in-place modify the prompt list
    # inside _encode_prompt corrupting the caller's data for subsequent use.
    if "prompt" in kwargs and isinstance(kwargs["prompt"], (list, tuple)):
        kwargs["prompt"] = list(kwargs["prompt"])

    # Save the initial latents before the pipeline consumes them
    initial_latents = kwargs.get("latents")
    if initial_latents is not None:
        initial_latents = initial_latents.clone()

    # If caller doesn't want images, set output_type to "latent" to skip VAE decode
    if not return_images:
        kwargs["output_type"] = "latent"

    # Install scheduler wrapper to collect intermediates
    stochastic = not deterministic
    wrapper = TrainingSchedulerWrapper(pipeline.scheduler, stochastic=stochastic)
    wrapper.install(initial_latents)

    # Install embedding capture to intercept encode_prompt output
    embedding_capture = EmbeddingCapture(pipeline)
    embedding_capture.install()

    # Install latent ID capture to intercept latent_image_ids from prepare_latents
    latent_id_capture = LatentIDCapture(pipeline)
    latent_id_capture.install()

    try:
        # Call the native pipeline — all encoding, timestep scheduling,
        # denoising loop, and decoding happen inside the pipeline itself
        result = pipeline(**kwargs)

        # Extract images from the native pipeline result
        images = result.images if return_images else None

        # Collect intermediates
        all_xt, all_x0, all_v = wrapper.collect()

        # Collect captured embeddings
        prompt_embeds, auxiliary_text_embeds = embedding_capture.collect()

        # Collect captured latent image IDs
        latent_image_ids = latent_id_capture.collect()
    finally:
        wrapper.uninstall()
        embedding_capture.uninstall()
        latent_id_capture.uninstall()

    pipeline.maybe_free_model_hooks()

    return (images, all_xt, all_x0, all_v, prompt_embeds, auxiliary_text_embeds,
            latent_image_ids)
