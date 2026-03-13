"""
Cache adapters for ZImageTransformer2DModel (FB Cache and Tea Cache).

Notes
-----
* ZImage Turbo / distilled variants use very few denoising steps (1-4).
  Caching assumes that consecutive-step activations are similar, which does NOT
  hold for aggressive step-distilled models. Using cache on ZImage Turbo will
  likely degrade generation quality. Pass ``warn_turbo=False`` only if you have
  verified this is safe for your use-case.

* The ``z_image`` Tea Cache polynomial coefficients in CacheContext are currently
  set to the identity [1.0, 0.0] (no rescaling). Calibrate them by profiling
  ZImage inference without cache: record l1_distance(modulated_t, modulated_{t-1})
  at each step, then fit a polynomial mapping raw L1 → accumulated distance scale.
"""

import functools
import unittest.mock
import warnings

import torch
from torch import nn

from xfuser.model_executor.cache import utils
from xfuser.model_executor.cache.diffusers_adapters.registry import TRANSFORMER_ADAPTER_REGISTRY


def _get_underlying_module(transformer):
    """
    Return the raw ZImageTransformer2DModel regardless of whether the input is
    the raw model or the xFuserZImageTransformer2DWrapper.
    """
    # Avoid a hard import of the wrapper class; check via attribute presence.
    if hasattr(transformer, "module"):
        return transformer.module
    return transformer


def _create_single_stream_cached_blocks(use_cache, blocks, rel_l1_thresh, num_steps, name, has_modulation):
    cached_class = {
        "Fb": utils.SingleStreamFBCachedBlocks,
        "Tea": utils.SingleStreamTeaCachedBlocks,
    }.get(use_cache)

    if not cached_class:
        raise ValueError(
            f"Unsupported use_cache value: {use_cache!r}. Choose 'Fb' (First-Block) or 'Tea' (TeaCache)."
        )

    kwargs = dict(
        rel_l1_thresh=rel_l1_thresh,
        num_steps=num_steps,
        name=name,
    )
    if use_cache == "Tea":
        kwargs["has_modulation"] = has_modulation

    return cached_class(list(blocks), **kwargs)


def apply_cache_on_transformer(
    transformer,
    *,
    rel_l1_thresh: float = 0.15,
    num_steps: int = 8,
    use_cache: str = "Fb",
    warn_turbo: bool = True,
):
    """
    Apply FB Cache or Tea Cache to a ZImageTransformer2DModel.

    Patches ``noise_refiner``, ``context_refiner``, and ``layers`` in-place
    by wrapping each group in a single ``SingleStreamFBCachedBlocks`` /
    ``SingleStreamTeaCachedBlocks`` instance.

    Parameters
    ----------
    transformer:
        A ``ZImageTransformer2DModel`` or ``xFuserZImageTransformer2DWrapper``.
    rel_l1_thresh:
        Relative L1 threshold for cache reuse.
        Lower → more conservative (fewer skips, better quality).
        Higher → more aggressive (more skips, faster inference).
        Tune this on a representative set of prompts.
    num_steps:
        Total number of denoising steps (required by Tea Cache for the
        reset-on-first/last-step logic).
    use_cache:
        ``"Fb"`` for First-Block cache or ``"Tea"`` for Tea Cache.
    warn_turbo:
        Emit a warning reminding users that cache is unsafe with ZImage Turbo.
        Set to ``False`` to suppress if you have consciously decided to proceed.
    """
    if warn_turbo:
        warnings.warn(
            "Cache (FBCache / TeaCache) is designed for standard multi-step diffusion "
            "and relies on the assumption that activations change slowly between consecutive "
            "denoising steps. ZImage Turbo uses step-distillation, which violates this "
            "assumption — applying cache to ZImage Turbo will likely produce degraded "
            "or incorrect outputs. Pass warn_turbo=False to suppress this warning.",
            UserWarning,
            stacklevel=2,
        )

    module = _get_underlying_module(transformer)
    name = TRANSFORMER_ADAPTER_REGISTRY.get(type(transformer), "default")

    # noise_refiner: modulation=True, forward(x, attn_mask, freqs_cis, adaln_input)
    cached_noise_refiner = nn.ModuleList([
        _create_single_stream_cached_blocks(
            use_cache, module.noise_refiner, rel_l1_thresh, num_steps, name, has_modulation=True
        )
    ])

    # context_refiner: modulation=False, forward(x, attn_mask, freqs_cis)  — no adaln_input
    cached_context_refiner = nn.ModuleList([
        _create_single_stream_cached_blocks(
            use_cache, module.context_refiner, rel_l1_thresh, num_steps, name, has_modulation=False
        )
    ])

    # layers (unified stream): modulation=True, forward(x, attn_mask, freqs_cis, adaln_input)
    cached_layers = nn.ModuleList([
        _create_single_stream_cached_blocks(
            use_cache, module.layers, rel_l1_thresh, num_steps, name, has_modulation=True
        )
    ])

    original_forward = transformer.forward

    @functools.wraps(original_forward)
    def new_forward(self, *args, **kwargs):
        with unittest.mock.patch.object(
            module, "noise_refiner", cached_noise_refiner,
        ), unittest.mock.patch.object(
            module, "context_refiner", cached_context_refiner,
        ), unittest.mock.patch.object(
            module, "layers", cached_layers,
        ):
            return original_forward(*args, **kwargs)

    transformer.forward = new_forward.__get__(transformer)
    return transformer
