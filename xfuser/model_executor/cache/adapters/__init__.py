"""
xDiT step-caching unified entry point.

Implementation files:
  flux.py:      Flux1 xDiT in-tree TeaCache (USP-native via all_reduce)
  flux2.py:     Flux2 xDiT in-tree FBCache (USP-native via all_reduce)
  cache_dit.py: cache-dit hooks: dbcache (optional dep: pip install cache-dit)

Routing:
  teacache  → flux.apply_teacache
  fbcache   → flux2.apply_fbcache
  dbcache   → cache_dit.apply_cache_dit_cache / apply_cache_dit_cache_multi
"""
import json
import logging
from typing import Any, Dict, Optional

import torch

logger = logging.getLogger(__name__)


def _resolve_threshold(
    preset_kwargs: Optional[Dict],
    cache_config: Optional[str],
    default: float = 0.12,
) -> float:
    """Extract residual_diff_threshold from preset_kwargs, overridden by cache_config JSON.

    Bad JSON raises (same as dbcache's _build_config) so a malformed --cache_config
    fails loudly instead of silently running with the default threshold.
    """
    threshold = (preset_kwargs or {}).get("residual_diff_threshold", default)
    if cache_config:
        try:
            override = json.loads(cache_config)
        except json.JSONDecodeError as e:
            raise ValueError(f"--cache_config is not valid JSON: {e}") from e
        threshold = override.get("residual_diff_threshold", threshold)
    return threshold


def apply_cache(
    cache_method: str,
    num_steps: int,
    pipe: Any,
    preset_kwargs: Optional[Any] = None,
    cache_config: Optional[str] = None,
    # CacheDitAdapterConfig or List[CacheDitAdapterConfig], dbcache only
    adapter_config: Optional[Any] = None,
    # Transformer attribute name on pipe, used for teacache/fbcache
    transformer_attr: str = "transformer",
) -> None:
    """Apply a step-caching method in place on pipe.

    For multi-transformer dbcache (list adapter_config), both transformers
    are patched via a single coordinated enable_cache() call.
    """
    if cache_method == "teacache":
        from xfuser.model_executor.cache.adapters.flux import apply_teacache
        transformer = getattr(pipe, transformer_attr)
        patched = apply_teacache(
            transformer,
            rel_l1_thresh=_resolve_threshold(preset_kwargs, cache_config),
            num_steps=num_steps,
        )
        setattr(pipe, transformer_attr, patched)
        return

    if cache_method == "fbcache":
        from xfuser.model_executor.cache.adapters.flux2 import apply_fbcache
        transformer = getattr(pipe, transformer_attr)
        patched = apply_fbcache(
            transformer,
            use_cache="Fb",
            rel_l1_thresh=_resolve_threshold(preset_kwargs, cache_config),
            return_hidden_states_first=False,
            num_steps=num_steps,
        )
        setattr(pipe, transformer_attr, patched)
        return

    if cache_method == "dbcache":
        from xfuser.model_executor.cache.adapters.cache_dit import (
            apply_cache_dit_cache,
            apply_cache_dit_cache_multi,
        )
        if isinstance(adapter_config, list):
            apply_cache_dit_cache_multi(
                pipe=pipe,
                num_steps=num_steps,
                adapter_configs=adapter_config,
                presets=preset_kwargs,
                cache_config=cache_config,
            )
            return
        attr = adapter_config.transformer_attr if adapter_config else transformer_attr
        transformer = getattr(pipe, attr, None)
        if transformer is None:
            raise RuntimeError(
                f"apply_cache (dbcache): pipe {type(pipe).__name__!r} has no attribute {attr!r}. "
                "Set adapter_config.transformer_attr to the correct pipe attribute."
            )
        patched = apply_cache_dit_cache(
            transformer,
            num_steps=num_steps,
            pipe=pipe,
            preset_kwargs=preset_kwargs,
            cache_config=cache_config,
            adapter_config=adapter_config,
        )
        setattr(pipe, attr, patched)
        return

    raise ValueError(
        f"Unknown cache_method: {cache_method!r}. "
        "Supported: 'teacache', 'fbcache', 'dbcache'."
    )
