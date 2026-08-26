"""Unified entry point for xDiT step-caching adapters."""
import json
import logging
from typing import Any, Dict, Optional

import torch

logger = logging.getLogger(__name__)


def _unwrap_fsdp(transformer):
    """Return the module wrapped by FSDP1, or the input module otherwise."""
    inner = getattr(transformer, "_fsdp_wrapped_module", None)
    if inner is not None:
        return inner
    if type(transformer).__name__ == "FullyShardedDataParallel":
        return getattr(transformer, "module", transformer)
    return transformer


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
            if not isinstance(override, dict):
                raise TypeError("cache_config must be a JSON object")
        except (json.JSONDecodeError, TypeError) as e:
            raise ValueError(f"--cache_config is not valid JSON: {e}") from e
        unknown = set(override) - {"residual_diff_threshold"}
        if unknown:
            raise ValueError(
                f"Unknown --cache_config keys for in-tree caching: {sorted(unknown)}"
            )
        threshold = override.get("residual_diff_threshold", threshold)
    return threshold


def apply_cache(
    cache_method: str,
    num_steps: int,
    pipe: Any,
    transformer: Optional[Any] = None,
    preset_kwargs: Optional[Any] = None,
    cache_config: Optional[str] = None,
    # CacheDitAdapterConfig or List[CacheDitAdapterConfig], dbcache only
    adapter_config: Optional[Any] = None,
    # Transformer attribute name on pipe, used for teacache/fbcache
    transformer_attr: str = "transformer",
) -> Optional[Any]:
    """Apply a step-caching method and return the patched transformer.

    For multi-transformer dbcache (list adapter_config), both transformers
    are patched via a single coordinated enable_cache() call and no single
    transformer is returned. When transformer is omitted, the patched module
    is also assigned back to pipe.
    """
    if cache_method == "teacache":
        from xfuser.model_executor.cache.adapters.flux import apply_teacache
        target = transformer if transformer is not None else getattr(pipe, transformer_attr)
        patch_target = _unwrap_fsdp(target)
        apply_teacache(
            patch_target,
            rel_l1_thresh=_resolve_threshold(preset_kwargs, cache_config),
            num_steps=num_steps,
        )
        if transformer is None:
            setattr(pipe, transformer_attr, target)
        return target

    if cache_method == "fbcache":
        from xfuser.envs import XDIT_FBCACHE_THRESH

        target = transformer if transformer is not None else getattr(pipe, transformer_attr)
        patch_target = _unwrap_fsdp(target)
        default_threshold = (
            float(XDIT_FBCACHE_THRESH)
            if XDIT_FBCACHE_THRESH
            else 0.12
        )
        if hasattr(patch_target, "single_stream_modulation"):
            from xfuser.model_executor.cache.adapters.flux2 import apply_fbcache
        else:
            from xfuser.model_executor.cache.adapters.flux import apply_fbcache
        apply_fbcache(
            patch_target,
            use_cache="Fb",
            rel_l1_thresh=_resolve_threshold(
                preset_kwargs,
                cache_config,
                default=default_threshold,
            ),
            return_hidden_states_first=False,
            num_steps=num_steps,
        )
        if transformer is None:
            setattr(pipe, transformer_attr, target)
        return target

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
        target = transformer if transformer is not None else getattr(pipe, attr, None)
        if target is None:
            raise RuntimeError(
                f"apply_cache (dbcache): pipe {type(pipe).__name__!r} has no attribute {attr!r}. "
                "Set adapter_config.transformer_attr to the correct pipe attribute."
            )
        patched = apply_cache_dit_cache(
            target,
            num_steps=num_steps,
            pipe=pipe,
            preset_kwargs=preset_kwargs,
            cache_config=cache_config,
            adapter_config=adapter_config,
        )
        if transformer is None:
            setattr(pipe, attr, patched)
        return patched

    raise ValueError(
        f"Unknown cache_method: {cache_method!r}. "
        "Supported: 'teacache', 'fbcache', 'dbcache'."
    )
