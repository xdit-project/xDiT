"""Import-compatible Cache-DiT adapter package."""

from xfuser.model_executor.cache.adapters.cache_dit.apply import (
    apply_cache_dit_cache,
    apply_cache_dit_cache_multi,
)

__all__ = ["apply_cache_dit_cache", "apply_cache_dit_cache_multi"]
