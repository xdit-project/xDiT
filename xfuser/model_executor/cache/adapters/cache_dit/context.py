"""Distributed cache-decision synchronization for cache-dit."""

import torch
import torch.distributed as dist

_SP_SYNC_PATCHED = False


def _install_sp_can_cache_sync() -> None:
    """Synchronize cache skip decisions across distributed ranks once."""
    global _SP_SYNC_PATCHED
    if not dist.is_available() or not dist.is_initialized() or dist.get_world_size() <= 1:
        return
    if _SP_SYNC_PATCHED:
        return
    try:
        from cache_dit.caching.cache_contexts.cache_manager import CachedContextManager
    except ImportError as e:
        raise ImportError(
            "Distributed dbcache requires cache-dit 1.5.x: "
            "the CachedContextManager synchronization hook is unavailable."
        ) from e

    from xfuser.core.distributed import get_world_group

    original_can_cache = CachedContextManager.can_cache

    @torch.compiler.disable
    def can_cache(self, *args, **kwargs):
        result = original_can_cache(self, *args, **kwargs)
        if not dist.is_available() or not dist.is_initialized():
            return result
        world_group = get_world_group()
        result_tensor = torch.tensor(
            [1 if result else 0],
            device=torch.cuda.current_device(),
            dtype=torch.int32,
        )
        world_group.broadcast(result_tensor, src=0)
        agreed = bool(result_tensor.item())
        if agreed and not getattr(self, "_xdit_cache_warmed", False):
            dist.barrier(group=world_group.device_group)
            self._xdit_cache_warmed = True
        return agreed

    CachedContextManager.can_cache = can_cache
    _SP_SYNC_PATCHED = True


def _is_parallelized_flag() -> bool:
    from xfuser.core.distributed import (
        get_pipeline_parallel_world_size,
        get_sequence_parallel_world_size,
    )

    return get_sequence_parallel_world_size() > 1 or get_pipeline_parallel_world_size() > 1
