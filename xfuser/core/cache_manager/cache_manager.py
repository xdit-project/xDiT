from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torch

from xfuser.logger import init_logger

logger = init_logger(__name__)


class CacheEntry:
    def __init__(
        self,
        cache_type: "str",
        num_cache_tensors: int = 1,
        tensors: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
    ):
        self.cache_type: str = cache_type
        if tensors is None:
            self.tensors: List[torch.Tensor] = [
                None,
            ] * num_cache_tensors
        elif isinstance(tensors, torch.Tensor):
            assert (
                num_cache_tensors == 1
            ), "num_cache_tensors must be 1 if you pass a single tensor to tensors argument"
            self.tensors = [
                tensors,
            ]
        elif isinstance(tensors, List):
            assert num_cache_tensors == len(
                tensors
            ), "num_cache_tensors must be equal to num of tensors"
            self.tensors = [
                tensors,
            ]


class CacheManager:
    supported_layer = ["attn"]
    supported_cache_type = ["naive_cache", "sequence_parallel_attn_cache"]

    def __init__(
        self,
    ):
        self.cache: Dict[Tuple[str, Any], CacheEntry] = {}

    def register_cache_entry(
        self, layer, layer_type: str, cache_type: str = "naive_cache"
    ):
        if layer_type not in self.supported_layer:
            raise ValueError(
                f"Layer type: {layer_type} is not supported. Supported layer type: {self.supported_layer}"
            )
        if cache_type not in self.supported_cache_type:
            raise ValueError(
                f"Cache type: {cache_type} is not supported. Supported cache type: {self.supported_cache_type}"
            )
        if self.cache.get((layer_type, layer), None) is not None:
            logger.warning(
                f"Cache for [layer_type, layer]: [{layer_type}, {layer.__class__}] is already initialized, resetting the cache..."
            )
        self.cache[layer_type, layer] = CacheEntry(cache_type)

    def update_and_get_kv_cache(
        self,
        new_kv: Union[torch.Tensor, List[torch.Tensor]],
        layer: Any,
        slice_dim: int = 1,
        layer_type: str = "attn",
        custom_get_kv: Optional[Callable[[Any, Any, str], torch.Tensor]] = None,
        **kwargs,
    ):
        return_list = False
        if isinstance(new_kv, List):
            return_list = True
            new_kv = torch.cat(new_kv, dim=-1)

        if custom_get_kv is not None:
            return custom_get_kv(self, new_kv, layer, slice_dim, layer_type, **kwargs)
        else:
            cache_type = self.cache[layer_type, layer].cache_type
            if cache_type == "naive_cache":
                kv_cache = self._naive_cache_update(
                    new_kv,
                    layer=layer,
                    slice_dim=slice_dim,
                    layer_type=layer_type,
                    **kwargs,
                )
            elif cache_type == "sequence_parallel_attn_cache":
                kv_cache = self._sequence_parallel_cache_update(
                    new_kv,
                    layer=layer,
                    slice_dim=slice_dim,
                    layer_type=layer_type,
                    **kwargs,
                )
            if return_list:
                return torch.chunk(kv_cache, 2, dim=-1)
            else:
                return kv_cache

    def _naive_cache_update(
        self,
        new_kv: Union[torch.Tensor, List[torch.Tensor]],
        layer,
        slice_dim: int = 1,
        layer_type: str = "attn",
    ):
        from xfuser.core.distributed.runtime_state import get_runtime_state

        if (
            get_runtime_state().num_pipeline_patch == 1
            or not get_runtime_state().patch_mode
        ):
            kv_cache = new_kv
            self.cache[layer_type, layer].tensors[0] = kv_cache
        else:
            start_token_idx = get_runtime_state().pp_patches_token_start_idx_local[
                get_runtime_state().pipeline_patch_idx
            ]
            end_token_idx = get_runtime_state().pp_patches_token_start_idx_local[
                get_runtime_state().pipeline_patch_idx + 1
            ]
            kv_cache = self.cache[layer_type, layer].tensors[0]
            kv_cache = self._update_kv_in_dim(
                kv_cache=kv_cache,
                new_kv=new_kv,
                dim=slice_dim,
                start_idx=start_token_idx,
                end_idx=end_token_idx,
            )
            self.cache[layer_type, layer].tensors[0] = kv_cache
        return kv_cache

    # work inside ring attn
    def _sequence_parallel_cache_update(
        self,
        new_kv: torch.Tensor,
        layer,
        slice_dim: int = 1,
        layer_type: str = "attn",
    ):
        from xfuser.core.distributed import (
            get_ulysses_parallel_world_size,
            get_runtime_state,
        )

        ulysses_world_size = get_ulysses_parallel_world_size()
        if get_runtime_state().num_pipeline_patch == 1:
            return new_kv
        elif not get_runtime_state().patch_mode:
            pp_patches_token_num = get_runtime_state().pp_patches_token_num
            kv_list = [
                kv.split(pp_patches_token_num, dim=slice_dim)
                for kv in torch.chunk(new_kv, ulysses_world_size, dim=slice_dim)
            ]
            kv_cache = torch.cat(
                [
                    kv_list[rank][pp_patch_idx]
                    for rank in range(ulysses_world_size)
                    for pp_patch_idx in range(len(pp_patches_token_num))
                ],
                dim=slice_dim,
            )
            self.cache[layer_type, layer].tensors[0] = kv_cache
        else:
            pp_patches_token_start_idx_local = (
                get_runtime_state().pp_patches_token_start_idx_local
            )
            pp_patch_idx = get_runtime_state().pipeline_patch_idx
            start_token_idx = (
                ulysses_world_size * pp_patches_token_start_idx_local[pp_patch_idx]
            )
            end_token_idx = (
                ulysses_world_size * pp_patches_token_start_idx_local[pp_patch_idx + 1]
            )
            # pp_patches_token_num = get_runtime_state().pp_patches_token_num
            # start_token_idx = ulysses_world_size * sum(pp_patches_token_num[:get_runtime_state().pipeline_patch_idx])
            # end_token_idx = ulysses_world_size * sum(pp_patches_token_num[:get_runtime_state().pipeline_patch_idx + 1])
            kv_cache = self.cache[layer_type, layer].tensors[0]
            kv_cache = self._update_kv_in_dim(
                kv_cache=kv_cache,
                new_kv=new_kv,
                dim=slice_dim,
                start_idx=start_token_idx,
                end_idx=end_token_idx,
            )
            self.cache[layer_type, layer].tensors[0] = kv_cache
        return kv_cache

    def _update_kv_in_dim(
        self,
        kv_cache: torch.Tensor,
        new_kv: torch.Tensor,
        dim: int,
        start_idx: int,
        end_idx: int,
    ):
        if dim < 0:
            dim += kv_cache.dim()
        if dim > kv_cache.dim():
            raise ValueError(
                f"'dim' argument {dim} can not bigger or equal than kv cache dimemsions: {kv_cache.dim()}"
            )

        if dim == 0:
            kv_cache[start_idx:end_idx, ...] = new_kv
        elif dim == 1:
            kv_cache[:, start_idx:end_idx:, ...] = new_kv
        elif dim == 2:
            kv_cache[:, :, start_idx:end_idx, ...] = new_kv
        elif dim == 3:
            kv_cache[:, :, :, start_idx:end_idx, ...] = new_kv
        return kv_cache
