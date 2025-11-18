# This file implements USP with torch version >= '2.5.0'
import torch
from torch.nn import functional as F

import torch.distributed._functional_collectives as ft_c

from torch.distributed.tensor.experimental._attention import _templated_ring_attention

if torch.cuda.is_available():
    from yunchang.globals import PROCESS_GROUP
else:
    PROCESS_GROUP = None

from xfuser.core.distributed import (
    get_sequence_parallel_world_size,
    get_ulysses_parallel_world_size,
    get_ring_parallel_world_size,
    get_sequence_parallel_rank,
    get_ulysses_parallel_rank,
)

from packaging.version import parse
from xfuser.envs import PACKAGES_CHECKER
from xfuser.core.cache_manager.cache_manager import get_cache_manager
env_info = PACKAGES_CHECKER.get_packages_info()
HAS_FLASH_ATTN = env_info["has_flash_attn"]
if HAS_FLASH_ATTN:
    import flash_attn

HAS_AITER = env_info["has_aiter"]
if HAS_AITER:
    import aiter
    import inspect
    HAS_ROUND_MODE = inspect.signature(aiter.ops.mha.flash_attn_func).parameters.get("how_v3_bf16_cvt") is not None
    if HAS_ROUND_MODE:
        import os
        HOW_V3_BF16_CVT = int(os.environ.get("HOW_V3_BF16_CVT", "2"))

aten = torch.ops.aten


def ring_attn(query, key, value, dropout_p=0.0, is_causal=False):
    if parse(torch.__version__).release >= parse("2.6.0").release:
        from torch.distributed.tensor.experimental._attention import _cp_options
        _cp_options.enable_load_balance = False
        kwargs = {
            "dropout_p": dropout_p,
            "is_causal": is_causal,
        }
        if HAS_AITER:
            out, *_ = _templated_ring_attention(
                PROCESS_GROUP.RING_PG,
                1,
                _aiter_attn_call,
                query,
                key,
                value,
                **kwargs,
            )
        elif HAS_FLASH_ATTN:
            out, *_ = _templated_ring_attention(
                PROCESS_GROUP.RING_PG,
                1,
                _flash_attn_call,
                query,
                key,
                value,
                **kwargs,
            )
        else:
            kwargs = {
                **kwargs,
                "attn_bias": None,
                "compute_log_sumexp": True,
            }
            out, *_ = _templated_ring_attention(
                PROCESS_GROUP.RING_PG,
                1,
                aten._scaled_dot_product_efficient_attention,
                query,
                key,
                value,
                **kwargs,
            )
    else:
        kwargs = {
            "dropout_p": dropout_p,
            "is_causal": is_causal,
        }
        if HAS_AITER:
            out, *_ = _templated_ring_attention(
                PROCESS_GROUP.RING_PG,
                1,
                _aiter_attn_call,
                query,
                key,
                value,
                **kwargs,
            )
        elif HAS_FLASH_ATTN:
            out, *_ = _templated_ring_attention(
                PROCESS_GROUP.RING_PG,
                _flash_attn_call,
                query,
                key,
                value,
                **kwargs
            )
        else:
            kwargs = {
                **kwargs,
                "attn_bias": None,
                "compute_log_sumexp": True,
            }
            out, *_ = _templated_ring_attention(
                PROCESS_GROUP.RING_PG,
                aten._scaled_dot_product_efficient_attention,
                query,
                key,
                value,
                **kwargs,
            )
    return out


def _maybe_wait(tensor: torch.Tensor) -> torch.Tensor:
    """
    When tracing the code, the result tensor is not an AsyncCollectiveTensor,
    so we cannot call ``wait()``.
    """
    if isinstance(tensor, ft_c.AsyncCollectiveTensor):
        return tensor.wait()
    return tensor


def _sdpa_all_to_all_single(x):
    x_shape = x.shape
    x = x.flatten()
    x = ft_c.all_to_all_single(x, output_split_sizes=None, input_split_sizes=None, group=PROCESS_GROUP.ULYSSES_PG)
    x = _maybe_wait(x)
    x = x.reshape(x_shape)
    return x


def _ft_c_input_all_to_all(x):
    world_size = get_ulysses_parallel_world_size()
    if world_size <= 1:
        return x

    assert x.ndim == 4, "x must have 4 dimensions, got {}".format(x.ndim)
    b, h, s, d = x.shape
    assert h % world_size == 0, "h must be divisible by world_size, got {} and {}".format(h, world_size)

    x = x.permute(1, 0, 2, 3).contiguous()
    x = _sdpa_all_to_all_single(x)
    x = x.reshape(world_size, h // world_size, b, -1, d).permute(2, 1, 0, 3, 4).reshape(b, h // world_size, -1, d)
    return x


def _ft_c_output_all_to_all(x):
    world_size = get_ulysses_parallel_world_size()
    if world_size <= 1:
        return x

    assert x.ndim == 4, "x must have 4 dimensions, got {}".format(x.ndim)
    b, h, s, d = x.shape
    assert s % world_size == 0, "s must be divisible by world_size, got {} and {}".format(s, world_size)

    x = x.permute(2, 0, 1, 3).contiguous()
    x = _sdpa_all_to_all_single(x)
    x = x.reshape(world_size, s // world_size, b, -1, d).permute(2, 0, 3, 1, 4).reshape(b, -1, s // world_size, d)
    return x


def _aiter_attn_call(query, key, value, dropout_p, is_causal):
    """
    Performs the necessary tensor permutes and
    then calls attention through AITER
    """
    query = torch.permute(query, [0, 2, 1, 3]).contiguous()
    key = torch.permute(key, [0, 2, 1, 3]).contiguous()
    value = torch.permute(value, [0, 2, 1, 3]).contiguous()
    if HAS_ROUND_MODE:
        output, softmax_lse = aiter.ops.mha.flash_attn_func(
            query,
            key,
            value,
            dropout_p=dropout_p,
            causal=is_causal,
            return_attn_probs=False,
            return_lse=True,
            how_v3_bf16_cvt=HOW_V3_BF16_CVT
        )
    else:
        output, softmax_lse = aiter.flash_attn_func(
            query,
            key,
            value,
            dropout_p=dropout_p,
            causal=is_causal,
            return_attn_probs=False,
            return_lse=True
        )
    output = torch.permute(output, [0, 2, 1, 3])
    return output, softmax_lse

def _flash_attn_call(query, key, value, dropout_p, is_causal):
    """
    Performs the necessary tensor permutes and
    then calls attention through flash_attn
    """
    query = torch.permute(query, [0, 2, 1, 3])
    key = torch.permute(key, [0, 2, 1, 3])
    value = torch.permute(value, [0, 2, 1, 3])
    output, softmax_lse, S_mask = flash_attn.flash_attn_func(
        query,
        key,
        value,
        dropout_p=dropout_p,
        causal=is_causal,
        return_attn_probs=True,
    )
    output = torch.permute(output, [0, 2, 1, 3])
    return output, softmax_lse

def _attention(query, key, value, dropout_p, is_causal):
    """
    Calls the correct attention mechanism based on the available libraries
    """
    if HAS_AITER:
        output, _ = _aiter_attn_call(query, key, value, dropout_p, is_causal)
        return output
    elif HAS_FLASH_ATTN:
        output, _ = _flash_attn_call(query, key, value, dropout_p, is_causal)
        return output
    else:
        return F.scaled_dot_product_attention(
            query, key, value, dropout_p=dropout_p, is_causal=is_causal
        )


def _preprocess_joint_tensors(joint_key, joint_value):
    """
    Preprocess the joint key and value tensors for Ulysses parallelism.
    """
    ulysses_world_size = get_ulysses_parallel_world_size()
    ulysses_rank = get_ulysses_parallel_rank()
    attn_heads_per_ulysses_rank = (
        joint_key.shape[1] // ulysses_world_size
    )
    joint_key = joint_key.transpose(1,2)
    joint_value = joint_value.transpose(1,2)
    joint_key = joint_key[
        ...,
        attn_heads_per_ulysses_rank
        * ulysses_rank : attn_heads_per_ulysses_rank
        * (ulysses_rank + 1),
        :, ].transpose(1,2)
    joint_value = joint_value[
        ...,
        attn_heads_per_ulysses_rank
        * ulysses_rank : attn_heads_per_ulysses_rank
        * (ulysses_rank + 1),
        :,
    ].transpose(1,2)
    return joint_key, joint_value

def _concat_joint_tensor(tensor, joint_tensor, joint_strategy, dim):
    """
    Concatenate the joint tensor to the main tensor based on the joint strategy.
    """
    if joint_strategy == "rear":
        tensor = torch.cat([tensor, joint_tensor], dim=dim)
    elif joint_strategy == "front":
        tensor = torch.cat([joint_tensor, tensor], dim=dim)
    else:
        raise ValueError(f"Invalid joint_strategy: {joint_strategy}")
    return tensor

def _update_and_get_kv_cache(key, value, attn_layer):
    """
    Update and get the key and value cache for pipeline parallelism.
    """
    key, value = get_cache_manager().update_and_get_kv_cache(
        new_kv=[key.transpose(1, 2), value.transpose(1, 2)],
        layer=attn_layer,
        slice_dim=1,
        layer_type="attn",
    )
    key = key.transpose(1, 2).contiguous()
    value = value.transpose(1, 2).contiguous()
    return key, value

def USP(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        dropout_p: float = 0.0,
        is_causal: bool = False,
        joint_query: torch.Tensor | None = None,
        joint_key: torch.Tensor | None = None,
        joint_value: torch.Tensor | None = None,
        joint_strategy: str | None = None,
        attn_layer=None,
    ):
    """
    Unified Sequence Parallelism (USP) attention call, supporting combinations of Ulysses and
    Ring attention. Also supports joint tensors and key-value caching for pipeline parallelism.
    """

    if joint_strategy:
        query = _concat_joint_tensor(query, joint_query, joint_strategy, dim=2)
        joint_key, joint_value = _preprocess_joint_tensors(joint_key, joint_value)

    if get_ulysses_parallel_world_size() > 1:
        query = _ft_c_input_all_to_all(query)
        key = _ft_c_input_all_to_all(key)
        value = _ft_c_input_all_to_all(value)

    if attn_layer:
        key, value = _update_and_get_kv_cache(key, value, attn_layer)
    if joint_strategy:
        key = _concat_joint_tensor(key, joint_key, joint_strategy, dim=2)
        value = _concat_joint_tensor(value, joint_value, joint_strategy, dim=2)

    if get_sequence_parallel_world_size() == 1: # No SP
        out = _attention(query, key, value, dropout_p=dropout_p, is_causal=is_causal)

    elif get_ulysses_parallel_world_size() == 1: # Ring only
        out = ring_attn(query, key, value, dropout_p=dropout_p, is_causal=is_causal)

    else:
        if get_ring_parallel_world_size() == 1: # Ulysses only
            out = _attention(query, key, value, dropout_p=dropout_p, is_causal=is_causal)
        else: # USP
            out = ring_attn(query, key, value, dropout_p=dropout_p, is_causal=is_causal)
        out = _ft_c_output_all_to_all(out)

    return out


