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
    get_runtime_state,
)
from xfuser.core.distributed.runtime_state import AttentionBackendType

from packaging.version import parse
from xfuser.envs import PACKAGES_CHECKER
from xfuser.core.cache_manager.cache_manager import get_cache_manager

env_info = PACKAGES_CHECKER.get_packages_info()
if env_info["has_aiter"]:
    from aiter import flash_attn_func as flash_attn_func_aiter
    import inspect
    try:
        AITER_HAS_ROUND_MODE = inspect.signature(flash_attn_func_aiter).parameters.get("how_v3_bf16_cvt") is not None
    except (AttributeError, TypeError):
        AITER_HAS_ROUND_MODE = False
    if AITER_HAS_ROUND_MODE:
        import os
        HOW_V3_BF16_CVT = int(os.environ.get("HOW_V3_BF16_CVT", "2"))

if env_info["has_flash_attn"]:
    from flash_attn import flash_attn_func as flash_attn_func_2
if env_info["has_flash_attn_3"]:
    from flash_attn_interface import flash_attn_func as flash_attn_func_3
if env_info["has_flash_attn_4"]:
    from flash_attn.cute.interface import flash_attn_fwd as flash_attn_func_4

aten = torch.ops.aten

# Attention function registry
_ATTENTION_FUNCTION_REGISTRY = {}

def register_attention_function(backend_type):
    """
    Decorator to register attention functions with their corresponding backend type.
    """
    def decorator(func):
        _ATTENTION_FUNCTION_REGISTRY[backend_type] = func
        return func
    return decorator


def ring_attn(attention_function, query, key, value, dropout_p=0.0, is_causal=False):
    kwargs = {
        "dropout_p": dropout_p,
        "is_causal": is_causal,
    }
    if parse(torch.__version__).release >= parse("2.6.0").release:
        from torch.distributed.tensor.experimental._attention import _cp_options
        _cp_options.enable_load_balance = False
        out, *_ = _templated_ring_attention(
            PROCESS_GROUP.RING_PG,
            1,
            attention_function,
            query,
            key,
            value,
            **kwargs,
        )
    else:
        out, *_ = _templated_ring_attention(
            PROCESS_GROUP.RING_PG,
            1,
            attention_function,
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


def _combined_qkv_all_to_all(q, k, v):
    """Concatenate query, key, value tensors and perform a single all-to-all communication."""
    world_size = get_ulysses_parallel_world_size()
    if world_size <= 1:
        return q, k, v

    assert q.ndim == 4, f"q must have 4 dimensions, got {q.ndim}"
    b, h, s, d = q.shape
    assert h % world_size == 0, f"h must be divisible by world_size, got {h} and {world_size}"

    # [3, b, h, s, d]
    qkv = torch.stack([q, k, v], dim=0)
    # [3, b, P, h/P, s, d]
    qkv = qkv.view(3, b, world_size, h // world_size, s, d)
    # [P, 3, b, h/P, s, d]
    qkv = qkv.permute(2, 0, 1, 3, 4, 5).contiguous()

    qkv = _sdpa_all_to_all_single(qkv)

    # [3, b, h/P, P, s, d]
    qkv = qkv.permute(1, 2, 3, 0, 4, 5).contiguous()
    # [3, b, h/P, P*s, d]
    qkv = qkv.view(3, b, h // world_size, -1, d)

    q, k, v = torch.unbind(qkv, dim=0)
    return q, k, v


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

@register_attention_function(AttentionBackendType.SDPA)
def _sdpa_attn_call(query, key, value, dropout_p, is_causal):
    """
    Performs attention through PyTorch's scaled_dot_product_attention.
    Allows Pytorch to decide which SDPA backend to use.
    """
    output = F.scaled_dot_product_attention(
        query, key, value, dropout_p=dropout_p, is_causal=is_causal
    )
    return output, None

@register_attention_function(AttentionBackendType.CUDNN)
def _cudnn_attn_call(query, key, value, dropout_p, is_causal):
    """
    Performs the necessary tensor permutes and
    then calls attention through cuDNN backend
    """
    query = torch.permute(query, [0, 2, 1, 3]).contiguous()
    key = torch.permute(key, [0, 2, 1, 3]).contiguous()
    value = torch.permute(value, [0, 2, 1, 3]).contiguous()
    output, softmax_lse = aten._scaled_dot_product_cudnn_attention(
        query,
        key,
        value,
        attn_bias=None,
        compute_logsumexp=True,
        dropout_p=dropout_p,
        is_causal=is_causal,
    )
    output = torch.permute(output, [0, 2, 1, 3])
    return output, softmax_lse

@register_attention_function(AttentionBackendType.FLASH_3)
def _flash_attn_3_call(query, key, value, dropout_p, is_causal):
    """
    Performs the necessary tensor permutes and
    then calls attention through flash_attn V3
    """
    query = torch.permute(query, [0, 2, 1, 3]).contiguous()
    key = torch.permute(key, [0, 2, 1, 3]).contiguous()
    value = torch.permute(value, [0, 2, 1, 3]).contiguous()
    output, softmax_lse = flash_attn_func_3.flash_attn_func(
        query,
        key,
        value,
        dropout_p=dropout_p,
        causal=is_causal,
        return_attn_probs=False,
        return_lse=True,
    )
    output = torch.permute(output, [0, 2, 1, 3])
    return output, softmax_lse

@register_attention_function(AttentionBackendType.FLASH_4)
def _flash_attn_4_call(query, key, value, dropout_p, is_causal):
    """
    Performs the necessary tensor permutes and
    then calls attention through flash_attn V4
    """

    ## TODO: check the dimensions
    # query = torch.permute(query, [0, 2, 1, 3]).contiguous()
    # key = torch.permute(key, [0, 2, 1, 3]).contiguous()
    # value = torch.permute(value, [0, 2, 1, 3]).contiguous()
    output, softmax_lse = flash_attn_func_4(
        query,
        key,
        value,
    )
    # output = torch.permute(output, [0, 2, 1, 3])
    ## TODO: Check if it really doesn't output lse
    softmax_lse = False
    return output, softmax_lse

@register_attention_function(AttentionBackendType.AITER)
def _aiter_attn_call(query, key, value, dropout_p, is_causal):
    """
    Performs the necessary tensor permutes and
    then calls attention through AITER
    """
    query = torch.permute(query, [0, 2, 1, 3]).contiguous()
    key = torch.permute(key, [0, 2, 1, 3]).contiguous()
    value = torch.permute(value, [0, 2, 1, 3]).contiguous()
    attn_kwargs = {
        "dropout_p": dropout_p,
        "causal": is_causal,
        "return_attn_probs": False,
        "return_lse": True,
    }
    if AITER_HAS_ROUND_MODE:
        attn_kwargs["how_v3_bf16_cvt"] = HOW_V3_BF16_CVT
    output, softmax_lse = flash_attn_func_aiter(
        query,
        key,
        value,
        **attn_kwargs
    )
    output = torch.permute(output, [0, 2, 1, 3])
    return output, softmax_lse

@register_attention_function(AttentionBackendType.FLASH)
def _flash_attn_call(query, key, value, dropout_p, is_causal):
    """
    Performs the necessary tensor permutes and
    then calls attention through flash_attn
    """
    query = torch.permute(query, [0, 2, 1, 3])
    key = torch.permute(key, [0, 2, 1, 3])
    value = torch.permute(value, [0, 2, 1, 3])
    output, softmax_lse, S_mask = flash_attn_func_2(
        query,
        key,
        value,
        dropout_p=dropout_p,
        causal=is_causal,
        return_attn_probs=True,
    )
    output = torch.permute(output, [0, 2, 1, 3])
    return output, softmax_lse


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

def _get_attention_function():
    """
    Get the attention function based on the runtime state.
    """
    attention_backend = get_runtime_state().attention_backend
    func = _ATTENTION_FUNCTION_REGISTRY.get(attention_backend, None)
    if func is None:
        raise NotImplementedError(f"Attention backend {attention_backend} not registered.")
    return func

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
        combine_qkv_a2a: bool | None = None,
    ):
    """
    Unified Sequence Parallelism (USP) attention call, supporting combinations of Ulysses and
    Ring attention. Also supports joint tensors and key-value caching for pipeline parallelism.
    """
    if combine_qkv_a2a is None:
        combine_qkv_a2a = False

    attention_function = _get_attention_function()

    if joint_strategy:
        query = _concat_joint_tensor(query, joint_query, joint_strategy, dim=2)
        joint_key, joint_value = _preprocess_joint_tensors(joint_key, joint_value)

    if get_ulysses_parallel_world_size() > 1:
        if combine_qkv_a2a and query.shape == key.shape == value.shape:
            query, key, value = _combined_qkv_all_to_all(query, key, value)
        else:
            query = _ft_c_input_all_to_all(query)
            key = _ft_c_input_all_to_all(key)
            value = _ft_c_input_all_to_all(value)

    if attn_layer:
        key, value = _update_and_get_kv_cache(key, value, attn_layer)
    if joint_strategy:
        key = _concat_joint_tensor(key, joint_key, joint_strategy, dim=2)
        value = _concat_joint_tensor(value, joint_value, joint_strategy, dim=2)

    if get_sequence_parallel_world_size() == 1: # No SP
        out, _ = attention_function(query, key, value, dropout_p=dropout_p, is_causal=is_causal)

    elif get_ulysses_parallel_world_size() == 1: # Ring only
        out = ring_attn(attention_function, query, key, value, dropout_p=dropout_p, is_causal=is_causal)

    else:
        if get_ring_parallel_world_size() == 1: # Ulysses only
            out, _ = attention_function(query, key, value, dropout_p=dropout_p, is_causal=is_causal)
        else: # USP
            out = ring_attn(attention_function, query, key, value, dropout_p=dropout_p, is_causal=is_causal)
        out = _ft_c_output_all_to_all(out)

    return out


