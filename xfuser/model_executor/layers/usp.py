import torch
from torch.nn import functional as F
from torch.distributed.tensor.experimental._attention import _templated_ring_attention
aten = torch.ops.aten

import torch.distributed._functional_collectives as ft_c
import torch.distributed as dist

from yunchang.globals import PROCESS_GROUP
from yunchang.comm.all_to_all import SeqAllToAll4D
from xfuser.core.distributed import (
    get_sequence_parallel_world_size,
    get_ulysses_parallel_world_size,
    get_ring_parallel_world_size,
)

def ring_attn(query, key, value, dropout_p=0.0, is_causal=False):
    out, *_ = _templated_ring_attention(
        PROCESS_GROUP.RING_PG,
        aten._scaled_dot_product_flash_attention,
        query,
        key,
        value,
        dropout_p=dropout_p,
        is_causal=is_causal
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


def USP(query, key, value, dropout_p=0.0, is_causal=False):
    if get_sequence_parallel_world_size() == 1:
        out = F.scaled_dot_product_attention(
            query, key, value, dropout_p=dropout_p, is_causal=is_causal
        )
    elif get_ulysses_parallel_world_size() == 1:
        out = ring_attn(query, key, value, dropout_p=dropout_p, is_causal=is_causal)
    elif get_ulysses_parallel_world_size() > 1:
        query = _ft_c_input_all_to_all(query)
        key = _ft_c_input_all_to_all(key)
        value = _ft_c_input_all_to_all(value)

        if get_ring_parallel_world_size() == 1:
            out = F.scaled_dot_product_attention(
                query, key, value, dropout_p=dropout_p, is_causal=is_causal
            )
        else:
            out = ring_attn(query, key, value, dropout_p=dropout_p, is_causal=is_causal)

        out = _ft_c_output_all_to_all(out)
        
    return out