#   Copyright 2025 Jintao Zhang, Chendong Xiang, Haofeng Huang
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#
#   This file has been modified from the upstream Apache-2.0 source at
#   https://github.com/thu-ml/SpargeAttn (spas_sage_attn/utils.py).

import torch
import triton
import triton.language as tl


# Module-level cache of broadcast threshold tensors keyed on a hashable
# Python tuple (so torch.compile / dynamo can constant-fold the lookup
# and bake the resulting tensor into the graph).
#
# Key layout: (float_value, num_heads, device_type, device_index)
#   - float_value is the scalar broadcast across all heads
#   - num_heads is the target shape[0]
#   - device_type / device_index distinguish per-GPU copies in multi-GPU runs
_HYPER_TENSOR_CACHE: dict[tuple, torch.Tensor] = {}


def _device_key(device: torch.device) -> tuple:
    # `torch.device` instances are not always hashable across versions; use
    # explicit (type, index) so the key is stable and dynamo-friendly.
    return (device.type, device.index if device.index is not None else -1)

def hyperparameter_check(
    hyper: float | torch.Tensor, H: int, device: torch.device
) -> torch.Tensor:
    if isinstance(hyper, (float, int)):
        key = (float(hyper), H, *_device_key(device))
        cached = _HYPER_TENSOR_CACHE.get(key)
        if cached is None:
            cached = torch.full((H,), float(hyper), device=device)
            _HYPER_TENSOR_CACHE[key] = cached
        return cached
    if isinstance(hyper, torch.Tensor):
        if hyper.dim() == 0:
            key = (float(hyper.item()), H, *_device_key(device))
            cached = _HYPER_TENSOR_CACHE.get(key)
            if cached is None:
                cached = torch.full((H,), hyper.item(), device=device)
                _HYPER_TENSOR_CACHE[key] = cached
            return cached
        assert hyper.dim() == 1 and hyper.numel() == H, (
            f"Hyperparameter tensor must have {H} elements, got shape "
            f"{tuple(hyper.shape)}"
        )
        return hyper.to(device)
    raise ValueError(
        f"Hyperparameter must be a float, int, or 0-D/1-D tensor; got {type(hyper)}"
    )


@triton.jit
def triton_bmm_pool_sim_simmean(
    x_ptr,
    pool_ptr,
    sim_ptr,
    simthreshd1_ptr,
    N: tl.constexpr,
    D: tl.constexpr,
    BS: tl.constexpr
):
    b, h, nb = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    B, H, NB = tl.num_programs(0), tl.num_programs(1), tl.num_programs(2)

    block_offset = b * H * N * D + h * N * D + nb * BS * D
    xmask = (nb*BS + tl.arange(0, BS)[:, None]) < N
    x_ptrs = x_ptr + block_offset + tl.arange(0, BS)[:, None] * D + tl.arange(0, D)[None, :]
    # Load the input block, xmask will return nan for out-of-bound elements
    x = tl.load(x_ptrs, mask = xmask)
    BS_ = BS if (N - nb*BS) >= BS else (N - nb*BS)

    cur_h1 = tl.load(simthreshd1_ptr + h)
    x_fp32 = x.to(tl.float32)
    # Check for NaN values
    is_nan = x_fp32 != x_fp32
    x_fp32 = tl.where(is_nan, 0.0, x_fp32)

    pool = (tl.sum(x_fp32, axis=0) / BS_)
    x_norm = tl.sqrt(tl.sum(x_fp32 * x_fp32, axis=1, keep_dims=True))
    x = (x / x_norm).to(tl.float16)  # norm at D dim
    # Check for NaN values after normalization
    is_nan = x != x
    x = tl.where(is_nan, 0.0, x)

    grams = tl.dot(x, tl.trans(x))
    sum_value = tl.sum(grams).to(tl.float32)
    cur_sim = (sum_value / (BS_ * BS_)) > cur_h1

    pool_block_offset = b * H * NB * D + h * NB * D + nb * D
    tl.store(pool_ptr + pool_block_offset + tl.arange(0, D), pool)
    sim_offset = b * H * NB + h * NB + nb
    tl.store(sim_ptr + sim_offset, cur_sim)


def get_pool_sim_triton_simmean(
    x, block_size, simthreshd1
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
    x: (B, H, N, D)
    block_size: int
    simthreshd1: (H,) tensor

    Steps:
    1. Pooling within each block
    2. Compute similarity within each block
    3. Return pooled tensor and similarity mask

    Note how 3rd dimension N is reduced to nblock = N // block_size.
    This way later in the algorithm we don't compute the full attention ( O(N^2) ), but only O(nblock^2).

    Returns:
    pool: (B, H, nblock, D) tensor
    sim_blocks: (B, H, nblock) bool tensor
    """
    x = x.contiguous()
    B, H, N, D = x.shape
    nblock = (N + block_size - 1) // block_size  # Number of blocks per feature map
    pool = torch.empty((B, H, nblock, D), device=x.device, dtype=x.dtype)
    sim_blocks = torch.empty((B, H, nblock), device=x.device, dtype=torch.bool)
    grid = (B, H, nblock)
    # Launch kernel
    triton_bmm_pool_sim_simmean[grid](x, pool, sim_blocks, simthreshd1, N=N, D=D, BS=block_size)
    return pool, sim_blocks


@triton.jit
def triton_fill_causal_mask(mask, BqdivBk):
    q, k = tl.program_id(0), tl.program_id(1)
    Q, K = tl.num_programs(0), tl.num_programs(1)
    if k >= (q + 1) * BqdivBk:
        tl.store(mask + q * K + k, 0)
    else:
        tl.store(mask + q * K + k, 1)


def fill_causal_mask_triton(mask: torch.Tensor, BqdivBk:float) -> torch.Tensor:
    assert mask.dim() == 2
    triton_fill_causal_mask[mask.shape](mask, BqdivBk)
    return mask


@triton.jit
def triton_fill_block_map_kernel(final_map, num_to_select, sorted_indices, NK: tl.constexpr):
    b, h, q = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    B, H, Q = tl.num_programs(0), tl.num_programs(1), tl.num_programs(2)
    cur_num_to_select = tl.load(num_to_select + b * H * Q + h * Q + q)
    cur_sorted_idx_ptr = sorted_indices + b * H * Q * NK + h * Q * NK + q * NK
    cur_final_map_ptr = final_map + b * H * Q * NK + h * Q * NK + q * NK
    cur_num_to_select = (cur_num_to_select + 1) if cur_num_to_select == 0 else cur_num_to_select
    for i in range(cur_num_to_select):
        cur_idx = tl.load(cur_sorted_idx_ptr + i)
        tl.store(cur_final_map_ptr + cur_idx, 1)


def fill_block_map_triton(final_map, num_to_select, sorted_indices):
    final_map = final_map.contiguous()
    num_to_select = num_to_select.contiguous()
    sorted_indices = sorted_indices.contiguous()
    B, H, Q, K = final_map.shape
    grid = (B, H, Q)
    triton_fill_block_map_kernel[grid](final_map, num_to_select, sorted_indices, K)
    return final_map


def get_block_map_meansim(
    q: torch.Tensor,
    k: torch.Tensor,
    is_causal: bool = False,
    BLKQ: int = 64,
    BLKK: int = 64,
    simthreshd1: float = 0.1,
    cdfthreshd: float = 0.9,
    attention_sink: bool = False
) -> torch.Tensor:
    Headnum = q.size(1)
    simthreshd1 = hyperparameter_check(simthreshd1, Headnum, q.device)
    cdfthreshd = hyperparameter_check(cdfthreshd, Headnum, q.device)
    nq = (q.shape[-2] + BLKQ - 1) // BLKQ
    nk = (k.shape[-2] + BLKK - 1) // BLKK
    pooled_qblocks, sim_qblocks = get_pool_sim_triton_simmean(q, BLKQ, simthreshd1)
    pooled_kblocks, sim_kblocks = get_pool_sim_triton_simmean(k, BLKK, simthreshd1)

    sim_kblocks = sim_kblocks.unsqueeze(-2).expand(-1, -1, nq, -1)  # faster than repeat
    sim_qblocks = sim_qblocks.unsqueeze(-1).expand(-1, -1, -1, nk)
    pooled_score = pooled_qblocks @ pooled_kblocks.transpose(-1, -2) * q.shape[-1] ** -0.5

    neg_inf = pooled_score.new_full((), float("-inf"))
    pooled_score = torch.where(sim_kblocks, pooled_score, neg_inf)
    if is_causal:
        nq = pooled_qblocks.shape[-2]
        nk = pooled_kblocks.shape[-2]
        empty_mask = torch.empty(nq, nk, device=q.device, dtype=torch.bool)
        causal_mask = fill_causal_mask_triton(empty_mask, BLKQ / BLKK)
        pooled_score = torch.where(causal_mask[None, None, ...], pooled_score, neg_inf)
    pooled_score = pooled_score.softmax(-1)
    sorted_score = torch.sort(pooled_score, dim=-1, descending=True)
    cdf = torch.cumsum(sorted_score.values, dim=-1)
    B, H, Q, K = cdf.shape

    cdfthreshd_ts = cdfthreshd.view(1, H, 1, 1)

    ge = cdf >= cdfthreshd_ts

    idx = ge.to(torch.uint8).argmax(dim=-1)
    any_ge = ge.any(dim=-1)
    # 0-D fallback value broadcasts in `where`; avoids a (B, H, Q) alloc.
    num_to_select = torch.where(any_ge, idx, idx.new_full((), K))
 
    final_map = torch.zeros_like(pooled_score, dtype=torch.bool)
    final_map = fill_block_map_triton(final_map, num_to_select, sorted_score.indices)
    final_map = final_map | (~sim_kblocks) | (~sim_qblocks)
    if is_causal:
        final_map = final_map * causal_mask[None, None, ...]

    if attention_sink:
        final_map[:, :, :, 0] = 1
    
    return final_map
