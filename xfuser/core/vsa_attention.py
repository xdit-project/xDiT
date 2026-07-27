"""Jenga-style block selection for the AITER CK-Tile VSA kernel.

The mask policy and kernel encoding are intentionally separate:

* ``build_jenga_block_mask`` implements Jenga's pooled Q/K CDF selection.
* ``block_mask_to_delta_lut`` converts a canonical boolean block mask to the
  delta-encoded LUT consumed by AITER's ``vsa_sparse_attention`` CK op.
* ``aiter_vsa_attention`` handles xDiT's Gilbert/static-mask setup and restores
  the original token order after the CK call.
"""

import math
from typing import Optional

import torch

from xfuser.core.sparge_attention.sparge import (
    mask_padded_kv_blocks,
    restore_sparge_output,
    setup_sparge,
)

_SEQSTART_CACHE: dict[tuple, torch.Tensor] = {}
_DELTA_LUT_WORKSPACE_CACHE: dict[
    tuple, tuple[torch.Tensor, torch.Tensor]
] = {}

try:
    import triton
    import triton.language as tl
except ImportError:  # CPU-only tests and installations without Triton.
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _block_mask_to_delta_lut_kernel(
        mask_ptr,
        lut_ptr,
        count_ptr,
        num_k_blocks: tl.constexpr,
    ):
        b = tl.program_id(0)
        h = tl.program_id(1)
        q = tl.program_id(2)
        num_heads = tl.num_programs(1)
        num_q_blocks = tl.num_programs(2)
        row = (b * num_heads * num_q_blocks + h * num_q_blocks + q)
        mask_ptr += row * num_k_blocks
        lut_ptr += row * num_k_blocks
        count_ptr += row

        valid = 0
        previous = 0
        for k_block in range(num_k_blocks):
            selected = tl.load(mask_ptr + k_block)
            if selected != 0:
                tl.store(lut_ptr + valid, k_block - previous)
                valid += 1
                previous = k_block
        tl.store(count_ptr, valid)


def _block_mask_to_delta_lut_torch(
    block_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Portable reference encoder used by CPU tests and Triton-free installs."""
    num_k_blocks = block_mask.shape[-1]
    absolute = torch.arange(
        num_k_blocks, dtype=torch.int32, device=block_mask.device
    ).view(1, 1, 1, -1)
    sentinel = torch.full_like(absolute, num_k_blocks)
    selected = torch.where(block_mask, absolute, sentinel)
    selected = selected.sort(dim=-1).values
    counts = block_mask.sum(dim=-1, dtype=torch.int32)

    valid = (
        torch.arange(num_k_blocks, device=block_mask.device)
        .view(1, 1, 1, -1)
        < counts.unsqueeze(-1)
    )
    selected = torch.where(valid, selected, torch.zeros_like(selected))
    deltas = selected.clone()
    if num_k_blocks > 1:
        deltas[..., 1:] = selected[..., 1:] - selected[..., :-1]
    return deltas.contiguous(), counts.contiguous()


def block_mask_to_delta_lut(
    block_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode ``[B,H,Qb,Kb]`` boolean mask for AITER VSA.

    The first valid entry is an absolute KV-block index; following entries are
    deltas from the previous selected block. ``counts`` gives the valid prefix
    length for every ``(batch, head, query-block)`` row.
    """
    if block_mask.ndim != 4:
        raise ValueError(
            f"VSA block mask must have shape [B,H,Qb,Kb], got {block_mask.shape}"
        )
    block_mask = block_mask.to(torch.bool).contiguous()
    b, h, q_blocks, k_blocks = block_mask.shape
    if k_blocks == 0:
        raise ValueError("VSA block mask must contain at least one KV block")

    if triton is None or block_mask.device.type != "cuda":
        return _block_mask_to_delta_lut_torch(block_mask)

    stream_id = torch.cuda.current_stream(block_mask.device).cuda_stream
    workspace_key = (
        block_mask.device.index,
        stream_id,
        b,
        h,
        q_blocks,
        k_blocks,
    )
    workspace = _DELTA_LUT_WORKSPACE_CACHE.get(workspace_key)
    if workspace is None:
        workspace = (
            torch.empty(
                (b, h, q_blocks, k_blocks),
                dtype=torch.int32,
                device=block_mask.device,
            ),
            torch.empty(
                (b, h, q_blocks),
                dtype=torch.int32,
                device=block_mask.device,
            ),
        )
        _DELTA_LUT_WORKSPACE_CACHE[workspace_key] = workspace
    lut, counts = workspace
    _block_mask_to_delta_lut_kernel[(b, h, q_blocks)](
        block_mask, lut, counts, k_blocks
    )
    return lut, counts


def jenga_scheduled_drop_rate(
    step_index: int,
    total_steps: int,
    drop_rates: list[float] | tuple[float, ...],
) -> float:
    """Reproduce Jenga's timestep-dependent self-attention drop rate."""
    if total_steps <= 0:
        raise ValueError(f"total_steps must be positive, got {total_steps}")
    if not drop_rates:
        raise ValueError("drop_rates must contain at least one value")
    if any(rate < 0.0 or rate > 1.0 for rate in drop_rates):
        raise ValueError(f"drop rates must be in [0,1], got {drop_rates}")

    step_index = min(max(int(step_index), 0), total_steps - 1)
    # Jenga switches after step 25 in its 50-step reference schedule. Express
    # that boundary relative to the requested run length so short runs also
    # exercise both configured rates while preserving the 50-step behavior.
    if len(drop_rates) == 1 or step_index <= total_steps // 2:
        base_drop_rate = float(drop_rates[0])
    else:
        base_drop_rate = float(drop_rates[1])

    # Jenga linearly warms the configured rate over the first 10% of steps.
    progress_x10 = (
        step_index / (total_steps - 1) * 10.0
        if total_steps > 1
        else 10.0
    )
    return min(base_drop_rate, progress_x10 * base_drop_rate)


def _first_frame_block_count(
    thw: tuple[int, int, int], block_size: int
) -> int:
    """Count blocks fully contained in the first temporal slice."""
    if block_size <= 0:
        raise ValueError(f"VSA block size must be positive, got {block_size}")
    time, height, width = map(int, thw)
    if time <= 0 or height <= 0 or width <= 0:
        raise ValueError(f"VSA thw dimensions must be positive, got {thw}")
    return (height * width) // block_size


def build_jenga_block_mask(
    query: torch.Tensor,
    key: torch.Tensor,
    *,
    block_size: int = 128,
    top_k: int = 1,
    prob_threshold: float = 0.9,
    static_block_mask: Optional[torch.Tensor] = None,
    first_frame_blocks: int = 0,
) -> torch.Tensor:
    """Build Jenga's pooled-Q/K importance mask.

    For every query block, select the smallest probability-sorted KV-block
    prefix whose cumulative softmax mass exceeds ``prob_threshold`` while
    keeping at least ``top_k`` blocks. Static neighbors and the first-frame
    relation are unioned into that dynamic mask.
    """
    if query.ndim != 4 or key.ndim != 4:
        raise ValueError("VSA expects query/key in [B,H,S,D] layout")
    if query.shape != key.shape:
        raise ValueError(
            f"VSA self-attention requires matching query/key shapes, got "
            f"{query.shape} and {key.shape}"
        )
    if block_size <= 0:
        raise ValueError(f"VSA block size must be positive, got {block_size}")
    if not 0.0 <= prob_threshold <= 1.0:
        raise ValueError(
            f"VSA probability threshold must be in [0,1], got {prob_threshold}"
        )

    batch, heads, sequence, head_dim = query.shape
    if sequence % block_size != 0:
        raise ValueError(
            f"VSA sequence length {sequence} is not divisible by {block_size}"
        )
    num_blocks = sequence // block_size
    top_k = min(max(int(top_k), 1), num_blocks)

    query_pool = query.reshape(
        batch, heads, num_blocks, block_size, head_dim
    ).mean(dim=-2)
    key_pool = key.reshape(
        batch, heads, num_blocks, block_size, head_dim
    ).mean(dim=-2)
    scores = torch.matmul(query_pool, key_pool.transpose(-1, -2))
    scores = scores * (head_dim ** -0.5)
    probabilities = torch.softmax(scores, dim=-1)
    sorted_probabilities, indices = probabilities.sort(
        dim=-1, descending=True
    )
    cumulative = sorted_probabilities.cumsum(dim=-1)
    needed = (cumulative <= prob_threshold).sum(dim=-1) + 1
    needed = needed.clamp(min=top_k, max=num_blocks)

    ranks = torch.arange(num_blocks, device=query.device).view(1, 1, 1, -1)
    selected_by_rank = ranks < needed.unsqueeze(-1)
    block_mask = torch.zeros(
        (batch, heads, num_blocks, num_blocks),
        dtype=torch.bool,
        device=query.device,
    )
    block_mask.scatter_(-1, indices, selected_by_rank)

    if static_block_mask is not None:
        static_block_mask = static_block_mask.to(
            device=query.device, dtype=torch.bool
        )
        q_blocks = min(num_blocks, static_block_mask.shape[0])
        k_blocks = min(num_blocks, static_block_mask.shape[1])
        block_mask[:, :, :q_blocks, :k_blocks] |= static_block_mask[
            None, None, :q_blocks, :k_blocks
        ]

    first_frame_blocks = min(max(int(first_frame_blocks), 0), num_blocks)
    if first_frame_blocks:
        block_mask[:, :, :first_frame_blocks, :first_frame_blocks] = True
    return block_mask


def aiter_vsa_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    thw: tuple[int, int, int],
    sp_size: int,
    block_size: int = 128,
    top_k: int = 1,
    top_k_ratio: float = 0.0,
    drop_rate: Optional[float] = None,
    prob_threshold: float = 0.9,
    reorder_sequence: bool = True,
    use_static_block_mask: bool = True,
    use_first_frame_mask: bool = True,
    collect_density: bool = False,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Run Jenga-mask sparse attention through AITER's CK-Tile VSA op."""
    if block_size != 128:
        raise ValueError(
            "The current AITER CK-Tile VSA build requires block_size=128"
        )
    if query.dtype not in (torch.bfloat16, torch.float16):
        raise TypeError(f"AITER VSA requires BF16/FP16 QKV, got {query.dtype}")
    if not (query.shape == key.shape == value.shape):
        raise ValueError("AITER VSA currently supports self-attention only")

    query, key, value, state, static_mask = setup_sparge(
        query,
        key,
        value,
        thw=thw,
        sp_size=sp_size,
        reorder_sequence=reorder_sequence,
        use_static_block_mask=use_static_block_mask,
        block_m=block_size,
        block_n=block_size,
        pad_block_divisible=True,
        use_sliced_gilbert=True,
    )
    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()
    if not 0.0 <= top_k_ratio <= 1.0:
        raise ValueError(
            f"VSA top-k ratio must be in [0,1], got {top_k_ratio}"
        )
    if top_k_ratio:
        top_k = max(
            int(top_k),
            math.ceil((query.shape[2] // block_size) * top_k_ratio),
        )
    if drop_rate is not None:
        if not 0.0 <= drop_rate <= 1.0:
            raise ValueError(f"VSA drop rate must be in [0,1], got {drop_rate}")
        # Match Jenga's ``int(num_blocks * (1 - sa_drop_rate))`` floor.
        top_k = max(
            int(top_k),
            int((query.shape[2] // block_size) * (1.0 - drop_rate)),
        )

    first_frame_blocks = 0
    if use_first_frame_mask:
        # Sliced Gilbert order keeps each temporal slice contiguous. Protect
        # only blocks fully contained in the first frame; a partial boundary
        # block remains dynamic because it also contains the next frame.
        first_frame_blocks = _first_frame_block_count(thw, block_size)

    # The probability mask depends on the current Q/K tensors and therefore
    # cannot be cached. Layout permutations and static-neighbor masks are
    # cached separately by setup_sparge().
    block_mask = build_jenga_block_mask(
        query,
        key,
        block_size=block_size,
        top_k=top_k,
        prob_threshold=prob_threshold,
        static_block_mask=static_mask,
        first_frame_blocks=first_frame_blocks,
    )
    block_mask = mask_padded_kv_blocks(block_mask, state, block_size)
    lut, valid_blocks = block_mask_to_delta_lut(block_mask)

    from aiter.ops.jenga_sparse_attention import vsa_sparse_attention

    batch, heads, sequence, head_dim = query.shape
    device_index = (
        query.device.index if query.device.index is not None else -1
    )
    seqstart_key = (query.device.type, device_index, sequence)
    seqstart = _SEQSTART_CACHE.get(seqstart_key)
    if seqstart is None:
        seqstart = torch.tensor(
            [0, sequence], dtype=torch.int32, device=query.device
        ).contiguous()
        _SEQSTART_CACHE[seqstart_key] = seqstart
    output = torch.empty_like(query).contiguous()
    output = vsa_sparse_attention(
        query,
        key,
        value,
        lut,
        valid_blocks,
        output,
        None,
        None,
        seqstart,
        seqstart,
        0,
        batch,
        heads,
        heads,
        sequence,
        sequence,
        head_dim,
        head_dim,
    )
    output = restore_sparge_output(output, state)
    density = block_mask.float().mean() if collect_density else None
    return output, density
