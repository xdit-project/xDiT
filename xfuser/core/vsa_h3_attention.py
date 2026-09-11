# SPDX-License-Identifier: Apache-2.0
"""Geometry and mask primitives for MiniMax-H3 Video Sparse Attention.

VSA-H3 uses segment-pure one-dimensional prefix tiles followed by three-
dimensional generated-video tiles. This module deliberately contains no
kernel dispatch; it is the shared, testable contract for a future portable
64-token block-sparse implementation.
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass

import torch
from torch.nn.attention.flex_attention import BlockMask, flex_attention


FASTH3_VSA_TILE_SHAPE = (4, 4, 4)
FASTH3_VSA_TILE_ELEMENTS = math.prod(FASTH3_VSA_TILE_SHAPE)
FASTH3_VSA_SPARSITY = 0.9
_compiled_flex_attention = torch.compile(flex_attention, dynamic=False)


@dataclass(frozen=True)
class MiniMaxH3VSAMetadata:
    """Cached mapping between packed H3 rows and the padded tile buffer."""

    total_seq_length: int
    num_prefix_tiles: int
    num_video_tiles: int
    variable_block_sizes: torch.Tensor
    packed_to_tiled_index: torch.Tensor
    tile_elements: int = FASTH3_VSA_TILE_ELEMENTS

    @property
    def num_tiles(self) -> int:
        return self.num_prefix_tiles + self.num_video_tiles

    @property
    def padded_seq_length(self) -> int:
        return self.num_tiles * self.tile_elements


def compute_h3_vsa_topk(sparsity: float, num_video_tiles: int) -> int:
    """Return the number of video-key tiles retained by VSA-H3."""
    if not 0.0 <= sparsity <= 1.0:
        raise ValueError(f"VSA-H3 sparsity must be in [0, 1], got {sparsity}.")
    if num_video_tiles < 1:
        raise ValueError(
            "VSA-H3 requires at least one generated-video tile, got "
            f"{num_video_tiles}."
        )
    return max(
        1,
        min(
            math.ceil((1.0 - sparsity) * num_video_tiles),
            num_video_tiles,
        ),
    )


def _axis_tile_sizes(length: int, tile: int, device: torch.device) -> torch.Tensor:
    count = math.ceil(length / tile)
    sizes = torch.full((count,), tile, dtype=torch.long, device=device)
    sizes[-1] = length - (count - 1) * tile
    return sizes


def _video_tile_sizes(
    video_shape: tuple[int, int, int],
    tile_shape: tuple[int, int, int],
    device: torch.device,
) -> torch.Tensor:
    t_sizes = _axis_tile_sizes(video_shape[0], tile_shape[0], device)
    h_sizes = _axis_tile_sizes(video_shape[1], tile_shape[1], device)
    w_sizes = _axis_tile_sizes(video_shape[2], tile_shape[2], device)
    return (
        t_sizes[:, None, None]
        * h_sizes[None, :, None]
        * w_sizes[None, None, :]
    ).reshape(-1)


def _video_tile_order(
    video_shape: tuple[int, int, int],
    tile_shape: tuple[int, int, int],
    device: torch.device,
) -> torch.Tensor:
    t, h, w = video_shape
    tile_t, tile_h, tile_w = tile_shape
    indices = torch.arange(t * h * w, device=device).reshape(t, h, w)
    tiles = []
    for t_start in range(0, t, tile_t):
        for h_start in range(0, h, tile_h):
            for w_start in range(0, w, tile_w):
                tiles.append(
                    indices[
                        t_start : t_start + tile_t,
                        h_start : h_start + tile_h,
                        w_start : w_start + tile_w,
                    ].flatten()
                )
    return torch.cat(tiles)


@functools.lru_cache(maxsize=16)
def build_h3_vsa_metadata(
    prefix_segments: tuple[int, ...],
    video_shape: tuple[int, int, int],
    device: torch.device,
    tile_shape: tuple[int, int, int] = FASTH3_VSA_TILE_SHAPE,
) -> MiniMaxH3VSAMetadata:
    """Build FastH3's packed-row to 64-token tile mapping.

    ``prefix_segments`` must contain the independent text, condition, and
    audio lengths. Zero-length segments are ignored, but adjacent non-empty
    segments are never merged into the same tile.
    """
    if any(segment < 0 for segment in prefix_segments):
        raise ValueError(
            f"VSA-H3 prefix segment lengths must be non-negative: {prefix_segments}."
        )
    if any(size < 1 for size in video_shape):
        raise ValueError(
            f"VSA-H3 generated-video shape must be positive: {video_shape}."
        )
    if any(size < 1 for size in tile_shape):
        raise ValueError(f"VSA-H3 tile shape must be positive: {tile_shape}.")

    tile_elements = math.prod(tile_shape)
    if tile_elements != FASTH3_VSA_TILE_ELEMENTS:
        raise ValueError(
            "FastH3 Preview v1 requires 64-token VSA tiles, got "
            f"{tile_shape} ({tile_elements} tokens)."
        )

    segments = tuple(int(segment) for segment in prefix_segments if segment)
    prefix_sizes = []
    for segment in segments:
        full_tiles, remainder = divmod(segment, tile_elements)
        prefix_sizes.extend([tile_elements] * full_tiles)
        if remainder:
            prefix_sizes.append(remainder)

    video_sizes = _video_tile_sizes(video_shape, tile_shape, device)
    variable_block_sizes = torch.cat(
        (
            torch.tensor(prefix_sizes, dtype=torch.long, device=device),
            video_sizes,
        )
    )

    prefix_length = sum(segments)
    tiled_to_packed = torch.cat(
        (
            torch.arange(prefix_length, device=device),
            _video_tile_order(video_shape, tile_shape, device) + prefix_length,
        )
    )
    non_pad_slots = (
        torch.arange(variable_block_sizes.numel(), device=device)[:, None]
        * tile_elements
        + torch.arange(tile_elements, device=device)[None, :]
    )
    valid = (
        torch.arange(tile_elements, device=device)[None, :]
        < variable_block_sizes[:, None]
    )
    non_pad_slots = non_pad_slots[valid]
    packed_to_tiled = non_pad_slots[torch.argsort(tiled_to_packed)]

    total_seq_length = prefix_length + math.prod(video_shape)
    if (
        int(variable_block_sizes.sum()) != total_seq_length
        or packed_to_tiled.numel() != total_seq_length
        or torch.unique(packed_to_tiled).numel() != total_seq_length
    ):
        raise ValueError(
            "Invalid VSA-H3 tile mapping for "
            f"prefix={prefix_segments}, video={video_shape}."
        )

    return MiniMaxH3VSAMetadata(
        total_seq_length=total_seq_length,
        num_prefix_tiles=len(prefix_sizes),
        num_video_tiles=video_sizes.numel(),
        variable_block_sizes=variable_block_sizes,
        packed_to_tiled_index=packed_to_tiled,
        tile_elements=tile_elements,
    )


def tile_h3_vsa_tensor(
    tensor: torch.Tensor,
    metadata: MiniMaxH3VSAMetadata,
) -> torch.Tensor:
    """Scatter ``[B, S, H, D]`` packed rows into a zero-padded tile buffer."""
    if tensor.ndim != 4 or tensor.shape[1] != metadata.total_seq_length:
        raise ValueError(
            "VSA-H3 expects [B, S, H, D] with S="
            f"{metadata.total_seq_length}, got {tuple(tensor.shape)}."
        )
    tiled = tensor.new_zeros(
        tensor.shape[0],
        metadata.padded_seq_length,
        tensor.shape[2],
        tensor.shape[3],
    )
    tiled[:, metadata.packed_to_tiled_index] = tensor
    return tiled


def untile_h3_vsa_tensor(
    tensor: torch.Tensor,
    metadata: MiniMaxH3VSAMetadata,
) -> torch.Tensor:
    """Restore packed row order from a padded VSA-H3 tile buffer."""
    if tensor.ndim != 4 or tensor.shape[1] != metadata.padded_seq_length:
        raise ValueError(
            "VSA-H3 expects a padded [B, S, H, D] tile buffer with S="
            f"{metadata.padded_seq_length}, got {tuple(tensor.shape)}."
        )
    return tensor[:, metadata.packed_to_tiled_index]


def build_h3_vsa_block_mask(
    scores: torch.Tensor,
    num_prefix_tiles: int,
    num_video_tiles: int,
    sparsity: float = FASTH3_VSA_SPARSITY,
) -> torch.Tensor:
    """Select FastH3's exempt-prefix, sparse-video key tiles.

    Every query retains every prefix key tile. The remaining budget is the
    top-k video key tiles selected independently for each query and head.
    """
    num_tiles = num_prefix_tiles + num_video_tiles
    if scores.ndim != 4 or scores.shape[-2:] != (num_tiles, num_tiles):
        raise ValueError(
            "VSA-H3 scores must be [B, H, tiles, tiles] with tiles="
            f"{num_tiles}, got {tuple(scores.shape)}."
        )

    video_topk = compute_h3_vsa_topk(sparsity, num_video_tiles)
    if video_topk == num_video_tiles:
        return torch.ones_like(scores, dtype=torch.bool)

    mask = torch.zeros_like(scores, dtype=torch.bool)
    video_indices = (
        scores[..., num_prefix_tiles:]
        .topk(video_topk, dim=-1)
        .indices
        + num_prefix_tiles
    )
    mask.scatter_(-1, video_indices, True)
    mask[..., :num_prefix_tiles] = True
    return mask


def _flex_block_mask(block_map: torch.Tensor, block_size: int) -> BlockMask:
    num_kv_blocks = block_map.shape[-1]
    indices = torch.arange(
        num_kv_blocks,
        dtype=torch.int32,
        device=block_map.device,
    ).view(1, 1, 1, -1)
    indices = indices.expand_as(block_map)
    sentinel = torch.full_like(indices, num_kv_blocks)
    kv_indices = torch.where(block_map, indices, sentinel).sort(dim=-1).values
    kv_num_blocks = block_map.sum(dim=-1, dtype=torch.int32)
    sequence_length = block_map.shape[-1] * block_size
    return BlockMask.from_kv_blocks(
        kv_num_blocks,
        kv_indices,
        BLOCK_SIZE=block_size,
        seq_lengths=(sequence_length, sequence_length),
        compute_q_blocks=False,
    )


def pool_h3_vsa_tiles(
    tensor: torch.Tensor,
    metadata: MiniMaxH3VSAMetadata,
) -> torch.Tensor:
    """Pool a padded BHSD tile buffer to fp32 ``[B, H, tiles, D]``."""
    if tensor.ndim != 4 or tensor.shape[2] != metadata.padded_seq_length:
        raise ValueError(
            "VSA-H3 pooling expects [B, H, S, D] with padded S="
            f"{metadata.padded_seq_length}, got {tuple(tensor.shape)}."
        )
    batch, heads, _, head_dim = tensor.shape
    pooled = tensor.view(
        batch,
        heads,
        metadata.num_tiles,
        metadata.tile_elements,
        head_dim,
    ).sum(dim=3, dtype=torch.float32)
    return pooled / metadata.variable_block_sizes.view(1, 1, -1, 1)


def flex_h3_vsa_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    metadata: MiniMaxH3VSAMetadata,
    sparsity: float = FASTH3_VSA_SPARSITY,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the portable 64-token VSA-H3 sparse branch with FlexAttention.

    Inputs and output use padded BHSD tile order. The returned compressed
    value is the dense pooled branch expanded back to token rows; callers
    apply the checkpoint's learned gate before adding it to the sparse output.
    """
    expected = (
        query.ndim == 4
        and query.shape == key.shape == value.shape
        and query.shape[2] == metadata.padded_seq_length
    )
    if not expected:
        raise ValueError(
            "VSA-H3 attention expects equally shaped padded BHSD tensors; "
            f"got q={tuple(query.shape)}, k={tuple(key.shape)}, "
            f"v={tuple(value.shape)}."
        )

    pooled_query = pool_h3_vsa_tiles(query, metadata)
    pooled_key = pool_h3_vsa_tiles(key, metadata)
    pooled_value = pool_h3_vsa_tiles(value, metadata)
    scores = torch.matmul(
        pooled_query,
        pooled_key.transpose(-2, -1),
    ) * (query.shape[-1] ** -0.5)
    block_map = build_h3_vsa_block_mask(
        scores,
        metadata.num_prefix_tiles,
        metadata.num_video_tiles,
        sparsity,
    )
    flex_block_mask = _flex_block_mask(block_map, metadata.tile_elements)
    variable_block_sizes = metadata.variable_block_sizes
    tile_elements = metadata.tile_elements

    def mask_padding(score, batch, head, query_index, key_index):
        del batch, head, query_index
        key_block = key_index // tile_elements
        valid = key_index % tile_elements < variable_block_sizes[key_block]
        return torch.where(valid, score, -float("inf"))

    sparse_output = _compiled_flex_attention(
        query,
        key,
        value,
        score_mod=mask_padding,
        block_mask=flex_block_mask,
        kernel_options={
            "BLOCK_M": tile_elements,
            "BLOCK_N": tile_elements,
            "ROWS_GUARANTEED_SAFE": True,
        },
    )
    compressed = torch.matmul(torch.softmax(scores, dim=-1), pooled_value)
    compressed = compressed.repeat_interleave(tile_elements, dim=2)
    return sparse_output, compressed
