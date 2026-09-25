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
import torch.nn.functional as F
from torch.nn.attention.flex_attention import BlockMask, flex_attention

from xfuser.logger import init_logger

logger = init_logger(__name__)


FASTH3_VSA_TILE_SHAPE = (4, 4, 4)
FASTH3_VSA_TILE_ELEMENTS = math.prod(FASTH3_VSA_TILE_SHAPE)
FASTH3_VSA_SPARSITY = 0.9

# Budget for one selection step's score matrix. The full matrix is
# [B, H, tiles, tiles] fp32, which runs to hundreds of MB at large video sizes,
# so selection walks query-tile chunks sized to stay under this. Expressed as a
# byte budget rather than a tile count so that small geometries take a single
# chunk instead of paying three kernel launches for an 8 MB tensor. It is also
# the knob to raise if a geometry ever gets large enough that the chunk falls to
# a handful of tiles and selection turns launch-bound.
FASTH3_VSA_SELECTION_SCORE_BYTES = 64 * 2**20

# The Triton kernel unrolls its KV loop by a tunable factor. Padding the index
# list to a multiple of the largest factor, with a sentinel tile whose slots are
# all invalid, removes the loop's tail guard entirely. Every sentinel tile is a
# tile's worth of key/value traffic the kernel does for nothing, so this tracks
# the largest unroll factor in the autotune space and no more.
FASTH3_VSA_KV_LIST_ALIGNMENT = 4

# FlexAttention kernel configuration. BLOCK_M/BLOCK_N are pinned by the 64-token
# tile: the Triton template asserts SPARSE_Q_BLOCK_SIZE >= BLOCK_M and
# SPARSE_KV_BLOCK_SIZE >= BLOCK_N, so neither can exceed the tile.
FASTH3_VSA_KERNEL_OPTIONS = {
    "BLOCK_M": FASTH3_VSA_TILE_ELEMENTS,
    "BLOCK_N": FASTH3_VSA_TILE_ELEMENTS,
    # Every query tile keeps all prefix tiles, so no row is ever fully masked.
    "ROWS_GUARANTEED_SAFE": True,
}

if torch.version.hip:
    # Inductor's ROCm default for FlexAttention is tuned for a 128-row M tile.
    # A 64-row tile gives each warp a quarter of the rows it was sized for, and
    # the default's warp count then costs more in scheduling than it recovers
    # in parallelism -- it measured several times slower than two warps here.
    # Scoped to the backend it was measured on: other backends have a different
    # default and would only be inheriting a foreign tuning.
    FASTH3_VSA_KERNEL_OPTIONS["num_warps"] = 2
    FASTH3_VSA_KERNEL_OPTIONS["num_stages"] = 2

_compiled_flex_attention = torch.compile(flex_attention, dynamic=False)
_KV_BLOCK_WORKSPACE_CACHE: dict[tuple, tuple[torch.Tensor, ...]] = {}


def _flex_attention_call(*args, **kwargs):
    """Dispatch FlexAttention without nesting torch.compile inside a graph.

    Calling a compiled wrapper from inside an outer compiled region is what
    stops the MiniMax-H3 transformer from tracing at ``fullgraph=True``.
    ``flex_attention`` is itself a traceable higher-order op, so hand the outer
    compile the raw call and keep the pre-compiled wrapper for eager runs.
    """
    if torch.compiler.is_compiling():
        return flex_attention(*args, **kwargs)
    return _compiled_flex_attention(*args, **kwargs)


@dataclass(frozen=True, eq=False)
class MiniMaxH3VSAMetadata:
    """Cached mapping between packed H3 rows and the padded tile buffer.

    ``eq=False`` keeps object identity for ``__eq__``/``__hash__``. A frozen
    dataclass otherwise synthesises both from the fields, and comparing or
    hashing the tensor fields raises. ``build_h3_vsa_metadata`` returns one
    shared instance per geometry, so identity comparison is also what callers
    want.

    Video tiles are ordered so that every full 64-token tile precedes every
    partial one. That turns "is this tile padded?" into a comparison against
    ``first_partial_video_tile`` instead of a lookup, which is what lets the
    selection path split FlexAttention's full and partial KV block lists
    without a gather.
    """

    total_seq_length: int
    num_prefix_tiles: int
    num_video_tiles: int
    num_full_video_tiles: int
    num_prefix_partial_tiles: int
    variable_block_sizes: torch.Tensor
    packed_to_tiled_index: torch.Tensor
    tiled_to_packed_index: torch.Tensor
    pad_slot_index: torch.Tensor
    packed_token_tile: torch.Tensor
    tiled_slot_valid: torch.Tensor
    tiled_to_packed_row: torch.Tensor
    prefix_partial_first_keys: torch.Tensor
    prefix_full_first_keys: torch.Tensor
    padding_mask_mod: object
    tile_elements: int = FASTH3_VSA_TILE_ELEMENTS

    @property
    def num_tiles(self) -> int:
        return self.num_prefix_tiles + self.num_video_tiles

    @property
    def padded_seq_length(self) -> int:
        return self.num_tiles * self.tile_elements

    @property
    def first_partial_video_tile(self) -> int:
        """Lowest tile id that is padded, or ``num_tiles`` when none is."""
        return self.num_prefix_tiles + self.num_full_video_tiles


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
    video_token_slices = torch.split(
        _video_tile_order(video_shape, tile_shape, device),
        video_sizes.tolist(),
    )

    # Sort full video tiles ahead of partial ones. This is a pure relabelling of
    # the padded buffer -- the token membership of each tile is untouched -- and
    # it makes the padded tile ids a contiguous suffix, so the selection path can
    # split FlexAttention's full and partial block lists with a comparison.
    video_order = torch.argsort(
        (video_sizes != tile_elements).to(torch.int8), stable=True
    ).tolist()
    video_sizes = video_sizes[video_order]
    num_full_video_tiles = int((video_sizes == tile_elements).sum())
    video_tokens = torch.cat([video_token_slices[index] for index in video_order])

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
            video_tokens + prefix_length,
        )
    )
    num_tiles = variable_block_sizes.numel()
    slot_in_tile = torch.arange(tile_elements, device=device)
    non_pad_slots = (
        torch.arange(num_tiles, device=device)[:, None] * tile_elements
        + slot_in_tile[None, :]
    )
    valid = slot_in_tile[None, :] < variable_block_sizes[:, None]
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

    # Gather-side mapping: padded slot -> packed row. Pad slots point at row 0
    # and are zeroed afterwards, which is cheaper than a zero-fill plus scatter
    # over the whole padded buffer.
    padded_seq_length = num_tiles * tile_elements
    tiled_slot_valid = valid.reshape(-1)
    tiled_to_packed_gather = torch.zeros(
        padded_seq_length, dtype=torch.long, device=device
    )
    tiled_to_packed_gather[packed_to_tiled] = torch.arange(
        total_seq_length, device=device
    )
    pad_slot_index = (~tiled_slot_valid).nonzero(as_tuple=True)[0]
    # Output-side mapping for the Triton kernel, which writes packed rows
    # directly. Padded slots point one past the end and are masked off.
    # One extra all-padding tile, so the Triton kernel's sentinel tile id is in
    # bounds and reads as padding.
    tiled_to_packed_row = torch.full(
        (padded_seq_length + tile_elements,),
        total_seq_length,
        dtype=torch.int32,
        device=device,
    )
    tiled_to_packed_row[packed_to_tiled] = torch.arange(
        total_seq_length, dtype=torch.int32, device=device
    )

    num_prefix_tiles = len(prefix_sizes)
    prefix_sizes_tensor = variable_block_sizes[:num_prefix_tiles]
    prefix_is_partial = prefix_sizes_tensor != tile_elements
    prefix_ids = torch.arange(
        num_prefix_tiles, dtype=torch.int32, device=device
    )
    # Sort keys that land one group ahead of the other: adding ``num_tiles`` to
    # the losing group makes a plain ascending sort produce a grouped, tile-id
    # ordered row. See build_h3_vsa_kv_blocks.
    prefix_partial_first_keys = prefix_ids + (
        (~prefix_is_partial).to(torch.int32) * num_tiles
    )
    prefix_full_first_keys = prefix_ids + (
        prefix_is_partial.to(torch.int32) * num_tiles
    )

    def padding_mask_mod(batch, head, query_index, key_index):
        """Drop the padded slots of partial tiles.

        A precomputed per-slot validity vector makes this one indexed load,
        against the divide-plus-gather-plus-compare a score_mod over
        ``variable_block_sizes`` would run on every score element of every
        block -- and FlexAttention skips it entirely on the full block list.
        """
        del batch, head, query_index
        return tiled_slot_valid[key_index]

    logger.info(
        # Sparsity is a per-call argument to selection, not part of the cached
        # geometry, so the retained tile count does not belong in this line.
        "VSA-H3 geometry: prefix=%s video=%s -> %d tokens, %d tiles "
        "(%d prefix, %d video, %d padded).",
        prefix_segments,
        video_shape,
        total_seq_length,
        num_tiles,
        num_prefix_tiles,
        video_sizes.numel(),
        num_tiles - num_prefix_tiles - num_full_video_tiles
        + int(prefix_is_partial.sum()),
    )
    return MiniMaxH3VSAMetadata(
        total_seq_length=total_seq_length,
        num_prefix_tiles=num_prefix_tiles,
        num_video_tiles=video_sizes.numel(),
        num_full_video_tiles=num_full_video_tiles,
        num_prefix_partial_tiles=int(prefix_is_partial.sum()),
        variable_block_sizes=variable_block_sizes,
        packed_to_tiled_index=packed_to_tiled,
        tiled_to_packed_index=tiled_to_packed_gather,
        pad_slot_index=pad_slot_index,
        packed_token_tile=packed_to_tiled // tile_elements,
        tiled_slot_valid=tiled_slot_valid,
        tiled_to_packed_row=tiled_to_packed_row,
        prefix_partial_first_keys=prefix_partial_first_keys,
        prefix_full_first_keys=prefix_full_first_keys,
        padding_mask_mod=padding_mask_mod,
        tile_elements=tile_elements,
    )


def tile_h3_vsa_tensor(
    tensor: torch.Tensor,
    metadata: MiniMaxH3VSAMetadata,
) -> torch.Tensor:
    """Scatter ``[B, S, H, D]`` packed rows into a zero-padded tile buffer.

    The readable reference for the scatter. The attention path runs on BHSD and
    uses ``tile_h3_vsa_bhsd``, which is pinned against this one in the tests.
    """
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


def tile_h3_vsa_bhsd(
    tensor: torch.Tensor,
    metadata: MiniMaxH3VSAMetadata,
) -> torch.Tensor:
    """Gather ``[B, H, S, D]`` packed rows into a zero-padded tile buffer.

    The gather form writes the padded buffer exactly once, where the scatter
    form in ``tile_h3_vsa_tensor`` costs a full-size zero fill plus a
    scattered write plus a transposed copy to get back to BHSD.
    """
    if tensor.ndim != 4 or tensor.shape[2] != metadata.total_seq_length:
        raise ValueError(
            "VSA-H3 expects [B, H, S, D] with S="
            f"{metadata.total_seq_length}, got {tuple(tensor.shape)}."
        )
    tiled = tensor.index_select(2, metadata.tiled_to_packed_index)
    # Pooling divides by the real token count per tile, so padded slots have to
    # hold zeros for the tile means to be correct. The attention kernel itself
    # does not care: mask_mod drops them.
    tiled.index_fill_(2, metadata.pad_slot_index, 0)
    return tiled


def untile_h3_vsa_tensor(
    tensor: torch.Tensor,
    metadata: MiniMaxH3VSAMetadata,
) -> torch.Tensor:
    """Restore packed row order from a padded VSA-H3 tile buffer.

    The BSHD inverse of ``tile_h3_vsa_tensor``; see that function.
    """
    if tensor.ndim != 4 or tensor.shape[1] != metadata.padded_seq_length:
        raise ValueError(
            "VSA-H3 expects a padded [B, S, H, D] tile buffer with S="
            f"{metadata.padded_seq_length}, got {tuple(tensor.shape)}."
        )
    return tensor[:, metadata.packed_to_tiled_index]


def untile_h3_vsa_bhsd(
    tensor: torch.Tensor,
    metadata: MiniMaxH3VSAMetadata,
) -> torch.Tensor:
    """Restore packed row order from a padded ``[B, H, S, D]`` tile buffer."""
    if tensor.ndim != 4 or tensor.shape[2] != metadata.padded_seq_length:
        raise ValueError(
            "VSA-H3 expects a padded [B, H, S, D] tile buffer with S="
            f"{metadata.padded_seq_length}, got {tuple(tensor.shape)}."
        )
    return tensor.index_select(2, metadata.packed_to_tiled_index)


def build_h3_vsa_block_mask(
    scores: torch.Tensor,
    num_prefix_tiles: int,
    num_video_tiles: int,
    sparsity: float = FASTH3_VSA_SPARSITY,
) -> torch.Tensor:
    """Select FastH3's exempt-prefix, sparse-video key tiles.

    Every query retains every prefix key tile. The remaining budget is the
    top-k video key tiles selected independently for each query and head.

    This is the readable reference for the selection policy. The attention path
    uses ``build_h3_vsa_kv_blocks``, which computes the same set without ever
    materialising a dense ``[B, H, tiles, tiles]`` mask.
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


def _kv_block_workspace(
    batch: int,
    heads: int,
    num_tiles: int,
    width: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-geometry KV index buffers, reused across the transformer's blocks.

    The buffers are handed out, not copied, so a geometry has exactly one live
    consumer: each attention layer selects, attends and is done with them before
    the next layer selects. Holding a returned buffer past that call -- stashing
    it for a later layer, or selecting for two layers before attending -- reads
    another layer's indices. The cache is keyed on the stream as well as the
    geometry so concurrent streams do not share one.

    Entries live for the process's lifetime. That is bounded by the number of
    distinct geometries, which for a generation run is one.
    """
    def allocate():
        return (
            torch.empty(
                (batch, heads, num_tiles, width),
                dtype=torch.int32,
                device=device,
            ),
            torch.empty(
                (batch, heads, num_tiles, width),
                dtype=torch.int32,
                device=device,
            ),
            torch.empty((batch, heads, num_tiles), dtype=torch.int32, device=device),
            torch.empty((batch, heads, num_tiles), dtype=torch.int32, device=device),
        )

    # Inductor plans its own buffer reuse, and a Python dict keyed on symbolic
    # sizes is not traceable, so the cache is an eager-only optimisation.
    if torch.compiler.is_compiling():
        return allocate()

    key = (
        device.type,
        device.index,
        torch.cuda.current_stream(device).cuda_stream
        if device.type == "cuda"
        else 0,
        batch,
        heads,
        num_tiles,
        width,
    )
    workspace = _KV_BLOCK_WORKSPACE_CACHE.get(key)
    if workspace is None:
        workspace = allocate()
        _KV_BLOCK_WORKSPACE_CACHE[key] = workspace
    return workspace


def _selection_chunk_tiles(batch: int, heads: int, num_tiles: int) -> int:
    """Query tiles per selection step, from the score-matrix byte budget."""
    row_bytes = batch * heads * num_tiles * 4
    budget = FASTH3_VSA_SELECTION_SCORE_BYTES // max(row_bytes, 1)
    # At a large batch or head count a single query tile's scores can already
    # exceed the budget; one tile per step is then the smallest step there is.
    return max(1, min(num_tiles, budget))


def _kv_list_workspace(
    batch: int,
    heads: int,
    num_tiles: int,
    width: int,
    device: torch.device,
) -> torch.Tensor:
    """Single-buffer sibling of ``_kv_block_workspace`` for the Triton path.

    Same single-live-consumer contract and same lifetime; see that function.
    """
    def allocate():
        return torch.empty(
            (batch, heads, num_tiles, width), dtype=torch.int32, device=device
        )

    if torch.compiler.is_compiling():
        return allocate()

    key = (
        "list",
        device.type,
        device.index,
        torch.cuda.current_stream(device).cuda_stream
        if device.type == "cuda"
        else 0,
        batch,
        heads,
        num_tiles,
        width,
    )
    workspace = _KV_BLOCK_WORKSPACE_CACHE.get(key)
    if workspace is None:
        workspace = allocate()
        _KV_BLOCK_WORKSPACE_CACHE[key] = workspace
    return workspace


def build_h3_vsa_kv_blocks(
    pooled_query: torch.Tensor,
    pooled_key: torch.Tensor,
    metadata: MiniMaxH3VSAMetadata,
    sparsity: float = FASTH3_VSA_SPARSITY,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Select VSA-H3 key tiles straight into FlexAttention's KV block lists.

    Every query tile keeps all ``P`` prefix tiles plus ``K`` top-k video tiles,
    so the retained count is the constant ``P + K`` and the index tensors are
    ``[B, H, tiles, P + K]`` rather than ``[B, H, tiles, tiles]``. Returns
    ``(kv_num_blocks, kv_indices, full_kv_num_blocks, full_kv_indices)``, with
    the padded tiles in the first pair -- those are the only ones the padding
    ``mask_mod`` has to run on.
    """
    batch, heads, num_tiles, head_dim = pooled_query.shape
    if num_tiles != metadata.num_tiles:
        raise ValueError(
            "VSA-H3 pooled tensors must have one row per tile, expected "
            f"{metadata.num_tiles}, got {num_tiles}."
        )

    num_prefix = metadata.num_prefix_tiles
    video_topk = compute_h3_vsa_topk(sparsity, metadata.num_video_tiles)
    width = num_prefix + video_topk
    partial_start = metadata.first_partial_video_tile
    device = pooled_query.device

    kv_indices, full_kv_indices, kv_num_blocks, full_kv_num_blocks = (
        _kv_block_workspace(batch, heads, num_tiles, width, device)
    )

    prefix_partial_first = metadata.prefix_partial_first_keys.view(
        1, 1, 1, num_prefix
    )
    prefix_full_first = metadata.prefix_full_first_keys.view(1, 1, 1, num_prefix)

    pooled_key_t = pooled_key.transpose(-2, -1)
    scale = head_dim**-0.5
    chunk = _selection_chunk_tiles(batch, heads, num_tiles)
    for start in range(0, num_tiles, chunk):
        stop = min(start + chunk, num_tiles)
        scores = torch.matmul(pooled_query[:, :, start:stop], pooled_key_t)
        scores.mul_(scale)
        video = (
            scores[..., num_prefix:]
            .topk(video_topk, dim=-1)
            .indices.to(torch.int32)
        )
        video += num_prefix
        is_partial = video >= partial_start
        offset = num_tiles
        prefix_shape = (batch, heads, stop - start, num_prefix)

        for prefix_keys, video_keys, destination in (
            (
                prefix_partial_first,
                video + (~is_partial).to(torch.int32) * offset,
                kv_indices,
            ),
            (
                prefix_full_first,
                video + is_partial.to(torch.int32) * offset,
                full_kv_indices,
            ),
        ):
            # Offsetting the losing group past every real tile id makes one
            # ascending sort both group the row and keep it in tile order.
            keys = torch.cat(
                (prefix_keys.expand(prefix_shape), video_keys), dim=-1
            ).sort(dim=-1).values
            destination[:, :, start:stop] = torch.where(
                keys >= offset, keys - offset, keys
            )

        partial_count = is_partial.sum(dim=-1, dtype=torch.int32)
        partial_count += metadata.num_prefix_partial_tiles
        kv_num_blocks[:, :, start:stop] = partial_count
        full_kv_num_blocks[:, :, start:stop] = width - partial_count

    return kv_num_blocks, kv_indices, full_kv_num_blocks, full_kv_indices


def build_h3_vsa_kv_list(
    pooled_query: torch.Tensor,
    pooled_key: torch.Tensor,
    metadata: MiniMaxH3VSAMetadata,
    sparsity: float = FASTH3_VSA_SPARSITY,
) -> torch.Tensor:
    """Select VSA-H3 key tiles as one ascending ``[B, H, tiles, P + K]`` list.

    Same policy and same retained set as ``build_h3_vsa_kv_blocks``, without the
    full/partial split. The split only exists to keep FlexAttention's padding
    ``mask_mod`` off the full blocks; the Triton kernel masks padding per tile
    from the tile map instead, so it wants the plain sorted list and saves one
    sort per call.

    The list is padded out to ``FASTH3_VSA_KV_LIST_ALIGNMENT`` with the sentinel
    tile id ``num_tiles``, whose slots read as invalid.
    """
    batch, heads, num_tiles, head_dim = pooled_query.shape
    if num_tiles != metadata.num_tiles:
        raise ValueError(
            "VSA-H3 pooled tensors must have one row per tile, expected "
            f"{metadata.num_tiles}, got {num_tiles}."
        )

    num_prefix = metadata.num_prefix_tiles
    video_topk = compute_h3_vsa_topk(sparsity, metadata.num_video_tiles)
    width = num_prefix + video_topk
    alignment = FASTH3_VSA_KV_LIST_ALIGNMENT
    padded_width = -(-width // alignment) * alignment
    kv_indices = _kv_list_workspace(
        batch, heads, num_tiles, padded_width, pooled_query.device
    )
    if padded_width > width:
        kv_indices[..., width:] = num_tiles

    prefix = torch.arange(
        num_prefix, dtype=torch.int32, device=pooled_query.device
    ).view(1, 1, 1, num_prefix)
    pooled_key_t = pooled_key.transpose(-2, -1)
    scale = head_dim**-0.5
    chunk = _selection_chunk_tiles(batch, heads, num_tiles)
    for start in range(0, num_tiles, chunk):
        stop = min(start + chunk, num_tiles)
        scores = torch.matmul(pooled_query[:, :, start:stop], pooled_key_t)
        scores.mul_(scale)
        video = (
            scores[..., num_prefix:]
            .topk(video_topk, dim=-1)
            .indices.to(torch.int32)
        )
        video += num_prefix
        # Ascending order is not needed for correctness, only for the locality
        # of the kernel's key/value loads. Only the video half is sorted: the
        # prefix half is arange(num_prefix) and every video id is >= num_prefix,
        # so concatenating the two is already the sort of the whole.
        kv_indices[:, :, start:stop, :num_prefix] = prefix
        kv_indices[:, :, start:stop, num_prefix:width] = video.sort(dim=-1).values

    return kv_indices


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

    Inputs and output use padded BHSD tile order. The second return value is
    the dense pooled branch, left at one row per tile as ``[B, H, tiles, D]``;
    callers broadcast it over the tile's tokens and apply the checkpoint's
    learned gate before adding it to the sparse output. Expanding it here would
    cost a multi-GB materialisation at production video sizes.
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

    # Selection stays in fp32 so the retained tile set matches the reference
    # policy in build_h3_vsa_block_mask bit for bit.
    kv_num_blocks, kv_indices, full_kv_num_blocks, full_kv_indices = (
        build_h3_vsa_kv_blocks(pooled_query, pooled_key, metadata, sparsity)
    )
    flex_block_mask = BlockMask.from_kv_blocks(
        kv_num_blocks,
        kv_indices,
        full_kv_num_blocks,
        full_kv_indices,
        BLOCK_SIZE=metadata.tile_elements,
        mask_mod=metadata.padding_mask_mod,
        seq_lengths=(metadata.padded_seq_length, metadata.padded_seq_length),
        compute_q_blocks=False,
    )

    sparse_output = _flex_attention_call(
        query,
        key,
        value,
        block_mask=flex_block_mask,
        kernel_options=FASTH3_VSA_KERNEL_OPTIONS,
    )

    # The compression branch is dense attention over the tile means. Pooling
    # stays fp32 so the retained tile set is exactly reproducible, but this
    # branch runs in the model dtype: that keeps the [tiles, tiles] probability
    # matrix inside a flash kernel instead of materialising it, which fp32
    # inputs would force. Measured against an fp32 reference at the production
    # tile count, the downcast moves the result by at most one bf16 ULP of the
    # value the gate multiplies, for roughly a third off the branch's time.
    compressed = F.scaled_dot_product_attention(
        pooled_query.to(query.dtype),
        pooled_key.to(query.dtype),
        pooled_value.to(query.dtype),
    )
    return sparse_output, compressed


def h3_vsa_triton_is_usable(device: torch.device) -> bool:
    """Whether the hand-written kernel can run on ``device``."""
    from . import triton_kernel as vsa_h3_triton

    # The kernel is launched on the tensors' own device, so a CPU tensor cannot
    # use it however Triton is built.
    return device.type == "cuda" and vsa_h3_triton.is_available()


def h3_vsa_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    gate: torch.Tensor,
    metadata: MiniMaxH3VSAMetadata,
    sparsity: float = FASTH3_VSA_SPARSITY,
    *,
    use_triton: bool,
) -> torch.Tensor:
    """VSA-H3 attention, gate applied, everything in packed row order.

    ``query``/``key``/``value``, ``gate`` and the result are all packed
    ``[B, H, S, D]``.

    The Triton kernel reads packed rows through the tile map and folds the
    compression branch and the gate into its epilogue, so tile order is never
    materialised. FlexAttention needs the padded tile buffers, so that path
    builds them. Both select the same key tiles.

    ``use_triton`` has no default: the attention backend the caller selected is
    what decides, and callers that check ``h3_vsa_triton_is_usable`` first are
    the ones that may fall back.
    """
    if use_triton:
        from .triton_kernel import (
            triton_h3_vsa_attention,
            triton_pool_h3_vsa_tiles,
        )

        pooled_query = triton_pool_h3_vsa_tiles(query, metadata)
        pooled_key = triton_pool_h3_vsa_tiles(key, metadata)
        pooled_value = triton_pool_h3_vsa_tiles(value, metadata)
        kv_indices = build_h3_vsa_kv_list(
            pooled_query, pooled_key, metadata, sparsity
        )
        # Model dtype, for the reason given in flex_h3_vsa_attention.
        compressed = F.scaled_dot_product_attention(
            pooled_query.to(query.dtype),
            pooled_key.to(query.dtype),
            pooled_value.to(query.dtype),
        )
        return triton_h3_vsa_attention(
            query, key, value, kv_indices, compressed, gate, metadata
        )

    sparse_output, compressed = flex_h3_vsa_attention(
        tile_h3_vsa_bhsd(query, metadata),
        tile_h3_vsa_bhsd(key, metadata),
        tile_h3_vsa_bhsd(value, metadata),
        metadata,
        sparsity,
    )
    packed_output = untile_h3_vsa_bhsd(sparse_output, metadata)
    return packed_output + (
        compressed.to(packed_output.dtype).index_select(
            2, metadata.packed_token_tile
        )
        * gate
    )
