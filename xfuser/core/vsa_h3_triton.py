# SPDX-License-Identifier: Apache-2.0
"""Portable 64-token block-sparse attention kernel for VSA-H3.

FlexAttention can express VSA-H3's mask but not its best schedule: the Triton
template asserts ``SPARSE_KV_BLOCK_SIZE >= BLOCK_N``, so a 64-token tile pins
the kernel to a 64-wide N tile and to one such tile per loop iteration, which
leaves the loop's index load and pointer arithmetic on the critical path. This
kernel keeps the 64-wide N tile but issues ``TILES_PER_ITER`` of them per
iteration from a statically unrolled body, so their loads and dots overlap and
the loop overhead is amortised across the group.

Three other things fall out of writing the kernel directly:

* every query tile keeps exactly ``P + K`` tiles, so the block count is a loop
  bound rather than a per-row load and there is no full/partial block list;
* the kernel reads packed rows through the tile map itself, so the padded tile
  buffers FlexAttention needs are never built -- no gather in, no un-tile out;
* the compression branch and the learned gate fold into the epilogue.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # CPU-only installs; callers fall back to FlexAttention.
    triton = None
    tl = None


_LOG2E = 1.4426950408889634


def is_available() -> bool:
    return triton is not None and torch.cuda.is_available()


if triton is not None:

    # Scores are masked with a large finite value rather than -inf so that an
    # all-padding N tile leaves the running max finite instead of yielding NaN.
    _MASK_SCORE = tl.constexpr(-1.0e30)

    def _configs():
        """Autotune space, pruned to the region that measured well.

        The kernel is bound by key/value load bandwidth rather than by matrix
        issue, so the decisive knob is occupancy -- it moved the production
        geometry far more than any block-shape change did. ``num_stages > 1``
        was uniformly worse: prefetching into shared memory competes with the
        occupancy the loads need. ``waves_per_eu`` is an AMD backend hint, so it
        only joins the space on ROCm.

        The space stops at a 64-row query tile on purpose. At the production
        geometry the kernel sits at both its bandwidth and its matrix-issue
        ceiling simultaneously, so trading one for the other is not a win.
        Letting neighbouring query tiles share one pass over the union of their
        key lists -- which divides the traffic but needs a 128-row tile -- was
        built and measured, and break-even sits near a union ratio of 0.58.
        Real runs of this checkpoint measure 0.63-0.67, i.e. the losing side.
        The ratio is a property of the checkpoint's selection coherence, not of
        the device: random tensors give 0.87 and would have predicted a win.
        """
        occupancy_hints = (
            [{"waves_per_eu": waves} for waves in (1, 2, 4)]
            if torch.version.hip
            else [{}]
        )
        return [
            triton.Config(
                {"TILES_PER_ITER": tiles_per_iter, **hint},
                num_warps=num_warps,
                num_stages=1,
            )
            for tiles_per_iter in (1, 2, 4)
            for num_warps in (2, 4)
            for hint in occupancy_hints
        ]

    @triton.jit
    def _vsa_h3_pool_kernel(
        X,
        TILED_TO_PACKED,
        SIZES,
        Out,
        stride_xz,
        stride_xh,
        stride_xm,
        stride_oz,
        stride_oh,
        stride_om,
        packed_seq_length,
        HEADS: tl.constexpr,
        TILE: tl.constexpr,
        HEAD_DIM: tl.constexpr,
    ):
        """Tile means of a packed ``[B, H, S, D]`` tensor, in fp32.

        Same gather as the attention kernel, so pooling costs one read of the
        sequence rather than a read plus a full padded-buffer write.
        """
        tile = tl.program_id(0)
        batch_head = tl.program_id(1)
        batch = batch_head // HEADS
        head = batch_head % HEADS

        rows = tl.load(TILED_TO_PACKED + tile * TILE + tl.arange(0, TILE))
        keep = rows < packed_seq_length
        values = tl.load(
            X
            + batch * stride_xz
            + head * stride_xh
            + rows[:, None] * stride_xm
            + tl.arange(0, HEAD_DIM)[None, :],
            mask=keep[:, None],
            other=0.0,
        )
        pooled = tl.sum(values.to(tl.float32), 0) / tl.load(SIZES + tile).to(
            tl.float32
        )
        tl.store(
            Out
            + batch * stride_oz
            + head * stride_oh
            + tile * stride_om
            + tl.arange(0, HEAD_DIM),
            pooled,
        )

    @triton.autotune(
        configs=_configs(),
        key=["num_kv_tiles", "HEAD_DIM", "TILE"],
    )
    @triton.jit
    def _vsa_h3_attention_kernel(
        Q,
        K,
        V,
        KV_IDX,
        TILED_TO_PACKED,
        COMPRESSED,
        GATE,
        Out,
        stride_qz,
        stride_qh,
        stride_qm,
        stride_iz,
        stride_ih,
        stride_im,
        stride_cz,
        stride_ch,
        stride_cm,
        stride_gz,
        stride_gh,
        stride_gm,
        stride_oz,
        stride_oh,
        stride_om,
        num_kv_tiles,
        packed_seq_length,
        qk_scale,
        HEADS: tl.constexpr,
        TILE: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        TILES_PER_ITER: tl.constexpr,
    ):
        query_tile = tl.program_id(0)
        batch_head = tl.program_id(1)
        batch = batch_head // HEADS
        head = batch_head % HEADS

        offs_t = tl.arange(0, TILE)
        offs_d = tl.arange(0, HEAD_DIM)

        # int64 for the per-(batch, head) bases: a full-size model at a long
        # sequence puts batch * stride_qz past 2^31, and these are scalars, so
        # widening them costs nothing. The per-row offsets below stay int32 and
        # are on the hot path; they span one head, and the launcher rejects any
        # layout that would let that reach 2^31.
        batch64 = batch.to(tl.int64)
        head64 = head.to(tl.int64)
        qkv_base = batch64 * stride_qz + head64 * stride_qh
        # Q/K/V are in packed row order; the tile map is the only thing that
        # puts them in tile order, and it costs one small cached load per tile
        # instead of three full-sequence gather passes by the caller. A slot
        # past the end of the sequence is padding.
        query_rows = tl.load(TILED_TO_PACKED + query_tile * TILE + offs_t)
        keep = query_rows < packed_seq_length
        query = tl.load(
            Q + qkv_base + query_rows[:, None] * stride_qm + offs_d[None, :],
            mask=keep[:, None],
            other=0.0,
        )

        index_base = KV_IDX + batch64 * stride_iz + head64 * stride_ih + (
            query_tile * stride_im
        )

        running_max = tl.full([TILE], _MASK_SCORE, tl.float32)
        running_sum = tl.zeros([TILE], tl.float32)
        accumulator = tl.zeros([TILE, HEAD_DIM], tl.float32)

        # The scale and the mask value both arrive as Python floats, which
        # Triton versions type differently in promotion: one of them widens the
        # score tile to fp64, which fails the loop's type check on running_max.
        # Neither needs the range, so the score path below pins fp32 explicitly.
        # Hoisting the scale as qk_scale.to(tl.float32) looks like the tidier
        # way to do it and is not: it is correct in eager and silently wrong
        # once Inductor compiles the call, so the cast goes on the product.
        mask_score = tl.full([TILE, TILE], _MASK_SCORE, tl.float32)

        # The index list is padded to a multiple of TILES_PER_ITER with a
        # sentinel tile whose slots are all invalid, so the loop needs no tail
        # guard: a sentinel leaves the running max untouched and adds zero
        # weight.
        for start in range(0, num_kv_tiles, TILES_PER_ITER):
            for step in tl.static_range(TILES_PER_ITER):
                # Scalar tile id, so the rows are a contiguous range and the
                # key/value loads stay wide block loads rather than per-lane
                # gathers. Unrolling by TILES_PER_ITER only gives the scheduler
                # more independent loads to overlap; the softmax stays online
                # per tile, which costs one rescale of the accumulator per
                # tile against the tile's own 2 MFLOP of matrix work.
                key_tile = tl.load(index_base + start + step)
                # Boundary tiles hold fewer than TILE real tokens; the tile map
                # sends their slots past the end of the sequence, which both
                # masks the load and keeps them out of the softmax.
                key_rows = tl.load(TILED_TO_PACKED + key_tile * TILE + offs_t)
                valid = key_rows < packed_seq_length
                key = tl.load(
                    K
                    + qkv_base
                    + key_rows[:, None] * stride_qm
                    + offs_d[None, :],
                    mask=valid[:, None],
                    other=0.0,
                )
                scores = (tl.dot(query, tl.trans(key)) * qk_scale).to(tl.float32)
                scores = tl.where(valid[None, :], scores, mask_score)

                tile_max = tl.maximum(running_max, tl.max(scores, 1))
                rescale = tl.exp2(running_max - tile_max)
                probabilities = tl.exp2(scores - tile_max[:, None])
                running_sum = running_sum * rescale + tl.sum(probabilities, 1)
                accumulator = accumulator * rescale[:, None]

                value = tl.load(
                    V
                    + qkv_base
                    + key_rows[:, None] * stride_qm
                    + offs_d[None, :],
                    mask=valid[:, None],
                    other=0.0,
                )
                accumulator = tl.dot(
                    probabilities.to(value.dtype), value, accumulator
                )
                running_max = tile_max

        accumulator = accumulator / running_sum[:, None]

        # Epilogue: add the gated compression branch and write straight to
        # packed row order, so the caller needs no separate gather, multiply or
        # un-tile pass over the full sequence.
        packed_rows = query_rows
        compressed = tl.load(
            COMPRESSED
            + batch64 * stride_cz
            + head64 * stride_ch
            + query_tile * stride_cm
            + offs_d
        )
        gate = tl.load(
            GATE
            + batch64 * stride_gz
            + head64 * stride_gh
            + packed_rows[:, None] * stride_gm
            + offs_d[None, :],
            mask=keep[:, None],
            other=0.0,
        )
        accumulator += compressed[None, :].to(tl.float32) * gate.to(tl.float32)
        tl.store(
            Out
            + batch64 * stride_oz
            + head64 * stride_oh
            + packed_rows[:, None] * stride_om
            + offs_d[None, :],
            accumulator.to(Out.dtype.element_ty),
            mask=keep[:, None],
        )


def triton_pool_h3_vsa_tiles(
    tensor: torch.Tensor,
    metadata,
) -> torch.Tensor:
    """Pool a packed ``[B, H, S, D]`` tensor to fp32 ``[B, H, tiles, D]``."""
    if triton is None:
        raise RuntimeError("VSA-H3's Triton kernel needs Triton installed.")
    batch, heads, sequence_length, head_dim = tensor.shape
    if sequence_length != metadata.total_seq_length:
        raise ValueError(
            "VSA-H3 pooling expects packed [B, H, S, D] with S="
            f"{metadata.total_seq_length}, got {tuple(tensor.shape)}."
        )
    pooled = tensor.new_empty(
        (batch, heads, metadata.num_tiles, head_dim), dtype=torch.float32
    )
    _vsa_h3_pool_kernel[(metadata.num_tiles, batch * heads)](
        tensor,
        metadata.tiled_to_packed_row,
        metadata.variable_block_sizes,
        pooled,
        tensor.stride(0),
        tensor.stride(1),
        tensor.stride(2),
        pooled.stride(0),
        pooled.stride(1),
        pooled.stride(2),
        sequence_length,
        HEADS=heads,
        TILE=metadata.tile_elements,
        HEAD_DIM=head_dim,
        num_warps=4,
    )
    return pooled


def triton_h3_vsa_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    kv_indices: torch.Tensor,
    compressed: torch.Tensor,
    gate: torch.Tensor,
    metadata,
) -> torch.Tensor:
    """Sparse attention plus gated compression, all in packed row order.

    ``query``/``key``/``value``, ``gate`` and the ``[B, H, S, D]`` result are
    all packed; the kernel reaches tile order through the metadata's tile map.
    """
    if triton is None:
        raise RuntimeError("VSA-H3's Triton kernel needs Triton installed.")

    batch, heads, sequence_length, head_dim = query.shape
    if sequence_length != metadata.total_seq_length:
        raise ValueError(
            "VSA-H3 Triton attention expects packed BHSD tensors with S="
            f"{metadata.total_seq_length}, got {tuple(query.shape)}."
        )
    if not (query.shape == key.shape == value.shape):
        raise ValueError(
            "VSA-H3 Triton attention expects equally shaped q/k/v, got "
            f"q={tuple(query.shape)}, k={tuple(key.shape)}, "
            f"v={tuple(value.shape)}."
        )
    # The kernel addresses q, k and v off one set of strides, so a k or v that
    # is laid out differently than q would be read at the wrong offsets.
    if not (query.stride() == key.stride() == value.stride()):
        raise ValueError(
            "VSA-H3 Triton attention expects q/k/v to share a layout, got "
            f"q={query.stride()}, k={key.stride()}, v={value.stride()}."
        )
    # Head-dim rows are loaded as a contiguous span.
    for name, tensor in (
        ("query", query),
        ("compressed", compressed),
        ("gate", gate),
        ("kv_indices", kv_indices),
    ):
        if tensor.stride(-1) != 1:
            raise ValueError(
                f"VSA-H3 Triton attention expects {name} contiguous in its last "
                f"dimension, got strides {tensor.stride()}."
            )
    # The kernel keeps its per-row offsets in int32; only the per-(batch, head)
    # bases are widened. One head's rows fit that for any layout worth running,
    # but a strided view could in principle span further, so check rather than
    # assume. The output is allocated below and is contiguous by construction.
    # The row index runs to sequence_length, not sequence_length - 1: padded
    # slots point one past the end, and the address is formed before the mask
    # drops the load.
    row_span = max(
        sequence_length * tensor.stride(2) + head_dim
        for tensor in (query, gate)
    )
    if row_span >= 2**31:
        raise ValueError(
            "VSA-H3 Triton attention addresses one head's rows in int32; this "
            f"layout spans {row_span} elements. Pass contiguous [B, H, S, D] "
            "tensors."
        )

    output = query.new_empty((batch, heads, sequence_length, head_dim))
    _vsa_h3_attention_kernel[(metadata.num_tiles, batch * heads)](
        query,
        key,
        value,
        kv_indices,
        metadata.tiled_to_packed_row,
        compressed,
        gate,
        output,
        query.stride(0),
        query.stride(1),
        query.stride(2),
        kv_indices.stride(0),
        kv_indices.stride(1),
        kv_indices.stride(2),
        compressed.stride(0),
        compressed.stride(1),
        compressed.stride(2),
        gate.stride(0),
        gate.stride(1),
        gate.stride(2),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        kv_indices.shape[-1],
        sequence_length,
        head_dim**-0.5 * _LOG2E,
        HEADS=heads,
        TILE=metadata.tile_elements,
        HEAD_DIM=head_dim,
    )
    return output
