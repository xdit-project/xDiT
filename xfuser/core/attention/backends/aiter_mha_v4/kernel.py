"""MHA v4 launchers. Imported when one of the family is selected, so AITER is
present and its enums can be translated once here rather than per call."""

from dataclasses import dataclass
from typing import Optional

import torch

from aiter import dtypes
from aiter.ops.mha_v4 import (
    AttentionFormat,
    AttentionPack,
    AttentionScaleMode,
    mha_v4,
    mha_v4_packed,
    native_fp8_format,
)

from xfuser.config.attention_a2a import AttentionA2AConfig
from xfuser.core.attention.numerics.layout import (
    from_bshd,
    make_contiguous,
    to_bshd,
)
from xfuser.core.attention.requirements import PARAM, device_arch
from xfuser.core.attention.sparsity.sparge import (
    SpargeConfig,
    build_block_mask,
    cost_sink_from,
    restore_sparge_output,
)
from xfuser.core.attention.spec import AttnCall
from xfuser.core.distributed.runtime_state import get_runtime_state, runtime_state_is_initialized

from .spec import FORMATS, Fmt, MhaV4Format, Scale

FORMAT = {f: getattr(AttentionFormat, f.name) for f in Fmt if f is not Fmt.NATIVE_FP8}
FORMAT[Fmt.NATIVE_FP8] = native_fp8_format()
SCALE = {s: getattr(AttentionScaleMode, s.name) for s in Scale}
_PACKED_FORMAT = {
    "bf16": Fmt.BF16,
    "int8": Fmt.INT8,
    "e4m3": Fmt.NATIVE_FP8,
    "mxfp8": Fmt.NATIVE_FP8,
    "mxfp4": Fmt.MXFP4,
    "mxfp6": Fmt.MXFP6,
}
_PACKED_SCALE = {
    "f32_per_tensor": Scale.F32_PER_TENSOR,
    "f32_per_channel": Scale.F32_PER_CHANNEL,
    "e8m0_per_1x32": Scale.E8M0_PER_1X32,
}


@dataclass(frozen=True)
class SparseGeometry:
    """How one format's sorted-sparse row cuts and bounds the keys.

    ``block_tile`` is the (q_tile, kv_tile) the Sparge mask must be cut at. The
    rows do not agree on it: at 256 query rows gfx950 routes BF16 and BF16/FP8 on
    64 keys and the rest on 128. ``ragged_kv`` says the row masks the keys past a
    short last block itself, so the tail needs no zero pad -- which is not just a
    copy saved, since a zero key inside a selected block is attended at score 0
    rather than skipped.
    """

    block_tile: tuple[int, int]
    ragged_kv: bool = False


def _legacy_kv_tile() -> int:
    # Not hoisted: optional, with an arch-derived fallback below.
    try:
        from aiter.ops.mha_v4 import mha_v4_kv_tile

        return int(mha_v4_kv_tile())
    except ImportError:
        return 64 if "gfx942" in device_arch() else 128


def _sparse_geometries(fmt: MhaV4Format) -> dict[Optional[tuple[int, int]], SparseGeometry]:
    """Every tile this GPU serves ``fmt``'s sparse row at, keyed by tile, plus
    its default under None. Empty when the format has no sparse row here."""
    # Not hoisted: older AITER builds have none of these, and answer one KV tile
    # for every row at a fixed 256 query rows.
    try:
        from aiter.ops.mha_v4 import (
            MHA_V4_SPARSE_MODE,
            AttentionPack,
            mha_v4_block_tile,
            mha_v4_block_tiles,
            mha_v4_operands,
            mha_v4_ragged_kv,
            scale_modes_for_formats,
        )
    except ImportError:
        return {None: SparseGeometry((256, _legacy_kv_tile()))}

    qk, v = FORMAT[fmt.qk], FORMAT[fmt.v]
    if fmt.qk_scale is None:
        scales = scale_modes_for_formats(qk, qk, v)
    else:
        scales = (SCALE[fmt.qk_scale], SCALE[fmt.qk_scale], SCALE[fmt.v_scale])
    # AITER repacks an MX V for the FP6 P operand on every row that has one,
    # dense and sparse alike, so the row is only found under that pack.
    pack = AttentionPack.V_FOR_FP6_P if fmt.v in (Fmt.MXFP6, Fmt.MXFP4) else AttentionPack.DEFAULT
    operands = mha_v4_operands(qk, qk, v, *scales, pack)
    default = mha_v4_block_tile(operands, MHA_V4_SPARSE_MODE)
    if not default[1]:
        return {}
    geometries = {
        tile: SparseGeometry(tile, mha_v4_ragged_kv(operands, MHA_V4_SPARSE_MODE, tile))
        for tile in mha_v4_block_tiles(operands, MHA_V4_SPARSE_MODE)
    }
    geometries[None] = geometries[default]
    return geometries


# Read once per format, here, because the kernel module is imported at backend
# selection: a manifest query per call would sit inside the traced region. A
# format this GPU has no sparse row for has no entries; its spec is unselectable
# there, so the lookup never happens.
SPARSE_GEOMETRY = {fmt.name: _sparse_geometries(fmt) for fmt in FORMATS if fmt.sparge_on is not None}

# Whether this AITER takes the mask's tile explicitly rather than assuming the
# recipe's default.
HAS_BLOCK_TILE = PARAM("aiter.ops.mha_v4:mha_v4", "block_tile").satisfied()


def check_sparse_block_size(fmt: MhaV4Format, block_size: tuple[int, int]) -> None:
    """Refuse a --sparseattn_block_size that ``fmt``'s sparse row is not built at."""
    served = sorted(tile for tile in SPARSE_GEOMETRY[fmt.name] if tile is not None)
    if HAS_BLOCK_TILE and tuple(block_size) in served:
        return
    q_tile, kv_tile = block_size
    if not served or not HAS_BLOCK_TILE:
        offered = "this AITER cannot select a tile"
    else:
        offered = "it serves " + ", ".join(f"{q}x{kv}" for q, kv in served)
    raise ValueError(
        f"--sparseattn_block_size {q_tile}x{kv_tile} has no AITER_{fmt.name}_SPARGE kernel on "
        f"{device_arch()}: {offered}."
    )


def _sparse_geometry(fmt: MhaV4Format) -> SparseGeometry:
    block_size = None
    if runtime_state_is_initialized():
        block_size = get_runtime_state().runtime_config.sparseattn_block_size
    return SPARSE_GEOMETRY[fmt.name][None if block_size is None else tuple(block_size)]


# Whether this AITER can be told how many keys each batch really has. Read once,
# here, because the kernel module is imported at backend selection -- evaluating
# it per call would put a signature probe inside the traced region.
HAS_SEQLENS_K = PARAM("aiter.ops.mha_v4:mha_v4", "seqlens_k").satisfied()
HAS_KV_SPLITS = PARAM("aiter.ops.mha_v4:mha_v4", "kv_splits").satisfied()


def kv_splits_kwargs():
    """``kv_splits`` for a dense mha_v4 call, from --mha_v4_kv_splits.

    Omitted on an AITER that predates split-KV, where "auto" means unsplit anyway.
    """
    splits = 1
    if runtime_state_is_initialized():
        splits = get_runtime_state().runtime_config.mha_v4_kv_splits
    if HAS_KV_SPLITS:
        return {"kv_splits": splits}
    if splits not in ("auto", 1):
        raise NotImplementedError(
            f"--mha_v4_kv_splits {splits} needs an AITER whose mha_v4 takes kv_splits; please update AITER"
        )
    return {}


def _launch(
    q, k, v, fmt: MhaV4Format, block_mask=None, block_tile=None, seqlens_k=None, return_lse=False, split_kv=False
):
    """One MHA v4 launch. Tensors are BSHD. Returns (output, lse-or-None)."""
    kwargs = kv_splits_kwargs() if split_kv else {}
    if fmt.qk_scale is not None:
        kwargs |= {
            "q_scale_mode": SCALE[fmt.qk_scale],
            "k_scale_mode": SCALE[fmt.qk_scale],
            "v_scale_mode": SCALE[fmt.v_scale],
        }
    if block_tile is not None and HAS_BLOCK_TILE:
        kwargs["block_tile"] = block_tile
    if seqlens_k is not None:
        kwargs["seqlens_k"] = seqlens_k
    if return_lse:
        kwargs["return_lse"] = True
    qk = FORMAT[fmt.qk]
    result = mha_v4(q, k, v, qk, qk, FORMAT[fmt.v], block_mask=block_mask, **kwargs)
    return result if return_lse else (result, None)


def mha_v4_attention_a2a_packed(
    q,
    k,
    v,
    q_scale,
    k_scale,
    v_scale,
    profile: str,
    softmax_scale: float,
):
    """Launch MHA v4 over Attention A2A's pre-quantized public buffers.

    This deliberately remains a normal Python function. Inductor traces through
    the static recipe selection and tensor views down to AITER's kernel custom
    op, matching the pre-quantized path on main.
    """
    config = AttentionA2AConfig(profile=profile)
    qk_codec, _, v_codec = config.consumer_codecs
    qk_format = FORMAT[_PACKED_FORMAT[qk_codec]]
    v_format = FORMAT[_PACKED_FORMAT[v_codec]]

    if qk_codec == "int8":
        q = q.view(torch.int8)
        k = k.view(torch.int8)
    elif qk_codec in ("e4m3", "mxfp8"):
        q = q.view(dtypes.fp8)
        k = k.view(dtypes.fp8)
    if v_codec in ("e4m3", "mxfp8"):
        v = v.view(dtypes.fp8)

    q_scale_mode, k_scale_mode, v_scale_mode = tuple(SCALE[_PACKED_SCALE[mode]] for mode in config.scale_modes)
    v_pack = AttentionPack.V_FOR_FP6_P if config.v_pack == "fp6_p" else AttentionPack.DEFAULT
    return mha_v4_packed(
        q,
        k,
        v,
        q_scale,
        k_scale,
        v_scale,
        qk_format,
        qk_format,
        v_format,
        q_scale_mode,
        k_scale_mode,
        v_scale_mode,
        v_pack=v_pack,
        softmax_scale=softmax_scale,
    )


def _shorten_keys(key, value, call: AttnCall, fmt: MhaV4Format):
    """Fold a key pack into a dense call. Returns (key, value, seqlens_k), BSHD.

    MHA v4 has no key-padding mask and does not need one. One sequence's valid
    keys are simply a shorter K/V, so attending over them is exact rather than
    approximate. Several are regrouped into a padded batch whose true lengths
    travel in seqlens_k, which the kernels read per batch, so the padding is
    never visited.

    ``valid_kv_len`` is one length for the whole call, so it can only describe a
    single sequence. A batch of several is gathered even when it declares one,
    because slicing would cut every row to the longest row's length and leave
    the shorter ones attending over their own pad. Nothing rejects that: the
    declaration passes both checks, since the longest segment genuinely is the
    valid count for one of the rows.

    Q is left alone either way: it is never packed, and a key-side length would
    be wrong for cross attention, where the two sequences differ.
    """
    valid = call.attention_kwargs.get("valid_kv_len")
    if valid is not None and key.shape[0] == 1:
        # A declared trailing pad on one sequence, which accepts has already
        # checked describes the pack truthfully. One copy, where gathering
        # costs two.
        return key[:, :valid].contiguous(), value[:, :valid].contiguous(), None

    # Gathered over K's own shape rather than through layout.pack_kv, which
    # flattens K against the *query* length. Equal for self attention, but this
    # path is reached by cross attention too, where the two differ.
    batch, seq_len, heads, head_dim = key.shape
    flat = (batch * seq_len, heads, head_dim)
    indices = call.varlen.indices_k
    k_packed = torch.index_select(key.reshape(flat), 0, indices)
    v_packed = torch.index_select(value.reshape(flat), 0, indices)

    if batch == 1:
        return (
            k_packed.reshape(1, -1, heads, head_dim),
            v_packed.reshape(1, -1, heads, head_dim),
            None,
        )

    if not HAS_SEQLENS_K:
        raise NotImplementedError(
            "this AITER build cannot express per-batch key lengths for MHA v4, "
            "so varlen packed keys with batch size > 1 are unsupported"
        )
    if fmt.qk is not Fmt.BF16:
        # AITER rejects the other objects rather than attend over the padding.
        # Name the ones that work instead of relaying a message about formats
        # the caller never chose.
        raise NotImplementedError(
            f"MHA v4 carries per-batch key lengths only on its BF16 Q/K rows, "
            f"so {fmt.name} Q/K cannot serve a batch of several padded "
            "sequences. Use AITER_BF16 or AITER_BF16FP8 for this model, or a "
            "configuration whose per-call batch is one."
        )

    # Scatter the packed rows back into a [batch, max_k, ...] buffer, one
    # sequence per row, and hand the true lengths over. The zeros are never
    # visited, which is what makes this exact rather than approximate.
    device = k_packed.device
    cu_k = call.varlen.cu_seqlens_k.to(device=device, dtype=torch.int32)
    lengths = (cu_k[1:] - cu_k[:-1]).contiguous()
    counts = lengths.to(torch.int64)
    rows = torch.arange(k_packed.shape[0], device=device)
    slot = rows - torch.repeat_interleave(cu_k[:-1].to(torch.int64), counts)
    row = torch.repeat_interleave(torch.arange(batch, device=device), counts)
    shape = (batch, int(call.varlen.max_seqlen_k), heads, head_dim)
    key, value = k_packed.new_zeros(shape), v_packed.new_zeros(shape)
    key[row, slot] = k_packed
    value[row, slot] = v_packed
    return key, value, lengths


def mha_v4_dense(query, key, value, call: AttnCall, *, fmt: MhaV4Format):
    # K/V stay views until they are shortened, so the pad is dropped in the
    # same copy that makes them contiguous rather than in one after it. Both
    # routes out of _shorten_keys are contiguous already: a slice and a gather
    # each materialise, and the scatter writes into a fresh buffer.
    q = to_bshd(query, contiguous=True)
    k, v = to_bshd(key, value)
    seqlens_k = None
    if call.varlen is not None:
        k, v, seqlens_k = _shorten_keys(k, v, call, fmt)
    else:
        k, v = make_contiguous(k, v)

    # Only ring needs the log-sumexp; asking for it otherwise buys a write the
    # caller discards. The degree comes off the call rather than the process
    # group, so this stays traceable and testable without one.
    output, softmax_lse = _launch(
        q,
        k,
        v,
        fmt,
        seqlens_k=seqlens_k,
        return_lse=call.ring_world_size > 1,
        split_kv=True,
    )
    # The kernel writes LSE as [batch, heads, Sq], already the layout the ring
    # merge expects, so only O is permuted back.
    return from_bshd(output), softmax_lse


def mha_v4_sparge(query, key, value, call: AttnCall, *, fmt: MhaV4Format):
    geometry = _sparse_geometry(fmt)
    q_tile, kv_tile = geometry.block_tile
    q, k, v, state, block_mask = build_block_mask(
        query,
        key,
        value,
        is_causal=call.is_causal,
        config=SpargeConfig.from_kwargs(call.attention_kwargs),
        block_m=q_tile,
        block_n=kv_tile,
        ulysses_world_size=call.ulysses_world_size,
        cost_sink=cost_sink_from(call.attention_kwargs),
        pad_block_divisible=not geometry.ragged_kv,
    )
    q, k, v = to_bshd(q, k, v, contiguous=True)
    output, _ = _launch(q, k, v, fmt, block_mask=block_mask, block_tile=geometry.block_tile)
    return restore_sparge_output(from_bshd(output), state), None
