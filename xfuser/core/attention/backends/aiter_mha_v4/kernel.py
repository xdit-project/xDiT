"""MHA v4 launchers. Imported when one of the family is selected, so AITER is
present and its enums can be translated once here rather than per call."""

import torch

from aiter.ops.mha_v4 import (
    AttentionFormat,
    AttentionScaleMode,
    mha_v4,
    native_fp8_format,
)

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

from .spec import Fmt, MhaV4Format, Scale

FORMAT = {f: getattr(AttentionFormat, f.name) for f in Fmt if f is not Fmt.NATIVE_FP8}
FORMAT[Fmt.NATIVE_FP8] = native_fp8_format()
SCALE = {s: getattr(AttentionScaleMode, s.name) for s in Scale}


def _read_kv_tile() -> int:
    """Sparge's KV tile must match the kernel's sparse geometry."""
    # Not hoisted: optional, with an arch-derived fallback below.
    try:
        from aiter.ops.mha_v4 import mha_v4_kv_tile

        return int(mha_v4_kv_tile())
    except ImportError:
        return 64 if "gfx942" in device_arch() else 128


KV_TILE = _read_kv_tile()

# Whether this AITER can be told how many keys each batch really has. Read once,
# here, because the kernel module is imported at backend selection -- evaluating
# it per call would put a signature probe inside the traced region.
HAS_SEQLENS_K = PARAM("aiter.ops.mha_v4:mha_v4", "seqlens_k").satisfied()


def _launch(q, k, v, fmt: MhaV4Format, block_mask=None, seqlens_k=None):
    """One MHA v4 launch. Tensors are BSHD."""
    kwargs = {}
    if fmt.qk_scale is not None:
        kwargs = {
            "q_scale_mode": SCALE[fmt.qk_scale],
            "k_scale_mode": SCALE[fmt.qk_scale],
            "v_scale_mode": SCALE[fmt.v_scale],
        }
    if seqlens_k is not None:
        kwargs["seqlens_k"] = seqlens_k
    qk = FORMAT[fmt.qk]
    return mha_v4(q, k, v, qk, qk, FORMAT[fmt.v],
                  block_mask=block_mask, **kwargs)


def _shorten_keys(key, value, call: AttnCall, fmt: MhaV4Format):
    """Fold a key pack into a dense call. Returns (key, value, seqlens_k), BSHD.

    MHA v4 has no key-padding mask and does not need one. One sequence's valid
    keys are simply a shorter K/V, so attending over them is exact rather than
    approximate. Several are regrouped into a padded batch whose true lengths
    travel in seqlens_k, which the kernels read per batch, so the padding is
    never visited.

    Q is left alone either way: it is never packed, and a key-side length would
    be wrong for cross attention, where the two sequences differ.
    """
    valid = call.attention_kwargs.get("valid_kv_len")
    if valid is not None:
        # A declared trailing pad, which accepts has already checked describes
        # the pack truthfully. Slicing costs no copy where gathering costs two.
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
    return from_bshd(_launch(q, k, v, fmt, seqlens_k=seqlens_k)), None


def mha_v4_sparge(query, key, value, call: AttnCall, *, fmt: MhaV4Format):
    q, k, v, state, block_mask = build_block_mask(
        query, key, value,
        is_causal=call.is_causal,
        config=SpargeConfig.from_kwargs(call.attention_kwargs),
        block_m=256, block_n=KV_TILE,
        ulysses_world_size=call.ctx.ulysses_world_size,
        cost_sink=cost_sink_from(call.attention_kwargs),
        pad_block_divisible=True,
    )
    q, k, v = to_bshd(q, k, v, contiguous=True)
    output = _launch(q, k, v, fmt, block_mask=block_mask)
    return restore_sparge_output(from_bshd(output), state), None


