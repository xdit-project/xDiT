"""MHA v4 launchers. Imported when one of the family is selected, so AITER is
present and its enums can be translated once here rather than per call."""

from aiter.ops.mha_v4 import (
    AttentionFormat,
    AttentionScaleMode,
    mha_v4,
    native_fp8_format,
)

from xfuser.core.attention.numerics.layout import from_bshd, to_bshd
from xfuser.core.attention.requirements import device_arch
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

def _launch(q, k, v, fmt: MhaV4Format, block_mask=None):
    """One MHA v4 launch. Tensors are BSHD."""
    kwargs = {}
    if fmt.qk_scale is not None:
        kwargs = {
            "q_scale_mode": SCALE[fmt.qk_scale],
            "k_scale_mode": SCALE[fmt.qk_scale],
            "v_scale_mode": SCALE[fmt.v_scale],
        }
    qk = FORMAT[fmt.qk]
    return mha_v4(q, k, v, qk, qk, FORMAT[fmt.v],
                  block_mask=block_mask, **kwargs)


def mha_v4_dense(query, key, value, call: AttnCall, *, fmt: MhaV4Format):
    # A declared trailing pad is served by shortening K/V rather than masking;
    # accepts has already established the pad is trailing and its length valid.
    # Q is left alone: it is never packed, and trimming it by a key-side length
    # would be wrong for cross attention, where the two sequences differ.
    if call.varlen is not None:
        valid = call.attention_kwargs["valid_kv_len"]
        key, value = key[:, :, :valid], value[:, :, :valid]

    q, k, v = to_bshd(query, key, value, contiguous=True)
    return from_bshd(_launch(q, k, v, fmt)), None


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


