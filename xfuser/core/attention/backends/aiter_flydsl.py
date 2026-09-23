"""
AITER FlyDSL: a gfx1201 MHA kernel
"""

import torch

from xfuser.core.attention.backends.sdpa import sdpa_flash
from xfuser.core.attention.constraints import NO_DROPOUT
from xfuser.core.attention.layout import from_bshd, to_bshd
from xfuser.core.attention.requirements import ARCH, SYMBOL
from xfuser.core.attention.spec import AttentionBackendType, AttnCall, Spec

_FLYDSL = "aiter.ops.flydsl:flydsl_flash_attn_func"


def _register_ops():
    """The custom ops still live in the legacy module. Importing it here (at
    call time, to avoid a cycle) guarantees they are registered. Phase 6 moves
    the registrations into this module when that file goes."""
    import xfuser.core.distributed.attention_backend  # noqa: F401


def _dispatch(query, key, value, call: AttnCall, op):
    """
    Anything the kernel would reject takes SDPA instead: causal cross-attn
    (ambiguous alignment, never occurs in diffusion), GQA or head-dim
    mismatches (the kernel is single-NUM_HEADS MHA), and head_dim outside its
    >=64, %32==0 tile constraint.
    """
    is_cross = query.shape[2] != key.shape[2]
    head_dim = query.shape[3]

    if (
        (is_cross and call.is_causal)
        or query.shape[1] != key.shape[1]
        or head_dim != key.shape[3]
        or head_dim < 64
        or head_dim % 32 != 0
    ):
        return sdpa_flash(query, key, value, call)

    _register_ops()
    q, k, v = to_bshd(query, key, value, contiguous=True)
    return from_bshd(op(q, k, v, call.is_causal)), None


def flydsl(query, key, value, call: AttnCall):
    _register_ops()
    return _dispatch(query, key, value, call, torch.ops.xfuser.flydsl_attn)


def flydsl_fp8(query, key, value, call: AttnCall):
    _register_ops()
    kwargs = call.attention_kwargs
    if kwargs.get("pre_quantized", False):
        if call.varlen is not None:
            raise NotImplementedError(
                "fp8 comms pre-quantized attention does not support varlen packing; "
                "the indices_k mask would be silently dropped and dense attention "
                "would run over padded keys."
            )
        q, k, v = to_bshd(query, key, value, contiguous=True)
        out = torch.ops.xfuser.flydsl_attn_fp8_prequant(
            q, k, v,
            kwargs["q_descale"], kwargs["k_descale"], kwargs["v_descale"],
            call.is_causal,
        )
        return from_bshd(out), None

    return _dispatch(query, key, value, call, torch.ops.xfuser.flydsl_attn_fp8)


SPECS = [
    Spec(AttentionBackendType.AITER_FLYDSL, impl=flydsl,
         requires=SYMBOL(_FLYDSL) & ARCH("gfx1201")),

    Spec(AttentionBackendType.AITER_FLYDSL_FP8, impl=flydsl_fp8, low_precision=True,
         accepts=NO_DROPOUT,
         requires=SYMBOL(_FLYDSL) & SYMBOL("aiter.ops.flydsl:flydsl_fp8_quant")
                & ARCH("gfx1201")),
]
