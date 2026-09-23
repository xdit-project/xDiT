"""
AITER FlyDSL: a gfx1201 MHA kernel
"""

import torch
from torch.library import custom_op, register_fake

from xfuser.logger import init_logger

from xfuser.core.attention.numerics import hadamard
from xfuser.core.attention.backends.sdpa import sdpa_flash
from xfuser.core.attention.constraints import NO_DROPOUT, NO_VARLEN
from xfuser.core.attention.numerics.layout import from_bshd, to_bshd
from xfuser.core.attention.requirements import ARCH, SYMBOL
from xfuser.core.attention.spec import AttentionBackendType, AttnCall, Spec

logger = init_logger(__name__)

_FLYDSL = "aiter.ops.flydsl:flydsl_flash_attn_func"


# ---------------------------------------------------------------------------
# custom ops
#
# Registered unconditionally: torch.library needs the function objects, not
# AITER, and the bodies import it lazily. A build without FlyDSL therefore has
# the ops but never reaches them, because the specs below refuse first.
#
# Two ops mirror the AITER / AITER_FP8 split: AITER_FLYDSL -> xfuser::flydsl_attn
# (bf16), AITER_FLYDSL_FP8 -> xfuser::flydsl_attn_fp8. fp8 is unfused (faster
# e2e) so it holds fp8 Q/K/V alongside the live bf16 Q/K/V -> higher peak VRAM;
# pick AITER_FLYDSL when tight.
# ---------------------------------------------------------------------------

def _kernel():
    from aiter.ops.flydsl import flydsl_flash_attn_func

    return flydsl_flash_attn_func


def _fp8_min_seq(head_dim: int, num_heads: int) -> int:
    """fp8 wins only above a sequence crossover (quant pre-pass cost vs K/V HBM
    bytes saved), which depends on (head_dim, num_heads). Measured on gfx1201:
    D64/H38 and D128/H<=32 flip at S~2560; D128 high head-count (wan H40) at
    S~3584. Below: fp8 loses 7-18%; above: wins <4%."""
    if head_dim >= 128 and num_heads > 32:
        return 3584
    return 2560


def _fp8_attn(query, key, value, is_causal):
    # flydsl_fp8_quant returns fp8 q/k/v + descales (real = fp8 * descale).
    from aiter.ops.flydsl import flydsl_fp8_quant

    qq, kk, vv, sq, sk, sv = flydsl_fp8_quant(query, key, value, rotation=True)
    return _kernel()(
        qq, kk, vv, causal=is_causal,
        q_descale=sq, k_descale=sk, v_descale=sv,
        waves_per_eu=2, daz=True,
    )


# Attn shape is constant across denoise steps, so log the chosen path once per shape.
_logged = set()


def _log_once(key_t, msg):
    if key_t in _logged:
        return
    _logged.add(key_t)
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        logger.info(msg)


@custom_op("xfuser::flydsl_attn", mutates_args=())
def _flydsl_attn(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, is_causal: bool
) -> torch.Tensor:
    B, S_real, H, D = query.shape
    is_cross = key.shape[1] != S_real
    _log_once(
        (B, S_real, H, D, is_cross, query.dtype),
        f"flydsl attn [B{B} S{S_real} H{H} D{D}] -> bf16",
    )
    return _kernel()(query, key, value, causal=is_causal, waves_per_eu=2, daz=True)


@register_fake("xfuser::flydsl_attn")
def _flydsl_attn_fake(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, is_causal: bool
) -> torch.Tensor:
    return torch.empty_like(query)


@custom_op("xfuser::flydsl_attn_fp8", mutates_args=())
def _flydsl_attn_fp8_kernel(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, is_causal: bool
) -> torch.Tensor:
    # fp8 only for bf16 self-attn above the crossover; else the bf16 kernel.
    B, S_real, H, D = query.shape
    is_cross = key.shape[1] != S_real
    min_seq = _fp8_min_seq(D, H)
    use_fp8 = query.dtype == torch.bfloat16 and not is_cross and S_real >= min_seq
    if use_fp8:
        msg = (
            f"flydsl attn [B{B} S{S_real} H{H} D{D}] -> fp8 (S>={min_seq}) "
            "(fp8 pre-pass adds a transient QKV copy -> higher peak VRAM)"
        )
    elif query.dtype == torch.bfloat16 and not is_cross:
        msg = f"flydsl attn [B{B} S{S_real} H{H} D{D}] -> bf16 (S<{min_seq})"
    else:
        msg = (
            f"flydsl attn [B{B} S{S_real} H{H} D{D}] -> bf16 "
            f"(not fp8-eligible: dtype={query.dtype}, cross={is_cross})"
        )
    _log_once((B, S_real, H, D, is_cross, query.dtype), msg)
    if use_fp8:
        return _fp8_attn(query, key, value, is_causal)
    return _kernel()(query, key, value, causal=is_causal, waves_per_eu=2, daz=True)


@register_fake("xfuser::flydsl_attn_fp8")
def _flydsl_attn_fp8_fake(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, is_causal: bool
) -> torch.Tensor:
    return torch.empty_like(query)


@custom_op("xfuser::flydsl_attn_fp8_prequant", mutates_args=())
def _flydsl_attn_fp8_prequant_kernel(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
    q_descale: torch.Tensor, k_descale: torch.Tensor, v_descale: torch.Tensor,
    is_causal: bool,
) -> torch.Tensor:
    return _kernel()(
        query, key, value, causal=is_causal,
        q_descale=q_descale, k_descale=k_descale, v_descale=v_descale,
        waves_per_eu=2, daz=True,
    )


@register_fake("xfuser::flydsl_attn_fp8_prequant")
def _flydsl_attn_fp8_prequant_fake(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
    q_descale: torch.Tensor, k_descale: torch.Tensor, v_descale: torch.Tensor,
    is_causal: bool,
) -> torch.Tensor:
    return torch.empty_like(query, dtype=torch.bfloat16)


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

    q, k, v = to_bshd(query, key, value, contiguous=True)
    return from_bshd(op(q, k, v, call.is_causal)), None


def flydsl(query, key, value, call: AttnCall):
    return _dispatch(query, key, value, call, torch.ops.xfuser.flydsl_attn)


def flydsl_fp8(query, key, value, call: AttnCall):
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
    Spec(AttentionBackendType.AITER_FLYDSL, impl=flydsl, accepts=NO_VARLEN,
         requires=SYMBOL(_FLYDSL) & ARCH("gfx1201")),

    Spec(AttentionBackendType.AITER_FLYDSL_FP8, impl=flydsl_fp8, low_precision=True,
         accepts=NO_DROPOUT & NO_VARLEN,
         accepts_prequantized=True,
         prequant_rotate=hadamard.rotate_qk,
         requires=SYMBOL(_FLYDSL) & SYMBOL("aiter.ops.flydsl:flydsl_fp8_quant")
                & ARCH("gfx1201")),
]
