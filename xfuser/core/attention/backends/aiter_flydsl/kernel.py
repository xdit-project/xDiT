"""
AITER FlyDSL: a gfx1201 MHA kernel
"""

import torch
from aiter.ops.flydsl import flydsl_flash_attn_func, flydsl_fp8_quant
from torch.library import custom_op, register_fake

from xfuser.logger import init_logger, log_once

from xfuser.core.attention.backends.sdpa.kernel import sdpa_flash
from xfuser.core.attention.numerics.layout import from_bshd, to_bshd
from xfuser.core.attention.spec import AttnCall
from .selection import fp8_eligible, fp8_min_seq

logger = init_logger(__name__)


# ---------------------------------------------------------------------------
# custom ops
#
# Two ops mirror the AITER / AITER_FP8 split: AITER_FLYDSL -> xfuser::flydsl_attn
# (bf16), AITER_FLYDSL_FP8 -> xfuser::flydsl_attn_fp8. fp8 is unfused (faster
# e2e) so it holds fp8 Q/K/V alongside the live bf16 Q/K/V -> higher peak VRAM;
# pick AITER_FLYDSL when tight.
# ---------------------------------------------------------------------------


def _fp8_attn(query, key, value, is_causal):
    # flydsl_fp8_quant returns fp8 q/k/v + descales (real = fp8 * descale).
    qq, kk, vv, sq, sk, sv = flydsl_fp8_quant(query, key, value, rotation=True)
    return flydsl_flash_attn_func(
        qq,
        kk,
        vv,
        causal=is_causal,
        q_descale=sq,
        k_descale=sk,
        v_descale=sv,
        waves_per_eu=2,
        daz=True,
    )


@custom_op("xfuser::flydsl_attn", mutates_args=())
def _flydsl_attn(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, is_causal: bool) -> torch.Tensor:
    B, S_real, H, D = query.shape
    is_cross = key.shape[1] != S_real
    # Attn shape is constant across denoise steps, so this logs once per shape.
    log_once(
        logger,
        (B, S_real, H, D, is_cross, query.dtype),
        f"flydsl attn [B{B} S{S_real} H{H} D{D}] -> bf16",
    )
    return flydsl_flash_attn_func(query, key, value, causal=is_causal, waves_per_eu=2, daz=True)


@register_fake("xfuser::flydsl_attn")
def _flydsl_attn_fake(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, is_causal: bool) -> torch.Tensor:
    return torch.empty_like(query)


@custom_op("xfuser::flydsl_attn_fp8", mutates_args=())
def _flydsl_attn_fp8_kernel(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, is_causal: bool
) -> torch.Tensor:
    # Use fp8 only for shapes with a measured end-to-end crossover.
    B, S_real, H, D = query.shape
    key_length = key.shape[1]
    is_cross = key_length != S_real
    min_seq = fp8_min_seq(D, H, is_cross=key_length > S_real)
    use_fp8 = fp8_eligible(query.dtype, S_real, key_length, H, D)
    if use_fp8:
        msg = (
            f"flydsl attn [B{B} Sq{S_real} Sk{key_length} H{H} D{D}] "
            f"-> fp8 (Sq>={min_seq}) "
            "(fp8 pre-pass adds a transient QKV copy -> higher peak VRAM)"
        )
    elif query.dtype == torch.bfloat16 and key_length >= S_real:
        reason = "no measured FP8 crossover" if min_seq is None else f"Sq<{min_seq}"
        msg = (
            f"flydsl attn [B{B} Sq{S_real} Sk{key_length} H{H} D{D}] "
            f"-> bf16 ({reason})"
        )
    else:
        msg = (
            f"flydsl attn [B{B} Sq{S_real} Sk{key_length} H{H} D{D}] "
            f"-> bf16 (not fp8-eligible: dtype={query.dtype}, cross={is_cross})"
        )
    log_once(logger, (B, S_real, key_length, H, D, query.dtype), msg)
    if use_fp8:
        return _fp8_attn(query, key, value, is_causal)
    return flydsl_flash_attn_func(query, key, value, causal=is_causal, waves_per_eu=2, daz=True)


@register_fake("xfuser::flydsl_attn_fp8")
def _flydsl_attn_fp8_fake(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, is_causal: bool) -> torch.Tensor:
    return torch.empty_like(query)


@custom_op("xfuser::flydsl_attn_fp8_prequant", mutates_args=())
def _flydsl_attn_fp8_prequant_kernel(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    q_descale: torch.Tensor,
    k_descale: torch.Tensor,
    v_descale: torch.Tensor,
    is_causal: bool,
) -> torch.Tensor:
    return flydsl_flash_attn_func(
        query,
        key,
        value,
        causal=is_causal,
        q_descale=q_descale,
        k_descale=k_descale,
        v_descale=v_descale,
        waves_per_eu=2,
        daz=True,
    )


@register_fake("xfuser::flydsl_attn_fp8_prequant")
def _flydsl_attn_fp8_prequant_fake(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    q_descale: torch.Tensor,
    k_descale: torch.Tensor,
    v_descale: torch.Tensor,
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
            q,
            k,
            v,
            kwargs["q_descale"],
            kwargs["k_descale"],
            kwargs["v_descale"],
            call.is_causal,
        )
        return from_bshd(out), None

    return _dispatch(query, key, value, call, torch.ops.xfuser.flydsl_attn_fp8)
