import functools
import os
import torch
import torch.nn as nn
from typing import Optional

try:
    import aiter
except ImportError:
    pass  # Error raised in base_model.py if fp8 enabled without AITER.


@functools.lru_cache(maxsize=1)
def _hip_quant_per_1x128():
    # aiter.get_hip_quant rebuilds a dispatch dict + functools.partial on every call (its
    # triton twin get_triton_quant is lru_cached; the hip one is not).
    return aiter.get_hip_quant(aiter.QuantType.per_1x128)

try:
    from aiter.ops.shuffle import shuffle_weight
    from aiter.ops.triton.gemm.basic.gemm_a8w8_blockscale import (
        gemm_a8w8_blockscale_preshuffle,
    )
    _HAS_PRESHUFFLE = True
except ImportError:
    _HAS_PRESHUFFLE = False  # older AITER without preshuffle blockscale GEMM

_FP8_BLOCK = 128
_PRESHUFFLE_LAYOUT = (16, 16)

# Preshuffle blockscale GEMM measured ~25% SLOWER than plain on gfx1201/RDNA4 (klein 2048^2:
# 20.4s vs 16.4s/iter), so default off. Root cause: no tuned block-128 preshuffle kernel exists
# for gfx1201 in AITER. The fast gluon/WMMA kernel is gfx1250-only; the core CK/asm dispatcher
# (gemm_a8w8_blockscale_bpreshuffle) has no gfx1201 code object (SIGSEGVs, so AITER gates it off).
# Both paths fall back to a generic triton kernel with a small-M (M_LEQ_8) config, mistuned for
# our large-M (~16k image tokens) workload, while plain gemm_a8w8_blockscale IS tuned for gfx1201
# large-M and wins. Set XFUSER_FP8_PRESHUFFLE=1 to re-enable on an arch with a real preshuffle
# kernel (e.g. gfx1250, or newer AITER that adds gfx1201 support).
_PRESHUFFLE_ENABLED = os.environ.get("XFUSER_FP8_PRESHUFFLE", "0") != "0"


def _fp8_dtype() -> torch.dtype:
    return aiter.dtypes.fp8


def _fp8_max() -> float:
    return torch.finfo(aiter.dtypes.fp8).max


def _quantize_weight_blocks(w_blocks: torch.Tensor, w_amax: torch.Tensor) -> torch.Tensor:
    fp8_max = _fp8_max()
    # One scale-mul + in-place clamp, then cast — avoids the extra full-size bf16 temps a
    # (divide, multiply, clamp) chain would materialise at load time.
    scale = (fp8_max / w_amax)[:, None, :, None]
    return (w_blocks * scale).clamp_(-fp8_max, fp8_max).to(_fp8_dtype())


def _pad_cols_to_multiple(t: torch.Tensor, block: int) -> tuple[torch.Tensor, bool]:
    c_pad = (-t.shape[-1]) % block
    if c_pad == 0:
        return t, False
    return torch.nn.functional.pad(t, (0, c_pad)), True


def quantize_weight_to_fp8_blockscale_plain(
    weight: torch.Tensor, device: Optional[torch.device] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Block-128 FP8 quantize a weight to the plain (non-preshuffle) layout.

    Returns (w_fp8 [N, K] fp8, w_scale [ceil(N/128), ceil(K/128)] float32). Mirrors
    xFuserFP8BlockScaleLinear._quantize_weights' plain branch; used by the transformers
    HfQuantizer load path (aiter_fp8_quantizer), which stores fp8 under a state-dict
    key rather than calling load_and_quantize_weights.
    """
    N, K = weight.shape
    n_blocks = (N + _FP8_BLOCK - 1) // _FP8_BLOCK
    k_blocks = (K + _FP8_BLOCK - 1) // _FP8_BLOCK
    target = device if device is not None else weight.device
    w = weight.to(device=target)
    r_pad = (-N) % _FP8_BLOCK
    c_pad = (-K) % _FP8_BLOCK
    if r_pad or c_pad:
        w = torch.nn.functional.pad(w, (0, c_pad, 0, r_pad))
    w_blocks = w.reshape(n_blocks, _FP8_BLOCK, k_blocks, _FP8_BLOCK)
    w_amax = w_blocks.abs().amax(dim=(1, 3)).clamp(min=1e-12)
    w_scale = (w_amax.float() / _fp8_max())
    w_q = _quantize_weight_blocks(w_blocks, w_amax)
    w_q = w_q.reshape(n_blocks * _FP8_BLOCK, k_blocks * _FP8_BLOCK)
    if r_pad or c_pad:
        w_q = w_q[:N, :K].contiguous()
    return w_q, w_scale


@torch.library.custom_op("mylib::fp8_blockscale_gemm", mutates_args=())
def _fp8_blockscale_gemm(
    x: torch.Tensor,
    w_fp8: torch.Tensor,
    w_scale: torch.Tensor,
) -> torch.Tensor:
    K = x.shape[1]
    x_padded, needs_pad = _pad_cols_to_multiple(x, _FP8_BLOCK)
    x_q, x_scale = _hip_quant_per_1x128()(x_padded, quant_dtype=_fp8_dtype())
    if needs_pad:
        x_q = x_q[:, :K].contiguous()
    return aiter.gemm_a8w8_blockscale(x_q, w_fp8, x_scale, w_scale)


@_fp8_blockscale_gemm.register_fake
def _(
    x: torch.Tensor,
    w_fp8: torch.Tensor,
    w_scale: torch.Tensor,
) -> torch.Tensor:
    M = x.shape[0]
    N = w_fp8.shape[0]
    return torch.empty(M, N, dtype=torch.bfloat16, device=x.device)


@torch.library.custom_op("mylib::fp8_blockscale_gemm_preshuffle", mutates_args=())
def _fp8_blockscale_gemm_preshuffle(
    x: torch.Tensor,
    w_shuffle: torch.Tensor,
    w_scale: torch.Tensor,
    n: int,
    k_padded: int,
) -> torch.Tensor:
    # Weight already in (N/16, K*16) fragment order from load. x_scale must be
    # transposed-contiguous — the preshuffle kernel reads it as is_x_scale_tranposed=True (sic).
    K = x.shape[1]
    if K < k_padded:
        x = torch.nn.functional.pad(x, (0, k_padded - K))
    x_q, x_scale = _hip_quant_per_1x128()(x, quant_dtype=_fp8_dtype())
    x_scale = x_scale.transpose(0, 1).contiguous().view(x_scale.shape[0], x_scale.shape[1])
    out = gemm_a8w8_blockscale_preshuffle(x_q, w_shuffle, x_scale, w_scale, dtype=torch.bfloat16)
    return out[:, :n].contiguous() if out.shape[1] != n else out


@_fp8_blockscale_gemm_preshuffle.register_fake
def _(
    x: torch.Tensor,
    w_shuffle: torch.Tensor,
    w_scale: torch.Tensor,
    n: int,
    k_padded: int,
) -> torch.Tensor:
    return torch.empty(x.shape[0], n, dtype=torch.bfloat16, device=x.device)


class xFuserFP8BlockScaleLinear(nn.Module):
    """
    Drop-in nn.Linear replacement, block-128 FP8 w8a8 on AITER.

    Weights pre-quantized at load time. Activations quantized inside the custom op
    via aiter.get_hip_quant(QuantType.per_1x128) — fused with GEMM under torch.compile.
    Plain gemm_a8w8_blockscale by default; preshuffle (offline weight reorder for
    gemm_a8w8_blockscale_preshuffle) is opt-in via XFUSER_FP8_PRESHUFFLE=1 — it measured
    ~25% slower on gfx1201/RDNA4.

    Scale shapes:
        weight_scale: [ceil(N/128), ceil(K/128)]
        x_scale (runtime): [M, K/128]
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None, preshuffle: bool = True):
        super().__init__()
        # Preshuffle needs a newer AITER and can be disabled via env for A/B measurement.
        # Fall back to the plain blockscale path when the kernel is absent or forced off.
        if preshuffle and (not _HAS_PRESHUFFLE or not _PRESHUFFLE_ENABLED):
            preshuffle = False
        self.in_features = in_features
        self.out_features = out_features
        self.preshuffle = preshuffle
        self.register_parameter("weight", None)
        if bias:
            self.bias = nn.Parameter(
                torch.empty(out_features, device=device, dtype=dtype)
            )
        else:
            self.register_parameter("bias", None)

    def load_and_quantize_weights(
        self, weight: torch.Tensor, bias: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        self._quantize_weights(weight, device=device)
        if bias is not None and self.bias is not None:
            target = device if device is not None else bias.device
            self.bias = torch.nn.Parameter(bias.to(device=target, dtype=bias.dtype).detach())

    def _quantize_weights(self, weight: torch.Tensor, device: Optional[torch.device] = None) -> None:
        N, K = weight.shape
        n_blocks = (N + _FP8_BLOCK - 1) // _FP8_BLOCK
        k_blocks = (K + _FP8_BLOCK - 1) // _FP8_BLOCK

        target = device if device is not None else weight.device
        w = weight.to(device=target)
        r_pad = (-N) % _FP8_BLOCK
        c_pad = (-K) % _FP8_BLOCK
        if r_pad or c_pad:
            w = torch.nn.functional.pad(w, (0, c_pad, 0, r_pad))
            was_padded = True
        else:
            was_padded = False

        w_blocks = w.reshape(n_blocks, _FP8_BLOCK, k_blocks, _FP8_BLOCK)
        w_amax = w_blocks.abs().amax(dim=(1, 3)).clamp(min=1e-12)  # [n_blocks, k_blocks]
        w_scale = (w_amax.float() / _fp8_max())

        w_q = _quantize_weight_blocks(w_blocks, w_amax)
        w_q = w_q.reshape(n_blocks * _FP8_BLOCK, k_blocks * _FP8_BLOCK)

        # bf16 weight is done with; drop it before shuffle/slice so the load peak isn't
        # bf16 + fp8(w_q) + fp8(w_shuffle) held at once.
        del w, w_blocks

        # Preshuffle stores the 128-padded weight permanently. For 128-aligned dims that
        # matches plain's numel (free speed), but for non-aligned dims the padding is pure
        # persistent VRAM waste. Only preshuffle when padding is zero; otherwise store exact
        # [N,K] plain (forward reads self.preshuffle, so flip it here too).
        if self.preshuffle and was_padded:
            self.preshuffle = False

        if self.preshuffle:
            # Dims are 128-aligned here (128-multiples satisfy the kernel's N%16, K%32 rule).
            n_pad = n_blocks * _FP8_BLOCK
            k_pad = k_blocks * _FP8_BLOCK
            w_shuffle = shuffle_weight(w_q, _PRESHUFFLE_LAYOUT).reshape(
                n_pad // _PRESHUFFLE_LAYOUT[0], k_pad * _PRESHUFFLE_LAYOUT[0]
            )
            self.weight_fp8 = nn.Parameter(w_shuffle, requires_grad=False)
            self.register_buffer("weight_scale", w_scale, persistent=True)
            return

        if was_padded:
            w_q = w_q[:N, :K].contiguous()

        # weight_fp8 as Parameter (not buffer) so FSDP2 shards it across ranks.
        # requires_grad=False: inference only, no gradient needed.
        self.weight_fp8 = nn.Parameter(w_q, requires_grad=False)
        self.register_buffer("weight_scale", w_scale, persistent=True)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        original_shape = input.shape
        x = input.reshape(-1, self.in_features)
        # DiT streaming path stores fp8 in `weight_fp8` (and nulls `weight`); the transformers
        # HfQuantizer path stores fp8 under `weight` (state-dict fill). Resolve either.
        weight_fp8 = getattr(self, "weight_fp8", None)
        if weight_fp8 is None:
            weight_fp8 = self.weight
        if weight_fp8 is None:
            raise RuntimeError(
                "FP8 weight not initialized. Call load_and_quantize_weights() or load a checkpoint first."
            )
        if self.preshuffle:
            k_padded = ((self.in_features + _FP8_BLOCK - 1) // _FP8_BLOCK) * _FP8_BLOCK
            output = torch.ops.mylib.fp8_blockscale_gemm_preshuffle(
                x, weight_fp8, self.weight_scale, self.out_features, k_padded,
            ).to(input.dtype)
        else:
            output = torch.ops.mylib.fp8_blockscale_gemm(
                x, weight_fp8, self.weight_scale,
            ).to(input.dtype)
        if self.bias is not None:
            output = output + self.bias
        return output.view(*original_shape[:-1], self.out_features)

    def extra_repr(self):
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}"
        )
