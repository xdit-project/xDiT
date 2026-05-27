import math
import torch
import torch.nn as nn
from typing import Optional

try:
    import aiter
except ImportError:
    pass  # Error raised in base_model.py if fp8 enabled without AITER.

_FP8_MAX = 448.0  # torch.finfo(torch.float8_e4m3fn).max
_FP8_BLOCK = 128


@torch.library.custom_op("mylib::fp8_blockscale_gemm", mutates_args=())
def _fp8_blockscale_gemm(
    a_q: torch.Tensor,
    a_scale: torch.Tensor,
    w_fp8: torch.Tensor,
    w_scale: torch.Tensor,
) -> torch.Tensor:
    return aiter.gemm_a8w8_blockscale(a_q, w_fp8, a_scale, w_scale)


@_fp8_blockscale_gemm.register_fake
def _(
    a_q: torch.Tensor,
    a_scale: torch.Tensor,
    w_fp8: torch.Tensor,
    w_scale: torch.Tensor,
) -> torch.Tensor:
    M = a_q.shape[0]
    N = w_fp8.shape[0]
    return torch.empty(M, N, dtype=torch.bfloat16, device=a_q.device)


def _pad_cols_to_multiple(t: torch.Tensor, block: int) -> tuple[torch.Tensor, bool]:
    """Pad last dim (cols) to a multiple of block. Returns (padded, was_padded)."""
    c_pad = (-t.shape[-1]) % block
    if c_pad == 0:
        return t, False
    return torch.nn.functional.pad(t, (0, c_pad)), True


class xFuserFP8BlockScaleLinear(nn.Module):
    """
    Drop-in nn.Linear replacement using AITER gemm_a8w8_blockscale (block-128 FP8 w8a8).

    Weights pre-quantized at load time to float8_e4m3fn with per-block-128 scales
    stored as persistent buffers (FSDP2-compatible, no Float8Tensor subclass or patches).
    Activations block-quantized dynamically in forward() — ops are visible to
    torch.compile/Inductor for fusion into ~2 kernels (reduction + elementwise).

    Scale shapes:
        weight_scale: [ceil(N/128), ceil(K/128)]
        x_scale (runtime): [M, ceil(K/128)]
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # weight is not allocated here — buffers are registered by load_and_quantize_weights
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
        n_blocks = math.ceil(N / _FP8_BLOCK)
        k_blocks = math.ceil(K / _FP8_BLOCK)

        # Move to target device; stay in BF16 — FP8 cast (3 mantissa bits) dominates error,
        # not BF16 intermediates (7 bits). Only w_amax is cast to FP32 for the stored scale.
        target = device if device is not None else weight.device
        w = weight.to(device=target)
        # Pad both dims for the [n_blocks, 128, k_blocks, 128] reshape
        r_pad = (-N) % _FP8_BLOCK
        c_pad = (-K) % _FP8_BLOCK
        if r_pad or c_pad:
            w = torch.nn.functional.pad(w, (0, c_pad, 0, r_pad))
            was_padded = True
        else:
            was_padded = False

        w_blocks = w.reshape(n_blocks, _FP8_BLOCK, k_blocks, _FP8_BLOCK)
        w_amax = w_blocks.abs().amax(dim=(1, 3)).clamp(min=1e-12)  # [n_blocks, k_blocks]
        w_scale = (w_amax.float() / _FP8_MAX)

        # Process one row-block at a time to avoid a full BF16 intermediate alongside the weight.
        # Peak: BF16 weight + FP8 output + one 128-row BF16 chunk (~0.75 MB for K=3072).
        w_q = torch.empty_like(w_blocks, dtype=torch.float8_e4m3fn)
        for i in range(n_blocks):
            w_q[i] = (w_blocks[i] / w_amax[i][None, :, None] * _FP8_MAX).clamp(-_FP8_MAX, _FP8_MAX).to(torch.float8_e4m3fn)
        w_q = w_q.reshape(n_blocks * _FP8_BLOCK, k_blocks * _FP8_BLOCK)
        if was_padded:
            w_q = w_q[:N, :K].contiguous()

        self.register_buffer("weight_fp8", w_q, persistent=True)
        self.register_buffer("weight_scale", w_scale, persistent=True)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if not hasattr(self, "weight_fp8"):
            raise RuntimeError(
                "weight_fp8 not initialized. Call load_and_quantize_weights() first."
            )

        original_shape = input.shape
        x = input.view(-1, self.in_features)
        M, K = x.shape
        k_blocks = math.ceil(K / _FP8_BLOCK)

        # aiter.get_hip_quant(aiter.QuantType.per_1x128) is the natural API here —
        # fused HIP kernel matching MXFP4's per_1x32 pattern — but module_quant JIT
        # fails to build on gfx1201 with ROCm 7.2.3:
        #   __builtin_amdgcn_raw_ptr_buffer_load_lds needs vmem-to-lds-load-insts
        # Manual block-128 quant; torch.compile/Inductor fuses these into ~2 kernels.
        x_padded, needs_pad = _pad_cols_to_multiple(x, _FP8_BLOCK)
        K_pad = x_padded.shape[1]

        x_blocks = x_padded.reshape(M, k_blocks, _FP8_BLOCK)
        x_amax = x_blocks.abs().amax(dim=-1).clamp(min=1e-12)  # [M, k_blocks]
        x_scale = (x_amax.float() / _FP8_MAX)
        x_q = (x_blocks / x_amax.unsqueeze(-1) * _FP8_MAX).clamp(-_FP8_MAX, _FP8_MAX)
        x_q = x_q.to(torch.float8_e4m3fn).reshape(M, K_pad)
        if needs_pad:
            x_q = x_q[:, :K].contiguous()

        output = torch.ops.mylib.fp8_blockscale_gemm(
            x_q, x_scale, self.weight_fp8, self.weight_scale,
        ).to(input.dtype)
        if self.bias is not None:
            output = output + self.bias
        return output.view(*original_shape[:-1], self.out_features)

    def extra_repr(self):
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}"
        )
