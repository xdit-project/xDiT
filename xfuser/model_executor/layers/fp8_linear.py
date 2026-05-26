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
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, **factory_kwargs)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)

    def load_and_quantize_weights(
        self, weights: torch.Tensor, bias: Optional[torch.Tensor] = None
    ) -> None:
        with torch.no_grad():
            self.weight.data.copy_(weights.data)
            if bias is not None and self.bias is not None:
                self.bias.data.copy_(bias.data)
        self._quantize_weights()

    def _quantize_weights(self) -> None:
        if self.weight is None:
            raise RuntimeError(
                "Cannot quantize: weight is None. Call load_and_quantize_weights() first."
            )
        N, K = self.weight.shape
        n_blocks = math.ceil(N / _FP8_BLOCK)
        k_blocks = math.ceil(K / _FP8_BLOCK)
        N_pad = n_blocks * _FP8_BLOCK
        K_pad = k_blocks * _FP8_BLOCK

        w = self.weight.to(torch.float32)
        if N_pad != N or K_pad != K:
            w_padded = torch.zeros(N_pad, K_pad, dtype=torch.float32, device=w.device)
            w_padded[:N, :K] = w
        else:
            w_padded = w

        # [n_blocks, _FP8_BLOCK, k_blocks, _FP8_BLOCK] -> amax per [n_blocks, k_blocks]
        w_blocks = w_padded.reshape(n_blocks, _FP8_BLOCK, k_blocks, _FP8_BLOCK)
        w_amax = w_blocks.abs().amax(dim=(1, 3)).clamp(min=1e-12)  # [n_blocks, k_blocks]
        w_scale = (w_amax / _FP8_MAX).float()

        w_q = (w_blocks / w_amax[:, None, :, None] * _FP8_MAX).clamp(-_FP8_MAX, _FP8_MAX)
        w_q = w_q.to(torch.float8_e4m3fn).reshape(N_pad, K_pad)[:N, :K].contiguous()

        delattr(self, "weight")
        self.register_parameter("weight", None)
        self.register_buffer("weight_fp8", w_q, persistent=True)
        self.register_buffer("weight_scale", w_scale, persistent=True)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if not hasattr(self, "weight_fp8"):
            self._quantize_weights()

        original_shape = input.shape
        x = input.view(-1, self.in_features)
        M, K = x.shape
        k_blocks = math.ceil(K / _FP8_BLOCK)
        K_pad = k_blocks * _FP8_BLOCK

        # aiter.get_hip_quant(aiter.QuantType.per_1x128) is the natural API here —
        # fused HIP kernel matching MXFP4's per_1x32 pattern — but module_quant JIT
        # fails to build on gfx1201 with ROCm 7.2.3:
        #   __builtin_amdgcn_raw_ptr_buffer_load_lds needs vmem-to-lds-load-insts
        # Manual block-128 quant; torch.compile/Inductor fuses these into ~2 kernels.
        x_f32 = x.float()
        if K_pad != K:
            x_padded = torch.zeros(M, K_pad, dtype=torch.float32, device=x_f32.device)
            x_padded[:, :K] = x_f32
        else:
            x_padded = x_f32

        x_blocks = x_padded.reshape(M, k_blocks, _FP8_BLOCK)
        x_amax = x_blocks.abs().amax(dim=-1).clamp(min=1e-12)  # [M_flat, k_blocks]
        x_scale = (x_amax / _FP8_MAX).float()
        x_q = (x_blocks / x_amax.unsqueeze(-1) * _FP8_MAX).clamp(-_FP8_MAX, _FP8_MAX)
        x_q = x_q.to(torch.float8_e4m3fn).reshape(M, K_pad)[:, :K].contiguous()

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
