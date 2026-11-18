import math
from typing import Optional

import torch
import torch.nn as nn

try:
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice
except ImportError:  # CPU-only tests and installations without Triton.
    triton = None

_K_TILE = 128
_PRESHUFFLE_LAYOUT = (16, 16)


def supports_aiter_int8_linear(in_features: int, out_features: int) -> bool:
    return in_features >= 2 * _K_TILE and in_features % _K_TILE == 0 and out_features % _PRESHUFFLE_LAYOUT[0] == 0


if triton is not None:

    @triton.jit
    def _quantize_rows_int8_kernel(
        output,
        scale_output,
        source,
        columns: tl.constexpr,
        source_stride,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        column = tl.arange(0, BLOCK)
        mask = column < columns
        values = tl.load(
            source + row * source_stride + column,
            mask=mask,
            other=0.0,
            cache_modifier=".cg",
        ).to(tl.float32)
        amax = tl.max(tl.abs(values), axis=0)
        scale = amax / 127.0
        reciprocal = tl.where(amax > 0.0, 1.0 / scale, 0.0)
        quantized = libdevice.rint(values * reciprocal)
        quantized = tl.maximum(tl.minimum(quantized, 127.0), -127.0)
        tl.store(scale_output + row, scale)
        tl.store(
            output + row * columns + column,
            quantized.to(tl.int8),
            mask=mask,
            cache_modifier=".cs",
        )


def _quantize_rows_int8(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    tensor = tensor.contiguous()
    quantized = torch.empty_like(tensor, dtype=torch.int8)
    scale = torch.empty(
        (tensor.shape[0],),
        dtype=torch.float32,
        device=tensor.device,
    )
    _quantize_rows_int8_kernel[(tensor.shape[0],)](
        quantized,
        scale,
        tensor,
        tensor.shape[1],
        tensor.stride(0),
        BLOCK=triton.next_power_of_2(tensor.shape[1]),
    )
    return quantized, scale


@torch.library.custom_op("xfuser::aiter_int8_gemm", mutates_args=())
def _aiter_int8_gemm(
    input_2d: torch.Tensor,
    weight_preshuffled: torch.Tensor,
    weight_scale: torch.Tensor,
) -> torch.Tensor:
    import aiter

    if input_2d.dim() != 2 or weight_preshuffled.dim() != 2:
        raise ValueError("AITER INT8 GEMM operands must be two-dimensional")
    if input_2d.dtype != torch.bfloat16:
        raise TypeError("AITER INT8 GEMM activation must use torch.bfloat16")
    k = input_2d.shape[1]
    weight_k = weight_preshuffled.shape[1]
    if weight_k != k:
        raise ValueError(f"AITER INT8 GEMM K mismatch: {k} and {weight_k}")
    x_quantized, x_scale = _quantize_rows_int8(input_2d)
    return aiter.gemm_a8w8_bpreshuffle(
        x_quantized,
        weight_preshuffled,
        x_scale.view(-1, 1),
        weight_scale.view(1, -1),
    )


@_aiter_int8_gemm.register_fake
def _(
    input_2d: torch.Tensor,
    weight_preshuffled: torch.Tensor,
    weight_scale: torch.Tensor,
) -> torch.Tensor:
    return torch.empty(
        (input_2d.shape[0], weight_preshuffled.shape[0]),
        dtype=torch.bfloat16,
        device=input_2d.device,
    )


class xFuserAiterInt8Linear(nn.Module):
    """Drop-in BF16 Linear backed by the gfx950 4-wave true-INT8 GEMM."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        dtype = torch.bfloat16 if dtype is None else dtype
        if dtype != torch.bfloat16:
            raise TypeError("AITER INT8 linear supports only torch.bfloat16")
        if not supports_aiter_int8_linear(in_features, out_features):
            raise ValueError(
                "AITER INT8 linear requires K >= 256, K % 128 == 0, "
                f"and N % 16 == 0; got N={out_features}, K={in_features}"
            )

        self.in_features = in_features
        self.out_features = out_features
        factory_kwargs = {"device": device, "dtype": dtype}
        self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    @staticmethod
    def _is_fsdp_managed_parameter(parameter) -> bool:
        try:
            from torch.distributed.tensor import DTensor
        except ImportError:
            return False
        return isinstance(parameter, DTensor)

    def _install_quantized_state(
        self,
        weight_preshuffled: torch.Tensor,
        weight_scale: torch.Tensor,
    ) -> None:
        if self.weight is not None and self._is_fsdp_managed_parameter(self.weight):
            raise RuntimeError("Quantize AITER INT8 weights before fully_shard")
        if hasattr(self, "weight_int8"):
            delattr(self, "weight_int8")
        if hasattr(self, "weight_scale"):
            delattr(self, "weight_scale")
        if self.weight is not None:
            delattr(self, "weight")
            self.register_parameter("weight", None)
        self.register_parameter(
            "weight_int8",
            nn.Parameter(weight_preshuffled.detach(), requires_grad=False),
        )
        self.register_buffer(
            "weight_scale",
            weight_scale.detach(),
            persistent=True,
        )

    def load_and_quantize_weights(
        self,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        *,
        device: torch.device | str | None = None,
    ) -> None:
        from aiter.ops.shuffle import shuffle_weight

        expected = (self.out_features, self.in_features)
        if tuple(weight.shape) != expected:
            raise ValueError(f"AITER INT8 weight must have shape {expected}, got {tuple(weight.shape)}")
        if weight.dtype != torch.bfloat16:
            raise TypeError("AITER INT8 weights must use torch.bfloat16")
        target = torch.device(device) if device is not None else weight.device
        if target.type == "meta":
            raise RuntimeError("AITER INT8 weights cannot be quantized on meta")
        full_weight = weight.detach().to(target)
        weight_quantized, weight_scale = _quantize_rows_int8(full_weight)
        weight_preshuffled = shuffle_weight(
            weight_quantized,
            layout=_PRESHUFFLE_LAYOUT,
        ).contiguous()
        self._install_quantized_state(weight_preshuffled, weight_scale)
        if bias is not None and self.bias is not None:
            self.bias = nn.Parameter(bias.detach().to(target))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if input.shape[-1] != self.in_features:
            raise ValueError(f"AITER INT8 input has {input.shape[-1]} features, expected {self.in_features}")
        if input.dtype != torch.bfloat16:
            raise TypeError("AITER INT8 activations must use torch.bfloat16")
        if not hasattr(self, "weight_int8"):
            self.load_and_quantize_weights(self.weight)

        original_shape = input.shape
        input_2d = input.reshape(-1, self.in_features).contiguous()
        output = torch.ops.xfuser.aiter_int8_gemm(
            input_2d,
            self.weight_int8,
            self.weight_scale,
        )
        if self.bias is not None:
            output = output + self.bias
        return output.reshape(*original_shape[:-1], self.out_features)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"
