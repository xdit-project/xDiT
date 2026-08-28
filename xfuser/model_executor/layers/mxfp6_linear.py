import math
from typing import Optional

import torch
import torch.nn as nn

try:
    from aiter.ops.gemm_op_a6w6 import (
        gemm_a6w6,
        mxfp6_gemm_pack_size,
        quant_mxfp6_gemm,
    )
except Exception:
    # The model preflight reports a detailed error before conversion is used.
    gemm_a6w6 = None
    mxfp6_gemm_pack_size = None
    quant_mxfp6_gemm = None


@torch.library.custom_op("xfuser::mxfp6_gemm", mutates_args=())
def _mxfp6_gemm(
    input_2d: torch.Tensor,
    weight_packed: torch.Tensor,
    weight_scale: torch.Tensor,
    out_features: int,
    in_features: int,
) -> torch.Tensor:
    activation_packed, activation_scale = quant_mxfp6_gemm(input_2d)
    return gemm_a6w6(
        activation_packed,
        weight_packed,
        activation_scale,
        weight_scale,
        input_2d.shape[0],
        out_features,
        in_features,
        dtype=torch.bfloat16,
    )


@_mxfp6_gemm.register_fake
def _(
    input_2d: torch.Tensor,
    weight_packed: torch.Tensor,
    weight_scale: torch.Tensor,
    out_features: int,
    in_features: int,
) -> torch.Tensor:
    return torch.empty(
        (input_2d.shape[0], out_features),
        dtype=torch.bfloat16,
        device=input_2d.device,
    )


class xFuserMXFP6Linear(nn.Module):
    """Drop-in BF16 linear using AITER's MXFP6 A6W6 GEMM."""

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
            raise TypeError("AITER MXFP6 linear supports only torch.bfloat16")

        self.in_features = in_features
        self.out_features = out_features
        factory_kwargs = {"device": device, "dtype": dtype}
        self.weight = nn.Parameter(
            torch.empty((out_features, in_features), **factory_kwargs)
        )
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

    @staticmethod
    def _destination_device(current, incoming) -> torch.device:
        return incoming.device if current.device.type == "meta" else current.device

    def _remove_packed_state(self) -> None:
        if hasattr(self, "weight_packed"):
            delattr(self, "weight_packed")
        if hasattr(self, "weight_scale"):
            delattr(self, "weight_scale")

    def _install_packed_state(
        self,
        weight_packed: torch.Tensor,
        weight_scale: torch.Tensor,
    ) -> None:
        current_packed = getattr(self, "weight_packed", None)
        if current_packed is not None and self._is_fsdp_managed_parameter(
            current_packed
        ):
            raise RuntimeError("Replace MXFP6 packed state before fully_shard")
        self._remove_packed_state()
        if self.weight is not None:
            delattr(self, "weight")
            self.register_parameter("weight", None)
        self.register_parameter(
            "weight_packed",
            nn.Parameter(weight_packed.detach(), requires_grad=False),
        )
        self.register_buffer("weight_scale", weight_scale.detach(), persistent=True)

    def load_and_quantize_weights(
        self,
        weights: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        *,
        device: torch.device | None = None,
    ) -> None:
        if weights.shape != (self.out_features, self.in_features):
            raise ValueError(
                "MXFP6 weight shape must be "
                f"({self.out_features}, {self.in_features}), got {tuple(weights.shape)}"
            )
        if weights.dtype != torch.bfloat16:
            raise TypeError("AITER MXFP6 weights must use torch.bfloat16")

        target = torch.device(device) if device is not None else weights.device
        if target.type == "meta":
            raise RuntimeError("MXFP6 weights cannot be quantized on the meta device")
        full_weight = weights.detach().to(target)
        packed = getattr(self, "weight_packed", None)
        if packed is not None and self._is_fsdp_managed_parameter(packed):
            raise RuntimeError("Quantize MXFP6 weights before fully_shard")
        if self.weight is not None and self._is_fsdp_managed_parameter(self.weight):
            raise RuntimeError("Quantize MXFP6 weights before fully_shard")
        self._remove_packed_state()
        if self.weight is not None:
            delattr(self, "weight")
        self.register_parameter("weight", nn.Parameter(full_weight))
        if bias is not None:
            self.bias = nn.Parameter(bias.detach().to(target))
        self._quantize_weights()

    def _quantize_weights(self) -> None:
        if self.weight is None:
            raise RuntimeError("MXFP6 full-precision weight is unavailable")
        if self._is_fsdp_managed_parameter(self.weight):
            raise RuntimeError("Quantize MXFP6 weights before fully_shard")
        weight_packed, weight_scale = quant_mxfp6_gemm(self.weight)
        self._install_packed_state(weight_packed, weight_scale)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        weight_key = prefix + "weight"
        packed_key = prefix + "weight_packed"
        scale_key = prefix + "weight_scale"
        has_packed = packed_key in state_dict or scale_key in state_dict

        if (packed_key in state_dict) != (scale_key in state_dict):
            raise RuntimeError(
                "MXFP6 packed state requires both weight_packed and weight_scale"
            )
        if has_packed and weight_key in state_dict:
            raise RuntimeError(
                "MXFP6 state cannot contain both full and packed weights"
            )

        destination = None
        if has_packed:
            incoming_packed = state_dict[packed_key]
            incoming_scale = state_dict[scale_key]
            expected_packed, expected_scale = mxfp6_gemm_pack_size(
                self.out_features, self.in_features
            )
            if (
                incoming_packed.dtype != torch.uint8
                or incoming_scale.dtype != torch.uint8
                or incoming_packed.numel() != expected_packed
                or incoming_scale.numel() != expected_scale
            ):
                raise ValueError("MXFP6 packed state has an invalid dtype or size")
            current = (
                self.weight_packed if hasattr(self, "weight_packed") else self.weight
            )
            if self._is_fsdp_managed_parameter(current):
                raise RuntimeError("Load MXFP6 packed state before fully_shard")
            destination = self._destination_device(current, incoming_packed)
            self._remove_packed_state()
            if self.weight is not None:
                delattr(self, "weight")
                self.register_parameter("weight", None)
            self.register_parameter(
                "weight_packed",
                nn.Parameter(
                    torch.empty_like(incoming_packed, device=destination),
                    requires_grad=False,
                ),
            )
            self.register_buffer(
                "weight_scale",
                torch.empty_like(incoming_scale, device=destination),
                persistent=True,
            )
        elif weight_key in state_dict:
            packed = getattr(self, "weight_packed", None)
            if packed is not None and self._is_fsdp_managed_parameter(packed):
                raise RuntimeError("Load MXFP6 full-precision state before fully_shard")
            incoming = state_dict[weight_key]
            current = packed if self.weight is None else self.weight
            destination = self._destination_device(current, incoming)
            if self.weight is None or self.weight.device.type == "meta":
                self._remove_packed_state()
                if self.weight is not None:
                    delattr(self, "weight")
                self.register_parameter(
                    "weight",
                    nn.Parameter(torch.empty_like(incoming, device=destination)),
                )

        bias_key = prefix + "bias"
        if (
            bias_key in state_dict
            and self.bias is not None
            and self.bias.device.type == "meta"
        ):
            incoming_bias = state_dict[bias_key]
            self.bias = nn.Parameter(
                torch.empty_like(
                    incoming_bias,
                    device=destination or incoming_bias.device,
                )
            )

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if input.shape[-1] != self.in_features:
            raise ValueError(
                f"MXFP6 input has {input.shape[-1]} features, "
                f"expected {self.in_features}"
            )
        if input.dtype != torch.bfloat16:
            raise TypeError("AITER MXFP6 activations must use torch.bfloat16")
        if not hasattr(self, "weight_packed"):
            self._quantize_weights()

        original_shape = input.shape
        output = torch.ops.xfuser.mxfp6_gemm(
            input.reshape(-1, self.in_features),
            self.weight_packed,
            self.weight_scale,
            self.out_features,
            self.in_features,
        )
        if self.bias is not None:
            output = output + self.bias
        return output.reshape(*original_shape[:-1], self.out_features)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, bias={self.bias is not None}"
        )
