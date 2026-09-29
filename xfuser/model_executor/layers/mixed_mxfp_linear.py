"""AITER A6W4 inference linear for gfx950."""

from __future__ import annotations

import math
import os

import torch
from torch import nn

try:
    from aiter.ops.gemm_op_a6w4 import (
        gemm_a6w4,
        mxfp4_gemm_pack_size,
        quant_mxfp4_gemm_hip_out,
    )
    from aiter.ops.gemm_op_a6w6 import (
        mxfp6_gemm_pack_size,
        quant_mxfp6_gemm,
    )
except Exception as exc:  # noqa: BLE001 - preserve the preflight failure
    _AITER_MIXED_IMPORT_ERROR = exc
    gemm_a6w4 = None
    mxfp4_gemm_pack_size = None
    mxfp6_gemm_pack_size = None
    quant_mxfp4_gemm_hip_out = None
    quant_mxfp6_gemm = None
else:
    _AITER_MIXED_IMPORT_ERROR = None


_SUPPORTED_DTYPES = frozenset({torch.bfloat16})
_PACK_TILE = 256
_PACK_K_TILE = 128
_PACK_GUARD_TILES = 2
_SCALE_TILE_BYTES = 1024
_FP4_TILE_BYTES = 16384
_FP6_TILE_BYTES = 24576


def _require_api(name: str, value):
    if callable(value):
        return value
    detail = (
        f": {type(_AITER_MIXED_IMPORT_ERROR).__name__}: " f"{_AITER_MIXED_IMPORT_ERROR}"
        if _AITER_MIXED_IMPORT_ERROR is not None
        else ""
    )
    raise RuntimeError(f"AITER mixed-MXFP API {name} is unavailable{detail}")


def _abi_pack_sizes(
    rows: int,
    in_features: int,
    *,
    format_: str,
) -> tuple[int, int]:
    padded_rows = ((rows + _PACK_TILE - 1) // _PACK_TILE) * _PACK_TILE
    padded_k = ((in_features + _PACK_K_TILE - 1) // _PACK_K_TILE) * _PACK_K_TILE
    row_tiles = padded_rows // _PACK_TILE
    guarded_k_tiles = padded_k // _PACK_K_TILE + _PACK_GUARD_TILES
    tile_bytes = _FP4_TILE_BYTES if format_ == "fp4" else _FP6_TILE_BYTES
    return (
        row_tiles * guarded_k_tiles * tile_bytes,
        row_tiles * guarded_k_tiles * _SCALE_TILE_BYTES,
    )


def _expected_pack_sizes(
    rows: int,
    in_features: int,
    *,
    format_: str,
) -> tuple[int, int]:
    fallback = _abi_pack_sizes(rows, in_features, format_=format_)
    size_fn = mxfp4_gemm_pack_size if format_ == "fp4" else mxfp6_gemm_pack_size
    if callable(size_fn):
        actual = tuple(int(value) for value in size_fn(rows, in_features))
        if actual != fallback:
            raise RuntimeError(
                f"AITER MXFP{format_[-1]} pack-size ABI mismatch: "
                f"got {actual}, expected {fallback}"
            )
        return actual
    return fallback


def _validate_compute_matrix(
    tensor: torch.Tensor,
    *,
    name: str,
    features: int,
) -> None:
    if tensor.ndim != 2 or tensor.shape[1] != features:
        raise ValueError(
            f"{name} must have shape [rows, {features}], got {tuple(tensor.shape)}"
        )
    if tensor.dtype not in _SUPPORTED_DTYPES:
        raise TypeError(f"{name} must use torch.bfloat16, got {tensor.dtype}")
    if tensor.device.type == "meta":
        raise RuntimeError(f"{name} cannot use the meta device")


def _validate_packed_pair(
    packed: torch.Tensor,
    scale: torch.Tensor,
    *,
    rows: int,
    in_features: int,
    format_: str,
    name: str,
) -> None:
    if packed.ndim != 1 or scale.ndim != 1:
        raise ValueError(f"{name} packed values and scales must be one-dimensional")
    if packed.dtype is not torch.uint8 or scale.dtype is not torch.uint8:
        raise TypeError(f"{name} packed values and scales must use torch.uint8")
    if not packed.is_contiguous() or not scale.is_contiguous():
        raise ValueError(f"{name} packed values and scales must be contiguous")
    if packed.device != scale.device:
        raise ValueError(f"{name} packed values and scales must share a device")
    expected = _expected_pack_sizes(rows, in_features, format_=format_)
    actual = (packed.numel(), scale.numel())
    if actual != expected:
        raise ValueError(
            f"{name} packed sizes {actual} do not match {expected} for "
            f"logical shape ({rows}, {in_features})"
        )


def probe_mixed_mxfp_apis() -> tuple[bool, str | None]:
    """Check the exact AITER surface before model allocation."""

    if os.getenv("AITER_TRITON_ONLY", "0") == "1":
        return False, "mixed FP6/FP4 GEMMs require AITER ASM"
    required = {
        "mxfp4_gemm_pack_size": mxfp4_gemm_pack_size,
        "mxfp6_gemm_pack_size": mxfp6_gemm_pack_size,
        "quant_mxfp4_gemm_hip_out": quant_mxfp4_gemm_hip_out,
        "quant_mxfp6_gemm": quant_mxfp6_gemm,
        "gemm_a6w4": gemm_a6w4,
    }
    missing = [name for name, value in required.items() if not callable(value)]
    if missing:
        detail = (
            f": {type(_AITER_MIXED_IMPORT_ERROR).__name__}: "
            f"{_AITER_MIXED_IMPORT_ERROR}"
            if _AITER_MIXED_IMPORT_ERROR is not None
            else ""
        )
        return False, f"missing AITER APIs: {', '.join(missing)}{detail}"
    try:
        _expected_pack_sizes(256, 128, format_="fp4")
        _expected_pack_sizes(256, 128, format_="fp6")
    except Exception as exc:  # noqa: BLE001
        return False, f"AITER mixed packed-layout ABI check failed: {exc}"
    try:
        arch = torch.cuda.get_device_properties(torch.cuda.current_device()).gcnArchName
    except Exception as exc:  # noqa: BLE001
        return False, f"cannot determine ROCm architecture: {exc}"
    if "gfx950" not in arch:
        return False, f"mixed FP6/FP4 ASM requires gfx950, detected {arch}"
    return True, None


def _validate_mixed_output(
    output: torch.Tensor,
    *,
    rows: int,
    out_features: int,
) -> torch.Tensor:
    if tuple(output.shape) != (rows, out_features):
        raise RuntimeError(
            f"AITER A6W4 returned {tuple(output.shape)}, expected "
            f"({rows}, {out_features})"
        )
    if output.dtype is not torch.bfloat16:
        raise RuntimeError(f"AITER A6W4 must return torch.bfloat16, got {output.dtype}")
    return output.contiguous()


def _run_mixed_gemm(
    input_2d: torch.Tensor,
    weight_packed: torch.Tensor,
    weight_scale: torch.Tensor,
    out_features: int,
    in_features: int,
) -> torch.Tensor:
    _validate_compute_matrix(
        input_2d,
        name="A6W4 activation",
        features=in_features,
    )
    _validate_packed_pair(
        weight_packed,
        weight_scale,
        rows=out_features,
        in_features=in_features,
        format_="fp4",
        name="A6W4 weight",
    )
    if input_2d.device != weight_packed.device:
        raise ValueError("A6W4 activation and weight must share a device")
    rows = input_2d.shape[0]
    if rows == 0:
        return torch.empty(
            (0, out_features),
            dtype=torch.bfloat16,
            device=input_2d.device,
        )
    activation_packed, activation_scale = _require_api(
        "quant_mxfp6_gemm", quant_mxfp6_gemm
    )(input_2d)
    _validate_packed_pair(
        activation_packed,
        activation_scale,
        rows=rows,
        in_features=in_features,
        format_="fp6",
        name="A6W4 activation",
    )
    output = _require_api("gemm_a6w4", gemm_a6w4)(
        activation_packed,
        weight_packed,
        activation_scale,
        weight_scale,
        rows,
        out_features,
        in_features,
    )
    return _validate_mixed_output(
        output,
        rows=rows,
        out_features=out_features,
    )


@torch.library.custom_op("xfuser::a6w4_linear", mutates_args=())
def _a6w4_linear(
    input_2d: torch.Tensor,
    weight_packed: torch.Tensor,
    weight_scale: torch.Tensor,
    out_features: int,
    in_features: int,
) -> torch.Tensor:
    return _run_mixed_gemm(
        input_2d,
        weight_packed,
        weight_scale,
        out_features,
        in_features,
    )


@_a6w4_linear.register_fake
def _(
    input_2d: torch.Tensor,
    weight_packed: torch.Tensor,
    weight_scale: torch.Tensor,
    out_features: int,
    in_features: int,
) -> torch.Tensor:
    del weight_packed, weight_scale, in_features
    return torch.empty(
        (input_2d.shape[0], out_features),
        dtype=torch.bfloat16,
        device=input_2d.device,
    )


class _xFuserMixedMXFPLinear(nn.Module):
    """Inference-only mixed-MXFP linear; subclasses choose operand formats."""

    kind: str
    weight_format: str

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        if in_features <= 0 or out_features <= 0:
            raise ValueError("mixed-MXFP feature dimensions must be positive")
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        dtype = torch.bfloat16 if dtype is None else dtype
        if dtype not in _SUPPORTED_DTYPES:
            raise TypeError("mixed-MXFP linear dtype must be torch.bfloat16")
        self._compute_dtype = dtype
        self.weight = nn.Parameter(
            torch.empty((out_features, in_features), device=device, dtype=dtype)
        )
        if bias:
            self.bias = nn.Parameter(
                torch.empty(out_features, device=device, dtype=dtype)
            )
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.weight.device.type == "meta":
            return
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            bound = 1 / math.sqrt(self.in_features)
            nn.init.uniform_(self.bias, -bound, bound)

    def _remove_packed_state(self) -> None:
        if hasattr(self, "weight_packed"):
            delattr(self, "weight_packed")
        if hasattr(self, "weight_scale"):
            delattr(self, "weight_scale")

    def load_and_quantize_weights(
        self,
        weights: torch.Tensor,
        bias: torch.Tensor | None = None,
        *,
        device: torch.device | str | None = None,
    ) -> None:
        expected = (self.out_features, self.in_features)
        if weights.ndim != 2 or tuple(weights.shape) != expected:
            raise ValueError(
                f"{self.kind.upper()} weight must have shape {expected}, got "
                f"{tuple(weights.shape)}"
            )
        if weights.dtype not in _SUPPORTED_DTYPES:
            raise TypeError(f"{self.kind.upper()} weight has unsupported dtype")
        target = torch.device(device) if device is not None else weights.device
        if target.type == "meta":
            raise RuntimeError(f"{self.kind.upper()} weights cannot be packed on meta")
        full_weight = weights.detach().to(target)
        packed_size, scale_size = _expected_pack_sizes(
            self.out_features,
            self.in_features,
            format_=self.weight_format,
        )
        packed = torch.zeros(packed_size, dtype=torch.uint8, device=target)
        scale = torch.zeros(scale_size, dtype=torch.uint8, device=target)
        _require_api("quant_mxfp4_gemm_hip_out", quant_mxfp4_gemm_hip_out)(
            full_weight, packed, scale
        )
        _validate_packed_pair(
            packed,
            scale,
            rows=self.out_features,
            in_features=self.in_features,
            format_=self.weight_format,
            name=f"{self.kind.upper()} weight",
        )
        if bias is not None:
            if self.bias is None:
                raise ValueError("source has bias but destination was bias-free")
            if tuple(bias.shape) != (self.out_features,) or bias.dtype != weights.dtype:
                raise ValueError(f"{self.kind.upper()} bias has incompatible metadata")
            self.bias = nn.Parameter(bias.detach().to(target))
        if hasattr(self, "weight"):
            delattr(self, "weight")
        self.register_parameter("weight", None)
        self.register_parameter(
            "weight_packed",
            nn.Parameter(packed, requires_grad=False),
        )
        self.register_buffer("weight_scale", scale, persistent=True)
        self._compute_dtype = weights.dtype

    def _quantize_weights(self) -> None:
        if self.weight is None:
            raise RuntimeError(
                f"{self.kind.upper()} full-precision weight is unavailable"
            )
        weight = self.weight.detach()
        bias = self.bias.detach() if self.bias is not None else None
        self.load_and_quantize_weights(weight, bias, device=weight.device)

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
                f"{self.kind.upper()} packed state requires values and scales"
            )
        if has_packed and weight_key in state_dict:
            raise RuntimeError(
                f"{self.kind.upper()} state cannot contain full and packed weights"
            )

        destination = None
        if has_packed:
            incoming_packed = state_dict[packed_key]
            incoming_scale = state_dict[scale_key]
            _validate_packed_pair(
                incoming_packed,
                incoming_scale,
                rows=self.out_features,
                in_features=self.in_features,
                format_=self.weight_format,
                name=f"{self.kind.upper()} checkpoint weight",
            )
            current = (
                self.weight_packed if hasattr(self, "weight_packed") else self.weight
            )
            destination = (
                incoming_packed.device
                if current is None or current.device.type == "meta"
                else current.device
            )
            self._remove_packed_state()
            if hasattr(self, "weight"):
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
            incoming = state_dict[weight_key]
            current = (
                self.weight_packed if hasattr(self, "weight_packed") else self.weight
            )
            destination = (
                incoming.device if current.device.type == "meta" else current.device
            )
            self._remove_packed_state()
            if self.weight is None:
                delattr(self, "weight")
                self.register_parameter(
                    "weight",
                    nn.Parameter(torch.empty_like(incoming, device=destination)),
                )
            elif self.weight.device.type == "meta":
                self.weight = nn.Parameter(
                    torch.empty_like(incoming, device=destination)
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

    def _add_bias(
        self,
        output: torch.Tensor,
        output_dtype: torch.dtype,
    ) -> torch.Tensor:
        if self.bias is not None:
            if self.bias.device != output.device or self.bias.dtype != output_dtype:
                raise ValueError(
                    f"{self.kind.upper()} bias must match output device and dtype"
                )
            output = output + self.bias
        return output.contiguous()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if input.ndim == 0 or input.shape[-1] != self.in_features:
            raise ValueError(
                f"{self.kind.upper()} input must end in {self.in_features}"
            )
        if input.dtype != self._compute_dtype:
            raise TypeError(
                f"{self.kind.upper()} input dtype {input.dtype} does not match "
                f"weight dtype {self._compute_dtype}"
            )
        if not hasattr(self, "weight_packed"):
            self._quantize_weights()
        if input.device != self.weight_packed.device:
            raise ValueError(
                f"{self.kind.upper()} input and weight must share a device"
            )
        original_shape = input.shape
        input_2d = input.reshape(-1, self.in_features).contiguous()
        output = torch.ops.xfuser.a6w4_linear(
            input_2d,
            self.weight_packed,
            self.weight_scale,
            self.out_features,
            self.in_features,
        ).to(input.dtype)
        output = self._add_bias(output, input.dtype)
        return output.reshape(*original_shape[:-1], self.out_features).contiguous()

    def extra_repr(self) -> str:
        return (
            f"kind={self.kind}, in_features={self.in_features}, "
            f"out_features={self.out_features}, bias={self.bias is not None}"
        )


class xFuserA6W4Linear(_xFuserMixedMXFPLinear):
    kind = "a6w4"
    weight_format = "fp4"


__all__ = [
    "probe_mixed_mxfp_apis",
    "xFuserA6W4Linear",
]
