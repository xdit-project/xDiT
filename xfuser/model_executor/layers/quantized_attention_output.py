"""Output projections that consume a quantized attention output directly."""

import math
from typing import NamedTuple

import torch
from torch import nn

E4M3_EXTERNAL_SCALE = "e4m3_external_scale"


class QuantizedAttentionOutput(NamedTuple):
    """Attention output already quantized for the following output projection.

    ``payload`` and ``scale`` hold this rank's sequence shard in the projection
    GEMM's activation layout. They alias transport buffers that are rewritten two
    attention calls later, so the projection must consume them immediately.
    """

    payload: torch.Tensor
    scale: torch.Tensor
    codec: str


def _active_linear(linear: nn.Module) -> nn.Module:
    active_linear = getattr(linear, "active_linear", None)
    return active_linear() if callable(active_linear) else linear


def _is_torchao_tensorwise_fp8(linear: nn.Module) -> bool:
    try:
        from torchao.quantization.quantize_.workflows.float8.float8_tensor import Float8Tensor
    except ImportError:
        return False
    weight = getattr(linear, "weight", None)
    return isinstance(weight, Float8Tensor) and weight.scale.numel() == 1


def e4m3_external_scale_from_amax(amax: torch.Tensor) -> torch.Tensor:
    """FP32 descales that TorchAO's dynamic per-tensor FP8 quantization derives from ``amax``.

    ``amax`` must keep the activation dtype: TorchAO clamps and divides in that dtype
    before converting the scale to FP32.
    """
    from xfuser.core.utils.runner_utils import FP8_ACTIVATION_SCALE_FLOOR

    scale = torch.clamp(amax, min=FP8_ACTIVATION_SCALE_FLOOR) / torch.finfo(torch.float8_e4m3fn).max
    return scale.to(torch.float32)


def attention_output_codec(linear: nn.Module) -> str | None:
    """Activation codec the projection GEMM consumes, or None if it needs BF16 input."""
    linear = _active_linear(linear)
    codec = getattr(linear, "prequantized_input_codec", None)
    if codec is not None:
        return codec
    if _is_torchao_tensorwise_fp8(linear):
        return E4M3_EXTERNAL_SCALE
    return None


def project_attention_output(
    linear: nn.Module,
    output: QuantizedAttentionOutput,
    logical_shape: tuple[int, ...],
) -> torch.Tensor:
    """Apply ``linear`` to ``output``, whose BF16 equivalent has ``logical_shape``."""
    linear = _active_linear(linear)
    codec = attention_output_codec(linear)
    if codec != output.codec:
        raise RuntimeError(f"attention output is quantized as {output.codec}, but the projection consumes {codec}")
    if codec == E4M3_EXTERNAL_SCALE:
        return _torchao_tensorwise_linear(linear, output.payload, output.scale, logical_shape)
    return linear.forward_prequantized(output.payload, output.scale, logical_shape)


def _torchao_tensorwise_linear(
    linear: nn.Module,
    payload: torch.Tensor,
    scale: torch.Tensor,
    logical_shape: tuple[int, ...],
) -> torch.Tensor:
    from torchao.quantization.quantize_.workflows.float8.float8_tensor import Float8Tensor

    weight = linear.weight
    rows, features = math.prod(logical_shape[:-1]), logical_shape[-1]
    activation = Float8Tensor(
        payload.view(weight.qdata.dtype).view(rows, features),
        scale.view(1, 1),
        block_size=[rows, features],
        mm_config=weight.mm_config,
        kernel_preference=weight.kernel_preference,
        dtype=weight.dtype,
    )
    # TorchAO rejects a quantized activation while the weight still requests activation quantization.
    weight = Float8Tensor(
        weight.qdata,
        weight.scale,
        block_size=weight.block_size,
        mm_config=weight.mm_config,
        kernel_preference=weight.kernel_preference,
        dtype=weight.dtype,
    )
    output = nn.functional.linear(activation, weight, linear.bias)
    return output.view(*logical_shape[:-1], linear.out_features)
