"""Output projections that consume a quantized attention output directly.

A consumer provides ``prequantized_input_codec``, the codec its GEMM consumes, and
``forward_prequantized(payload, scale, logical_shape)``, which applies the GEMM to an
activation already packed in that codec's layout. xDiT's quantized linears implement
both; modules that xDiT does not own get them from a registered adapter.
"""

import math
from collections.abc import Callable
from typing import NamedTuple, Protocol

import torch
from torch import nn

from xfuser.config.attention_a2a_codecs import E4M3_EXTERNAL_SCALE


class QuantizedAttentionOutput(NamedTuple):
    """Attention output already quantized for the following output projection.

    ``payload`` and ``scale`` hold this rank's sequence shard in the activation
    layout of the projection GEMM that consumes ``codec``. They alias transport
    buffers that are rewritten two attention calls later, so the projection must
    consume them immediately.
    """

    payload: torch.Tensor
    scale: torch.Tensor
    codec: str


class PrequantizedConsumer(Protocol):
    prequantized_input_codec: str | None

    def forward_prequantized(
        self,
        payload: torch.Tensor,
        scale: torch.Tensor,
        logical_shape: tuple[int, ...],
    ) -> torch.Tensor: ...


_ADAPTERS: list[Callable[[nn.Module], PrequantizedConsumer | None]] = []


def register_prequantized_adapter(adapter):
    """Register ``adapter(module)``, which returns a consumer for the modules it supports and None otherwise."""
    _ADAPTERS.append(adapter)
    return adapter


def prequantized_consumer(linear: nn.Module) -> PrequantizedConsumer | None:
    """The consumer that applies ``linear`` to a quantized activation, or None if its GEMM needs BF16 input."""
    if getattr(linear, "prequantized_input_codec", None) is not None:
        return linear
    for adapter in _ADAPTERS:
        consumer = adapter(linear)
        if consumer is not None:
            return consumer
    return None


def attention_output_codec(linear: nn.Module) -> str | None:
    """Activation codec the projection GEMM consumes, or None if it needs BF16 input."""
    consumer = prequantized_consumer(linear)
    return None if consumer is None else consumer.prequantized_input_codec


def project_attention_output(
    linear: nn.Module,
    output: QuantizedAttentionOutput,
    logical_shape: tuple[int, ...],
) -> torch.Tensor:
    """Apply ``linear`` to ``output``, whose BF16 equivalent has ``logical_shape``."""
    consumer = prequantized_consumer(linear)
    codec = None if consumer is None else consumer.prequantized_input_codec
    if codec != output.codec:
        raise RuntimeError(f"attention output is quantized as {output.codec}, but the projection consumes {codec}")
    return consumer.forward_prequantized(output.payload, output.scale, logical_shape)


class _TorchAOTensorwiseFP8:
    """A TorchAO per-tensor FP8 linear applied to an ``e4m3_external_scale`` activation."""

    prequantized_input_codec = E4M3_EXTERNAL_SCALE

    def __init__(self, linear: nn.Module) -> None:
        self.linear = linear

    def forward_prequantized(
        self,
        payload: torch.Tensor,
        scale: torch.Tensor,
        logical_shape: tuple[int, ...],
    ) -> torch.Tensor:
        from torchao.quantization.quantize_.workflows.float8.float8_tensor import Float8Tensor

        weight = self.linear.weight
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
        output = nn.functional.linear(activation, weight, self.linear.bias)
        return output.view(*logical_shape[:-1], self.linear.out_features)


@register_prequantized_adapter
def _torchao_tensorwise_fp8(linear: nn.Module) -> _TorchAOTensorwiseFP8 | None:
    try:
        from torchao.quantization.quantize_.workflows.float8.float8_tensor import Float8Tensor
    except ImportError:
        return None
    weight = getattr(linear, "weight", None)
    if isinstance(weight, Float8Tensor) and weight.scale.numel() == 1:
        return _TorchAOTensorwiseFP8(linear)
    return None
