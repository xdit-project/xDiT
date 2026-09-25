"""An FP8 linear must survive an input that is entirely zeros.

Zeros are not a corner case here. A model that zero-pads its text sequence up to a multiple of the
sequence-parallel world size hands whole padding-only chunks to late ranks, so at eight ranks
Qwen-Image fed one of its text MLP layers an all-zero activation. The dynamic per-tensor scale is
max_abs / 448, which is zero for that input, and quantizing divides by the scale: the layer returned
NaN, the NaN spread to every rank through the attention all-to-all, and every FP8 render came out
pure black while bf16 at the same rank count was fine.
"""

import pytest
import torch

from xfuser.core.utils.runner_utils import FP8_ACTIVATION_SCALE_FLOOR

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="FP8 quantization needs a device")


def _quantized_linear(**overrides):
    from torchao.quantization.granularity import PerTensor
    from torchao.quantization.quant_api import (
        Float8DynamicActivationFloat8WeightConfig,
        quantize_,
    )

    torch.manual_seed(0)
    # Wide enough to take the same tensorwise kernel the models use; a 128-wide layer selects a
    # blockwise one instead and would be testing a path no model runs here.
    linear = torch.nn.Linear(1024, 1024).to(device="cuda", dtype=torch.bfloat16)
    settings = {
        "granularity": PerTensor(),
        "activation_value_lb": FP8_ACTIVATION_SCALE_FLOOR,
        **overrides,
    }
    quantize_(linear, Float8DynamicActivationFloat8WeightConfig(**settings))
    return linear


def test_an_all_zero_activation_gives_the_bias_rather_than_nan():
    linear = _quantized_linear()
    zeros = torch.zeros(4, 1024, device="cuda", dtype=torch.bfloat16)

    output = linear(zeros)

    assert torch.isfinite(output).all()
    assert torch.allclose(output, linear.bias.expand_as(output), atol=1e-2)


def test_the_floor_is_what_prevents_it():
    """Without the floor the same input returns all NaN, which is what the floor prevents."""
    unfloored = _quantized_linear(activation_value_lb=None)
    zeros = torch.zeros(4, 1024, device="cuda", dtype=torch.bfloat16)

    assert torch.isnan(unfloored(zeros)).all()


def test_ordinary_activations_are_untouched_by_the_floor():
    """The floor may only bind below itself: a change to normal numerics would be a regression."""
    floored = _quantized_linear()
    unfloored = _quantized_linear(activation_value_lb=None)
    torch.manual_seed(1)
    activations = torch.randn(4, 1024, device="cuda", dtype=torch.bfloat16)

    assert torch.equal(floored(activations), unfloored(activations))
