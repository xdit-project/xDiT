"""Projecting a quantized attention output matches the projection's BF16 forward bit for bit."""

import pytest
import torch
from torch import nn

from xfuser.model_executor.layers.quantized_attention_output import (
    E4M3_EXTERNAL_SCALE,
    QuantizedAttentionOutput,
    attention_output_codec,
    e4m3_external_scale_from_amax,
    project_attention_output,
)

pytestmark = [pytest.mark.accelerator, pytest.mark.rocm]

# Wan2.2 T2V keeps 1184 sequence rows per rank at Ulysses degree 8, which is not a
# multiple of the 256-row MX packing tile.
ROWS = 1184
IN_FEATURES = 1024
OUT_FEATURES = 512


@pytest.fixture
def activation():
    if not torch.cuda.is_available():
        pytest.skip("requires a ROCm device")
    arch = torch.cuda.get_device_properties(torch.cuda.current_device()).gcnArchName.split(":")[0]
    if arch != "gfx950":
        pytest.skip(f"quantized attention-output consumers require gfx950, found {arch}")
    torch.manual_seed(0)
    return torch.randn(1, ROWS, IN_FEATURES, dtype=torch.bfloat16, device="cuda")


def _project(linear, activation, payload, scale, codec):
    assert attention_output_codec(linear) == codec
    output = QuantizedAttentionOutput(payload.view(torch.uint8), scale.view(torch.uint8), codec)
    return project_attention_output(linear, output, tuple(activation.shape))


def test_torchao_fp8_projection_matches_dynamic_activation_quantization(activation):
    torchao_quantization = pytest.importorskip("torchao.quantization")
    from torchao.quantization.granularity import PerTensor

    from xfuser.core.utils.runner_utils import FP8_ACTIVATION_SCALE_FLOOR, _get_fp8_kernel_preference

    linear = nn.Linear(IN_FEATURES, OUT_FEATURES, dtype=torch.bfloat16, device="cuda")
    torchao_quantization.quantize_(
        linear,
        torchao_quantization.Float8DynamicActivationFloat8WeightConfig(
            granularity=PerTensor(),
            kernel_preference=_get_fp8_kernel_preference(),
            activation_value_lb=FP8_ACTIVATION_SCALE_FLOOR,
        ),
    )
    scale = e4m3_external_scale_from_amax(activation.abs().amax())
    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    payload = (activation.float() / scale).clamp(-fp8_max, fp8_max).to(torch.float8_e4m3fn)

    output = QuantizedAttentionOutput(payload.view(torch.uint8), scale.view(1, 1), E4M3_EXTERNAL_SCALE)
    assert attention_output_codec(linear) == E4M3_EXTERNAL_SCALE
    assert torch.equal(project_attention_output(linear, output, tuple(activation.shape)), linear(activation))


def test_mxfp4_projection_matches_forward(activation):
    aiter = pytest.importorskip("aiter")
    from xfuser.model_executor.layers.mxfp4_linear import xFuserMXFP4Linear

    linear = xFuserMXFP4Linear(IN_FEATURES, OUT_FEATURES, device="cuda", dtype=torch.bfloat16)
    linear._quantize_weights()
    quantize = aiter.get_hip_quant(aiter.QuantType.per_1x32)
    payload, scale = quantize(activation.view(-1, IN_FEATURES), shuffle=True)

    assert torch.equal(_project(linear, activation, payload, scale, "mxfp4"), linear(activation))


def test_mxfp6_projection_matches_forward(activation):
    pytest.importorskip("aiter")
    from aiter.ops.gemm_op_a6w6 import quant_mxfp6_gemm

    from xfuser.model_executor.layers.mxfp6_linear import xFuserMXFP6Linear

    linear = xFuserMXFP6Linear(IN_FEATURES, OUT_FEATURES, device="cuda")
    linear._quantize_weights()
    payload, scale = quant_mxfp6_gemm(activation.view(-1, IN_FEATURES))

    assert torch.equal(_project(linear, activation, payload, scale, "mxfp6"), linear(activation))


def test_a6w4_projection_matches_forward(activation):
    pytest.importorskip("aiter")
    from aiter.ops.gemm_op_a6w6 import quant_mxfp6_gemm

    from xfuser.model_executor.layers.mixed_mxfp_linear import xFuserA6W4Linear

    linear = xFuserA6W4Linear(IN_FEATURES, OUT_FEATURES, device="cuda", dtype=torch.bfloat16)
    linear._quantize_weights()
    payload, scale = quant_mxfp6_gemm(activation.view(-1, IN_FEATURES))

    assert torch.equal(_project(linear, activation, payload, scale, "mxfp6"), linear(activation))
