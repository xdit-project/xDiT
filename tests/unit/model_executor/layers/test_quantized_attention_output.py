"""Output projections that consume a quantized attention output, with fake AITER kernels."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from xfuser.model_executor.layers import mixed_mxfp_linear, mxfp4_linear, mxfp6_linear
from xfuser.model_executor.layers.quantized_attention_output import (
    E4M3_EXTERNAL_SCALE,
    QuantizedAttentionOutput,
    attention_output_codec,
    e4m3_external_scale_from_amax,
    project_attention_output,
)

IN_FEATURES = 64
OUT_FEATURES = 16


def _fake_gemm(activation_packed, rows, out_features):
    """A GEMM result that depends on every packed activation byte."""
    total = activation_packed.view(torch.uint8).to(torch.int64).sum().item()
    return torch.full((rows, out_features), float(total % 251), dtype=torch.bfloat16)


def _fake_pack(tensor, packed_size, scale_size):
    data = (tensor.float().abs() * 7).to(torch.int64).flatten() % 256
    packed = data.repeat(packed_size // data.numel() + 1)[:packed_size].to(torch.uint8)
    return packed, torch.ones(scale_size, dtype=torch.uint8)


@pytest.fixture
def fake_mxfp4(monkeypatch):
    def get_hip_quant(_):
        def quantize(tensor, shuffle=True):
            assert shuffle
            packed, scale = _fake_pack(tensor, tensor.shape[0] * tensor.shape[1] // 2, tensor.shape[0])
            return packed.view(tensor.shape[0], -1), scale.view(tensor.shape[0], 1)

        return quantize

    def gemm_a4w4(a_quant, w_quant, a_scale, w_scale, bpreshuffle=True, bias=None):
        assert bpreshuffle and bias is None
        return _fake_gemm(a_quant, a_quant.shape[0], w_quant.shape[0])

    fake = SimpleNamespace(
        QuantType=SimpleNamespace(per_1x32=object()),
        dtypes=SimpleNamespace(fp4x2=torch.uint8, fp8_e8m0=torch.uint8),
        get_hip_quant=get_hip_quant,
        gemm_a4w4=gemm_a4w4,
    )
    monkeypatch.setattr(mxfp4_linear, "aiter", fake, raising=False)
    monkeypatch.setattr(mxfp4_linear, "shuffle_weight", lambda weight, layout: weight, raising=False)
    return fake


@pytest.fixture
def fake_mxfp6(monkeypatch):
    def pack_size(rows, features):
        return rows * features * 3 // 4, rows * features // 32

    def quantize(tensor):
        return _fake_pack(tensor, *pack_size(*tensor.shape))

    def gemm_a6w6(
        activation_packed, weight_packed, activation_scale, weight_scale, rows, out_features, in_features, dtype
    ):
        return _fake_gemm(activation_packed, rows, out_features)

    monkeypatch.setattr(mxfp6_linear, "mxfp6_gemm_pack_size", pack_size)
    monkeypatch.setattr(mxfp6_linear, "quant_mxfp6_gemm", quantize)
    monkeypatch.setattr(mxfp6_linear, "gemm_a6w6", gemm_a6w6)
    return quantize


def _activation(rows=6):
    torch.manual_seed(0)
    return torch.randn(1, rows, IN_FEATURES, dtype=torch.bfloat16)


def _project(linear, payload, scale, codec, logical_shape):
    return project_attention_output(linear, QuantizedAttentionOutput(payload, scale, codec), logical_shape)


class _Consumer(nn.Module):
    def __init__(self, codec):
        super().__init__()
        self.prequantized_input_codec = codec
        self.calls = []

    def forward_prequantized(self, payload, scale, logical_shape):
        self.calls.append((payload, scale, logical_shape))
        return payload


@pytest.mark.parametrize(
    ("layer", "codec"),
    [
        (lambda: mxfp4_linear.xFuserMXFP4Linear(IN_FEATURES, OUT_FEATURES), "mxfp4"),
        (lambda: mxfp6_linear.xFuserMXFP6Linear(IN_FEATURES, OUT_FEATURES), "mxfp6"),
        (lambda: mixed_mxfp_linear.xFuserA6W4Linear(IN_FEATURES, OUT_FEATURES), "mxfp6"),
        (lambda: nn.Linear(IN_FEATURES, OUT_FEATURES), None),
    ],
)
def test_codec_is_the_activation_format_of_the_projection_gemm(layer, codec):
    assert attention_output_codec(layer()) == codec


@pytest.mark.parametrize(("use_high_precision", "codec"), [(True, "mxfp6"), (False, "mxfp4")])
def test_hybrid_projection_follows_the_active_linear(monkeypatch, use_high_precision, codec):
    monkeypatch.setattr(
        mxfp4_linear,
        "get_runtime_state",
        lambda: SimpleNamespace(use_high_precision_gemm=use_high_precision),
    )
    high, low = _Consumer("mxfp6"), _Consumer("mxfp4")
    hybrid = mxfp4_linear.xFuserHybridMXFP4Linear(high, low)
    payload, scale = torch.zeros(2, dtype=torch.uint8), torch.ones(1, dtype=torch.uint8)

    assert attention_output_codec(hybrid) == codec
    assert _project(hybrid, payload, scale, codec, (1, 2, IN_FEATURES)) is payload
    active, inactive = (high, low) if use_high_precision else (low, high)
    assert active.calls == [(payload, scale, (1, 2, IN_FEATURES))]
    assert inactive.calls == []


def test_projection_rejects_an_output_quantized_for_another_gemm():
    payload, scale = torch.zeros(2, dtype=torch.uint8), torch.ones(1, dtype=torch.uint8)

    with pytest.raises(RuntimeError, match="quantized as mxfp4, but the projection consumes mxfp6"):
        _project(_Consumer("mxfp6"), payload, scale, "mxfp4", (1, 2, IN_FEATURES))
    with pytest.raises(RuntimeError, match="the projection consumes None"):
        _project(nn.Linear(IN_FEATURES, OUT_FEATURES), payload, scale, "mxfp4", (1, 2, IN_FEATURES))


def test_torchao_tensorwise_fp8_projection_uses_external_scales():
    pytest.importorskip("torchao")
    from torchao.quantization.quantize_.workflows.float8.float8_tensor import Float8Tensor

    def linear_with_scale(scale):
        linear = nn.Linear(IN_FEATURES, OUT_FEATURES, bias=False)
        qdata = torch.zeros(OUT_FEATURES, IN_FEATURES, dtype=torch.float8_e4m3fn)
        block_size = [1 if scale.numel() > 1 else OUT_FEATURES, IN_FEATURES]
        weight = Float8Tensor(qdata, scale, block_size=block_size, dtype=torch.bfloat16)
        linear.weight = nn.Parameter(weight, requires_grad=False)
        return linear

    assert attention_output_codec(linear_with_scale(torch.ones(1, 1))) == E4M3_EXTERNAL_SCALE
    assert attention_output_codec(linear_with_scale(torch.ones(OUT_FEATURES, 1))) is None


def test_external_scale_divides_in_the_activation_dtype():
    amax = torch.tensor([3.140625, 0.0], dtype=torch.bfloat16)

    scale = e4m3_external_scale_from_amax(amax)

    assert scale.dtype is torch.float32
    assert torch.equal(scale, (amax.clamp(min=1e-12) / 448.0).float())
    assert scale[0] != torch.tensor(3.140625) / 448.0


def test_mxfp4_prequantized_forward_matches_forward(fake_mxfp4):
    linear = mxfp4_linear.xFuserMXFP4Linear(IN_FEATURES, OUT_FEATURES)
    activation = _activation()
    payload, scale = fake_mxfp4.get_hip_quant(None)(activation.view(-1, IN_FEATURES), shuffle=True)

    assert torch.equal(_project(linear, payload, scale, "mxfp4", tuple(activation.shape)), linear(activation))
    with pytest.raises(ValueError, match="does not match logical input shape"):
        _project(linear, payload[1:], scale, "mxfp4", tuple(activation.shape))


def test_mxfp6_prequantized_forward_matches_forward(fake_mxfp6):
    linear = mxfp6_linear.xFuserMXFP6Linear(IN_FEATURES, OUT_FEATURES)
    activation = _activation()
    payload, scale = fake_mxfp6(activation.view(-1, IN_FEATURES))

    assert torch.equal(_project(linear, payload, scale, "mxfp6", tuple(activation.shape)), linear(activation))
    with pytest.raises(ValueError, match="do not match logical input shape"):
        _project(linear, payload[1:], scale, "mxfp6", tuple(activation.shape))
