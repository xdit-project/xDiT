"""Focused lifecycle tests for the A6W4 inference linear."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture(scope="module")
def runtime():
    torch = pytest.importorskip("torch")
    path = Path(__file__).resolve().parents[4] / "xfuser" / "model_executor" / "layers" / "mixed_mxfp_linear.py"
    spec = importlib.util.spec_from_file_location("mixed_mxfp_linear_under_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return SimpleNamespace(torch=torch, module=module)


@pytest.fixture
def fake_aiter(runtime, monkeypatch):
    torch, module = runtime.torch, runtime.module
    calls = {"activation": [], "weight": [], "gemm": []}

    def size4(rows, features):
        return module._abi_pack_sizes(rows, features, format_="fp4")

    def size6(rows, features):
        return module._abi_pack_sizes(rows, features, format_="fp6")

    def pack_activation(tensor):
        calls["activation"].append(tensor)
        packed, scale = size6(*tensor.shape)
        return (
            torch.zeros(packed, dtype=torch.uint8, device=tensor.device),
            torch.ones(scale, dtype=torch.uint8, device=tensor.device),
        )

    def pack_weight(tensor, packed, scale, *_args):
        calls["weight"].append(tensor)
        packed.zero_()
        scale.fill_(1)

    def gemm(a, w, a_scale, w_scale, rows, out_features, in_features):
        calls["gemm"].append((rows, out_features, in_features))
        del w, a_scale, w_scale
        return torch.zeros(
            (rows, out_features),
            dtype=torch.bfloat16,
            device=a.device,
        )

    monkeypatch.setattr(module, "mxfp4_gemm_pack_size", size4)
    monkeypatch.setattr(module, "mxfp6_gemm_pack_size", size6)
    monkeypatch.setattr(module, "quant_mxfp6_gemm", pack_activation)
    monkeypatch.setattr(module, "quant_mxfp4_gemm_hip_out", pack_weight)
    monkeypatch.setattr(module, "gemm_a6w4", gemm)
    return calls


def test_a6w4_weight_activation_and_output_contract(runtime, fake_aiter):
    torch, cls = runtime.torch, runtime.module.xFuserA6W4Linear
    layer = cls(8, 4, bias=True, dtype=torch.bfloat16)
    layer.load_and_quantize_weights(
        torch.randn(4, 8, dtype=torch.bfloat16),
        torch.randn(4, dtype=torch.bfloat16),
    )

    assert layer.weight is None
    assert layer.weight_packed.dtype is torch.uint8
    assert len(fake_aiter["weight"]) == 1

    output = layer(torch.randn(2, 3, 8, dtype=torch.bfloat16).transpose(0, 1))
    assert output.shape == (3, 2, 4)
    assert output.is_contiguous()
    assert len(fake_aiter["activation"]) == 1
    assert fake_aiter["gemm"] == [(6, 4, 8)]


def test_a6w4_dynamic_rows_compile_fullgraph(runtime, fake_aiter):
    torch, cls = runtime.torch, runtime.module.xFuserA6W4Linear
    if not hasattr(torch, "compile"):
        pytest.skip("torch.compile unavailable")
    layer = cls(8, 4, bias=False, dtype=torch.bfloat16)
    layer.load_and_quantize_weights(torch.randn(4, 8, dtype=torch.bfloat16))
    compiled = torch.compile(layer, backend="eager", fullgraph=True, dynamic=True)
    assert compiled(torch.randn(2, 8, dtype=torch.bfloat16)).shape == (2, 4)
    assert compiled(torch.randn(5, 8, dtype=torch.bfloat16)).shape == (5, 4)


def test_a6w4_packed_state_round_trip(runtime, fake_aiter):
    torch, cls = runtime.torch, runtime.module.xFuserA6W4Linear
    source = cls(8, 4, bias=True, dtype=torch.bfloat16)
    source.load_and_quantize_weights(
        torch.randn(4, 8, dtype=torch.bfloat16),
        torch.randn(4, dtype=torch.bfloat16),
    )

    restored = cls(8, 4, bias=True, dtype=torch.bfloat16, device="meta")
    restored.load_state_dict(source.state_dict())
    assert restored.weight is None
    assert torch.equal(restored.weight_packed, source.weight_packed)
    assert torch.equal(restored.weight_scale, source.weight_scale)


def test_a6w4_probe_reports_missing_surface(runtime, monkeypatch):
    monkeypatch.setattr(runtime.module, "gemm_a6w4", None)
    available, reason = runtime.module.probe_mixed_mxfp_apis()
    assert not available
    assert "gemm_a6w4" in reason
