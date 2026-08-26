"""MXFP4 registration tests that do not require AITER kernels."""

from types import SimpleNamespace

import pytest


@pytest.fixture(scope="module")
def runtime():
    torch = pytest.importorskip("torch", reason="PyTorch is required for MXFP4 tests")
    from xfuser.model_executor.layers import mxfp4_linear

    return SimpleNamespace(torch=torch, module=mxfp4_linear)


@pytest.fixture
def fake_aiter_quantization(runtime, monkeypatch):
    torch = runtime.torch
    quant_type = SimpleNamespace(per_1x32=object())

    def get_hip_quant(_):
        def quantize(weight, shuffle=True):
            assert shuffle
            packed = torch.zeros(
                (weight.shape[0], weight.shape[1] // 2), dtype=torch.uint8
            )
            scale = torch.ones((weight.shape[0], 1), dtype=torch.float32)
            return packed, scale

        return quantize

    monkeypatch.setattr(
        runtime.module,
        "aiter",
        SimpleNamespace(QuantType=quant_type, get_hip_quant=get_hip_quant),
        raising=False,
    )
    monkeypatch.setattr(
        runtime.module, "shuffle_weight", lambda weight, layout: weight, raising=False
    )


def test_quantized_weight_is_non_trainable_parameter_and_scale_is_buffer(
    runtime,
    fake_aiter_quantization,
):
    layer = runtime.module.xFuserMXFP4Linear(8, 4, bias=False)

    layer._quantize_weights()

    parameters = dict(layer.named_parameters())
    buffers = dict(layer.named_buffers())
    assert layer.weight is None
    assert parameters["weight_shuffle"] is layer.weight_shuffle
    assert not layer.weight_shuffle.requires_grad
    assert "weight_shuffle" not in buffers
    assert buffers["weight_scale"] is layer.weight_scale
    assert set(layer.state_dict()) == {"weight_shuffle", "weight_scale"}

    layer.to("meta")
    assert layer.weight_shuffle.device.type == "meta"
    assert layer.weight_scale.device.type == "meta"


def test_forward_uses_registered_quantized_parameter_without_aiter_kernel(
    runtime,
    fake_aiter_quantization,
):
    torch = runtime.torch
    layer = runtime.module.xFuserMXFP4Linear(8, 4, bias=False)
    layer._quantize_weights()
    seen = {}

    def fake_mm(inputs, weight, scale, bias):
        seen.update(weight=weight, scale=scale, bias=bias)
        return torch.zeros((inputs.shape[0], layer.out_features), dtype=inputs.dtype)

    layer.mm = fake_mm
    output = layer(torch.ones(2, 3, 8))

    assert output.shape == (2, 3, 4)
    assert seen["weight"] is layer.weight_shuffle
    assert seen["scale"] is layer.weight_scale
    assert seen["bias"] is None


def test_quantized_state_dict_round_trips(runtime, fake_aiter_quantization):
    torch = runtime.torch
    source = runtime.module.xFuserMXFP4Linear(8, 4, bias=False)
    destination = runtime.module.xFuserMXFP4Linear(8, 4, bias=False)
    source._quantize_weights()
    with torch.no_grad():
        source.weight_shuffle.fill_(7)
        source.weight_scale.fill_(0.25)

    state = source.state_dict()
    result = destination.load_state_dict(state)

    assert not result.missing_keys
    assert not result.unexpected_keys
    assert destination.weight is None
    assert destination.weight_shuffle.device.type == "cpu"
    assert destination.weight_shuffle.dtype == state["weight_shuffle"].dtype
    assert destination.weight_scale.dtype == state["weight_scale"].dtype
    assert not destination.weight_shuffle.requires_grad
    assert torch.equal(destination.weight_shuffle, source.weight_shuffle)
    assert torch.equal(destination.weight_scale, source.weight_scale)


def test_cpu_quantized_state_load_preserves_destination_device(
    runtime, fake_aiter_quantization
):
    source = runtime.module.xFuserMXFP4Linear(8, 4, bias=False)
    source._quantize_weights()
    destination = runtime.module.xFuserMXFP4Linear(
        8, 4, bias=False, device="cpu", dtype=runtime.torch.float16
    )

    destination.load_state_dict(source.state_dict())

    assert destination.weight_shuffle.device.type == "cpu"
    assert destination.weight_shuffle.dtype == source.weight_shuffle.dtype
    assert destination.weight_scale.device.type == "cpu"
    assert destination.weight_scale.dtype == source.weight_scale.dtype


def test_cuda_quantized_state_load_keeps_destination_on_cuda(
    runtime, fake_aiter_quantization
):
    torch = runtime.torch
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for destination-device state loading")
    source = runtime.module.xFuserMXFP4Linear(8, 4, bias=False)
    source._quantize_weights()
    destination = runtime.module.xFuserMXFP4Linear(
        8, 4, bias=False, device="cuda", dtype=torch.float16
    )

    destination.load_state_dict(source.state_dict())

    assert destination.weight_shuffle.device.type == "cuda"
    assert destination.weight_scale.device.type == "cuda"
    assert torch.equal(destination.weight_shuffle.cpu(), source.weight_shuffle)
    assert torch.equal(destination.weight_scale.cpu(), source.weight_scale)


def test_packed_state_materializes_fresh_meta_destination(
    runtime, fake_aiter_quantization
):
    torch = runtime.torch
    source = runtime.module.xFuserMXFP4Linear(8, 4, bias=False)
    source._quantize_weights()
    destination = runtime.module.xFuserMXFP4Linear(8, 4, bias=False, device="meta")

    result = destination.load_state_dict(source.state_dict())

    assert not result.missing_keys
    assert not result.unexpected_keys
    assert destination.weight is None
    assert destination.weight_shuffle.device.type == "cpu"
    assert destination.weight_scale.device.type == "cpu"
    assert torch.equal(destination.weight_shuffle, source.weight_shuffle)
    assert torch.equal(destination.weight_scale, source.weight_scale)


def test_managed_packed_parameter_rejects_both_state_layout_transitions(
    runtime, fake_aiter_quantization, monkeypatch
):
    packed_source = runtime.module.xFuserMXFP4Linear(8, 4, bias=False)
    packed_source._quantize_weights()
    full_source = runtime.module.xFuserMXFP4Linear(8, 4, bias=False)
    destination = runtime.module.xFuserMXFP4Linear(8, 4, bias=False)
    destination._quantize_weights()
    monkeypatch.setattr(
        destination, "_is_fsdp_managed_parameter", lambda parameter: True
    )

    with pytest.raises(RuntimeError, match="packed state cannot be loaded after FSDP"):
        destination.load_state_dict(packed_source.state_dict())
    with pytest.raises(
        RuntimeError,
        match="full-precision state cannot replace an FSDP-managed packed parameter",
    ):
        destination.load_state_dict(full_source.state_dict())


def test_unquantized_state_dict_still_loads_strictly(runtime):
    torch = runtime.torch
    source = runtime.module.xFuserMXFP4Linear(8, 4, bias=True)
    destination = runtime.module.xFuserMXFP4Linear(8, 4, bias=True)
    with torch.no_grad():
        source.weight.fill_(0.5)
        source.bias.fill_(-0.25)

    result = destination.load_state_dict(source.state_dict())

    assert not result.missing_keys
    assert not result.unexpected_keys
    assert torch.equal(destination.weight, source.weight)
    assert torch.equal(destination.bias, source.bias)
    assert not hasattr(destination, "weight_shuffle")
