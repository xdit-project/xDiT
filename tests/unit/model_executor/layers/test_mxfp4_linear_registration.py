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
            packed = torch.zeros((weight.shape[0], weight.shape[1] // 2), dtype=torch.uint8)
            scale = torch.ones((weight.shape[0], 1), dtype=torch.float32)
            return packed, scale

        return quantize

    monkeypatch.setattr(
        runtime.module,
        "aiter",
        SimpleNamespace(QuantType=quant_type, get_hip_quant=get_hip_quant),
        raising=False,
    )
    monkeypatch.setattr(runtime.module, "shuffle_weight", lambda weight, layout: weight, raising=False)


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


def test_cpu_quantized_state_load_preserves_destination_device(runtime, fake_aiter_quantization):
    source = runtime.module.xFuserMXFP4Linear(8, 4, bias=False)
    source._quantize_weights()
    destination = runtime.module.xFuserMXFP4Linear(8, 4, bias=False, device="cpu", dtype=runtime.torch.float16)

    destination.load_state_dict(source.state_dict())

    assert destination.weight_shuffle.device.type == "cpu"
    assert destination.weight_shuffle.dtype == source.weight_shuffle.dtype
    assert destination.weight_scale.device.type == "cpu"
    assert destination.weight_scale.dtype == source.weight_scale.dtype


@pytest.mark.accelerator
def test_cuda_quantized_state_load_keeps_destination_on_cuda(runtime, fake_aiter_quantization):
    torch = runtime.torch
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for destination-device state loading")
    source = runtime.module.xFuserMXFP4Linear(8, 4, bias=False)
    source._quantize_weights()
    destination = runtime.module.xFuserMXFP4Linear(8, 4, bias=False, device="cuda", dtype=torch.float16)

    destination.load_state_dict(source.state_dict())

    assert destination.weight_shuffle.device.type == "cuda"
    assert destination.weight_scale.device.type == "cuda"
    assert torch.equal(destination.weight_shuffle.cpu(), source.weight_shuffle)
    assert torch.equal(destination.weight_scale.cpu(), source.weight_scale)


def test_packed_state_materializes_fresh_meta_destination(runtime, fake_aiter_quantization):
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


def test_managed_packed_parameter_rejects_both_state_layout_transitions(runtime, fake_aiter_quantization, monkeypatch):
    packed_source = runtime.module.xFuserMXFP4Linear(8, 4, bias=False)
    packed_source._quantize_weights()
    full_source = runtime.module.xFuserMXFP4Linear(8, 4, bias=False)
    destination = runtime.module.xFuserMXFP4Linear(8, 4, bias=False)
    destination._quantize_weights()
    monkeypatch.setattr(destination, "_is_fsdp_managed_parameter", lambda parameter: True)

    with pytest.raises(RuntimeError, match="packed state cannot be loaded after FSDP"):
        destination.load_state_dict(packed_source.state_dict())
    with pytest.raises(
        RuntimeError,
        match="full-precision state cannot replace an FSDP-managed packed parameter",
    ):
        destination.load_state_dict(full_source.state_dict())


def test_concatenated_stacks_packed_weights_and_scales(runtime, fake_aiter_quantization):
    torch = runtime.torch
    layers = [runtime.module.xFuserMXFP4Linear(8, n, bias=False) for n in (4, 6)]
    for value, layer in enumerate(layers):
        layer._quantize_weights()
        with torch.no_grad():
            layer.weight_shuffle.fill_(value)
            layer.weight_scale.fill_(value + 0.5)

    fused = runtime.module.xFuserMXFP4Linear.concatenated(layers)

    assert (fused.in_features, fused.out_features) == (8, 10)
    assert fused.weight is None and fused.bias is None
    assert torch.equal(fused.weight_shuffle, torch.cat([layer.weight_shuffle for layer in layers]))
    assert torch.equal(fused.weight_scale, torch.cat([layer.weight_scale for layer in layers]))
    assert not fused.weight_shuffle.requires_grad
    assert set(fused.state_dict()) == {"weight_shuffle", "weight_scale"}


def test_concatenated_refuses_unquantized_or_biased_layers(runtime, fake_aiter_quantization):
    unquantized = runtime.module.xFuserMXFP4Linear(8, 4, bias=False)
    biased = runtime.module.xFuserMXFP4Linear(8, 4, bias=True)
    biased._quantize_weights()

    with pytest.raises(ValueError, match="Quantize"):
        runtime.module.xFuserMXFP4Linear.concatenated([unquantized])
    with pytest.raises(ValueError, match="bias-free"):
        runtime.module.xFuserMXFP4Linear.concatenated([biased])


def test_concatenated_refuses_padded_scales_before_the_last_layer(runtime, fake_aiter_quantization):
    torch = runtime.torch
    layers = [runtime.module.xFuserMXFP4Linear(8, 128, bias=False) for _ in range(2)]
    for layer in layers:
        layer._quantize_weights()
        # AITER's shuffled scale has a multiple of 256 rows.
        layer.weight_scale = torch.ones((256, 1), dtype=torch.uint8)

    with pytest.raises(ValueError, match="padded scale rows"):
        runtime.module.xFuserMXFP4Linear.concatenated(layers)

    layers[0].weight_scale = torch.ones((128, 1), dtype=torch.uint8)
    fused = runtime.module.xFuserMXFP4Linear.concatenated(layers)
    assert fused.weight_scale.shape == (384, 1)


@pytest.mark.accelerator
def test_concatenated_matches_quantizing_the_stacked_weight():
    # The packing and scale preshuffle must be local to row blocks for stacking to be exact.
    torch = pytest.importorskip("torch")
    pytest.importorskip("aiter")
    if not torch.cuda.is_available():
        pytest.skip("AITER MXFP4 kernels need a GPU")
    from xfuser.model_executor.layers.mxfp4_linear import xFuserMXFP4Linear

    torch.manual_seed(0)
    widths = (3072, 3072, 3072, 18432)
    weights = [torch.randn(n, 3072, device="cuda", dtype=torch.bfloat16) for n in widths]
    layers = []
    for weight in weights:
        layer = xFuserMXFP4Linear(3072, weight.shape[0], bias=False, device="cuda", dtype=torch.bfloat16)
        layer.load_and_quantize_weights(weight)
        layers.append(layer)
    stacked = xFuserMXFP4Linear(3072, sum(widths), bias=False, device="cuda", dtype=torch.bfloat16)
    stacked.load_and_quantize_weights(torch.cat(weights))

    fused = xFuserMXFP4Linear.concatenated(layers)

    assert torch.equal(fused.weight_shuffle, stacked.weight_shuffle)
    assert torch.equal(fused.weight_scale, stacked.weight_scale)
    x = torch.randn(522, 3072, device="cuda", dtype=torch.bfloat16)
    actual = fused(x).float()
    expected = torch.cat([layer(x) for layer in layers], dim=-1).float()
    # Another GEMM tile may round some outputs one bf16 step differently.
    assert ((actual - expected).norm() / expected.norm()).item() < 1e-3


def test_stacked_along_input_refuses_layers_it_cannot_join(runtime, fake_aiter_quantization):
    narrow = runtime.module.xFuserMXFP4Linear(8, 4, bias=False)
    narrow._quantize_weights()
    unquantized = runtime.module.xFuserMXFP4Linear(256, 4, bias=False)

    with pytest.raises(ValueError, match="multiples of 256"):
        runtime.module.xFuserMXFP4Linear.stacked_along_input([narrow, narrow])
    with pytest.raises(ValueError, match="Quantize"):
        runtime.module.xFuserMXFP4Linear.stacked_along_input([unquantized])
    with pytest.raises(ValueError, match="same output width"):
        runtime.module.xFuserMXFP4Linear.stacked_along_input([narrow, runtime.module.xFuserMXFP4Linear(8, 6)])


@pytest.mark.accelerator
def test_stacked_along_input_matches_quantizing_the_joined_weight():
    # The packing and both preshuffles must keep each row block's K chunks in order.
    torch = pytest.importorskip("torch")
    pytest.importorskip("aiter")
    if not torch.cuda.is_available():
        pytest.skip("AITER MXFP4 kernels need a GPU")
    from xfuser.model_executor.layers.mxfp4_linear import xFuserMXFP4Linear

    torch.manual_seed(0)
    widths = (3072, 9216)
    weights = [torch.randn(3072, k, device="cuda", dtype=torch.bfloat16) for k in widths]
    layers = []
    for weight in weights:
        layer = xFuserMXFP4Linear(weight.shape[1], 3072, bias=False, device="cuda", dtype=torch.bfloat16)
        layer.load_and_quantize_weights(weight)
        layers.append(layer)
    joined = xFuserMXFP4Linear(sum(widths), 3072, bias=False, device="cuda", dtype=torch.bfloat16)
    joined.load_and_quantize_weights(torch.cat(weights, dim=1))

    fused = xFuserMXFP4Linear.stacked_along_input(layers)

    assert (fused.in_features, fused.out_features) == (sum(widths), 3072)
    assert torch.equal(fused.weight_shuffle, joined.weight_shuffle)
    assert torch.equal(fused.weight_scale, joined.weight_scale)
    xs = [torch.randn(522, k, device="cuda", dtype=torch.bfloat16) for k in widths]
    actual = fused(torch.cat(xs, dim=1)).float()
    expected = sum(layer(x).float() for layer, x in zip(layers, xs))
    # One GEMM accumulates in fp32 what the pair rounds to bf16 twice.
    assert ((actual - expected).norm() / expected.norm()).item() < 1e-2


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
