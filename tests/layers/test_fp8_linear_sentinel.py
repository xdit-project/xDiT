"""Unit tests for the `weight` sentinel of xFuserFP8BlockScaleLinear.

Once a layer is quantized, `weight` is a 0-element plain attribute rather than a parameter, so that
it stays out of named_parameters (the replicated broadcast walks those in lockstep across ranks)
while still answering the dtype/device reads that T5-family text encoders make. Being neither a
parameter nor a buffer, it is invisible to nn.Module's own device/dtype plumbing, which is what
these tests pin. CPU-only: no AITER kernel is invoked, and "meta" stands in for a second device.

Run with:
    pytest tests/layers/fp8_linear_sentinel_test.py -v
"""

import pytest

torch = pytest.importorskip("torch")

from xfuser.model_executor.layers.fp8_linear import xFuserFP8BlockScaleLinear


def make_quantized_layer(in_features=8, out_features=4, bias=True):
    """A layer in its post-quantization shape, without running an AITER kernel to get there."""
    layer = xFuserFP8BlockScaleLinear(
        in_features,
        out_features,
        bias=bias,
        device="cpu",
        dtype=torch.bfloat16,
        preshuffle=False,
    )
    if bias:
        layer.bias = torch.nn.Parameter(torch.zeros(out_features, dtype=torch.bfloat16))
    # Stand-ins for the quantizer's outputs: only their device/dtype matter here.
    layer.weight_fp8 = torch.nn.Parameter(
        torch.zeros(out_features, in_features, dtype=torch.bfloat16),
        requires_grad=False,
    )
    layer.register_buffer("weight_scale", torch.ones(1, 1, dtype=torch.float32))
    layer._install_weight_sentinel()
    return layer


def make_layer_with_meta_sentinel(in_features=8, out_features=4):
    """A layer in the shape the replicated broadcast leaves it: quantized tensors filled with real
    data from rank 0, but the sentinel still on meta because a plain __dict__ attribute is not
    something the broadcast (which walks parameters and buffers) can rehome.
    """
    with torch.device("meta"):
        layer = xFuserFP8BlockScaleLinear(
            in_features, out_features, bias=False, dtype=torch.bfloat16, preshuffle=False
        )
        layer.weight_fp8 = torch.nn.Parameter(
            torch.zeros(out_features, in_features, dtype=torch.bfloat16), requires_grad=False
        )
        layer.register_buffer("weight_scale", torch.ones(1, 1, dtype=torch.float32))
        layer._install_weight_sentinel()
    assert layer.weight.is_meta, "precondition: quantizing under meta leaves a meta sentinel"
    layer.weight_fp8 = torch.nn.Parameter(
        torch.zeros(out_features, in_features, dtype=torch.bfloat16), requires_grad=False
    )
    layer.weight_scale = torch.ones(1, 1, dtype=torch.float32)
    return layer


def test_sentinel_is_invisible_to_named_parameters_and_buffers():
    """A sentinel counted as a tensor would desync the rank0 broadcast's lockstep walk."""
    layer = make_quantized_layer()
    names = {n for n, _ in layer.named_parameters()} | {
        n for n, _ in layer.named_buffers()
    }
    assert "weight" not in names
    assert layer.weight.numel() == 0


def test_sentinel_follows_a_module_move():
    """_apply only walks params and buffers, so without the override the sentinel keeps a stale
    device and sends a `weight`-probing caller (e.g. T5's wo.weight.dtype cast) elsewhere.
    """
    layer = make_quantized_layer().to("meta")
    assert layer.weight_fp8.device.type == "meta"
    assert layer.weight.device.type == "meta"


def test_sentinel_follows_a_dtype_cast():
    layer = make_quantized_layer().to(torch.float32)
    assert layer.weight.dtype == torch.float32


def test_sentinel_survives_a_move_when_absent():
    """Before quantization `weight` is still a (None) parameter, so the move must not trip on it."""
    layer = xFuserFP8BlockScaleLinear(
        8, 4, bias=False, device="cpu", dtype=torch.bfloat16
    )
    layer.to("meta")
    assert layer.weight is None


def test_move_fp8_weights_leaves_other_tensors_alone():
    """Load-time eviction runs while `bias` is still meta and unmovable, so it moves only the
    quantized tensors — and the sentinel, which advertises where they are."""
    layer = make_quantized_layer()
    layer.move_fp8_weights_to("meta")
    assert layer.weight_fp8.device.type == "meta"
    assert layer.weight_scale.device.type == "meta"
    assert layer.weight.device.type == "meta"
    assert layer.bias.device.type == "cpu"


def test_move_off_meta_rebuilds_the_sentinel_rather_than_copying_it():
    """Regression: the replicated placement quantizes the text encoder under meta, so the move that
    follows the broadcast fill used to copy a meta sentinel and raise "Cannot copy out of meta
    tensor". Only the AITER block-scale path has a sentinel, and the contract limits that path to
    RDNA4, so no gfx950 run could reach this.
    """
    layer = make_layer_with_meta_sentinel()
    layer.to("cpu")
    assert layer.weight.device.type == "cpu"
    assert layer.weight.dtype == torch.bfloat16
    assert layer.weight.numel() == 0


def test_move_fp8_weights_off_meta_rebuilds_the_sentinel():
    """The same hazard by the other route: eviction moves the quantized tensors directly."""
    layer = make_layer_with_meta_sentinel()
    layer.move_fp8_weights_to("cpu")
    assert layer.weight.device.type == "cpu"
    assert layer.weight.numel() == 0


def test_absorb_moves_loader_stored_fp8_out_of_weight():
    """The transformers loader must write fp8 to `weight`; absorbing restores the DiT layout."""
    fp8_dtype = pytest.importorskip("aiter").dtypes.fp8
    layer = xFuserFP8BlockScaleLinear(
        8, 4, bias=False, device="cpu", dtype=torch.bfloat16, preshuffle=False
    )
    layer.register_parameter(
        "weight",
        torch.nn.Parameter(torch.zeros(4, 8, dtype=fp8_dtype), requires_grad=False),
    )
    layer.absorb_fp8_weight_from_weight_attr()
    assert layer.weight_fp8.dtype == fp8_dtype
    assert layer.weight.numel() == 0
    assert "weight" not in {n for n, _ in layer.named_parameters()}
