"""The block-scale GEMM's operands must be on one device by the time it is launched.

The kernel takes the weight and its scale as raw pointers and reports only an argument index when
one of them is unreachable ("Pointer argument (at 4) cannot be accessed from Triton"), which says
nothing about which tensor or how it got there. The two are carried by different mechanisms —
weight_fp8 is a parameter FSDP2 shards and all-gathers itself, weight_scale a replicated buffer that
only a module move carries — so a run that drives both at once (a sharded load under
--enable_model_cpu_offload) can leave them apart.

CPU-only: no kernel is launched, and "meta" stands in for the accelerator.
"""

import pytest

torch = pytest.importorskip("torch")

from xfuser.model_executor.layers.fp8_linear import xFuserFP8BlockScaleLinear


def make_quantized_layer(in_features=8, out_features=4):
    """A layer in its post-quantization shape, without running an AITER kernel to get there."""
    layer = xFuserFP8BlockScaleLinear(
        in_features, out_features, bias=False, device="cpu", dtype=torch.bfloat16, preshuffle=False
    )
    layer.weight_fp8 = torch.nn.Parameter(
        torch.zeros(out_features, in_features, dtype=torch.bfloat16), requires_grad=False
    )
    layer.register_buffer("weight_scale", torch.ones(1, 1, dtype=torch.float32))
    layer._install_weight_sentinel()
    return layer


def test_a_scale_left_behind_is_brought_to_the_activation():
    """The reported failure: the weight reaches the device, its scale is still on the host."""
    layer = make_quantized_layer()
    layer.weight_fp8 = torch.nn.Parameter(layer.weight_fp8.to("meta"), requires_grad=False)
    activation = torch.zeros(2, 8, dtype=torch.bfloat16, device="meta")
    assert layer.weight_scale.device.type == "cpu", "precondition: scale lags the weight"

    weight, scale = layer._gemm_operands(activation)

    assert weight.device == activation.device
    assert scale.device == activation.device


def test_the_rehomed_scale_is_kept_so_the_move_is_paid_once():
    """Rehoming per forward would put a host-to-device copy in front of every GEMM."""
    layer = make_quantized_layer()
    layer.weight_fp8 = torch.nn.Parameter(layer.weight_fp8.to("meta"), requires_grad=False)
    activation = torch.zeros(2, 8, dtype=torch.bfloat16, device="meta")

    layer._gemm_operands(activation)

    assert layer.weight_scale.device.type == "meta"
    assert "weight_scale" in layer._buffers, "it has to stay a buffer to keep following module moves"


def test_a_misplaced_weight_is_named_rather_than_left_to_the_kernel():
    """Too big to move under the caller's feet, so the only useful thing to do is say where it is."""
    layer = make_quantized_layer()
    activation = torch.zeros(2, 8, dtype=torch.bfloat16, device="meta")

    with pytest.raises(RuntimeError, match="FP8 weight is on cpu.*activation is on meta"):
        layer._gemm_operands(activation)


def test_operands_already_together_are_left_alone():
    """The check must not touch a layer that is already correct."""
    layer = make_quantized_layer()
    scale_before = layer.weight_scale
    activation = torch.zeros(2, 8, dtype=torch.bfloat16)

    weight, scale = layer._gemm_operands(activation)

    assert weight is layer.weight_fp8
    assert scale is scale_before


def test_an_unquantized_layer_still_says_so():
    """The pre-existing complaint about a layer nobody loaded must survive the placement checks."""
    layer = xFuserFP8BlockScaleLinear(
        8, 4, bias=False, device="cpu", dtype=torch.bfloat16, preshuffle=False
    )
    layer.register_parameter("weight", None)

    with pytest.raises(RuntimeError, match="FP8 weight not initialized"):
        layer._gemm_operands(torch.zeros(2, 8, dtype=torch.bfloat16))
