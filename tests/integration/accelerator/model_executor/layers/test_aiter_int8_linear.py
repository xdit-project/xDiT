import pytest
import torch

from xfuser.model_executor.layers.aiter_int8_linear import (
    _quantize_rows_int8,
    xFuserAiterInt8Linear,
)
from xfuser.model_executor.models.runner_models.loading.format_backends import (
    _probe_aiter_int8_apis,
)


def _aiter_int8_available() -> tuple[bool, str | None]:
    if not torch.cuda.is_available() or torch.version.hip is None:
        return False, "requires ROCm"
    return _probe_aiter_int8_apis()


_AVAILABLE, _REASON = _aiter_int8_available()
pytestmark = pytest.mark.skipif(not _AVAILABLE, reason=f"AITER INT8 GEMMs unavailable: {_REASON}")


@torch.no_grad()
def test_aiter_int8_quantizer_uses_round_to_nearest():
    values = torch.tensor(
        [[0.49, 0.51, 1.49, 1.51, -0.49, -0.51, -1.49, -1.51, 3.0]],
        device="cuda",
        dtype=torch.float32,
    )
    quantized, scale = _quantize_rows_int8(values)
    expected = (values / scale[:, None]).round().clamp(-127, 127).to(torch.int8)
    assert torch.equal(quantized, expected)


@torch.no_grad()
def test_aiter_int8_linear_shape_accuracy_and_determinism():
    generator = torch.Generator(device="cuda").manual_seed(17)
    m, k, n = 131, 512, 272
    x = torch.randn(
        (2, m, k),
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )
    weight = torch.randn(
        (n, k),
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )
    bias = torch.randn(
        (n,),
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )
    layer = xFuserAiterInt8Linear(
        k,
        n,
        bias=True,
        device="meta",
        dtype=torch.bfloat16,
    )
    layer.load_and_quantize_weights(weight, bias, device="cuda")

    output = layer(x)
    repeat = layer(x)
    compiled = torch.compile(layer, fullgraph=True)
    compiled_output = compiled(x)
    reference = torch.nn.functional.linear(x, weight, bias)

    assert output.shape == (2, m, n)
    assert output.dtype == torch.bfloat16
    assert torch.equal(output, repeat)
    assert torch.equal(output, compiled_output)
    assert torch.isfinite(output.float()).all()
    cosine = torch.nn.functional.cosine_similarity(
        output.float().flatten(),
        reference.float().flatten(),
        dim=0,
    )
    assert cosine > 0.999
    torch._dynamo.reset()


@torch.no_grad()
def test_aiter_int8_linear_zero_activation_returns_bias():
    k, n = 512, 272
    weight = torch.randn((n, k), device="cuda", dtype=torch.bfloat16)
    bias = torch.randn((n,), device="cuda", dtype=torch.bfloat16)
    layer = xFuserAiterInt8Linear(
        k,
        n,
        bias=True,
        device="meta",
        dtype=torch.bfloat16,
    )
    layer.load_and_quantize_weights(weight, bias, device="cuda")
    output = layer(torch.zeros((4, k), device="cuda", dtype=torch.bfloat16))

    assert torch.isfinite(output.float()).all()
    assert torch.equal(output, bias.expand_as(output))
