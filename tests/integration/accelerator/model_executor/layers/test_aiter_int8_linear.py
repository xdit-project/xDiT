import pytest
import torch

from xfuser.model_executor.layers.aiter_int8_linear import _quantize_rows_int8, xFuserAiterInt8Linear
from xfuser.model_executor.models.runner_models.loading.format_backends import _probe_aiter_int8_apis

_AVAILABLE, _REASON = (
    _probe_aiter_int8_apis() if torch.cuda.is_available() and torch.version.hip else (False, "requires ROCm")
)
pytestmark = pytest.mark.skipif(not _AVAILABLE, reason=f"AITER INT8 GEMMs unavailable: {_REASON}")


@torch.no_grad()
def test_aiter_int8_quantizer_uses_round_to_nearest():
    values = torch.tensor([[0.49, 0.51, 1.49, 1.51, -0.49, -0.51, -1.49, -1.51, 3.0]], device="cuda")
    quantized, scale = _quantize_rows_int8(values)
    assert torch.equal(quantized, (values / scale[:, None]).round().clamp(-127, 127).to(torch.int8))


@torch.no_grad()
def test_aiter_int8_linear_matches_bf16_linear():
    generator = torch.Generator(device="cuda").manual_seed(17)
    m, k, n = 131, 512, 272
    x, weight, bias = (
        torch.randn(shape, device="cuda", dtype=torch.bfloat16, generator=generator)
        for shape in ((2, m, k), (n, k), (n,))
    )
    # Zero rows, like Attention A2A's sequence padding, must give exactly the bias.
    x[:, -3:] = 0
    layer = xFuserAiterInt8Linear(k, n, device="meta", dtype=torch.bfloat16)
    layer.load_and_quantize_weights(weight, bias, device="cuda")

    output = layer(x)
    reference = torch.nn.functional.linear(x, weight, bias)
    cosine = torch.nn.functional.cosine_similarity(output.float().flatten(), reference.float().flatten(), dim=0)
    assert output.shape == reference.shape and output.dtype == torch.bfloat16
    assert cosine > 0.999
    assert torch.equal(output[:, -3:], bias.expand(2, 3, n))
    assert torch.equal(layer(x), output)
    assert torch.equal(torch.compile(layer, fullgraph=True)(x), output)
    torch._dynamo.reset()
