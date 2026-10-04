"""The cuDNN attention backend must not return NaN for float32 inputs.

xDiT calls the aten cuDNN kernel directly, which bypasses the checks PyTorch's
own dispatcher applies. That kernel only implements float16 and bfloat16, yet
for float32 inputs it returns NaN instead of raising, and cuDNN is the default
backend on NVIDIA GPUs. Float32 calls must reach a kernel that serves them.
"""

import pytest
import torch

from xfuser.core.attention import registry
from xfuser.core.attention.spec import AttentionBackendType, AttnCall

pytestmark = pytest.mark.nvidia


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["float32", "bfloat16"])
def test_cudnn_backend_matches_reference_attention(dtype):
    spec = registry.get(AttentionBackendType.CUDNN)
    unavailable = spec.unavailable()
    if unavailable is not None:
        pytest.skip(unavailable)

    generator = torch.Generator(device="cuda").manual_seed(0)
    query, key, value = (torch.randn(2, 3, 64, 8, device="cuda", generator=generator).to(dtype) for _ in range(3))
    expected = torch.ops.aten._scaled_dot_product_attention_math(query.float(), key.float(), value.float())[0]

    output, lse = spec.run(query, key, value, AttnCall())

    assert output.dtype == dtype
    tolerance = 1e-4 if dtype == torch.float32 else 2e-2
    torch.testing.assert_close(output.float(), expected, rtol=tolerance, atol=tolerance)
    assert lse is not None and torch.isfinite(lse).all()
