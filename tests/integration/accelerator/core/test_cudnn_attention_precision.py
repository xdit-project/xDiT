"""The cuDNN attention backend must not return NaN for float32 inputs.

xDiT calls the aten cuDNN kernel directly, which bypasses the checks PyTorch's
own dispatcher applies. That kernel only implements float16 and bfloat16, yet
for float32 inputs it returns NaN instead of raising, and cuDNN is the default
backend on NVIDIA GPUs. Float32 calls must reach a kernel that serves them.
"""

import pytest
import torch

from xfuser.core.attention import registry
from xfuser.core.attention.spec import AttentionBackendType, AttnCall, VarlenPacking
from xfuser.model_executor.layers.attention_mask import make_attn_mask_with_meta

pytestmark = pytest.mark.nvidia


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["float32", "bfloat16"])
def test_cudnn_backend_matches_reference_attention(dtype):
    spec = registry.get(AttentionBackendType.CUDNN)
    unavailable = spec.unavailable()
    if unavailable is not None:
        pytest.skip(unavailable)

    # Not a multiple of 32: the memory-efficient kernel pads its LSE to one.
    seq_len = 50
    generator = torch.Generator(device="cuda").manual_seed(0)
    query, key, value = (torch.randn(2, 3, seq_len, 8, device="cuda", generator=generator).to(dtype) for _ in range(3))
    scores = query.float() @ key.float().transpose(-1, -2) * query.shape[-1] ** -0.5
    expected = torch.softmax(scores, dim=-1) @ value.float()

    output, lse = spec.run(query, key, value, AttnCall())

    assert output.dtype == dtype
    tolerance = 1e-4 if dtype == torch.float32 else 2e-2
    torch.testing.assert_close(output.float(), expected, rtol=tolerance, atol=tolerance)
    torch.testing.assert_close(lse[..., :seq_len].float(), torch.logsumexp(scores, dim=-1), rtol=1e-3, atol=1e-3)


def test_cudnn_float32_fallback_preserves_masked_packing():
    """The float32 fallback must preserve the mask that accompanies packed keys."""
    spec = registry.get(AttentionBackendType.CUDNN)
    unavailable = spec.unavailable()
    if unavailable is not None:
        pytest.skip(unavailable)

    # Exercise both the efficient kernel's padded LSE and its bias row alignment.
    q_len, kv_len = 50, 77
    generator = torch.Generator(device="cuda").manual_seed(0)
    shapes = ((2, 3, q_len, 64), (2, 3, kv_len, 64), (2, 3, kv_len, 64))
    query, key, value = (
        torch.randn(shape, device="cuda", dtype=torch.float32, generator=generator) for shape in shapes
    )
    positions = torch.arange(kv_len, device="cuda")
    valid = torch.stack((positions < 20, positions % 3 != 0))
    meta = make_attn_mask_with_meta(valid)
    kwargs = {
        "attn_mask": meta.attn_mask,
        "indices_k": meta.indices_k,
        "cu_seqlens_k": meta.cu_seqlens_k,
        "max_seqlen_k": meta.max_seqlen_k,
    }

    output, lse = spec.run(
        query, key, value, AttnCall(varlen=VarlenPacking.from_kwargs(kwargs), attention_kwargs=kwargs)
    )

    expected_outputs, expected_lses = [], []
    for batch in range(query.shape[0]):
        q = query[batch : batch + 1]
        k, v = (tensor[batch : batch + 1, :, valid[batch]] for tensor in (key, value))
        scores = q @ k.transpose(-1, -2) * q.shape[-1] ** -0.5
        expected_outputs.append(torch.softmax(scores, dim=-1) @ v)
        expected_lses.append(torch.logsumexp(scores, dim=-1))

    assert output.dtype == torch.float32
    torch.testing.assert_close(output, torch.cat(expected_outputs), rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(lse, torch.cat(expected_lses), rtol=1e-3, atol=1e-3)
