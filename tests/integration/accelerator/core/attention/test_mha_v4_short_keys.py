"""AITER_BF16 attends correctly over any number of keys.

Its MHA v4 kernel gets calls of 64 keys or fewer wrong whatever the query
length, as MiniMax-H3's text refiner makes them for a short prompt, so the
backend has to hand those calls to its fallback.
"""

import pytest
import torch
import torch.nn.functional as F

from xfuser.core.attention import registry
from xfuser.core.attention.spec import AttentionBackendType, AttnCall, VarlenPacking

pytestmark = pytest.mark.rocm


@pytest.fixture
def aiter_bf16():
    if not torch.cuda.is_available():
        pytest.skip("requires an accelerator")
    spec = registry.get(AttentionBackendType.AITER_BF16)
    reason = spec.unavailable()
    if reason is not None:
        pytest.skip(f"AITER_BF16: {reason}")
    spec.resolved()  # as backend selection does, resolving the fallback too
    return spec


def _rel_l2(actual, expected):
    return ((actual.float() - expected).norm() / expected.norm()).item()


@pytest.mark.parametrize("kv_len", [1, 53, 64, 65, 200])
def test_short_key_calls_match_sdpa(aiter_bf16, kv_len):
    generator = torch.Generator(device="cuda").manual_seed(0)
    query = torch.randn(1, 8, 256, 128, device="cuda", dtype=torch.bfloat16, generator=generator)
    key, value = (
        torch.randn(1, 8, kv_len, 128, device="cuda", dtype=torch.bfloat16, generator=generator) for _ in range(2)
    )

    output = aiter_bf16.run(query, key, value, AttnCall())[0]
    expected = F.scaled_dot_product_attention(query.float(), key.float(), value.float())

    assert _rel_l2(output, expected) < 1e-2


def test_a_short_pack_behind_a_long_pad_matches_sdpa(aiter_bf16):
    """A trailing pad can make a call long while it attends over few keys."""
    generator = torch.Generator(device="cuda").manual_seed(0)
    query, key, value = (
        torch.randn(1, 8, 128, 128, device="cuda", dtype=torch.bfloat16, generator=generator) for _ in range(3)
    )
    valid = 60
    call = AttnCall(
        varlen=VarlenPacking(
            indices_k=torch.arange(valid, device="cuda"),
            cu_seqlens_k=torch.tensor([0, valid], dtype=torch.int32, device="cuda"),
            max_seqlen_k=valid,
        ),
        attention_kwargs={"valid_kv_len": valid},
    )

    output = aiter_bf16.run(query, key, value, call)[0]
    expected = F.scaled_dot_product_attention(query.float(), key[:, :, :valid].float(), value[:, :, :valid].float())

    assert _rel_l2(output, expected) < 1e-2
