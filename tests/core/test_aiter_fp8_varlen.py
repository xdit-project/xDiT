"""AITER_FP8 routing: which of its three paths a call takes, and what each
hands the kernel.

The Hadamard matrix is replaced by a feature reversal, which is orthonormal
and its own inverse, so "was Q/K rotated" is a flip rather than a tolerance.
"""

from types import SimpleNamespace

import pytest
import torch

from xfuser.core.attention import registry
from xfuser.core.attention.spec import AttentionBackendType, AttnCall, VarlenPacking


def _feature_reversal_matrix():
    return torch.flip(torch.eye(128), dims=[1])


@pytest.fixture
def fp8_kernel(monkeypatch):
    """The AITER_FP8 kernel module with its vendor calls stubbed out."""
    kernel = pytest.importorskip(
        "xfuser.core.attention.backends.aiter_fp8.kernel",
        reason="AITER_FP8 needs aiter",
    )
    monkeypatch.setattr(
        kernel.hadamard, "matrix", lambda block_r, device: _feature_reversal_matrix().to(device)
    )
    return kernel


def _run(query, key, value, **kwargs):
    spec = registry.get(AttentionBackendType.AITER_FP8)
    return spec.impl(query, key, value, AttnCall(**kwargs)) \
        if not hasattr(spec.impl, "target") else \
        spec.resolved()(query, key, value, AttnCall(**kwargs))


def test_varlen_call_packs_keys_and_keeps_every_query(fp8_kernel, monkeypatch):
    """Q is never filtered; K/V are gathered by indices_k. Getting this wrong
    runs dense attention over padded keys, which is wrong rather than slow."""
    calls = {}

    # Patched at the custom op rather than inside it: the op dispatches through
    # torch._ops, so a stub on the module global it reads would not be seen.
    def fake_varlen_op(query, key, value, cu_seqlens_q, cu_seqlens_k,
                       max_seqlen_q, max_seqlen_k, softmax_scale, is_causal):
        calls.update(query=query, key=key,
                     shapes=(query.shape, key.shape, value.shape),
                     max_seqlen_q=max_seqlen_q, max_seqlen_k=max_seqlen_k)
        return query

    monkeypatch.setattr(fp8_kernel, "_varlen_op", fake_varlen_op)

    query = torch.randn(1, 2, 4, 128, device="cuda")
    key = torch.randn(1, 2, 4, 128, device="cuda")
    value = torch.randn(1, 2, 4, 128, device="cuda")
    # The packing is a field on the call, not a kwargs entry: usp builds it
    # once so a kernel never re-derives it from the dict.
    output, lse = fp8_kernel.aiter_fp8(query, key, value, AttnCall(
        varlen=VarlenPacking(
            indices_k=torch.tensor([0, 1, 2], device="cuda"),
            cu_seqlens_k=torch.tensor([0, 3], dtype=torch.int32, device="cuda"),
            max_seqlen_k=3,
        ),
    ))

    assert output.shape == query.shape
    assert lse is None
    assert calls["shapes"] == ((4, 2, 128), (3, 2, 128), (3, 2, 128))
    assert calls["max_seqlen_q"] == 4
    assert calls["max_seqlen_k"] == 3

    rotated = query.permute(0, 2, 1, 3).reshape(4, 2, 128).flip(-1)
    assert calls["query"].device.type == "cuda"
    assert torch.equal(calls["query"], rotated)


def test_mha_v4_path_hands_over_unrotated_qk(fp8_kernel, monkeypatch):
    """MHA v4 rotates and quantises internally, so passing it pre-rotated Q/K
    would apply the rotation twice."""
    calls = {}

    def mha_v4(query, key, value, *formats):
        calls.update(query=query, key=key, formats=formats)
        return query

    monkeypatch.setattr(fp8_kernel, "_USE_MHA_V4", True)
    monkeypatch.setattr(fp8_kernel, "mha_v4", mha_v4)
    monkeypatch.setattr(fp8_kernel, "native_fp8_format", lambda: 4)

    query = torch.randn(1, 2, 4, 128, device="cuda")
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    output, _ = fp8_kernel.aiter_fp8(query, key, value, AttnCall())

    assert torch.equal(calls["query"], query.permute(0, 2, 1, 3).contiguous())
    assert torch.equal(calls["key"], key.permute(0, 2, 1, 3).contiguous())
    assert calls["formats"] == (4, 4, 4)


def test_dense_path_rotates_qk_before_quantising(fp8_kernel, monkeypatch):
    """The dense and varlen kernels expect pre-rotated Q/K, unlike MHA v4."""
    calls = {}
    rotation = _feature_reversal_matrix()

    monkeypatch.setattr(fp8_kernel, "_USE_MHA_V4", False)
    monkeypatch.setattr(fp8_kernel, "_dense_op",
                        lambda q, k, v, scale, causal: calls.update(query=q, key=k) or q)

    query = torch.randn(1, 2, 4, 128)
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    fp8_kernel.aiter_fp8(query, key, value, AttnCall())

    assert torch.equal(calls["query"], torch.matmul(query.permute(0, 2, 1, 3), rotation))
    assert torch.equal(calls["key"], torch.matmul(key.permute(0, 2, 1, 3), rotation))
