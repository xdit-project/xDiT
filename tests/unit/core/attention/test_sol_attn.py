"""Sol-Attn backend contract checks that do not launch the kernel."""

from types import SimpleNamespace

import torch
import torch.nn.functional as F

from xfuser.core.attention import registry
from xfuser.core.attention.spec import AttentionBackendType, AttnCall, VarlenPacking


def _spec():
    return registry.get(AttentionBackendType.SOL_ATTN)


def _qkv(batch=1, heads=2, length=16, head_dim=128, dtype=torch.bfloat16):
    shape = (batch, heads, length, head_dim)
    return (
        torch.zeros(shape, dtype=dtype),
        torch.zeros(shape, dtype=dtype),
        torch.zeros(shape, dtype=dtype),
    )


def test_sol_attn_is_registered_without_a_logsumexp():
    spec = _spec()
    assert spec.fallback is AttentionBackendType.SDPA
    assert spec.ring.unmet() is not None
    assert spec.sparsity is None


def test_sol_attn_accepts_bf16_self_attention():
    q, k, v = _qkv()
    assert _spec().rejects(q, k, v, AttnCall()) is None


def test_sol_attn_rejects_calls_the_kernel_cannot_serve():
    spec = _spec()
    q, k, v = _qkv(dtype=torch.float16)
    assert "bfloat16" in spec.rejects(q, k, v, AttnCall())

    q, k, v = _qkv(head_dim=64)
    assert "head dimension" in spec.rejects(q, k, v, AttnCall())

    q, k, v = _qkv()
    k = torch.zeros(1, 2, 8, 128, dtype=torch.bfloat16)
    v = torch.zeros(1, 2, 8, 128, dtype=torch.bfloat16)
    assert "equal query, key, and value" in spec.rejects(q, k, v, AttnCall())

    q, k, v = _qkv()
    assert "causal" in spec.rejects(q, k, v, AttnCall(is_causal=True))
    assert "dropout" in spec.rejects(q, k, v, AttnCall(dropout_p=0.1))
    packing = VarlenPacking(
        indices_k=torch.zeros(4, dtype=torch.int64),
        cu_seqlens_k=torch.tensor([0, 4], dtype=torch.int32),
        max_seqlen_k=4,
    )
    assert "varlen" in spec.rejects(q, k, v, AttnCall(varlen=packing))


def test_ineligible_call_falls_back_to_sdpa():
    q, k, v = _qkv(head_dim=64, dtype=torch.float32)
    q.normal_()
    k.normal_()
    v.normal_()
    out, lse = _spec().run(q, k, v, AttnCall())
    ref = F.scaled_dot_product_attention(q, k, v)
    assert lse is None
    assert torch.allclose(out, ref)


def test_runtime_config_is_forwarded_to_kernel(monkeypatch):
    import pytest

    pytest.importorskip("sol_attn")
    from xfuser.core.attention.backends.sol_attn import kernel

    runtime_config = SimpleNamespace(
        sol_attn_tau=2.5,
        sol_attn_thresh_type="diag",
        sol_attn_kv_splits="4",
        sol_attn_sink_tokens=32,
        sol_attn_sink_start=0,
    )
    runtime_state = SimpleNamespace(runtime_config=runtime_config)
    monkeypatch.setattr(kernel, "get_runtime_state", lambda: runtime_state)
    received = {}

    def fake_op(query, key, value, tau, thresh_type, kv_splits, sink_tokens, sink_start):
        received["config"] = (tau, thresh_type, kv_splits, sink_tokens, sink_start)
        return query

    monkeypatch.setattr(kernel, "_sol_attn_op", fake_op)
    q, k, v = _qkv()
    out, lse = kernel.sol_attention(q, k, v, AttnCall())
    assert received["config"] == (2.5, 0, 4, 32, 0)
    assert out.shape == q.shape
    assert lse is None
