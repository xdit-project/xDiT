"""Sol-Attn backend: contract checks, and a kernel smoke test when it can run."""

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


def test_call_kwargs_override_the_defaults():
    from xfuser.core.attention.backends.sol_attn.kernel import _configured

    call = AttnCall(
        attention_kwargs={
            "sol_attn_tau": 2.5,
            "sol_attn_thresh_type": "exact",
            "sol_attn_kv_splits": "4",
            "sol_attn_sink_tokens": 32,
            "sol_attn_sink_start": 0,
        }
    )
    cfg = _configured(call)
    assert cfg["sol_attn_tau"] == 2.5
    assert cfg["sol_attn_thresh_type"] == "exact"
    assert cfg["sol_attn_kv_splits"] == "4"
    assert cfg["sol_attn_sink_tokens"] == 32
    assert cfg["sol_attn_sink_start"] == 0


def test_kernel_returns_the_query_shape():
    spec = _spec()
    reason = spec.unavailable()
    if reason is not None:
        import pytest

        pytest.skip(reason)

    torch.manual_seed(0)
    q, k, v = _qkv(heads=4, length=128)
    q = torch.randn_like(q)
    k = torch.randn_like(k)
    v = torch.randn_like(v)
    q, k, v = q.cuda(), k.cuda(), v.cuda()
    out, lse = spec.run(
        q,
        k,
        v,
        AttnCall(
            attention_kwargs={
                "sol_attn_tau": 1.0,
                "sol_attn_thresh_type": "diag",
                "sol_attn_kv_splits": 1,
                "sol_attn_sink_tokens": 0,
                "sol_attn_sink_start": None,
            }
        ),
    )
    assert out.shape == q.shape
    assert out.dtype == torch.bfloat16
    assert lse is None
    assert torch.isfinite(out).all()
