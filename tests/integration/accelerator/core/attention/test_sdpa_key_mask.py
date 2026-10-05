"""cuDNN, the default backend on NVIDIA without FlashAttention, and the aten
memory-efficient kernel serve calls that carry a key-padding mask.

LTX-2 text cross attention hands the backend an ``attn_mask`` together with
the varlen packing derived from it. Both backends used to refuse the packing and,
on calls without one, to drop the mask; they now apply the mask as their bias.
"""

import pytest
import torch

from xfuser.core.attention import registry
from xfuser.core.attention.spec import AttentionBackendType, AttnCall, VarlenPacking
from xfuser.model_executor.layers.attention_mask import make_attn_mask_with_meta
from xfuser.model_executor.layers.usp import attention

# cuDNN attention exists only on NVIDIA; the aten memory-efficient kernel runs on any accelerator.
BACKENDS = pytest.mark.parametrize(
    "backend",
    [
        pytest.param(AttentionBackendType.CUDNN, marks=pytest.mark.nvidia),
        AttentionBackendType.SDPA_EFFICIENT,
    ],
)


def _qkv(batch, heads, q_len, kv_len, head_dim=64):
    generator = torch.Generator(device="cuda").manual_seed(0)
    shapes = ((batch, heads, q_len, head_dim), (batch, heads, kv_len, head_dim), (batch, heads, kv_len, head_dim))
    return [torch.randn(s, device="cuda", dtype=torch.bfloat16, generator=generator) for s in shapes]


def _valid(kv_len, counts):
    valid = torch.zeros(len(counts), kv_len, dtype=torch.bool, device="cuda")
    for row, count in enumerate(counts):
        valid[row, :count] = True
    return valid


def _reference(query, key, value, valid):
    """Each sample attending to its valid keys alone, in float32."""
    outs, lses = [], []
    for b in range(query.shape[0]):
        q, k, v = (t[b : b + 1].float() for t in (query, key, value))
        k, v = k[:, :, valid[b]], v[:, :, valid[b]]
        scores = q @ k.transpose(-1, -2) * q.shape[-1] ** -0.5
        outs.append(torch.softmax(scores, dim=-1) @ v)
        lses.append(torch.logsumexp(scores, dim=-1))
    return torch.cat(outs), torch.cat(lses)


@BACKENDS
@pytest.mark.parametrize("kv_len, counts", [(77, (20, 76)), (128, (1, 128))])
def test_masked_packed_keys_attend_only_to_valid_keys(backend, kv_len, counts):
    query, key, value = _qkv(2, 4, 96, kv_len)
    valid = _valid(kv_len, counts)
    meta = make_attn_mask_with_meta(valid)
    kwargs = {
        "attn_mask": meta.attn_mask,
        "indices_k": meta.indices_k,
        "cu_seqlens_k": meta.cu_seqlens_k,
        "max_seqlen_k": meta.max_seqlen_k,
    }

    out = attention(query, key, value, backend=backend, attention_kwargs=kwargs)

    expected, _ = _reference(query, key, value, valid)
    torch.testing.assert_close(out.float(), expected, atol=2e-2, rtol=2e-2)


@BACKENDS
def test_masked_call_log_sum_exp_covers_only_valid_keys(backend):
    """Ring attention merges per-rank partials on the LSE, so it must agree too."""
    query, key, value = _qkv(2, 4, 64, 77)
    valid = _valid(77, (30, 77))
    meta = make_attn_mask_with_meta(valid)
    kwargs = {"attn_mask": meta.attn_mask}
    spec = registry.get(backend)

    out, lse = spec.run(query, key, value, AttnCall(varlen=VarlenPacking.from_kwargs(kwargs), attention_kwargs=kwargs))

    expected_out, expected_lse = _reference(query, key, value, valid)
    torch.testing.assert_close(out.float(), expected_out, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(lse.float(), expected_lse, atol=1e-3, rtol=1e-3)


@BACKENDS
def test_additive_mask_without_packing_is_applied(backend):
    query, key, value = _qkv(2, 4, 64, 40)
    valid = _valid(40, (10, 33))
    bias = torch.zeros(valid.shape, dtype=torch.float32, device="cuda").masked_fill(~valid, -10000.0)

    out = attention(query, key, value, backend=backend, attention_kwargs={"attn_mask": bias[:, None, None, :]})

    expected, _ = _reference(query, key, value, valid)
    torch.testing.assert_close(out.float(), expected, atol=2e-2, rtol=2e-2)
