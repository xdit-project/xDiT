"""FlyDSL QK norm and RoPE writing Q/K straight into the Ulysses all-to-all send layout."""

import pytest
import torch

pytestmark = pytest.mark.rocm

fused = pytest.importorskip("xfuser.model_executor.layers.fused_qk_rope_flydsl")
if not fused._HAS_FLYDSL or not torch.cuda.is_available():
    pytest.skip("requires FlyDSL on a ROCm device", allow_module_level=True)

from diffusers.models.normalization import RMSNorm  # noqa: E402

HEADS, HEAD_DIM = 24, 128


def _inputs(seq_len, rope_style):
    torch.manual_seed(0)
    # q/k are chunks of a fused QKV projection, so their rows are strided.
    qkv = torch.randn(1, seq_len, 3, HEADS, HEAD_DIM, device="cuda", dtype=torch.bfloat16) * 3
    norm_q = RMSNorm(HEAD_DIM, eps=1e-6).to("cuda", torch.bfloat16)
    norm_k = RMSNorm(HEAD_DIM, eps=1e-6).to("cuda", torch.bfloat16)
    with torch.no_grad():
        norm_q.weight.copy_(torch.randn(HEAD_DIM) * 0.2 + 1)
        norm_k.weight.copy_(torch.randn(HEAD_DIM) * 0.2 + 1)
    angles = torch.rand(seq_len, HEAD_DIM // 2, device="cuda") * 6
    if rope_style == "neox":
        cos, sin = angles.cos().repeat(1, 2), angles.sin().repeat(1, 2)
        kwargs = {"rotary_dim": HEAD_DIM, "neox_tables": fused.prepare_neox_rope_tables(cos, sin, HEAD_DIM)}
    else:
        cos, sin = angles.cos().repeat_interleave(2, -1), angles.sin().repeat_interleave(2, -1)
        kwargs = {}
    return qkv[:, :, 0], qkv[:, :, 1], norm_q, norm_k, (cos, sin), kwargs


@pytest.mark.parametrize("rope_style", ["neox", "gptj"])
@pytest.mark.parametrize(("seq_len", "world"), [(1099, 4), (550, 8), (64, 2)])
def test_send_layout_is_the_contiguous_output_regrouped_per_peer(rope_style, seq_len, world):
    query, key, norm_q, norm_k, rope, kwargs = _inputs(seq_len, rope_style)

    with torch.no_grad():
        flat = fused.flydsl_fused_qk_norm_rope(query, key, norm_q, norm_k, rope, rope_style=rope_style, **kwargs)
        sent = fused.flydsl_fused_qk_norm_rope(
            query, key, norm_q, norm_k, rope, rope_style=rope_style, send_world=world, **kwargs
        )

    local_heads = HEADS // world
    for got, want in zip(sent, flat):
        assert got.shape == (world, seq_len, local_heads, HEAD_DIM)
        expected = want.view(seq_len, world, local_heads, HEAD_DIM).transpose(0, 1)
        # Only the store address changes, so the values must be bit-identical.
        assert torch.equal(got, expected)


def test_send_layout_stays_within_bf16_rounding_of_the_eager_reference():
    """Eager rotates in bf16 (three roundings), the kernel in fp32 (one), so they differ by a few bf16 steps."""
    query, key, norm_q, norm_k, rope, kwargs = _inputs(1099, "neox")

    with torch.no_grad():
        got_q, got_k = fused.flydsl_fused_qk_norm_rope(
            query, key, norm_q, norm_k, rope, rope_style="neox", send_world=4, **kwargs
        )
        want = fused._reference_neox(query, key, norm_q, norm_k, rope)

    for got, ref in zip((got_q, got_k), want):
        ref = ref.view(1099, 4, HEADS // 4, HEAD_DIM).transpose(0, 1)
        torch.testing.assert_close(got, ref, atol=2**-4, rtol=2**-6)
        assert (got.float() - ref.float()).abs().mean() < 2**-8


def test_send_layout_under_torch_compile_matches_eager():
    query, key, norm_q, norm_k, rope, kwargs = _inputs(550, "neox")

    def prologue(q, k):
        return fused.flydsl_fused_qk_norm_rope(q, k, norm_q, norm_k, rope, rope_style="neox", send_world=8, **kwargs)

    with torch.no_grad():
        eager = prologue(query, key)
        compiled = torch.compile(prologue, fullgraph=True)(query, key)

    for got, want in zip(compiled, eager):
        assert torch.equal(got, want)
