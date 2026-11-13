"""AITER FAv3 forward: query rows past the last full wave are computed split-K."""

import pytest
import torch

pytestmark = [pytest.mark.accelerator, pytest.mark.rocm]

aiter_kernel = pytest.importorskip("xfuser.core.attention.backends.aiter.kernel")
from xfuser.core.attention.spec import AttnCall  # noqa: E402


def _compute_units():
    props = torch.cuda.get_device_properties(0)
    if props.gcnArchName.split(":")[0] not in aiter_kernel._V3_ARCHS:
        pytest.skip("the FAv3 tile geometry is only known for gfx942 and gfx950")
    return props.multi_processor_count


def _bhsd(batch, heads, seq, generator):
    return torch.randn(batch, heads, seq, 128, device="cuda", dtype=torch.bfloat16, generator=generator)


@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("compiled", [False, True])
def test_last_wave_split_matches_dense(batch, compiled):
    heads = 3
    # Two tiles past a full wave per batch, the Qwen-Image-Edit Ulysses-8 layout on MI355X.
    seq_q = (_compute_units() // heads) * 256 + 10
    seq_k = seq_q - 10
    generator = torch.Generator(device="cuda").manual_seed(0)
    query = _bhsd(batch, heads, seq_q, generator)
    key = _bhsd(batch, heads, seq_k, generator)
    value = _bhsd(batch, heads, seq_k, generator)
    call = AttnCall()
    bshd = [t.transpose(1, 2).contiguous() for t in (query, key, value)]
    assert aiter_kernel._tail_split_plan(*bshd, call)[0] > 0

    attention = aiter_kernel.aiter_attention
    if compiled:
        # Qwen-Image-Edit's CFG branches differ in length, so its graph is dynamic.
        attention = torch.compile(attention, dynamic=True, fullgraph=True)
    out, lse = attention(query, key, value, call)
    ref_out, ref_lse = aiter_kernel._dense(*bshd, 0.0, False)

    torch.testing.assert_close(out.transpose(1, 2), ref_out, atol=2e-3, rtol=0)
    torch.testing.assert_close(lse, ref_lse, atol=1e-4, rtol=0)


def test_full_last_wave_is_left_to_the_dense_kernel():
    heads = 3
    seq = (_compute_units() // heads) * 256
    generator = torch.Generator(device="cuda").manual_seed(0)
    bshd = [_bhsd(1, heads, seq, generator).transpose(1, 2).contiguous() for _ in range(3)]
    assert aiter_kernel._tail_split_plan(*bshd, AttnCall()) == (0, 0)
    assert aiter_kernel._tail_split_plan(*bshd, AttnCall(is_causal=True)) == (0, 0)
