"""Sparge and VSA backends ignore a trailing sequence-parallel pad when the model publishes its layout.

USP hands these backends the pad untrimmed, and setup_sparge strips it against
`thw` before building the mask. The real rows must therefore come out as if the
pad had never been there, and the pad rows as zeros. Without reordering or a
static mask the strip leaves sequence slices, a layout some kernels misread.
"""

import pytest
import torch

from xfuser.core.attention import registry
from xfuser.core.attention.spec import AttentionBackendType, AttnCall, Sparsity
from xfuser.model_executor.layers import usp

THW, SP_PAD, HEADS, HEAD_DIM = (4, 16, 16), 4, 4, 128
SPARSE_BACKENDS = [
    backend
    for backend in AttentionBackendType
    if (spec := registry.find(backend)) is not None and spec.sparsity in (Sparsity.SPARGE, Sparsity.VSA)
]
LAYOUTS = {
    "reordered": {},
    "linear-order": {
        "spargeattn_reorder_sequence": False,
        "use_spargeattn_static_block_mask": False,
        "vsa_reorder_sequence": False,
        "use_vsa_static_block_mask": False,
    },
}


def _qkv():
    """Real Q/K/V, and the same with the pad as Wan appends it: zero Q/K rows (zero RoPE), non-zero V rows."""
    seq_len = THW[0] * THW[1] * THW[2]
    generator = torch.Generator().manual_seed(0)
    real = [torch.randn(1, HEADS, seq_len, HEAD_DIM, generator=generator) for _ in range(3)]
    zeros = torch.zeros(1, HEADS, SP_PAD, HEAD_DIM)
    pad_values = torch.randn(1, HEADS, SP_PAD, HEAD_DIM, generator=generator)
    padded = [torch.cat([real[0], zeros], 2), torch.cat([real[1], zeros], 2), torch.cat([real[2], pad_values], 2)]
    return [[tensor.to("cuda", torch.bfloat16) for tensor in tensors] for tensors in (real, padded)]


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("backend", SPARSE_BACKENDS, ids=lambda backend: backend.name)
def test_padded_input_gives_the_unpadded_output_plus_zero_rows(backend, layout):
    if not torch.cuda.is_available():
        pytest.skip("requires an accelerator")
    spec = registry.get(backend)
    real, padded = _qkv()
    reason = spec.unavailable() or spec.rejects(*padded, AttnCall())
    if reason is not None:
        pytest.skip(reason)

    # Only `thw`, not valid_kv_len: usp.attention would slice K/V by it before setup_sparge sees them.
    def attend(query, key, value):
        return usp.attention(query, key, value, backend=backend, attention_kwargs={"thw": THW, **LAYOUTS[layout]})

    expected = attend(*real)
    actual = attend(*padded)

    seq_len = real[0].shape[2]
    assert actual.shape == padded[0].shape
    # Bit-identical: after the strip both calls see the same Q/K/V, so the same mask and the same kernel launch.
    difference = (actual[:, :, :seq_len].float() - expected.float()).abs().max().item()
    assert torch.equal(actual[:, :, :seq_len], expected), f"real rows differ by up to {difference:.3e}"
    assert not actual[:, :, seq_len:].any()
