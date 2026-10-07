"""TRITON_BSA: Prism's block-sparse attention as an xDiT attention backend.

Keeping every key block must reproduce dense attention, which pins down the
grid padding, the padded-key mask and the block reordering without needing
Prism's own implementation. The backend wrapper must run calls that publish no
token grid densely, and must hand back the Ulysses padding rows it was given.
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")

if not torch.cuda.is_available():
    pytest.skip("requires an accelerator", allow_module_level=True)


def _qkv(n, dtype, heads=4, head_dim=32, seed=0):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    return tuple(torch.randn(1, heads, n, head_dim, device="cuda", dtype=dtype, generator=generator) for _ in range(3))


@pytest.mark.parametrize("thw", [(8, 8, 12), (7, 7, 11)], ids=["whole-blocks", "padded"])
def test_keeping_every_block_is_dense_attention(thw):
    from xfuser.core.attention.backends.triton_bsa.bsa import block_sparse_attention_3d

    q, k, v = _qkv(thw[0] * thw[1] * thw[2], torch.float32)
    actual = block_sparse_attention_3d(q, k, v, thw, (4, 4, 4), sparsity=0.0, cdf_threshold=None)
    expected = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


def test_sparse_selection_drops_blocks():
    from xfuser.core.attention.backends.triton_bsa.bsa import block_sparse_attention_3d

    thw = (8, 8, 12)
    q, k, v = _qkv(thw[0] * thw[1] * thw[2], torch.float32)
    dense = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    sparse = block_sparse_attention_3d(q, k, v, thw, (4, 4, 4), sparsity=0.75, cdf_threshold=0.2)
    assert not sparse.isnan().any()
    assert (sparse - dense).abs().max() > 1e-2


def test_calls_without_a_grid_run_dense():
    from xfuser.core.attention.backends.triton_bsa.kernel import triton_bsa
    from xfuser.core.attention.spec import AttnCall

    q, k, v = _qkv(96, torch.bfloat16)
    actual, _ = triton_bsa(q, k[:, :, :40], v[:, :, :40], AttnCall())
    expected = torch.nn.functional.scaled_dot_product_attention(q, k[:, :, :40], v[:, :, :40])
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)


def test_gathered_padding_rows_come_back_as_zeros():
    """After the Ulysses exchange the queries still carry the sequence padding
    while USP has already cut it from the keys."""
    from xfuser.core.attention.backends.triton_bsa.bsa import block_sparse_attention_3d
    from xfuser.core.attention.backends.triton_bsa.kernel import triton_bsa
    from xfuser.core.attention.spec import AttnCall

    thw = (7, 7, 11)
    n = thw[0] * thw[1] * thw[2]
    q, k, v = _qkv(n + 3, torch.bfloat16)
    call = AttnCall(attention_kwargs={"bsa_thw": thw, "bsa_sparsity": 0.75, "bsa_cdf_threshold": 0.2})
    actual, _ = triton_bsa(q, k[:, :, :n], v[:, :, :n], call)
    expected = block_sparse_attention_3d(q[:, :, :n], k[:, :, :n], v[:, :, :n], thw, (4, 4, 4), 0.75, 0.2)
    torch.testing.assert_close(actual[:, :, :n], expected, rtol=0, atol=0)
    assert actual[:, :, n:].abs().max() == 0
