"""SSTA block masks have the layout the block-sparse kernels accept."""

import pytest

torch = pytest.importorskip("torch")

from xfuser.core.distributed.ssta import get_sparse_mask, setup_ssta  # noqa: E402

_HEADS = 4
_THW = (4, 4, 4)
_TILE = (2, 2, 2)
_TEXT_LEN = 16


def _attn_kwargs(share, sparse_type):
    return {
        "ssta_threshold": 0.0,
        "ssta_lambda": 0.7,
        "ssta_sampling_type": "importance",
        "ssta_adaptive_pool": None,
        "attn_pad_type": "zero",
        "attn_use_text_mask": 0,
        "text_mask": None,
        "attn_mask_share_within_head": share,
        "attn_sparse_type": sparse_type,
        "encoder_sequence_length": _TEXT_LEN,
        "ssta_topk": 6,
        "thw": _THW,
        "tile_size": list(_TILE),
        "win_size": [[1, 1, 1]],
        "sp_size": 1,
        "sparse_text_to_image": False,
    }


def _block_sparse_reference(q, k, v, block_size, block_mask):
    """Dense attention under the block mask, accepting the [b, h or 1, nq, nk] layout of flex_block_attn."""
    assert block_mask.dim() == 4, f"block mask must be [b, h, q_blocks, kv_blocks], got {tuple(block_mask.shape)}"
    mask = block_mask.repeat_interleave(block_size, dim=-2).repeat_interleave(block_size, dim=-1)
    return torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask)


@pytest.mark.parametrize("sparse_type", ["ssta", "moba"])
@pytest.mark.parametrize("share", [0, 1])
def test_block_mask_is_batch_head_query_key(share, sparse_type):
    generator = torch.Generator().manual_seed(0)
    seq_len = _THW[0] * _THW[1] * _THW[2] + _TEXT_LEN
    query, key, value = (torch.randn(2, _HEADS, seq_len, 8, generator=generator) for _ in range(3))

    q, k, v, mask_config, _ = setup_ssta(query, key, value, _attn_kwargs(share, sparse_type))
    block_mask = get_sparse_mask(mask_config, sparse_type=sparse_type)

    block_size = _TILE[0] * _TILE[1] * _TILE[2]
    blocks = seq_len // block_size
    assert tuple(block_mask.shape) == (2, 1 if share else _HEADS, blocks, blocks)
    out = _block_sparse_reference(q, k, v, block_size, block_mask)
    assert out.shape == q.shape
    assert torch.isfinite(out).all()
