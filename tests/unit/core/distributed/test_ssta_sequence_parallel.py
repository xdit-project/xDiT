"""SSTA under Ulysses drops the text's sequence-parallel padding.

A prompt that does not split evenly across the ranks is padded before it is
sharded. After the all-to-all and de-interleaving, that padding ends the text.
SSTA must tile the text as one device does, so the padding never serves as a
key, and must hand back an output in the per-rank layout, padding included.
"""

import pytest

torch = pytest.importorskip("torch")

from xfuser.core.distributed.ssta import get_sparse_mask, setup_ssta, untile_ssta_output  # noqa: E402

_SP = 2
_HEADS = 2
_HEAD_DIM = 8
_THW = (2, 4, 4)
_TILE = (1, 2, 2)
_TEXT = 11  # padded to 12 to shard over two ranks


def _attn_kwargs(*, sp_size, encoder_sequence_length, **extra):
    return {
        "ssta_threshold": 0.0,
        "ssta_lambda": 0.7,
        "ssta_sampling_type": "importance",
        "ssta_adaptive_pool": None,
        "attn_pad_type": "zero",
        "attn_use_text_mask": 0,
        "text_mask": None,
        "attn_mask_share_within_head": 0,
        "attn_sparse_type": "ssta",
        "encoder_sequence_length": encoder_sequence_length,
        "ssta_topk": 4,
        "thw": _THW,
        "tile_size": list(_TILE),
        "win_size": [[1, 1, 1]],
        "sp_size": sp_size,
        "sparse_text_to_image": False,
        **extra,
    }


def _ulysses_layout(x, text_pad, image_pad, thw, sp_size):
    """[image, text] as the Ulysses all-to-all hands it to SSTA: each rank's
    [image, text] chunk in rank order, with the text padded at its end."""
    image_len = thw[0] * thw[1] * thw[2]
    image, text = x[:, :, :image_len], x[:, :, image_len:]
    image = torch.cat([image, image_pad], dim=2)
    text = torch.cat([text, text_pad], dim=2)
    chunks = zip(image.chunk(sp_size, dim=2), text.chunk(sp_size, dim=2))
    return torch.cat([part for pair in chunks for part in pair], dim=2)


@pytest.mark.parametrize(
    "thw,text_len,sp_size",
    [
        pytest.param(_THW, _TEXT, _SP, id="text-padding"),
        pytest.param((1, 3, 3), 11, 2, id="image-and-text-padding"),
        pytest.param((1, 3, 3), 12, 2, id="image-padding-divisible-text"),
        pytest.param((1, 1, 1), 1, 4, id="padding-spans-ranks"),
    ],
)
def test_sp_padding_is_dropped_before_tiling_and_restored_after(thw, text_len, sp_size):
    generator = torch.Generator().manual_seed(0)
    image_len = thw[0] * thw[1] * thw[2]
    shape = (1, _HEADS, image_len + text_len, _HEAD_DIM)
    q, k, v = (torch.randn(*shape, generator=generator) for _ in range(3))
    text_pad = (-text_len) % sp_size
    image_pad = (-image_len) % sp_size
    # Padded tokens are not zero once they have been through the blocks.
    text_pads = [torch.randn(1, _HEADS, text_pad, _HEAD_DIM, generator=generator) for _ in range(3)]
    image_pads = [torch.randn(1, _HEADS, image_pad, _HEAD_DIM, generator=generator) for _ in range(3)]

    ref_q, ref_k, ref_v, ref_config, ref_state = setup_ssta(
        q, k, v, _attn_kwargs(sp_size=1, encoder_sequence_length=text_len, thw=thw)
    )
    local_text_len = (text_len + text_pad) // sp_size
    sp_kwargs = _attn_kwargs(
        sp_size=sp_size, encoder_sequence_length=local_text_len, encoder_sp_padding=text_pad, thw=thw
    )
    sp_q, sp_k, sp_v, sp_config, sp_state = setup_ssta(
        *(_ulysses_layout(x, text, image, thw, sp_size) for x, text, image in zip((q, k, v), text_pads, image_pads)),
        sp_kwargs,
    )

    torch.testing.assert_close(sp_q, ref_q, rtol=0, atol=0)
    torch.testing.assert_close(sp_k, ref_k, rtol=0, atol=0)
    torch.testing.assert_close(sp_v, ref_v, rtol=0, atol=0)
    assert torch.equal(get_sparse_mask(sp_config, "ssta"), get_sparse_mask(ref_config, "ssta"))

    tiled_output = torch.randn(ref_q.shape, generator=generator)
    expected = untile_ssta_output(tiled_output, ref_state, text_len, 1)
    actual = untile_ssta_output(tiled_output, sp_state, local_text_len, sp_size)
    zero_text = torch.zeros(1, _HEADS, text_pad, _HEAD_DIM)
    zero_image = torch.zeros(1, _HEADS, image_pad, _HEAD_DIM)
    torch.testing.assert_close(actual, _ulysses_layout(expected, zero_text, zero_image, thw, sp_size), rtol=0, atol=0)
