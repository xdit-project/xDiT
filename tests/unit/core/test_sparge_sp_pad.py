"""Sparge strips the sequence-parallel pad whenever the model publishes its layout.

USP leaves the pad in K/V for Sparge (it cannot drop keys ahead of the
reordering), so setup_sparge must remove it: a zero pad key gets logit 0 and
would otherwise take attention mass from every real query.
"""

import itertools

import pytest
import torch

from xfuser.core.sparge_attention.sparge import restore_sparge_output, setup_sparge

THW = (2, 3, 4)
SP_PAD = 4


def _qkv(seq_len):
    generator = torch.Generator().manual_seed(0)
    return [torch.randn(1, 2, seq_len, 8, generator=generator) for _ in range(3)]


def _with_sp_pad(query, key, value):
    """Append the pad as Wan does: zero Q/K rows, since its RoPE tables are zero-padded too, but V rows that are not.

    Zero V rows would be indistinguishable from the zero rows Sparge itself appends to fill its last block.
    """
    zeros = torch.zeros(1, 2, SP_PAD, 8)
    pad_values = torch.randn(1, 2, SP_PAD, 8, generator=torch.Generator().manual_seed(1))
    return torch.cat([query, zeros], dim=2), torch.cat([key, zeros], dim=2), torch.cat([value, pad_values], dim=2)


@pytest.mark.parametrize(
    "reorder_sequence, use_static_block_mask, pad_block_divisible", list(itertools.product((False, True), repeat=3))
)
def test_padded_input_is_set_up_as_the_unpadded_one(reorder_sequence, use_static_block_mask, pad_block_divisible):
    real = _qkv(THW[0] * THW[1] * THW[2])
    padded = _with_sp_pad(*real)
    options = dict(
        thw=THW,
        sp_size=1,
        reorder_sequence=reorder_sequence,
        use_static_block_mask=use_static_block_mask,
        block_m=16,
        block_n=16,
        pad_block_divisible=pad_block_divisible,
    )

    *expected, _, _ = setup_sparge(*real, **options)
    *actual, state, _ = setup_sparge(*padded, **options)

    for expected_tensor, actual_tensor in zip(expected, actual):
        assert torch.equal(actual_tensor, expected_tensor)
    restored = restore_sparge_output(actual[0], state)
    assert restored.shape == padded[0].shape
    # The caller's output keeps one row per input row; the pad rows come back as zeros.
    assert not restored[:, :, -SP_PAD:].any()


def test_a_sequence_shorter_than_the_published_layout_is_refused():
    query, key, value = _qkv(THW[0] * THW[1] * THW[2] - 1)
    with pytest.raises(ValueError, match="shorter than thw product"):
        setup_sparge(query, key, value, thw=THW, sp_size=1)
