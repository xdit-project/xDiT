import unittest

import torch

from xfuser.core.sparge_attention.gilbert import (
    sliced_gilbert_block_neighbor_mapping,
    _sliced_gilbert_block_neighbor_mapping,
)


def _tridiagonal(n: int) -> torch.Tensor:
    band = torch.eye(n, dtype=torch.bool)
    band[:-1, 1:] |= torch.eye(n - 1, dtype=torch.bool)
    band[1:, :-1] |= torch.eye(n - 1, dtype=torch.bool)
    return band


class TestSlicedGilbertBlockNeighborMapping(unittest.TestCase):
    """Tests for sliced_gilbert_block_neighbor_mapping with known expected results."""

    # A row of points in identity order, which is the one layout whose neighbours can be written
    # down by hand: each point touches the one either side of it, so a block touches itself and the
    # two beside it and nothing else. A 2x2x2 cube cannot show this - every point there is within
    # one step of every other, so every block neighbours every block.
    def test_identity_mapping_gives_a_banded_mask(self):
        """Identity linear_to_hilbert and block_m == block_n -> diagonal and its two neighbours."""
        t, h, w = 1, 1, 8
        block_m = block_n = 2
        linear_to_hilbert = torch.arange(8, dtype=torch.int64)
        mask = _sliced_gilbert_block_neighbor_mapping(
            t, h, w, block_m, block_n, linear_to_hilbert
        )
        expected = _tridiagonal(4)
        self.assertEqual(tuple(mask.shape), (4, 4))
        self.assertEqual(mask.dtype, torch.bool)
        self.assertTrue(
            torch.equal(mask, expected),
            f"Expected a banded mask, got\n{mask}",
        )

    def test_public_api_identity_mapping_gives_the_same_banded_mask(self):
        """Public API with identity (lth, htl) yields the same banded mask."""
        t, h, w = 1, 1, 8
        block_m = block_n = 2
        device = torch.device("cpu")
        linear_to_hilbert = torch.arange(8, dtype=torch.int64)
        hilbert_to_linear = torch.arange(8, dtype=torch.int64)
        mask = sliced_gilbert_block_neighbor_mapping(
            t, h, w, block_m, block_n, device,
            gilbert_mapping=(linear_to_hilbert, hilbert_to_linear),
        )
        expected = _tridiagonal(4)
        self.assertEqual(tuple(mask.shape), (4, 4))
        self.assertEqual(mask.dtype, torch.bool)
        self.assertTrue(
            torch.equal(mask, expected),
            f"Expected a banded mask, got\n{mask}",
        )

    def test_a_volume_small_enough_that_everything_touches_is_all_true(self):
        # The case the two above used to assert a diagonal for: in a 2x2x2 cube every point is a
        # neighbour of every other, so nothing about the mapping can separate the blocks.
        linear_to_hilbert = torch.arange(8, dtype=torch.int64)
        mask = _sliced_gilbert_block_neighbor_mapping(2, 2, 2, 2, 2, linear_to_hilbert)
        self.assertTrue(bool(mask.all()))

    def test_shape_and_dtype_non_square_blocks(self):
        """Shape and dtype for non-square block grid (qblocks != kblocks)."""
        t, h, w = 1, 4, 4
        block_m, block_n = 4, 2
        linear_to_hilbert = torch.arange(16, dtype=torch.int64)
        mask = _sliced_gilbert_block_neighbor_mapping(
            t, h, w, block_m, block_n, linear_to_hilbert
        )
        self.assertEqual(tuple(mask.shape), (4, 8))
        self.assertEqual(mask.dtype, torch.bool)


if __name__ == "__main__":
    unittest.main()
