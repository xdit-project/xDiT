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

    # For eight points in a line, each interior point is adjacent only to its immediate predecessor
    # and successor. With two points per block, the expected block-neighbor mask is tridiagonal.
    # A 2x2x2 grid is unsuitable because every block is adjacent to every other block.
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

    def test_public_api_identity_mapping_returns_tridiagonal_mask(self):
        """The public API returns a tridiagonal mask for identity mappings."""
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
        # In a 2x2x2 cube every point neighbors every other point, so the mask is fully connected.
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
