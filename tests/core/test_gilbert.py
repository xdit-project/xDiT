import unittest

import torch

from xfuser.core.sparge_attention.gilbert import (
    sliced_gilbert_block_neighbor_mapping,
    _sliced_gilbert_block_neighbor_mapping,
)


class TestSlicedGilbertBlockNeighborMapping(unittest.TestCase):
    """Tests for sliced_gilbert_block_neighbor_mapping with known expected results."""

    def test_identity_mapping_gives_block_diagonal_mask(self):
        """Identity linear_to_hilbert and block_m == block_n -> strictly diagonal mask."""
        t, h, w = 2, 2, 2
        block_m = block_n = 2
        linear_to_hilbert = torch.arange(8, dtype=torch.int64)
        mask = _sliced_gilbert_block_neighbor_mapping(
            t, h, w, block_m, block_n, linear_to_hilbert
        )
        expected = torch.eye(4, dtype=torch.bool)
        self.assertEqual(tuple(mask.shape), (4, 4))
        self.assertEqual(mask.dtype, torch.bool)
        self.assertTrue(
            torch.equal(mask, expected),
            f"Expected block-diagonal mask, got\n{mask}",
        )

    def test_public_api_identity_mapping_gives_same_block_diagonal_mask(self):
        """Public API with identity (lth, htl) yields same block-diagonal mask."""
        t, h, w = 2, 2, 2
        block_m = block_n = 2
        device = torch.device("cpu")
        linear_to_hilbert = torch.arange(8, dtype=torch.int64)
        hilbert_to_linear = torch.arange(8, dtype=torch.int64)
        mask = sliced_gilbert_block_neighbor_mapping(
            t, h, w, block_m, block_n, device,
            gilbert_mapping=(linear_to_hilbert, hilbert_to_linear),
        )
        expected = torch.eye(4, dtype=torch.bool)
        self.assertEqual(tuple(mask.shape), (4, 4))
        self.assertEqual(mask.dtype, torch.bool)
        self.assertTrue(
            torch.equal(mask, expected),
            f"Expected block-diagonal mask, got\n{mask}",
        )

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
