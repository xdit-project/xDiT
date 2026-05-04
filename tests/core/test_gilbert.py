import unittest

try:
    import numpy
    from xfuser.core.sparge_attention.gilbert import (
        sliced_gilbert_block_neighbor_mapping,
        _sliced_gilbert_block_neighbor_mapping,
    )
    _GILBERT_IMPORT_OK = True
except ImportError:
    _GILBERT_IMPORT_OK = False


@unittest.skipUnless(_GILBERT_IMPORT_OK, "numba/numpy and gilbert module required")
class TestSlicedGilbertBlockNeighborMapping(unittest.TestCase):
    """Tests for sliced_gilbert_block_neighbor_mapping with known expected results."""

    def test_identity_mapping_gives_block_diagonal_mask(self):
        """Identity linear_to_hilbert and block_m == block_n -> strictly diagonal mask."""
        t, h, w = 2, 2, 2
        block_m = block_n = 2
        linear_to_hilbert = numpy.arange(8, dtype=numpy.int64)
        mask = _sliced_gilbert_block_neighbor_mapping(
            t, h, w, block_m, block_n, linear_to_hilbert
        )
        expected = numpy.eye(4, dtype=numpy.bool_)
        self.assertEqual(mask.shape, (4, 4))
        self.assertEqual(mask.dtype, numpy.bool_)
        self.assertTrue(
            numpy.array_equal(mask, expected),
            f"Expected block-diagonal mask, got\n{mask}",
        )

    def test_public_api_identity_mapping_gives_same_block_diagonal_mask(self):
        """Public API with identity (lth, htl) yields same block-diagonal mask."""
        t, h, w = 2, 2, 2
        block_m = block_n = 2
        linear_to_hilbert = numpy.arange(8, dtype=numpy.int64)
        hilbert_to_linear = numpy.arange(8, dtype=numpy.int64)
        mask = sliced_gilbert_block_neighbor_mapping(
            t, h, w, block_m, block_n,
            gilbert_mapping=(linear_to_hilbert, hilbert_to_linear),
        )
        expected = numpy.eye(4, dtype=numpy.bool_)
        self.assertEqual(mask.shape, (4, 4))
        self.assertEqual(mask.dtype, numpy.bool_)
        self.assertTrue(
            numpy.array_equal(mask, expected),
            f"Expected block-diagonal mask, got\n{mask}",
        )

    def test_shape_and_dtype_non_square_blocks(self):
        """Shape and dtype for non-square block grid (qblocks != kblocks)."""
        t, h, w = 1, 4, 4
        block_m, block_n = 4, 2
        linear_to_hilbert = numpy.arange(16, dtype=numpy.int64)
        mask = _sliced_gilbert_block_neighbor_mapping(
            t, h, w, block_m, block_n, linear_to_hilbert
        )
        self.assertEqual(mask.shape, (4, 8))
        self.assertEqual(mask.dtype, numpy.bool_)


if __name__ == "__main__":
    unittest.main()
