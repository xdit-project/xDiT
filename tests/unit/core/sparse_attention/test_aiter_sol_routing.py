"""Sol-Attn's host-side bookkeeping: the block tile and the K/V tail.

The tile is passed explicitly, as --sparseattn_block_size hands it over, so these need no AITER
manifest.
"""

import pytest
import torch

from xfuser.core.attention.sparsity import sol

TILE = (256, 128)


def test_an_explicit_block_tile_is_used_as_given():
    """check_sol_attn_recipe has already validated it, so there is nothing left to ask aiter."""
    assert sol.sol_attn_block_tile(sol._RECIPES["fp8"], block_tile=[64, 64]) == (64, 64)


@pytest.mark.parametrize("tile", [TILE, (64, 64)])
@pytest.mark.parametrize("ragged_kv", [True, False])
def test_kv_is_padded_to_the_tile_only_for_a_row_that_cannot_bound_its_last_block(monkeypatch, tile, ragged_kv):
    """A zero key scores 0 rather than -inf, so a pad the row does not need takes softmax mass.
    The gfx950 rows mask a short last block themselves; a row that indexes KV in whole tiles does
    not."""
    monkeypatch.setattr(sol, "_takes_ragged_kv", lambda recipe, block_tile: ragged_kv)
    kv_tile = tile[1]
    key = torch.randn(1, 2 * kv_tile + 5, 2, 128)

    padded_key, padded_value = sol._pad_kv_to_tile(key, key, sol._RECIPES["f6f4"], tile)

    if ragged_kv:
        assert padded_key is key and padded_value is key
    else:
        assert padded_key.shape[1] == 3 * kv_tile
        torch.testing.assert_close(padded_key[:, : key.shape[1]], key)
        assert not padded_key[:, key.shape[1] :].any()


def test_a_whole_number_of_blocks_is_never_padded(monkeypatch):
    monkeypatch.setattr(sol, "_takes_ragged_kv", lambda recipe, block_tile: False)
    key = torch.randn(1, 2 * TILE[1], 2, 128)

    padded_key, _ = sol._pad_kv_to_tile(key, key, sol._RECIPES["fp8"], TILE)

    assert padded_key is key
