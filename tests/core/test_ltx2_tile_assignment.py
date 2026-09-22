import importlib.util
from pathlib import Path

import pytest


MODULE_PATH = (
    Path(__file__).parents[2]
    / "xfuser/model_executor/layers/ltx2/tile_assignment.py"
)
SPEC = importlib.util.spec_from_file_location("ltx2_tile_assignment", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)
assign_tiles_to_ranks = MODULE.assign_tiles_to_ranks


LTX2_1024_1536_121_TILE_VOLUMES = [
    368640,
    368640,
    61440,
    153600,
    153600,
    25600,
    304128,
    304128,
    50688,
    126720,
    126720,
    21120,
]


def _rank_loads(volumes, owners, world_size):
    loads = [0] * world_size
    for volume, owner in zip(volumes, owners):
        loads[owner] += volume
    return loads


def test_balances_ltx2_default_decode_tiles():
    owners = assign_tiles_to_ranks(LTX2_1024_1536_121_TILE_VOLUMES, 8)

    assert owners == [0, 1, 6, 4, 5, 4, 2, 3, 7, 6, 7, 5]
    assert _rank_loads(LTX2_1024_1536_121_TILE_VOLUMES, owners, 8) == [
        368640,
        368640,
        304128,
        304128,
        179200,
        174720,
        188160,
        177408,
    ]


def test_ties_are_resolved_by_tile_then_rank():
    assert assign_tiles_to_ranks([10, 10, 10, 10], 2) == [0, 1, 0, 1]


def test_supports_more_ranks_than_tiles():
    assert assign_tiles_to_ranks([3, 2], 4) == [0, 1]


@pytest.mark.parametrize("world_size", [0, -1])
def test_rejects_non_positive_world_size(world_size):
    with pytest.raises(ValueError, match="world_size must be positive"):
        assign_tiles_to_ranks([1], world_size)


def test_rejects_negative_volume():
    with pytest.raises(ValueError, match="tile volumes must be non-negative"):
        assign_tiles_to_ranks([1, -1], 2)
