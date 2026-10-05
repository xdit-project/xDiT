"""Deterministic load balancing for tile-parallel model stages."""

from __future__ import annotations

from collections.abc import Sequence


def assign_tiles_to_ranks(tile_volumes: Sequence[int], world_size: int) -> list[int]:
    """Assign tiles using deterministic longest-processing-time scheduling.

    Tile volume is used as a proxy for decode cost. Tiles are considered from
    largest to smallest and assigned to the rank with the least accumulated
    volume. Ties are resolved by tile index and then rank, making every rank
    independently derive the same ownership map.

    Returns one owner rank per tile in the original tile order.
    """
    if world_size <= 0:
        raise ValueError(f"world_size must be positive, got {world_size}")
    if any(volume < 0 for volume in tile_volumes):
        raise ValueError("tile volumes must be non-negative")

    owners = [0] * len(tile_volumes)
    rank_loads = [0] * world_size

    tile_order = sorted(
        range(len(tile_volumes)),
        key=lambda tile_idx: (-tile_volumes[tile_idx], tile_idx),
    )
    for tile_idx in tile_order:
        owner = min(
            range(world_size),
            key=lambda rank: (rank_loads[rank], rank),
        )
        owners[tile_idx] = owner
        rank_loads[owner] += tile_volumes[tile_idx]

    return owners
