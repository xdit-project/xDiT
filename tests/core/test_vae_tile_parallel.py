"""Dealing a tiled decode's calls out to a group, over gloo, without a GPU or a VAE in sight"""

import socket
import unittest
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from xfuser.core.utils import vae_tile_parallel

# Shapes differing by rank share and by size, the way a tile grid's edges do: the last two are
# clipped, and no rank holds both of them.
SHAPES = ((2, 3), (2, 3), (2, 3), (2, 3), (1, 3), (2, 1))


def _result(index: int) -> torch.Tensor:
    """What call `index` returns: its own number, in a shape only it has"""
    return torch.full(SHAPES[index], float(index), dtype=torch.float32)


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _dispatch_in_a_group(rank: int, world_size: int, port: int, calls_made: int) -> None:
    """One rank of the group, asserting for itself; mp.spawn re-raises what it fails on"""
    dist.init_process_group(
        "gloo",
        rank=rank,
        world_size=world_size,
        init_method=f"tcp://127.0.0.1:{port}",
        timeout=timedelta(seconds=120),
    )
    try:
        made = []

        def call(index):
            made.append(index)
            return _result(index)

        indices = list(range(calls_made))
        results = vae_tile_parallel.dispatch_over(dist.group.WORLD)(
            [lambda index=index: call(index) for index in indices]
        )

        assert len(results) == len(indices), f"{len(results)} results for {len(indices)} calls"
        for index, result in zip(indices, results):
            # Every rank ends holding every result, whoever computed it, in the order the calls
            # were given rather than the order they were made.
            torch.testing.assert_close(result, _result(index), rtol=0, atol=0)

        if len(indices) < world_size:
            # Too few to divide, so each rank makes them all rather than leaving a rank with
            # nothing to send.
            assert made == indices, f"rank {rank} made {made}, not all of {indices}"
        else:
            assert made == indices[rank::world_size], f"rank {rank} made {made}"
    finally:
        dist.destroy_process_group()


class TestDispatchOverAGroup(unittest.TestCase):
    """A rank makes its share of the calls and comes away with what every other rank made"""

    def _spawn(self, world_size: int, calls: int) -> None:
        mp.spawn(
            _dispatch_in_a_group,
            args=(world_size, _free_port(), calls),
            nprocs=world_size,
            join=True,
        )

    def test_the_calls_are_divided_and_the_results_shared(self):
        # Six calls over two ranks divides evenly, over four it does not, and the odd rank out
        # sends one call's worth less than the others.
        for world_size in (2, 4):
            with self.subTest(world_size=world_size):
                self._spawn(world_size, len(SHAPES))

    def test_a_group_of_one_makes_every_call_itself(self):
        self._spawn(1, len(SHAPES))

    def test_fewer_calls_than_ranks_leaves_every_rank_making_them_all(self):
        self._spawn(4, 3)


if __name__ == "__main__":
    unittest.main()
