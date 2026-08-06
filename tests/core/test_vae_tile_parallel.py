"""Dealing a tiled decode's calls out to a group, over gloo, without a GPU or a VAE in sight"""

import itertools
import random
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


def _every_split(tiles: int, world_size: int):
    """Every way to cut `tiles` into `world_size` contiguous non-empty runs"""
    for cuts in itertools.combinations(range(1, tiles), world_size - 1):
        edges = (0,) + cuts + (tiles,)
        yield list(zip(edges, edges[1:]))


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


BAND_VAES = {
    "AutoencoderKL": (
        dict(
            block_out_channels=[8, 8, 16, 16],
            layers_per_block=1,
            latent_channels=4,
            norm_num_groups=8,
            sample_size=256,
            down_block_types=["DownEncoderBlock2D"] * 4,
            up_block_types=["UpDecoderBlock2D"] * 4,
        ),
        False,
    ),
    "AutoencoderKLWan": (
        dict(base_dim=8, z_dim=4, dim_mult=[1, 2, 4, 4], num_res_blocks=1),
        True,
    ),
    "AutoencoderKLQwenImage": (
        dict(base_dim=8, z_dim=4, dim_mult=[1, 2, 4, 4], num_res_blocks=1),
        True,
    ),
}
# Two windows down by three across comes out as a 3x4 grid of tiles at a quarter overlap. It is
# deliberately not square and deliberately not a multiple of the ranks: twelve tiles over four
# ranks is three each against four columns, so every rank's run starts and ends mid-row, which is
# the case a split by whole rows would never reach. The tiles are decoded on a CPU here, so a
# wider grid costs minutes rather than the coverage it looks like it buys.
WINDOWS_DOWN, WINDOWS_ACROSS = 2, 3


def _tiled_vae(name: str):
    """The same small VAE and latents on every rank, at a window several tiles across"""
    import diffusers

    from xfuser.core.utils import vae_parallel, vae_tiling

    kwargs, video = BAND_VAES[name]
    # Importing xfuser on ROCm swaps torch's GroupNorm for AITER's, which faults on the CPU
    # tensors these tiles are made of. The revert has to land before the VAE is assembled.
    vae_parallel.restore_torch_group_norm()
    # Seeded because every rank builds its own and they have to agree to the bit.
    torch.manual_seed(0)
    vae = getattr(diffusers, name)(**kwargs).eval()
    vae.enable_tiling()
    _, plan = vae_tiling.snap_tile_window(vae, vae_tiling.tile_window(vae) // 4)
    vae_tiling.apply_tile_plan(vae, plan)

    window = vae_tiling.latent_rows(vae, plan)
    down, across = window * WINDOWS_DOWN, window * WINDOWS_ACROSS
    shape = (1, 4, 2, down, across) if video else (1, 4, down, across)
    torch.manual_seed(1)
    return vae, torch.randn(*shape)


def _bands_in_a_group(rank: int, world_size: int, port: int, name: str) -> None:
    """One rank blending its own band, checked against the whole grid blended by one rank"""
    from xfuser.core.utils import vae_tile_parallel, vae_tiling

    dist.init_process_group(
        "gloo",
        rank=rank,
        world_size=world_size,
        init_method=f"tcp://127.0.0.1:{port}",
        timeout=timedelta(seconds=300),
    )
    try:
        vae, latents = _tiled_vae(name)
        with torch.no_grad():
            expected = vae.tiled_decode(latents).sample

            dispatch, assemble = vae_tile_parallel.sharing(dist.group.WORLD)
            # One tile per call, so that what is compared is where the tiles were decoded and
            # blended and not how many of them shared a call.
            decode = vae_tiling.tiled_decode_for(vae, 0, dispatch, assemble)
            assert decode is not None, f"no reimplemented loop for {name}"
            got = decode(latents).sample

        assert got.shape == expected.shape, f"{got.shape} != {expected.shape}"
        # Bit-exact, not close: a band replays the blending its neighbour would have done on the
        # same values, so there is no reordering to excuse a difference.
        torch.testing.assert_close(got, expected, rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


class TestBands(unittest.TestCase):
    """Tiles split into a contiguous run per rank, blended locally, gathered back whole"""

    def test_tiles_of_equal_weight_are_split_as_evenly_as_they_divide(self):
        self.assertEqual(vae_tile_parallel.runs([1] * 4, 2), [(0, 2), (2, 4)])
        self.assertEqual(
            vae_tile_parallel.runs([1] * 5, 4), [(0, 2), (2, 3), (3, 4), (4, 5)]
        )
        self.assertEqual(vae_tile_parallel.runs([1] * 3, 1), [(0, 3)])

    def test_the_heaviest_run_is_the_lightest_it_can_be(self):
        # Against every contiguous split there is, at sizes small enough to enumerate them all.
        random.seed(7)
        for tiles in range(2, 9):
            for world_size in range(2, min(tiles, 5) + 1):
                for _ in range(20):
                    weights = [random.randint(1, 9) for _ in range(tiles)]
                    mine = max(
                        sum(weights[start:stop])
                        for start, stop in vae_tile_parallel.runs(weights, world_size)
                    )
                    best = min(
                        max(sum(weights[start:stop]) for start, stop in split)
                        for split in _every_split(tiles, world_size)
                    )
                    self.assertEqual(mine, best, f"{weights} over {world_size}")

    def test_the_lighter_tiles_at_the_end_do_not_all_land_on_one_rank(self):
        # The case that a run split by count got wrong: the latent bounds clip the last row and
        # the last column, so an equal count of tiles is an unequal amount of work.
        weights = [9, 9, 4, 9, 9, 4, 4, 4, 2]
        held = [sum(weights[start:stop]) for start, stop in vae_tile_parallel.runs(weights, 2)]
        self.assertLessEqual(max(held) / min(held), 1.1, held)

    def test_every_rank_holds_a_tile_and_every_tile_is_held_once(self):
        random.seed(11)
        for tiles in range(1, 12):
            for world_size in range(1, min(tiles, 6) + 1):
                weights = [random.randint(1, 9) for _ in range(tiles)]
                split = vae_tile_parallel.runs(weights, world_size)
                self.assertEqual(len(split), world_size, f"{weights}/{world_size}")
                self.assertTrue(all(stop > start for start, stop in split), split)
                covered = [n for start, stop in split for n in range(start, stop)]
                self.assertEqual(covered, list(range(tiles)), f"{tiles}/{world_size}")

    def test_a_band_decode_is_what_one_rank_blending_everything_gives(self):
        for name in BAND_VAES:
            for world_size in (2, 4):
                with self.subTest(vae=name, world_size=world_size):
                    mp.spawn(
                        _bands_in_a_group,
                        args=(world_size, _free_port(), name),
                        nprocs=world_size,
                        join=True,
                    )


if __name__ == "__main__":
    unittest.main()
