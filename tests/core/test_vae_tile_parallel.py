"""Dealing a tiled decode's calls out to a group, over gloo, without a GPU or a VAE in sight"""

import itertools
import os
import random
import socket
import unittest
from datetime import timedelta
from typing import List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from xfuser.core.utils import vae_tile_parallel

# Shapes differing by rank share and by size, the way a tile grid's edges do: the last two are
# clipped, and no rank holds both of them.
SHAPES = ((2, 3), (2, 3), (2, 3), (2, 3), (1, 3), (2, 1))


def _result(index: int, device=None) -> torch.Tensor:
    """What call `index` returns: its own number, in a shape only it has"""
    return torch.full(SHAPES[index], float(index), dtype=torch.float32, device=device)


def _load(weights, owner, world_size: int) -> List[int]:
    """What each rank carries under an assignment"""
    load = [0] * world_size
    for n, weight in enumerate(weights):
        load[owner[n]] += weight
    return load


def _by_runs(weights, world_size: int) -> List[int]:
    """The assignment before any tile is moved to level it"""
    return [
        rank
        for rank, (start, stop) in enumerate(vae_tile_parallel.runs(weights, world_size))
        for _ in range(start, stop)
    ]


def _every_split(tiles: int, world_size: int):
    """Every way to cut `tiles` into `world_size` contiguous non-empty runs"""
    for cuts in itertools.combinations(range(1, tiles), world_size - 1):
        edges = (0,) + cuts + (tiles,)
        yield list(zip(edges, edges[1:]))


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _spawn_group(target, world_size: int, *rest, attempts: int = 5) -> None:
    """Run `target` across `world_size` processes, on a port the rendezvous can actually have

    A port is picked by binding one and immediately letting it go, so between picking it and the
    rendezvous binding it for real there is a window where anything else on the machine can take
    it - and on a busy box something occasionally does. That fails the run with EADDRINUSE, which
    says nothing about what was being tested and makes the suite intermittently red. Retried on
    that one error only: when the rendezvous is what failed, nothing under test has run yet.
    """
    for attempt in range(attempts):
        try:
            mp.spawn(
                target,
                args=(world_size, _free_port(), *rest),
                nprocs=world_size,
                join=True,
            )
            return
        except Exception as error:
            if attempt == attempts - 1 or "EADDRINUSE" not in str(error):
                raise


def _cores_allowed() -> int:
    """The cores this process may actually use, which is not the number it can see

    Under a container CPU limit the kernel enforces a quota rather than an affinity mask, so
    `os.cpu_count()` reports the whole host - 128 where the quota was 8 - and anything sizing a
    thread pool from it asks for sixteen times the machine it has been given.
    """
    try:
        quota, period = open("/sys/fs/cgroup/cpu.max").read().split()
        if quota != "max":
            return max(1, int(quota) // int(period))
    except (OSError, ValueError):
        pass
    return os.cpu_count() or 1


def _share_the_cores(world_size: int) -> None:
    """Take a share of what this process may use, since the other ranks are here too

    Every rank is a process of its own, and four of them each sizing a thread pool from the whole
    host put 512 threads on an 8-core quota. The four-rank decodes then ran an order of magnitude
    longer than the two-rank ones and looked for all the world like a deadlock. These tests check
    what the assembly computes, not how fast it computes it.
    """
    torch.set_num_threads(max(1, _cores_allowed() // world_size))


def _backend_for(world_size: int) -> Tuple[str, Optional[str]]:
    """The collective backend to use and the device to put this rank's tensors on

    Gloo on the CPU runs anywhere, which is why these tests were written for it, but it is neither
    the fast path nor the one shipped: a real decode gathers over RCCL between devices. Where the
    group can have a device each, use that - it exercises the collective that will actually carry
    the tiles, and a decode that takes minutes on CPU takes seconds. Where it cannot, gloo still
    checks the arithmetic, which is what these tests are for.
    """
    if torch.cuda.is_available() and torch.cuda.device_count() >= world_size:
        return dist.Backend.NCCL, "cuda"
    return dist.Backend.GLOO, None


def _dispatch_in_a_group(rank: int, world_size: int, port: int, calls_made: int) -> None:
    """One rank of the group, asserting for itself; mp.spawn re-raises what it fails on"""
    _share_the_cores(world_size)
    backend, device = _backend_for(world_size)
    if device is not None:
        torch.cuda.set_device(rank)
    dist.init_process_group(
        backend,
        rank=rank,
        world_size=world_size,
        init_method=f"tcp://127.0.0.1:{port}",
        timeout=timedelta(seconds=120),
    )
    try:
        made = []

        def call(index):
            made.append(index)
            return _result(index, device)

        indices = list(range(calls_made))
        results = vae_tile_parallel.dispatch_over(dist.group.WORLD)(
            [lambda index=index: call(index) for index in indices]
        )

        assert len(results) == len(indices), f"{len(results)} results for {len(indices)} calls"
        for index, result in zip(indices, results):
            # Every rank ends holding every result, whoever computed it, in the order the calls
            # were given rather than the order they were made.
            torch.testing.assert_close(result, _result(index, device), rtol=0, atol=0)

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
        _spawn_group(_dispatch_in_a_group, world_size, calls)

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


RUN_VAES = {
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


def _blend(deep_down: int, deep_across: int):
    """A Blend carrying nothing but its depths, which is all `_wanted` reads"""
    return vae_tile_parallel.Blend(
        down=None,
        across=None,
        deep_down=deep_down,
        deep_across=deep_across,
        crop=None,
        tile_down=64,
        tile_across=64,
    )


def _tiled_vae(name: str, device=None, overlap: Optional[float] = None):
    """The same small VAE and latents on every rank, at a window several tiles across"""
    import diffusers

    from xfuser.core.utils import vae_parallel, vae_tiling

    kwargs, video = RUN_VAES[name]
    # Importing xfuser on ROCm swaps torch's GroupNorm for AITER's, which faults on the CPU
    # tensors these tiles are made of. The revert has to land before the VAE is assembled, and it
    # stays reverted on a device too so that both paths compute the same arithmetic.
    vae_parallel.restore_torch_group_norm()
    # Seeded because every rank builds its own and they have to agree to the bit.
    torch.manual_seed(0)
    vae = getattr(diffusers, name)(**kwargs).eval()
    vae.enable_tiling()
    _, plan = vae_tiling.snap_tile_window(vae, vae_tiling.tile_window(vae) // 4)
    vae_tiling.apply_tile_plan(vae, plan)
    if overlap is not None:
        step = vae_tiling.tile_overlap_plan(vae, overlap)
        assert step is not None, f"{name} cannot step its tiles at {overlap}"
        vae_tiling.apply_tile_plan(vae, step)

    window = vae_tiling.latent_rows(vae, plan)
    down, across = window * WINDOWS_DOWN, window * WINDOWS_ACROSS
    shape = (1, 4, 2, down, across) if video else (1, 4, down, across)
    torch.manual_seed(1)
    latents = torch.randn(*shape)
    if device is not None:
        vae, latents = vae.to(device), latents.to(device)
    return vae, latents


def _runs_in_a_group(
    rank: int, world_size: int, port: int, name: str, overlap: Optional[float] = None
) -> None:
    """One rank blending its own run, checked against the whole grid blended by one rank"""
    from xfuser.core.utils import vae_tile_parallel, vae_tiling

    _share_the_cores(world_size)
    backend, device = _backend_for(world_size)
    if device is not None:
        torch.cuda.set_device(rank)
        device = f"cuda:{rank}"
    dist.init_process_group(
        backend,
        rank=rank,
        world_size=world_size,
        init_method=f"tcp://127.0.0.1:{port}",
        timeout=timedelta(seconds=300),
    )
    try:
        vae, latents = _tiled_vae(name, device, overlap)
        with torch.no_grad():
            expected = vae.tiled_decode(latents).sample

            dispatch, assemble = vae_tile_parallel.sharing(dist.group.WORLD)
            decode = vae_tiling.tiled_decode_for(vae, dispatch, assemble)
            assert decode is not None, f"no reimplemented loop for {name}"
            got = decode(latents).sample

        assert got.shape == expected.shape, f"{got.shape} != {expected.shape}"
        # Bit-exact, not close: a run replays the blending its neighbour would have done on the
        # same values, so there is no reordering to excuse a difference.
        #
        # That holds on gloo, which is what -TestGpus 1 runs, and it holds on four devices over
        # RCCL. It was for a while thought not to: AutoencoderKL at two ranks missed by 2.1e-06
        # on a device while passing at four and passing on Wan and Qwen-Image at both, and that
        # was written up here as an accelerator picking its convolution differently. It was not.
        # Every one of those runs installed a branch xDiT over the DistVAE the image happened to
        # ship, and DistVAE is what swaps GroupNorm; with both repos pinned to matching commits
        # the whole family is bit-exact on four devices. So read a failure here as this code, or
        # as a mismatched pair - not as the hardware.
        torch.testing.assert_close(got, expected, rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


class TestRuns(unittest.TestCase):
    """Tiles split into a contiguous run per rank, blended locally, gathered back whole"""

    def test_tiles_of_equal_weight_are_split_as_evenly_as_they_divide(self):
        # Evenly means no run heavier than it has to be, which for equal weights is the share
        # rounded up. It does not mean the runs are the same length: ten tiles over three ranks
        # divides 4, 4, 2, and the 2 costs nothing because the 4s are what the decode waits for.
        for tiles, world_size in ((4, 2), (5, 4), (3, 1), (9, 4), (10, 3), (12, 5)):
            split = vae_tile_parallel.runs([1] * tiles, world_size)
            self.assertEqual(len(split), world_size, split)
            self.assertEqual(split[0][0], 0, split)
            self.assertEqual(split[-1][1], tiles, split)
            for (_, stop), (start, _) in zip(split, split[1:]):
                self.assertEqual(stop, start, split)
            longest = max(stop - start for start, stop in split)
            self.assertEqual(longest, -(-tiles // world_size), split)

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
        # the last column, so an equal count of tiles is an unequal amount of work. Weighing the
        # split is not enough on its own here - the best contiguous cut of these nine is 31
        # against 23 - so it takes the levelling to move one tile across and even them up.
        weights = [9, 9, 4, 9, 9, 4, 4, 4, 2]
        owner = vae_tile_parallel.shares(weights, 2)
        held = [
            sum(weight for n, weight in enumerate(weights) if owner[n] == rank)
            for rank in range(2)
        ]
        self.assertLessEqual(max(held) / min(held), 1.1, held)

    def test_levelling_never_leaves_a_rank_worse_off_than_the_runs_it_started_from(self):
        random.seed(13)
        for tiles in range(2, 20):
            for world_size in range(2, min(tiles, 6) + 1):
                for _ in range(10):
                    weights = [random.randint(1, 9) for _ in range(tiles)]
                    owner = vae_tile_parallel.shares(weights, world_size)
                    self.assertEqual(set(owner), set(range(world_size)), weights)
                    self.assertEqual(len(owner), tiles)
                    self.assertLessEqual(
                        max(_load(weights, owner, world_size)),
                        max(_load(weights, _by_runs(weights, world_size), world_size)),
                        f"{weights} over {world_size}",
                    )

    def test_a_tile_moves_across_where_a_run_cannot_be_levelled(self):
        # The grid measured on four ranks: nine tiles, the last row and column clipped. Contiguity
        # alone leaves the heaviest rank a quarter above the lightest possible; a tile moving
        # across takes that back.
        weights = [16384, 16384, 8192, 16384, 16384, 8192, 8192, 8192, 4096]
        by_runs = max(_load(weights, _by_runs(weights, 4), 4))
        levelled = max(_load(weights, vae_tile_parallel.shares(weights, 4), 4))
        self.assertEqual(by_runs, 32768)
        self.assertEqual(levelled, 28672)

    def test_the_edges_asked_for_are_the_edges_the_blending_reaches_for(self):
        random.seed(17)
        for rows in range(1, 6):
            for columns in range(1, 6):
                for world_size in range(2, min(rows * columns, 5) + 1):
                    weights = [random.randint(1, 9) for _ in range(rows * columns)]
                    owner = vae_tile_parallel.shares(weights, world_size)
                    blend = _blend(1, 1)
                    wanted = vae_tile_parallel._wanted(owner, columns, blend)
                    reaches = set()
                    for n in range(rows * columns):
                        row, column = divmod(n, columns)
                        if row > 0 and owner[n - columns] != owner[n]:
                            reaches.add(n - columns)
                            if column:
                                reaches.add(n - columns - 1)
                        if column > 0 and owner[n - 1] != owner[n]:
                            reaches.add(n - 1)
                            if row:
                                reaches.add(n - columns - 1)
                    # Equal, not merely a superset. Asked only to contain what the blending
                    # reads, this passed for a _wanted that returned the whole grid - which
                    # changes no arithmetic and so passes the decode tests too, while putting
                    # every tile on the wire. Over-fetching is the failure this can see and
                    # nothing else can.
                    self.assertEqual(
                        wanted, reaches, f"{rows}x{columns} over {world_size}"
                    )

    def test_a_blend_no_rows_deep_asks_for_no_edges_on_that_axis(self):
        # A wide enough stride leaves the tiles touching rather than overlapping, and then there
        # is nothing to blend and nothing to send. Worth its own case because a depth of zero is
        # not inert where the edges are sliced: `tile[..., -0:, :]` is the whole tile, so a loop
        # that only skipped the blending would still put the entire grid on the wire.
        weights = [1] * 12
        owner = vae_tile_parallel.shares(weights, 4)
        self.assertEqual(vae_tile_parallel._wanted(owner, 4, _blend(0, 0)), set())
        # One axis at a time, since the strides are set per axis and only one of them can run
        # out. Each asks for less than both do - the corner tile a blended edge is rebuilt from
        # is only reached when both blends happen - and neither asks for anything the pair does
        # not.
        both = vae_tile_parallel._wanted(owner, 4, _blend(1, 1))
        for depths in ((1, 0), (0, 1)):
            wanted = vae_tile_parallel._wanted(owner, 4, _blend(*depths))
            self.assertTrue(wanted, depths)
            self.assertTrue(wanted < both, depths)

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

    def test_a_run_decode_is_what_one_rank_blending_everything_gives(self):
        for name in RUN_VAES:
            for world_size in (2, 4):
                with self.subTest(vae=name, world_size=world_size):
                    _spawn_group(_runs_in_a_group, world_size, name, None)

    def test_tiles_that_do_not_overlap_at_all_still_assemble(self):
        # --vae_tile_overlap can widen the stride until the tiles touch rather than overlap, and
        # then the blends are no rows deep. Checked against the same VAE's own loop at the same
        # stride, so what this proves is that dividing the work changes nothing: a depth of zero
        # has to mean no blending and no edges, and not the whole tile taken as its own edge.
        for name in RUN_VAES:
            with self.subTest(vae=name):
                _spawn_group(_runs_in_a_group, 4, name, 0.0)


if __name__ == "__main__":
    unittest.main()
