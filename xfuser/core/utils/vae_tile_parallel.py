"""Dealing a tiled VAE's tiles out to the ranks of a group, a whole tile at a time.

Tiling and sharding both split a VAE decode, and composing them splits it twice. DistVAE shards
the rows of whatever it is handed, and a tiled decode hands it one tile at a time, so every tile
pays its own Patchify, a halo exchange per convolution, a reduction per norm and a gather to put
the rows back. That bill is per tile and not per pixel, so narrowing the window multiplies it
while the arithmetic each rank does shrinks, and past a certain tile count more ranks stop
buying anything at all.

Tiles are independent, which the rows inside a tile are not. Dealing whole tiles out costs two
exchanges for the whole decode however many tiles there are, and leaves each rank decoding a
tile the way one GPU would.

Nothing here knows what a tile is: a caller builds one thunk per decoder call its own loop would
have made, and gets back what all of those calls returned, on every rank, in order.
"""

import functools
import math
from typing import Callable, Dict, List, NamedTuple, Optional, Sequence, Set, Tuple

import torch
import torch.distributed as dist

# Recorded on the VAE itself, because the decision is made when the decoder would otherwise be
# sharded and acted on later, when the tile window is settled and the decode is installed.
GROUP_ATTR = "_xfuser_tile_parallel_group"

Call = Callable[[], torch.Tensor]
Dispatch = Callable[[Sequence[Call]], List[torch.Tensor]]

Where = Tuple[int, int]
Decode = Callable[[Sequence[Where]], Dict[Where, torch.Tensor]]

# Both diffusers tiling loops blend down the second-from-last axis and across the last one, on a
# 4D sample and a 5D one alike, so the assembly below needs no axis of its own to be told.
DOWN, ACROSS = -2, -1


class Blend(NamedTuple):
    """How a tiling loop stitches its tiles together, as both diffusers loops spell it"""

    down: Callable  # blend_v: mixes a tile's first `deep_down` rows with the tile above's last
    across: Callable  # blend_h: mixes its first `deep_across` columns with the left tile's last
    deep_down: int
    deep_across: int
    crop: Callable[[torch.Tensor], torch.Tensor]  # the corner of a blended tile that is kept
    # How big a whole tile is, which decides whether a run can be blended alone at all. Taken
    # from the window rather than from a decoded tile, because every rank has to reach the same
    # answer: one rank falling back while the others gather would hang the decode, not fail it.
    tile_down: int
    tile_across: int


def mark(vae, group) -> None:
    """Record that this VAE's tiles go out to `group`, rather than each tile's rows being sharded"""
    setattr(vae, GROUP_ATTR, group)


def group_of(vae):
    """The group this VAE's tiles go out to, None where its decode is not parallel this way"""
    return getattr(vae, GROUP_ATTR, None)


def in_order(calls: Sequence[Call]) -> List[torch.Tensor]:
    """Every call, here, in order: what a decode that is not parallel at all does"""
    return [call() for call in calls]


def dispatch_over(group) -> Dispatch:
    """A dispatcher giving each rank of `group` its share of the calls and every rank the results"""
    world_size = dist.get_world_size(group)
    if world_size < 2:
        return in_order
    rank = dist.get_rank(group)

    def dispatch(calls: Sequence[Call]) -> List[torch.Tensor]:
        # Fewer calls than ranks and some rank contributes nothing to the exchange, with no
        # tensor of its own to take a dtype and a device from. A decode that small is a tile or
        # two, so every rank simply making every call costs less than arranging not to.
        if len(calls) < world_size:
            return in_order(calls)
        made = [call() if n % world_size == rank else None for n, call in enumerate(calls)]
        return _share(made, group, world_size)

    return dispatch


def sharing(group) -> Tuple[Dispatch, Callable]:
    """The two ways a group divides a tiled decode: by run where it can, by call where it can't

    Runs divide the blending as well as the decoding and send back the image once rather than
    every overlapping tile, so they are what a tiled decode should use. They need a tile per rank
    and tiles wider and deeper than two blends, and those are why the other one is still here.
    """
    return dispatch_over(group), functools.partial(assemble_in_runs, group)


def runs(weights: Sequence[int], world_size: int) -> List[Tuple[int, int]]:
    """Tiles split into one contiguous run per rank, as evenly by `weights` as they divide

    Contiguous in the order the tiling loop walks, which is what makes a run cheap to blend: its
    tiles' neighbours are mostly its own. Split by tile rather than by row, because a row is too
    coarse a unit to balance with - three rows over two ranks is a two-to-one split, and the rank
    left waiting costs more than dividing the blending saves.

    Weighed by area rather than counted, because the two disagree in exactly the way a contiguous
    run is worst placed to survive. The latent bounds clip the last row and the last column, so
    the cheap tiles are not spread through the grid but gathered at the end of it, and an equal
    count of them hands the last rank the lightest work every time.

    The split minimises the heaviest run, since the decode waits for that one. Found by asking
    whether a given ceiling can be met, which is a greedy walk, and halving the interval of
    ceilings around it.
    """
    if world_size < 2:
        return [(0, len(weights))]
    low, high = max(weights), sum(weights)
    while low < high:
        middle = (low + high) // 2
        if len(_greedy(weights, middle)) <= world_size:
            high = middle
        else:
            low = middle + 1
    return _widen(_greedy(weights, low), world_size)


def shares(weights: Sequence[int], world_size: int) -> List[int]:
    """Which rank decodes each tile: contiguous runs, levelled by moving a few tiles across

    A run is the cheap shape to blend, since its tiles' neighbours are mostly its own, but it is
    a coarse shape to balance. Nine tiles over four ranks split by weight as evenly as contiguity
    allows still leaves the heaviest rank a quarter above the average, because the tiles are large
    against the share and a run cannot skip one. No weighing fixes that; only a finer assignment.

    So the runs are a starting point rather than the answer. A tile at a time moves from the
    heaviest rank to the lightest wherever that lowers the heaviest, which is what the decode
    waits for. Each move costs an exchange - the tile's neighbours are now somewhere else - and
    that is why the runs are worth starting from, and why the moves prefer a tile already beside
    the rank taking it.

    A rank down to its last tile never gives it up, because handing over everything it has cannot
    lower the higher of the two loads.
    """
    owner: List[int] = []
    for rank, (start, stop) in enumerate(runs(weights, world_size)):
        owner.extend([rank] * (stop - start))
    if world_size < 2:
        return owner

    load = [0] * world_size
    for n, weight in enumerate(weights):
        load[owner[n]] += weight

    # Bounded by the tiles: every move strictly lowers the heaviest load, so the sorted loads
    # fall each time and cannot return to where they were.
    for _ in range(len(weights)):
        heavy = max(range(world_size), key=lambda r: (load[r], -r))
        light = min(range(world_size), key=lambda r: (load[r], r))
        best = None
        for n, weight in enumerate(weights):
            if owner[n] != heavy:
                continue
            after = max(load[heavy] - weight, load[light] + weight)
            if after >= load[heavy]:
                continue
            beside = any(
                0 <= m < len(weights) and owner[m] == light for m in (n - 1, n + 1)
            )
            key = (after, 0 if beside else 1, n)
            if best is None or key < best[0]:
                best = (key, n)
        if best is None:
            break
        moved = best[1]
        owner[moved] = light
        load[heavy] -= weights[moved]
        load[light] += weights[moved]
    return owner


def _greedy(weights: Sequence[int], ceiling: int) -> List[Tuple[int, int]]:
    """The fewest contiguous runs none of which weighs more than `ceiling`"""
    out, start, carried = [], 0, 0
    for at, weight in enumerate(weights):
        if carried and carried + weight > ceiling:
            out.append((start, at))
            start, carried = at, 0
        carried += weight
    out.append((start, len(weights)))
    return out


def _widen(split: List[Tuple[int, int]], world_size: int) -> List[Tuple[int, int]]:
    """Enough runs for every rank, by halving the ones holding most tiles

    A ceiling that a few ranks can meet leaves the rest with nothing, and a rank holding no tile
    has no tensor of its own to take a dtype and a device from. Halving cannot raise the heaviest
    run, so nothing found above is given up here.
    """
    while len(split) < world_size:
        widest = max(range(len(split)), key=lambda n: split[n][1] - split[n][0])
        start, stop = split[widest]
        if stop - start < 2:
            break  # fewer tiles than ranks, which the caller declines before asking
        middle = (start + stop) // 2
        split[widest : widest + 1] = [(start, middle), (middle, stop)]
    return split


def assemble_in_runs(
    group,
    rows: int,
    columns: int,
    decode: Decode,
    blend: Blend,
    weights: Sequence[int],
) -> Optional[torch.Tensor]:
    """Assemble a tile grid with each rank decoding and blending its own run, None if it can't

    Dealing tiles out divides the decoding and leaves the blending on every rank, a cost that
    does not shrink however many ranks join the group. Giving a rank a share of neighbouring
    tiles lets it blend its own and send only the finished pieces, so the blending divides too.

    The reason a share can be blended alone is a property of the two blends. `blend_v` writes a
    tile's *first* rows and `blend_h` its *first* columns, so neither ever writes the last rows or
    the last columns - and those are the only parts of a tile that the tiles after it read. A
    tile's edges are therefore final while it is still raw, and one exchange of raw edges lets
    every rank blend its run exactly as a single rank walking the whole grid would, waiting on
    nobody else's blending.

    What comes back is each rank's cropped tiles, which are disjoint and tile the image exactly,
    so the gather carries the image once rather than every overlapping tile.

    Where a tile is smaller than twice the blend the argument fails, because the rows and columns
    the blends write would reach into the ones their neighbours read. Every reason to decline is
    one every rank reaches the same way, from the grid and the window rather than from the tiles
    a rank happens to hold: a rank that fell back alone would leave the others waiting in a
    gather it never joins, which hangs a decode rather than failing it.
    """
    world_size = dist.get_world_size(group)
    if world_size < 2:
        return None
    order = [(i, j) for i in range(rows) for j in range(columns)]
    # Fewer tiles than ranks and some rank would hold nothing, with no tensor of its own to take a
    # dtype and a device from. A decode that small has nothing worth dividing anyway.
    if len(order) < world_size:
        return None
    if blend.tile_down < 2 * blend.deep_down or blend.tile_across < 2 * blend.deep_across:
        return None

    rank = dist.get_rank(group)
    owner = shares(weights, world_size)
    mine = decode([at for n, at in enumerate(order) if owner[n] == rank])

    # Exchanged before anything is blended, both because the edges are raw at that point and
    # because a rank waiting on a neighbour's blending would serialise what this is dividing.
    edges = _share_edges(
        order, mine, owner, rank, _wanted(owner, columns, blend), group, world_size, blend
    )

    blended: Dict[Where, torch.Tensor] = {}
    kept: List[Optional[torch.Tensor]] = [None] * len(order)
    for n, (i, j) in enumerate(order):
        if owner[n] != rank:
            continue
        tile = mine[(i, j)]
        # A blend no rows deep is one the tiles do not overlap enough to need, which a wide
        # enough stride leaves. Skipped rather than called with a zero depth, because a depth of
        # zero reads as "the whole tile" everywhere an edge is sliced off the end of one.
        if i > 0 and blend.deep_down:
            # The neighbour itself where this rank blended it, and its edge rebuilt from the raw
            # ones otherwise. Both carry the same values; only the cost differs.
            above = blended.get((i - 1, j))
            tile = blend.down(
                above if above is not None else _edge_above(edges, i, j, blend),
                tile,
                blend.deep_down,
            )
        if j > 0 and blend.deep_across:
            left = blended.get((i, j - 1))
            tile = blend.across(
                left if left is not None else _edge_left(edges, i, j, blend),
                tile,
                blend.deep_across,
            )
        blended[(i, j)] = tile
        kept[n] = blend.crop(tile)

    shared = _share(kept, group, world_size)
    return torch.cat(
        [
            torch.cat(shared[i * columns : (i + 1) * columns], dim=ACROSS)
            for i in range(rows)
        ],
        dim=DOWN,
    )


def assemble_here(rows: int, columns: int, decode: Decode, blend: Blend) -> torch.Tensor:
    """Assemble the whole grid on this rank, which is what diffusers' own loop does"""
    mine = decode([(i, j) for i in range(rows) for j in range(columns)])
    # Both blends write into the tile they are handed, so each tile is blended against neighbours
    # that were themselves already blended, and the scan order that makes is part of the result.
    made = []
    above: Optional[List[torch.Tensor]] = None
    for i in range(rows):
        row = [mine[(i, j)] for j in range(columns)]
        kept = []
        for j, tile in enumerate(row):
            if above is not None:
                tile = blend.down(above[j], tile, blend.deep_down)
            if j > 0:
                tile = blend.across(row[j - 1], tile, blend.deep_across)
            row[j] = tile
            kept.append(blend.crop(tile))
        made.append(torch.cat(kept, dim=ACROSS))
        above = row
    return torch.cat(made, dim=DOWN)


def _wanted(owner: Sequence[int], columns: int, blend: Blend) -> Set[int]:
    """The tiles whose raw edges a rank other than their own will read

    Read off what the blending below asks for, tile by tile, rather than reasoned about from the
    shape of a rank's share: a rank blending a tile reaches for the one above and the one to its
    left, and only where one of those is somewhere else does anything have to travel. Where the
    shares are runs that is about a row of tiles per rank however large the grid, and where a
    tile has been moved across to level the load it is that tile's neighbours as well.

    A blend no rows deep asks for nothing, so a stride wide enough to leave the tiles touching
    rather than overlapping sends no edges at all on that axis.
    """
    wanted: Set[int] = set()
    for n, rank in enumerate(owner):
        row, column = divmod(n, columns)
        above = blend.deep_down and row > 0 and owner[n - columns] != rank
        left = blend.deep_across and column > 0 and owner[n - 1] != rank
        if above:
            wanted.add(n - columns)
        if left:
            wanted.add(n - 1)
        # The tile up and to the left is read for either of them, since blending down and then
        # across reaches diagonally through the two, and it is asked for once rather than from
        # inside each branch.
        if (above or left) and blend.deep_down and blend.deep_across:
            if row > 0 and column > 0:
                wanted.add(n - columns - 1)
    return wanted


def _share_edges(
    order: Sequence[Where],
    mine: Dict[Where, torch.Tensor],
    owner: Sequence[int],
    rank: int,
    wanted: Set[int],
    group,
    world_size: int,
    blend: Blend,
) -> Dict[Where, Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]]:
    """The last rows and last columns of the tiles another rank will read, raw, on every rank

    The edges rather than the tiles: what a neighbour reads is one blend deep, so this carries a
    fraction of what dealing the tiles themselves round would have to.
    """
    sending: List[Optional[torch.Tensor]] = [None] * (2 * len(order))
    for n in wanted:
        if owner[n] != rank:
            continue
        tile = mine[order[n]]
        # Cloned because the blending below writes into the tiles these came from, and an edge is
        # only the edge a neighbour needs while it is still raw. Guarded on the depth because a
        # slice from -0 is the whole tile rather than none of it, which would send the grid
        # itself round in place of its seams.
        if blend.deep_down:
            sending[2 * n] = tile[..., -blend.deep_down :, :].clone()
        if blend.deep_across:
            sending[2 * n + 1] = tile[..., -blend.deep_across :].clone()
    shared = _share(sending, group, world_size, next(iter(mine.values())))
    return {at: (shared[2 * n], shared[2 * n + 1]) for n, at in enumerate(order)}


def _edge_above(edges, i: int, j: int, blend: Blend) -> torch.Tensor:
    """The last rows of the tile above, as its own rank would have blended them

    Only its blending across the columns reaches its last rows, and that reads its left
    neighbour's last columns, which nothing writes. One blend of two raw edges rebuilds it.
    """
    below, _ = edges[(i - 1, j)]
    if j == 0 or not blend.deep_across:
        return below
    left, _ = edges[(i - 1, j - 1)]
    return blend.across(left, below.clone(), blend.deep_across)


def _edge_left(edges, i: int, j: int, blend: Blend) -> torch.Tensor:
    """The last columns of the tile to the left, as its own rank would have blended them

    Only its blending down the rows reaches its last columns, and that reads the corner where the
    tile above it meets the tile above and to its left - raw on both counts.
    """
    _, beside = edges[(i, j - 1)]
    if i == 0 or not blend.deep_down:
        return beside
    above, _ = edges[(i - 1, j - 1)]
    return blend.down(above[..., -blend.deep_across :], beside.clone(), blend.deep_down)


def _share(
    made: List[Optional[torch.Tensor]],
    group,
    world_size: int,
    like: Optional[torch.Tensor] = None,
) -> List[torch.Tensor]:
    """Fill in the calls this rank did not make from the ranks that did

    `like` says what to send from where this rank has nothing of its own to send, which happens
    only where what is being shared is edges: the last run has no run after it to read its own.
    """
    mine = [(n, tensor) for n, tensor in enumerate(made) if tensor is not None]

    # A rank cannot work out the shape of a call it did not make: tiles at the right and bottom
    # edges are clipped by the latent bounds, and a rank can hold none of them. One object
    # exchange settles that for the whole decode.
    manifest: List = [None] * world_size
    dist.all_gather_object(manifest, [(n, tuple(t.shape)) for n, t in mine], group=group)

    # Nothing to send is a real answer here, not an empty group: the last run has no run after it
    # to read its edges. Its rank still joins the exchange above, so nobody is left waiting.
    width = max(sum(math.prod(shape) for _, shape in entries) for entries in manifest)
    if width == 0:
        return list(made)

    # Then one tensor exchange for the results themselves, flattened together and padded to the
    # largest share, since all_gather wants every rank sending the same count. Ranks differ by at
    # most one call, so the padding is at most one call's worth of the traffic.
    #
    # Filled a call at a time rather than concatenated into the buffer, which would hold a second
    # copy of everything this rank decoded while the first was still alive. Left uninitialised
    # past what this rank sends: the manifest bounds what each share is read back out of, so the
    # padding is never looked at.
    sample = mine[0][1] if mine else like
    sending = torch.empty(width, dtype=sample.dtype, device=sample.device)
    at = 0
    for _, tensor in mine:
        sending[at : at + tensor.numel()] = tensor.reshape(-1)
        at += tensor.numel()
    received = [torch.empty_like(sending) for _ in range(world_size)]
    dist.all_gather(received, sending, group=group)

    shared: List[torch.Tensor] = list(made)
    for entries, buffer in zip(manifest, received):
        at = 0
        for n, shape in entries:
            size = math.prod(shape)
            # What this rank decoded itself is kept as it decoded it, rather than read back out
            # of its own copy in the buffer.
            if shared[n] is None:
                shared[n] = buffer[at : at + size].view(shape)
            at += size
    return shared
