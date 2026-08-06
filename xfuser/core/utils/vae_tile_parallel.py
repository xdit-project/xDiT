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
from typing import Callable, Dict, List, NamedTuple, Optional, Sequence, Tuple

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
    """The two ways a group divides a tiled decode: by band where it can, by call where it can't

    Bands divide the blending as well as the decoding and send back a disjoint piece of the image
    rather than every tile, so they are what a tiled decode should use. They need at least one
    row of tiles per rank, and a grid that coarse is why the other one is still here.
    """
    return dispatch_over(group), functools.partial(assemble_in_bands, group)


def bands(rows: int, world_size: int) -> List[Tuple[int, int]]:
    """`rows` of tiles split into one contiguous run per rank, as evenly as they divide"""
    base, extra = divmod(rows, world_size)
    out, at = [], 0
    for rank in range(world_size):
        take = base + (1 if rank < extra else 0)
        out.append((at, at + take))
        at += take
    return out


def assemble_in_bands(
    group, rows: int, columns: int, decode: Decode, blend: Blend
) -> Optional[torch.Tensor]:
    """Assemble a tile grid with each rank decoding and blending one band of rows, None if it can't

    Dealing tiles out divides the decoding and leaves the blending on every rank, which is a cost
    that does not shrink with the group however many ranks join it. Giving a rank a contiguous
    band of tile rows instead lets it blend its own band and send only that, so the blending
    divides too and what comes back is a disjoint piece of the image rather than every tile.

    The reason a band can be blended alone is a property of the two blends. `blend_v` writes a
    tile's *first* rows and `blend_h` its *first* columns, so neither ever writes the last rows of
    a tile - which is the only part of the row above that the row below reads. The bottom edge of
    a band is therefore already final before any blending starts, and one exchange of those edges,
    made while the tiles are still raw, is all a rank needs to blend its band exactly as a single
    rank walking the whole grid would have. No rank waits for another to finish blending.

    Where the tiles are shallower than twice the blend, that argument fails: the rows `blend_v`
    writes would reach into the rows the band below reads. None comes back and the caller falls
    back to a scheme that does not divide the blending.
    """
    world_size = dist.get_world_size(group)
    if world_size < 2:
        return None
    # Fewer rows than ranks and some rank would hold no band at all. Round-robin over the calls
    # still divides that decode; bands cannot.
    if rows < world_size:
        return None

    rank = dist.get_rank(group)
    start, stop = bands(rows, world_size)[rank]
    mine = decode([(i, j) for i in range(start, stop) for j in range(columns)])

    # Sent before anything is blended, both because the edge is raw at that point and because a
    # rank that had to wait for its neighbour's blending would serialise the very thing this is
    # dividing.
    sending = _bottom_edge(mine, stop - 1, columns, blend.deep_down, read=stop < rows)
    if sending is None:
        return None
    received = [torch.empty_like(sending) for _ in range(world_size)]
    dist.all_gather(received, sending, group=group)

    above = None
    if start > 0:
        widths = [mine[(start, j)].shape[ACROSS] for j in range(columns)]
        above = list(torch.split(received[rank - 1], widths, dim=ACROSS))
        # The edge arrives as the tiles decoded it, so the row above's own blending across its
        # columns is replayed here. It reads each left neighbour's *last* columns, which no blend
        # writes, so replaying it needs nothing further from anyone.
        for j in range(1, columns):
            above[j] = blend.across(above[j - 1], above[j], blend.deep_across)

    band = _blend_rows(range(start, stop), columns, mine, blend, above)
    return _share_bands(band, group, world_size)


def assemble_here(rows: int, columns: int, decode: Decode, blend: Blend) -> torch.Tensor:
    """Assemble the whole grid on this rank, which is what diffusers' own loop does"""
    mine = decode([(i, j) for i in range(rows) for j in range(columns)])
    return _blend_rows(range(rows), columns, mine, blend, None)


def _blend_rows(
    which: Sequence[int],
    columns: int,
    mine: Dict[Where, torch.Tensor],
    blend: Blend,
    above: Optional[List[torch.Tensor]],
) -> torch.Tensor:
    """Diffusers' assembly, over the rows given rather than all of them

    `above` is the row before the first, blended, or None where there is none. Both blends write
    into the tile they are handed, so each tile is blended against neighbours that were themselves
    already blended, and the scan order that produces is part of the result.
    """
    made = []
    for i in which:
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


def _bottom_edge(
    mine: Dict[Where, torch.Tensor], row: int, columns: int, deep: int, read: bool
) -> Optional[torch.Tensor]:
    """The last `deep` rows of a band's bottom row of tiles, as one tensor across the columns

    `read` is whether any band sits below this one. The last band's edge is still sent, because
    all_gather asks every rank for the same shape, but nothing is read out of it, so a bottom row
    the latent bounds clipped short is padded rather than refused.
    """
    tiles = [mine[(row, j)] for j in range(columns)]
    # A tile no deeper than two blends would have the rows blend_v writes reaching into the rows
    # the band below reads, and then the edge sent here would not be the edge that band needs.
    if read and any(tile.shape[DOWN] < 2 * deep for tile in tiles):
        return None
    edge = torch.cat([tile[..., -deep:, :] for tile in tiles], dim=ACROSS).clone()
    short = deep - edge.shape[DOWN]
    if short > 0:
        edge = torch.nn.functional.pad(edge, (0, 0, 0, short))
    return edge


def _share_bands(band: torch.Tensor, group, world_size: int) -> torch.Tensor:
    """Every rank's band, stacked back into the whole image on every rank"""
    # Bands differ in height wherever the rows do not divide by the ranks, so the shapes are
    # exchanged before the pixels, and the send is padded to the largest.
    manifest: List = [None] * world_size
    dist.all_gather_object(manifest, tuple(band.shape), group=group)
    width = max(math.prod(shape) for shape in manifest)

    sending = torch.empty(width, dtype=band.dtype, device=band.device)
    sending[: band.numel()] = band.reshape(-1)
    received = [torch.empty_like(sending) for _ in range(world_size)]
    dist.all_gather(received, sending, group=group)
    return torch.cat(
        [
            buffer[: math.prod(shape)].view(shape)
            for shape, buffer in zip(manifest, received)
        ],
        dim=DOWN,
    )


def _share(
    made: List[Optional[torch.Tensor]], group, world_size: int
) -> List[torch.Tensor]:
    """Fill in the calls this rank did not make from the ranks that did"""
    mine = [(n, tensor) for n, tensor in enumerate(made) if tensor is not None]

    # A rank cannot work out the shape of a call it did not make: tiles at the right and bottom
    # edges are clipped by the latent bounds, and a rank can hold none of them. One object
    # exchange settles that for the whole decode.
    manifest: List = [None] * world_size
    dist.all_gather_object(manifest, [(n, tuple(t.shape)) for n, t in mine], group=group)

    # Then one tensor exchange for the results themselves, flattened together and padded to the
    # largest share, since all_gather wants every rank sending the same count. Ranks differ by at
    # most one call, so the padding is at most one call's worth of the traffic.
    width = max(sum(math.prod(shape) for _, shape in entries) for entries in manifest)
    # Filled a call at a time rather than concatenated into the buffer, which would hold a second
    # copy of everything this rank decoded while the first was still alive. Left uninitialised
    # past what this rank sends: the manifest bounds what each share is read back out of, so the
    # padding is never looked at.
    sample = mine[0][1]
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
