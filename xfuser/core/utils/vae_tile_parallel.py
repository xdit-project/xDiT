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

import math
from typing import Callable, List, Optional, Sequence

import torch
import torch.distributed as dist

# Recorded on the VAE itself, because the decision is made when the decoder would otherwise be
# sharded and acted on later, when the tile window is settled and the decode is installed.
GROUP_ATTR = "_xfuser_tile_parallel_group"

Call = Callable[[], torch.Tensor]
Dispatch = Callable[[Sequence[Call]], List[torch.Tensor]]


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
    flat = torch.cat([tensor.reshape(-1) for _, tensor in mine])
    width = max(sum(math.prod(shape) for _, shape in entries) for entries in manifest)
    # Left uninitialised past what this rank is sending: the manifest bounds what each rank's
    # share is read back out of, so the padding is never looked at.
    sending = torch.empty(width, dtype=flat.dtype, device=flat.device)
    sending[: flat.numel()] = flat
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
