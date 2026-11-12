"""What the Ulysses exchange (ulysses.py) needs from a transport.

A transport moves this rank's data into its peers' buffers, one-sided, and tells each peer when
it has: one transport per process group, created collectively. ulysses.py uses only what is
described here; transport.py implements it with copies queued on the SDMA engines from the host.

Copies and signals:
- A copy is (source offset, destination buffer, destination offset, bytes), the source offset
  relative to the tensor a call sends. A route is the copies of one send to every peer,
  registered once and sent per call, so the per-call host cost does not grow with the copies.
- A route may raise a signal on a channel behind its copies at every peer. A channel is an id
  in [0, channels); the exchange gives each of its waits one (for example, a head group's
  input). How a signal is stored is the transport's business.
- `Call.wait(channel)` makes a stream wait until every peer's latest signal on the channel has
  landed. This rank's own data is copied on the stream, so it is not waited for.

Ordering contract: a channel's signal lands at a peer only after every copy this rank sent to
that peer since the channel's previous signal, including routes sent with `signal=None` earlier
in the same call. A head group's tensors can therefore go in several sends, with the signal on
the last.

A route's copies read their source once the sending stream reaches the send; the caller keeps
the source alive until the peers have it (the exchange knows when: its protocol says when a
peer has received this rank's data). The transport caches routes by key and rebuilds one when
a buffer it names was re-created, so callers never see addresses.
"""

from typing import Callable, Hashable, List, Optional, Protocol, Sequence, Tuple

import torch

# One copy of a route: (source offset, destination buffer, destination offset, bytes).
Copy = Tuple[int, "SymmetricBuffer", int, int]


class SymmetricBuffer(Protocol):
    """Device memory of the same size on every rank of the group, written by peers."""

    nbytes: int
    # This rank's buffer as a uint8 tensor.
    local: torch.Tensor


class Route(Protocol):
    """A registered send: per peer, copies and an optional signal (`PeerTransport.route`)."""


class Call(Protocol):
    """One Ulysses call: its sends and waits, from `PeerTransport.begin` to `end`."""

    def send(self, route: Route, src: torch.Tensor, stream: torch.cuda.Stream) -> None:
        """Start `route`'s copies from `src` once `stream` reaches this point."""

    def wait(self, channel: int, stream: torch.cuda.Stream) -> None:
        """Make `stream` wait until every peer's latest signal on `channel` has landed."""

    def end(self) -> None:
        """Close the call; its sends and waits are all issued."""


class PeerTransport(Protocol):
    """One process group's transport: peers in send order, buffers, routes and calls."""

    rank: int
    world: int
    # The other ranks of the group in the order routes list them.
    peers: List[int]
    # Channel ids are [0, channels).
    channels: int
    # True: copies are queued from the host, so a CUDA graph capture would not replay them;
    # False: a stream-ordered kernel issues them, and they can be captured.
    host_issued: bool

    def buffer(self, name: str, nbytes: int) -> SymmetricBuffer:
        """The buffer `name`, at least `nbytes`; collective when it must be (re)created."""

    def route(self, key: Hashable, build: Callable[[], Sequence[Sequence[Copy]]], signal: Optional[int]) -> Route:
        """The route `key`: `build()` gives each peer's copies in `peers` order, called only when
        the route is not cached or names a re-created buffer. With `signal`, a channel id, the
        route raises this rank's signal on that channel at each peer behind its copies."""

    def begin(self) -> Call:
        """Open a call; the transport takes whatever per-call resources its sends and waits need."""
