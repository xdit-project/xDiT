"""Ulysses attention with its all-to-alls on the SDMA engines, pipelined over head groups.

Ulysses sends each rank's Q/K/V for head block p to rank p, which then attends over
the whole sequence for its heads and sends each rank's rows back. This does both
exchanges with the SDMA transport (transport.py) and splits each rank's heads into
head groups: group g's attention starts once group g's tensors have landed, while
later groups are still moving, and its output is pushed back as soon as it is done,
while later groups attend.

What one call exchanges is an `Exchange`: tensors with their own row counts and head
counts (Q first). That covers joint-text attention (Q carries the joint rows), K/V
with fewer heads (grouped-query attention), and extra per-head tensors that ride
with Q. Head groups split the heads in units that keep every tensor's heads whole.

A call is `begin`, then `send` once per head group with that group's packed tensor
(`pack_group`), then `finish`, which attends group by group and returns the output.
The steps are separate so the packing, or even the projection that produces a group,
can run in between (and be fused there by torch.compile).

Layouts, for P ranks, batch b, head dim d; tensor t has s_t rows on this rank and
h_tg heads in group g:

- packed group g (this rank's): [P][t][b][s_t][h_tg][d], so the rows for each peer
  are one contiguous run.
- inbox (shared), group g, tensor t: [b][R_t][h_tg][d] with R_t = P * s_t plus rows
  reserved for this rank's own joint K/V. Rank r's block lands at its row offset, so
  each tensor is bshd over the whole sequence, which is the layout attention kernels
  read: nothing is unpacked before attention.
- attention output of group g: bshd [b][P * s_q][h_qg][d]: the rows rank p gets back
  are one contiguous [s_q, h_qg, d] block per batch item.
- out inbox (shared): [g][P][b][s_q][h_qg][d], unpacked with one copy per group into
  the [b, s_q, H, d] output, which the output projection reads as [b, s_q, H * d].

Ordering needs no barrier after the first call: each rank pushes a call's data only
once its stream has received every peer's output of its previous call, and a peer
sends that output only after its attention, which reads its inbox, is done, and
after this rank's earlier pushes into it have landed. So the inboxes and the flags
(reset by the waits) are free whenever they are written. That holds as long as
every Ulysses attention of the group goes through this module. For the same reason a
packed tensor is free to reuse once `finish` has returned (the call holds it until then:
under torch.compile its producer's buffer may be freed as soon as `send` returns), and an
attention output is held until the next call.
"""

import functools
import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.distributed as dist

from xfuser.core.distributed.sdma import transport

# What `attend` gets: one head group's tensors, each a [b, h, rows, d] view of a bshd
# tensor (Q, K, V, then the extras); it returns the group's attention output as
# [b, h_q, rows_q, d].
Attend = Callable[..., torch.Tensor]

# Head groups at most: the flag buffer is sized for this many, so changing the group count
# between calls needs no new buffer, and each group has a trigger for its input and output.
MAX_HEAD_GROUPS = 16


def split_heads(heads: int, groups: int) -> List[int]:
    """Sizes of `groups` head groups covering `heads` heads, as even as possible, larger first."""
    groups = max(1, min(groups, heads, MAX_HEAD_GROUPS))
    return [heads // groups + (1 if i < heads % groups else 0) for i in range(groups)]


def head_split(heads: Sequence[int], groups: int) -> Tuple[Tuple[int, ...], ...]:
    """Per head group, each tensor's heads per rank in it: the groups split the heads in units
    that keep every tensor whole (one unit is one K/V head and its query heads)."""
    units = math.gcd(*heads)
    return tuple(tuple(u * h // units for h in heads) for u in split_heads(units, groups))


def encode(world, batch, head_dim, rows, heads, reserved, head_groups) -> List[int]:
    """An exchange as integers, for a custom op's arguments (its dtype travels with a tensor)."""
    out = [world, batch, head_dim, head_groups, len(rows)]
    for r, h, (before, after) in zip(rows, heads, reserved):
        out += [r, h, before, after]
    return out


def _prefix(sizes):
    out = [0]
    for size in sizes:
        out.append(out[-1] + size)
    return out


@dataclass(frozen=True)
class Exchange:
    """The tensors one Ulysses call exchanges, and how they are split into head groups.

    `rows[t]` is tensor t's rows on this rank and `heads[t]` its heads per rank (Q is
    tensor 0). `reserved[t]` is (rows before, rows after) kept in tensor t's inbox for
    this rank's own rows, such as the joint-text K/V that every rank holds.
    """

    world: int
    batch: int
    head_dim: int
    dtype: torch.dtype
    rows: Tuple[int, ...]
    heads: Tuple[int, ...]
    reserved: Tuple[Tuple[int, int], ...]
    head_groups: int
    # Derived in __post_init__.
    sizes: Tuple[Tuple[int, ...], ...] = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        object.__setattr__(self, "sizes", head_split(self.heads, self.head_groups))

    @staticmethod
    def make(world, batch, head_dim, dtype, rows, heads, reserved, head_groups) -> "Exchange":
        return _exchange(world, batch, head_dim, dtype, tuple(rows), tuple(heads), tuple(reserved), head_groups)

    def encode(self) -> List[int]:
        return encode(self.world, self.batch, self.head_dim, self.rows, self.heads, self.reserved, self.head_groups)

    @staticmethod
    def decode(spec: Sequence[int], dtype: torch.dtype) -> "Exchange":
        world, batch, head_dim, head_groups, n = spec[:5]
        per = [spec[5 + 4 * t : 9 + 4 * t] for t in range(n)]
        return Exchange.make(
            world,
            batch,
            head_dim,
            dtype,
            [p[0] for p in per],
            [p[1] for p in per],
            [(p[2], p[3]) for p in per],
            head_groups,
        )

    @property
    def groups(self) -> int:
        return len(self.sizes)

    @property
    def tensors(self) -> int:
        return len(self.rows)

    def head_offset(self, g: int, t: int) -> int:
        return sum(self.sizes[i][t] for i in range(g))

    def inbox_rows(self, t: int) -> int:
        return self.world * self.rows[t] + sum(self.reserved[t])

    def piece_elems(self, g: int, t: int) -> int:
        """Elements of tensor t's [b, s_t, h_tg, d] piece for one peer in group g."""
        return self.batch * self.rows[t] * self.sizes[g][t] * self.head_dim

    def peer_elems(self, g: int) -> int:
        return sum(self.piece_elems(g, t) for t in range(self.tensors))

    def inbox_elems(self, g: int, t: int) -> int:
        return self.batch * self.inbox_rows(t) * self.sizes[g][t] * self.head_dim

    @functools.cached_property
    def inbox_offsets(self) -> List[List[int]]:
        """Element offset of (group, tensor) in the inbox."""
        flat = _prefix([self.inbox_elems(g, t) for g in range(self.groups) for t in range(self.tensors)])
        return [[flat[g * self.tensors + t] for t in range(self.tensors)] for g in range(self.groups)]

    def out_block_elems(self, g: int) -> int:
        """One batch item's [s_q, h_qg, d] block of group g's output."""
        return self.rows[0] * self.sizes[g][0] * self.head_dim

    @functools.cached_property
    def out_offsets(self) -> List[int]:
        return _prefix([self.world * self.batch * self.out_block_elems(g) for g in range(self.groups)])


_exchanges: Dict[tuple, Exchange] = {}


def _exchange(*key) -> Exchange:
    if key not in _exchanges:
        _exchanges[key] = Exchange(*key)
    return _exchanges[key]


def pieces_from_heads(x: torch.Tensor, world: int, offset: int, count: int) -> torch.Tensor:
    """Heads p * H/P + offset .. + count of `x` ([b, H, s, d], any strides) for every rank p, as a
    [P, b, s, count, d] view: one group's piece of one tensor, in `pack_group`'s order."""
    b, heads, s, d = x.shape
    return x.view(b, world, heads // world, s, d)[:, :, offset : offset + count].permute(1, 0, 3, 2, 4)


def pieces_from_grouped(x: torch.Tensor, world: int) -> torch.Tensor:
    """A group's tensor computed on its own as [b, s, P * h, d] (heads ordered by rank) as the
    [P, b, s, h, d] view `pack_group` takes."""
    b, s, heads, d = x.shape
    return x.view(b, s, world, heads // world, d).permute(2, 0, 1, 3, 4)


def pack_group(pieces: Sequence[torch.Tensor]) -> torch.Tensor:
    """A head group's packed tensor: per peer, every tensor's [b, s_t, h_tg, d] piece in turn.

    `pieces[t]` is [P, b, s_t, h_tg, d] with any strides. Traced by torch.compile, this is
    one `cat` that Inductor writes straight from the kernels producing the pieces; eager,
    one copy per tensor into the result.
    """
    P = pieces[0].shape[0]
    if torch.compiler.is_compiling():
        return torch.cat([p.reshape(P, -1) for p in pieces], dim=1).reshape(-1)
    sizes = [p[0].numel() for p in pieces]
    out = torch.empty(P * sum(sizes), dtype=pieces[0].dtype, device=pieces[0].device)
    rows = out.view(P, -1)
    start = 0
    for piece, n in zip(pieces, sizes):
        rows[:, start : start + n].view(piece.shape).copy_(piece)
        start += n
    return out


class _Call:
    """One open call on a Ulysses group, from `begin` to `finish`."""

    def __init__(self, exchange, slot, call, rccl):
        self.exchange, self.slot, self.call, self.rccl = exchange, slot, call, rccl
        # The packed tensors sent, held until `finish` returns: the engines read them until every
        # peer has the data, which this rank knows only once the peers' outputs are back.
        self.packed: List[torch.Tensor] = []

    @property
    def sent(self) -> int:
        return len(self.packed)


class _State:
    """Per Ulysses group: plans and the attention outputs held for one call."""

    def __init__(self):
        self.open: Optional[_Call] = None
        self.held: List[torch.Tensor] = []
        # (exchange, direction, group) -> (buffer generations, transport.Plan)
        self.plans: Dict[tuple, tuple] = {}


# Keyed by the group's ranks, as transport.group_for shares its buffers. A call that runs over
# RCCL never creates the PeerGroup: under CUDA graph trees its first allocations would land in
# the graph pool and outlive the graph.
_states: Dict[tuple, _State] = {}
_state_keys: Dict[dist.ProcessGroup, tuple] = {}
_side_streams: Dict[torch.device, List[torch.cuda.Stream]] = {}


def _state(process_group: dist.ProcessGroup) -> _State:
    key = _state_keys.get(process_group)
    if key is None:
        key = _state_keys[process_group] = tuple(dist.get_process_group_ranks(process_group))
    state = _states.get(key)
    if state is None:
        state = _states[key] = _State()
    return state


def _streams(device: torch.device, count: int) -> List[torch.cuda.Stream]:
    """`count` side streams on `device`, the first at high priority."""
    streams = _side_streams.setdefault(device, [])
    while len(streams) < count:
        streams.append(torch.cuda.Stream(device=device, priority=-1 if not streams else 0))
    return streams


def _buffers(g: transport.PeerGroup, ex: Exchange):
    elt = torch.empty((), dtype=ex.dtype).element_size()
    inbox = g.buffer(
        "ulysses_inbox", ex.inbox_offsets[-1][-1] * elt + ex.inbox_elems(ex.groups - 1, ex.tensors - 1) * elt
    )
    outbox = g.buffer("ulysses_outbox", ex.out_offsets[-1] * elt)
    # Words [g * P, (g + 1) * P): group g's input flags; from MAX_HEAD_GROUPS * P on, its output
    # flags; word p is rank p's.
    flags = g.buffer("ulysses_flags", 2 * MAX_HEAD_GROUPS * g.world * 4, memory="uncached")
    return inbox, outbox, flags, elt


def _in_flag(ex, g, p, world):
    return (g * world + p) * 4


def _out_flag(ex, g, p, world):
    return ((MAX_HEAD_GROUPS + g) * world + p) * 4


def _plan(state, g, ex, direction, gi, inbox, outbox, flags, elt) -> transport.Plan:
    """The plan for group gi's input ("in") or output ("out") push, made for the current buffers."""
    key = (ex, direction, gi)
    generations = (inbox.generation, outbox.generation, flags.generation)
    cached = state.plans.get(key)
    if cached is not None and cached[0] == generations:
        return cached[1]
    if cached is not None:
        cached[1].free()
    P, r, b, d = g.world, g.rank, ex.batch, ex.head_dim
    copies, flag_words = [], []
    for i, p in enumerate(g.peers):
        mine = []
        if direction == "in":
            offsets = ex.inbox_offsets[gi]
            start = 0
            for t in range(ex.tensors):
                h, s, R = ex.sizes[gi][t], ex.rows[t], ex.inbox_rows(t)
                block = s * h * d
                before = ex.reserved[t][0]
                for bi in range(b):
                    src = (p * ex.peer_elems(gi) + start + bi * block) * elt
                    dst = inbox.peer_bases[i] + (offsets[t] + (bi * R + before + r * s) * h * d) * elt
                    mine.append((src, dst, block * elt))
                start += ex.piece_elems(gi, t)
            flag_words.append(flags.peer_bases[i] + _in_flag(ex, gi, r, P))
        else:
            h, s = ex.sizes[gi][0], ex.rows[0]
            block = ex.out_block_elems(gi)
            for bi in range(b):
                src = ((bi * P * s + p * s) * h * d) * elt
                dst = outbox.peer_bases[i] + (ex.out_offsets[gi] + (r * b + bi) * block) * elt
                mine.append((src, dst, block * elt))
            flag_words.append(flags.peer_bases[i] + _out_flag(ex, gi, r, P))
        copies.append(mine)
    plan = transport.Plan(g, copies, flag_words)
    state.plans[key] = (generations, plan)
    return plan


def begin(process_group: dist.ProcessGroup, exchange: Exchange, rccl: bool = False) -> None:
    """Open a call on `process_group`. With `rccl`, or while a CUDA graph is being captured (host-
    issued copies would not be replayed), the call exchanges the same packed tensors with RCCL."""
    state = _state(process_group)
    if state.open is not None:
        raise RuntimeError("an SDMA Ulysses call is already open on this group")
    world = dist.get_world_size(process_group)
    if exchange.world != world:
        raise ValueError(f"exchange for {exchange.world} ranks on a group of {world}")
    rccl = rccl or torch.cuda.is_current_stream_capturing()
    if rccl:
        state.open = _Call(exchange, None, None, True)
        return
    g = transport.group_for(process_group)
    ex, P, r = exchange, g.world, g.rank
    inbox, outbox, flags, elt = _buffers(g, ex)
    n_data = sum(
        _plan(state, g, ex, direction, gi, inbox, outbox, flags, elt).copies
        for direction in ("in", "out")
        for gi in range(ex.groups)
    )
    # Exactly the signals the call uses: the next use of the slot waits for each to complete.
    slot, call = transport.start(n_data, 2 * ex.groups * (P - 1))
    words = flags.local[: 2 * MAX_HEAD_GROUPS * P * 4].view(torch.int32).view(2, MAX_HEAD_GROUPS, P)
    words[:, : ex.groups, r].fill_(1)
    state.open = _Call(ex, slot, call, False)


def send(process_group: dist.ProcessGroup, gi: int, packed: torch.Tensor) -> None:
    """Push head group `gi`'s packed tensor (`pack_group`) once the stream has produced it."""
    state = _state(process_group)
    c = state.open
    if c is None or gi != c.sent:
        raise RuntimeError(f"SDMA Ulysses send of group {gi} out of order")
    c.packed.append(packed)
    if c.rccl:
        return
    g = transport.group_for(process_group)
    ex, P, r, b, d = c.exchange, g.world, g.rank, c.exchange.batch, c.exchange.head_dim
    inbox, outbox, flags, elt = _buffers(g, ex)
    n_peers = P - 1
    first = sum(_plan(state, g, ex, "in", i, inbox, outbox, flags, elt).copies for i in range(gi))
    # Queued before the trigger is released, so the host issues them while the stream finishes.
    _plan(state, g, ex, "in", gi, inbox, outbox, flags, elt).issue(c.slot, gi, first, gi * n_peers, packed.data_ptr())
    transport.release(c.slot, gi, torch.cuda.current_stream())
    # This rank's own blocks, on the stream.
    mine = packed.view(P, -1)[r]
    offsets = ex.inbox_offsets[gi]
    start = 0
    for t in range(ex.tensors):
        h, s, R, before = ex.sizes[gi][t], ex.rows[t], ex.inbox_rows(t), ex.reserved[t][0]
        n = ex.piece_elems(gi, t)
        _inbox(inbox, offsets[t], b, R, h, d, ex.dtype)[:, before + r * s : before + (r + 1) * s].copy_(
            mine[start : start + n].view(b, s, h, d)
        )
        start += n


def _inbox(buf, offset, b, rows, h, d, dtype):
    elt = torch.empty((), dtype=dtype).element_size()
    return buf.local[offset * elt : (offset + b * rows * h * d) * elt].view(dtype).view(b, rows, h, d)


def finish(
    process_group: dist.ProcessGroup,
    attend: Attend,
    local: Optional[Sequence[Optional[torch.Tensor]]] = None,
    concurrent: bool = False,
) -> torch.Tensor:
    """Attend group by group and exchange the output back; returns this rank's rows for every head
    as a contiguous [b, s_q, H, d].

    `local[t]`, if given, is this rank's own
    rows for tensor t's reserved inbox rows, [b, heads[t], before + after, d], heads split by
    group like the exchanged ones. With `concurrent`, each head group attends on a stream of its
    own (the first at high priority), so a group whose data has landed fills the GPU while the
    one before it drains: fewer heads per kernel leave more of its last wave idle.
    """
    state = _state(process_group)
    c = state.open
    if c is None or c.sent != c.exchange.groups:
        raise RuntimeError("SDMA Ulysses finish before every group was sent")
    state.open = None
    if c.rccl:
        return _finish_rccl(process_group, c.exchange, c.packed, attend, local)
    g = transport.group_for(process_group)
    ex, P, r, b, d, G = c.exchange, g.world, g.rank, c.exchange.batch, c.exchange.head_dim, c.exchange.groups
    inbox, outbox, flags, elt = _buffers(g, ex)
    stream = torch.cuda.current_stream()
    n_peers = P - 1
    in_copies = sum(_plan(state, g, ex, "in", i, inbox, outbox, flags, elt).copies for i in range(G))
    out_first = [
        in_copies + sum(_plan(state, g, ex, "out", i, inbox, outbox, flags, elt).copies for i in range(gi))
        for gi in range(G)
    ]
    if concurrent:
        ready = torch.cuda.Event()
        ready.record(stream)
        side = _streams(stream.device, G)
    held = []
    for gi in range(G):
        group_stream = stream
        if concurrent:
            group_stream = side[gi]
            group_stream.wait_event(ready)
        with torch.cuda.stream(group_stream):
            transport.wait(group_stream, flags.ptr + _in_flag(ex, gi, 0, P), P, gi, c.call)
            offsets = ex.inbox_offsets[gi]
            views = []
            for t in range(ex.tensors):
                h = ex.sizes[gi][t]
                x = _inbox(inbox, offsets[t], b, ex.inbox_rows(t), h, d, ex.dtype)
                before, after = ex.reserved[t]
                if local is not None and local[t] is not None and before + after:
                    own = local[t][:, ex.head_offset(gi, t) : ex.head_offset(gi, t) + h].transpose(1, 2)
                    if before:
                        x[:, :before].copy_(own[:, :before])
                    if after:
                        x[:, ex.inbox_rows(t) - after :].copy_(own[:, before:])
                views.append(x.transpose(1, 2))
            out = attend(*views).transpose(1, 2)
            if out.dtype != ex.dtype:
                out = out.to(ex.dtype)
            if not out.is_contiguous():
                out = out.contiguous()
            held.append(out)
            transport.release(c.slot, MAX_HEAD_GROUPS + gi, group_stream)
            _plan(state, g, ex, "out", gi, inbox, outbox, flags, elt).issue(
                c.slot, MAX_HEAD_GROUPS + gi, out_first[gi], (G + gi) * n_peers, out.data_ptr()
            )
            h, s = ex.sizes[gi][0], ex.rows[0]
            _outbox(outbox, ex, gi, elt)[r].copy_(out.view(b, P, s, h, d)[:, r])
    if concurrent:
        for gi in range(G):
            stream.wait_stream(side[gi])
    # The previous call's outputs have been read by every peer's engines: each peer sent this
    # call's data only after receiving them, and the waits above are behind that data.
    state.held = held

    s = ex.rows[0]
    result = torch.empty(b, s, P, ex.heads[0], d, dtype=ex.dtype, device=stream.device)
    for gi in range(G):
        transport.wait(stream, flags.ptr + _out_flag(ex, gi, 0, P), P, G + gi, c.call)
        off, h = ex.head_offset(gi, 0), ex.sizes[gi][0]
        result[:, :, :, off : off + h].copy_(_outbox(outbox, ex, gi, elt).permute(1, 2, 0, 3, 4))
    return result.view(b, s, P * ex.heads[0], d)


def _outbox(outbox, ex, gi, elt):
    """Group gi's out inbox, [P][b][s_q][h_qg][d]."""
    P, b, s, h, d = ex.world, ex.batch, ex.rows[0], ex.sizes[gi][0], ex.head_dim
    start = ex.out_offsets[gi] * elt
    return outbox.local[start : start + P * b * s * h * d * elt].view(ex.dtype).view(P, b, s, h, d)


def _finish_rccl(process_group, ex, packed, attend, local):
    """`finish` for a call that exchanges its packed tensors with RCCL all-to-alls."""
    P, b, d = ex.world, ex.batch, ex.head_dim
    s = ex.rows[0]
    result = torch.empty(b, s, P, ex.heads[0], d, dtype=ex.dtype, device=packed[0].device)
    for gi in range(ex.groups):
        got = torch.empty_like(packed[gi])
        dist.all_to_all_single(got, packed[gi], group=process_group)
        rows = got.view(P, -1)
        views, start = [], 0
        for t in range(ex.tensors):
            h, st = ex.sizes[gi][t], ex.rows[t]
            n = ex.piece_elems(gi, t)
            x = rows[:, start : start + n].view(P, b, st, h, d).permute(1, 0, 2, 3, 4).reshape(b, P * st, h, d)
            before, after = ex.reserved[t]
            if local is not None and local[t] is not None and before + after:
                own = local[t][:, ex.head_offset(gi, t) : ex.head_offset(gi, t) + h].transpose(1, 2)
                x = torch.cat([own[:, :before], x, own[:, before:]], dim=1)
            views.append(x.transpose(1, 2))
            start += n
        out = attend(*views).transpose(1, 2).to(ex.dtype)
        h = ex.sizes[gi][0]
        back = out.reshape(b, P, s, h, d).permute(1, 0, 2, 3, 4).contiguous()
        returned = torch.empty_like(back)
        dist.all_to_all_single(returned, back, group=process_group)
        off = ex.head_offset(gi, 0)
        result[:, :, :, off : off + h].copy_(returned.permute(1, 2, 0, 3, 4))
    return result.view(b, s, P * ex.heads[0], d)


def attention(
    tensors: Sequence[torch.Tensor],
    attend: Attend,
    process_group: dist.ProcessGroup,
    head_groups: int = 2,
    concurrent: bool = False,
    reserved: Optional[Sequence[Tuple[int, int]]] = None,
    local: Optional[Sequence[Optional[torch.Tensor]]] = None,
    rccl: bool = False,
) -> torch.Tensor:
    """Ulysses attention over `process_group`: begin, pack and send every group, finish.

    `tensors` are this rank's [b, H_t, s_t, d] (any strides; Q first), all with one batch, head
    dim and dtype; every H_t must divide by the group size. Returns [b, s_q, H_q, d].
    """
    world = dist.get_world_size(process_group)
    q = tensors[0]
    ex = Exchange.make(
        world,
        q.shape[0],
        q.shape[3],
        q.dtype,
        [x.shape[2] for x in tensors],
        [x.shape[1] // world for x in tensors],
        reserved or [(0, 0)] * len(tensors),
        head_groups,
    )
    begin(process_group, ex, rccl)
    for gi in range(ex.groups):
        pieces = [pieces_from_heads(x, world, ex.head_offset(gi, t), ex.sizes[gi][t]) for t, x in enumerate(tensors)]
        send(process_group, gi, pack_group(pieces))
    return finish(process_group, attend, local, concurrent)
