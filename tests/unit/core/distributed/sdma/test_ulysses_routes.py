"""The routes and waits a Ulysses exchange asks of its transport, on a recording fake."""

from types import SimpleNamespace

import pytest
import torch

from xfuser.core.distributed.sdma import ulysses


class _Buffer:
    def __init__(self, name, nbytes):
        self.name, self.nbytes = name, nbytes
        self.local = torch.zeros(nbytes, dtype=torch.uint8)


class _Call:
    def __init__(self, log):
        self.log = log

    def send(self, route, src, stream):
        self.log.append(("send", route))

    def wait(self, channel, stream):
        self.log.append(("wait", channel))

    def end(self):
        self.log.append(("end",))


class _Transport:
    """Two ranks; records every send as (copies with buffer names, signal) and every wait."""

    channels = 64
    host_issued = True

    def __init__(self, rank):
        self.rank, self.world, self.peers = rank, 2, [1 - rank]
        self.buffers = {}
        self.log = []

    def buffer(self, name, nbytes):
        if name not in self.buffers or self.buffers[name].nbytes < nbytes:
            self.buffers[name] = _Buffer(name, nbytes)
        return self.buffers[name]

    def route(self, key, build, signal):
        return [[(src, dst.name, offset, n) for src, dst, offset, n in peer] for peer in build()], signal

    def begin(self):
        return _Call(self.log)


@pytest.fixture
def fake(monkeypatch):
    def install(rank):
        transport = _Transport(rank)
        group = object()
        monkeypatch.setattr(
            ulysses,
            "dist",
            SimpleNamespace(get_world_size=lambda g: 2, get_process_group_ranks=lambda g: [0, 1]),
        )
        monkeypatch.setattr(ulysses, "transport", SimpleNamespace(group_for=lambda g: transport, PeerGroup=_Transport))
        monkeypatch.setattr(torch.cuda, "current_stream", lambda *args, **kwargs: None)
        monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
        monkeypatch.setattr(ulysses, "_states", {})
        return transport, group

    return install


def _run(group, exchange, sends):
    ulysses.begin(group, exchange)
    for gi, tensors in sends:
        elems = sum(exchange.piece_elems(gi, t) for t in tensors or range(exchange.tensors)) * exchange.world
        ulysses.send(group, gi, torch.zeros(elems, dtype=exchange.dtype), tensors)
    ulysses.finish(group, lambda q, *rest: q)


def test_joint_text_split_send(fake):
    # Rank 1 of 2, bf16, head dim 2. Q: 3 rows, 2 heads; K/V: 2 rows, 1 head, behind one joint
    # row in front of their inbox. One head group, sent as Q/K and then V.
    transport, group = fake(1)
    exchange = ulysses.Exchange.make(2, 1, 2, torch.bfloat16, [3, 2, 2], [2, 1, 1], [(0, 0), (1, 0), (1, 0)], 1)
    _run(group, exchange, [(0, (0, 1)), (0, (2,))])
    # Inbox: Q 6 rows x 2 heads x 2 at 0, K 5 x 1 x 2 at element 24, V at element 34 (bytes x2).
    # Rank 1's rows sit after rank 0's: Q row 3, K/V row 1 + 2 (behind the joint row).
    assert transport.log == [
        ("send", ([[(0, "ulysses_inbox", 24, 24), (24, "ulysses_inbox", 60, 8)]], None)),
        ("send", ([[(0, "ulysses_inbox", 80, 8)]], 0)),
        ("wait", 0),
        # Rank 0's 3 output rows, 2 heads, go to the second block of its out inbox.
        ("send", ([[(0, "ulysses_outbox", 24, 24)]], ulysses.MAX_HEAD_GROUPS)),
        ("wait", ulysses.MAX_HEAD_GROUPS),
        ("end",),
    ]


def test_grouped_query_head_groups(fake):
    # Rank 0 of 2, bf16, head dim 2, 2 rows each. Q: 4 heads, K/V: 2 heads, in 2 head groups of
    # 2 Q heads and 1 K/V head.
    transport, group = fake(0)
    exchange = ulysses.Exchange.make(2, 1, 2, torch.bfloat16, [2, 2, 2], [4, 2, 2], [(0, 0)] * 3, 2)
    _run(group, exchange, [(0, None), (1, None)])
    sends = [entry[1] for entry in transport.log if entry[0] == "send"]
    # Group 1: peer 1's Q, K, V pieces (8, 4, 4 elements) follow peer 0's 16 in the packed
    # tensor; in the inbox, group 1's Q, K, V start at elements 32, 48, 56 and rank 0's rows
    # come first.
    assert sends[1] == (
        [[(32, "ulysses_inbox", 64, 16), (48, "ulysses_inbox", 96, 8), (56, "ulysses_inbox", 112, 8)]],
        1,
    )
    # Group 1's output: peer 1's 2 rows after rank 0's 2, into group 1's out inbox (element 16).
    assert sends[3] == ([[(16, "ulysses_outbox", 32, 16)]], ulysses.MAX_HEAD_GROUPS + 1)
    waits = [entry[1] for entry in transport.log if entry[0] == "wait"]
    assert waits == [0, 1, ulysses.MAX_HEAD_GROUPS, ulysses.MAX_HEAD_GROUPS + 1]


def test_a_send_out_of_order_leaves_the_call_open(fake):
    transport, group = fake(0)
    exchange = ulysses.Exchange.make(2, 1, 2, torch.bfloat16, [2, 2, 2], [2, 2, 2], [(0, 0)] * 3, 1)
    ulysses.begin(group, exchange)
    with pytest.raises(ValueError):
        ulysses.send(group, 0, torch.zeros(16, dtype=torch.bfloat16), (1,))
    assert transport.log == []
    ulysses.send(group, 0, torch.zeros(48, dtype=torch.bfloat16))
    ulysses.finish(group, lambda q, *rest: q)
    assert transport.log[-1] == ("end",)
