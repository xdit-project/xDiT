"""Each sequence-parallel rank projects a share of the blocks' modulation, and the ranks gather the row.

The gathered row must hold what every block's own linear computes, for any rank count,
including counts that leave uneven or empty shares.
"""

import copy
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from xfuser.model_executor.layers import sharded_modulation
from xfuser.model_executor.layers.sharded_modulation import ShardedModulation, SlicedModulation

_WIDTHS = (6, 6, 3, 3, 3)


def _linears():
    torch.manual_seed(0)
    return [torch.nn.Linear(4, width) for width in _WIDTHS]


def _shard(monkeypatch, linears, rank, size):
    """Shard `linears` as rank `rank` of `size`, with an all-gather that returns the rank's own share."""
    monkeypatch.setattr(sharded_modulation, "get_sequence_parallel_rank", lambda: rank)
    monkeypatch.setattr(sharded_modulation, "get_sequence_parallel_world_size", lambda: size)
    monkeypatch.setattr(
        sharded_modulation, "get_sp_group", lambda: SimpleNamespace(all_gather=lambda local, dim: local)
    )
    return ShardedModulation(linears)


@pytest.mark.parametrize("size", [1, 2, 3, 8])
def test_the_ranks_shares_gather_to_every_linears_projection(monkeypatch, size):
    stock = _linears()
    embedding = torch.randn(2, 4, generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        expected = [linear(embedding) for linear in stock]
    ranks = [copy.deepcopy(stock) for _ in range(size)]
    shards = [_shard(monkeypatch, linears, rank, size) for rank, linears in enumerate(ranks)]

    # Equal shares, as all-gather needs; with 8 ranks the last one holds only padding.
    assert len({shard.weight.shape for shard in shards}) == 1
    with torch.no_grad():
        gathered = torch.cat([F.linear(embedding, shard.weight, shard.bias) for shard in shards], dim=-1)
    group = SimpleNamespace(all_gather=lambda local, dim: gathered)
    monkeypatch.setattr(sharded_modulation, "get_sp_group", lambda: group)

    for shard, linears in zip(shards, ranks):
        with torch.no_grad():
            shard(embedding)
        for linear, output in zip(linears, expected):
            assert torch.allclose(linear(embedding), output, atol=1e-6)


def test_the_linears_return_their_slice_and_keep_their_weights(monkeypatch):
    linears = _linears()
    before = [copy.deepcopy(linear.state_dict()) for linear in linears]
    embedding = torch.randn(1, 4, generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        expected = [linear(embedding) for linear in linears]

    shard = _shard(monkeypatch, linears, 0, 1)
    with torch.no_grad():
        shard(embedding)

    assert all(type(linear) is SlicedModulation for linear in linears)
    for linear, state, output in zip(linears, before, expected):
        assert linear.state_dict().keys() == state.keys()
        assert all(torch.equal(linear.state_dict()[key], state[key]) for key in state)
        # The input is ignored: the row was projected from the embedding the shard was given.
        assert torch.allclose(linear(torch.zeros(1, 4)), output, atol=1e-6)


def test_a_linear_handed_the_leading_samples_returns_their_rows(monkeypatch):
    linears = _linears()
    embedding = torch.randn(4, 4, generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        expected = [linear(embedding[:2]) for linear in linears]

    shard = _shard(monkeypatch, linears, 0, 1)
    with torch.no_grad():
        shard(embedding)

    for linear, output in zip(linears, expected):
        assert torch.allclose(linear(embedding[:2]), output, atol=1e-6)


def test_the_share_is_a_buffer_outside_the_state_dict(monkeypatch):
    shard = _shard(monkeypatch, _linears(), 1, 2)

    assert shard.weight.shape == (11, 4) and shard.bias.shape == (11,)
    assert dict(shard.named_buffers()).keys() == {"weight", "bias"}
    assert not shard.state_dict()
