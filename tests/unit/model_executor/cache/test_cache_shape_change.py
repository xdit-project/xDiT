"""Step caches recompute when the token count changes without an explicit reset (#548)."""

from unittest.mock import patch

import pytest
import torch

from xfuser.model_executor.cache import utils


class _Norm(torch.nn.Module):
    def forward(self, hidden_states, emb):
        return hidden_states + emb[:, None, :], None, None, None, None


class _Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = _Norm()
        self.calls = 0

    def forward(self, hidden_states, encoder_hidden_states, *args, **kwargs):
        self.calls += 1
        return hidden_states * 1.5 + 1.0, encoder_hidden_states + 1.0


def _expected(hidden, encoder, num_blocks):
    for _ in range(num_blocks):
        hidden, encoder = hidden * 1.5 + 1.0, encoder + 1.0
    return hidden, encoder


@pytest.fixture
def cache_on_cpu():
    with (
        patch.object(utils, "get_device", return_value=torch.device("cpu")),
        patch.object(utils, "get_sequence_parallel_world_size", return_value=1),
    ):
        yield


def _make(cls, blocks):
    return cls(blocks, rel_l1_thresh=0.5, num_steps=4, name="default")


def _step(cached, seq_len):
    hidden = torch.ones(1, seq_len, 4)
    encoder = torch.ones(1, 3, 4)
    temb = torch.ones(1, 4)
    return hidden, encoder, cached(hidden, encoder, temb=temb)


@pytest.mark.parametrize("cls", [utils.TeaCachedTransformerBlocks, utils.FBCachedTransformerBlocks])
def test_new_token_count_recomputes_then_caches_again(cache_on_cpu, cls):
    blocks = [_Block(), _Block()]
    cached = _make(cls, blocks)

    # Run at 16 tokens until the cache is in use.
    for _ in range(3):
        _step(cached, 16)

    # Change to 9 tokens without a request-boundary reset: every block must
    # run rather than reuse residuals of the old shape.
    calls_before = blocks[1].calls
    hidden, encoder, (out_hidden, out_encoder) = _step(cached, 9)

    assert blocks[1].calls == calls_before + 1
    expected_hidden, expected_encoder = _expected(hidden, encoder, len(blocks))
    torch.testing.assert_close(out_hidden, expected_hidden)
    torch.testing.assert_close(out_encoder, expected_encoder)

    # Identical steps at the new size reach the cache again.
    calls_before = blocks[1].calls
    for _ in range(3):
        _, _, (out_hidden, _) = _step(cached, 9)
    assert blocks[1].calls < calls_before + 3
    torch.testing.assert_close(out_hidden, expected_hidden)
