"""Step caches restart with every request and follow its step count."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from diffusers import FlowMatchEulerDiscreteScheduler

from xfuser.model_executor.cache import utils
from xfuser.model_executor.cache.adapters import apply_cache


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


class _Transformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer_blocks = torch.nn.ModuleList([_Block(), _Block()])
        self.single_transformer_blocks = torch.nn.ModuleList()


class _SingleBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, hidden_states, encoder_hidden_states=None, *args, **kwargs):
        self.calls += 1
        if encoder_hidden_states is None:
            # diffusers < 0.35 passes the concatenated streams.
            return hidden_states + 1.0
        return encoder_hidden_states + 1.0, hidden_states + 1.0


@pytest.fixture
def cache_on_cpu():
    with (
        patch.object(utils, "get_device", return_value=torch.device("cpu")),
        patch.object(utils, "get_sequence_parallel_world_size", return_value=1),
    ):
        yield


def _cached_pipe(cache_method, pipe=None):
    transformer = _Transformer()
    last_block = transformer.transformer_blocks[-1]
    if pipe is None:
        pipe = SimpleNamespace(scheduler=FlowMatchEulerDiscreteScheduler())
    pipe.transformer = transformer
    # A threshold no step reaches, so only forced computes run the blocks.
    apply_cache(
        cache_method=cache_method,
        num_steps=4,
        pipe=pipe,
        cache_config='{"residual_diff_threshold": 100.0}',
    )
    return pipe, transformer.transformer_blocks[0], last_block


def _request(pipe, cached_blocks, last_block, num_steps):
    """Run one request with identical inputs at every step; return the computed steps."""
    pipe.scheduler.set_timesteps(num_inference_steps=num_steps)
    # FBCache swaps the two streams, so give them one shape.
    hidden = torch.ones(1, 8, 4)
    encoder = torch.ones(1, 8, 4)
    temb = torch.ones(1, 4)
    computed = []
    for step in range(len(pipe.scheduler.timesteps)):
        calls = last_block.calls
        cached_blocks(hidden, encoder, temb=temb)
        if last_block.calls > calls:
            computed.append(step)
    return computed


def test_teacache_forces_first_and_last_step_of_each_request(cache_on_cpu):
    pipe, cached_blocks, last_block = _cached_pipe("teacache")

    # The cache was built for 4 steps; requests ask for other step counts.
    assert _request(pipe, cached_blocks, last_block, 6) == [0, 5]
    assert _request(pipe, cached_blocks, last_block, 3) == [0, 2]
    assert _request(pipe, cached_blocks, last_block, 6) == [0, 5]


def test_fbcache_does_not_reuse_the_previous_request(cache_on_cpu):
    pipe, cached_blocks, last_block = _cached_pipe("fbcache")

    assert _request(pipe, cached_blocks, last_block, 3) == [0]
    # The previous request ended on an identical input, but its residuals
    # belong to another request.
    assert _request(pipe, cached_blocks, last_block, 3) == [0]


def test_replacing_the_transformer_restarts_only_the_new_transformer(cache_on_cpu):
    pipe, old_blocks, _ = _cached_pipe("teacache")
    pipe, cached_blocks, last_block = _cached_pipe("teacache", pipe)

    old_blocks.cnt = torch.tensor(2)
    assert _request(pipe, cached_blocks, last_block, 6) == [0, 5]
    assert int(old_blocks.cnt) == 2


@pytest.mark.parametrize("with_single_blocks", [False, True])
def test_reapplying_teacache_to_the_same_transformer(cache_on_cpu, with_single_blocks):
    transformer = _Transformer()
    last_block = transformer.transformer_blocks[-1]
    if with_single_blocks:
        last_block = _SingleBlock()
        transformer.single_transformer_blocks.append(last_block)
    pipe = SimpleNamespace(scheduler=FlowMatchEulerDiscreteScheduler(), transformer=transformer)

    # Reapply on this same model after requests have populated its cache. A
    # changed threshold must take effect without nesting or losing blocks.
    for threshold, expected_steps in [(100.0, [0, 5]), (100.0, [0, 5]), (0.0, list(range(6)))]:
        apply_cache(
            cache_method="teacache",
            num_steps=4,
            pipe=pipe,
            cache_config=f'{{"residual_diff_threshold": {threshold}}}',
        )
        assert pipe.transformer is transformer
        cached_blocks = transformer.transformer_blocks[0]
        assert _request(pipe, cached_blocks, last_block, 6) == expected_steps
        assert _request(pipe, cached_blocks, last_block, 3) == ([0, 2] if threshold else [0, 1, 2])


class _Flux2Block(_Block):
    def forward(self, hidden_states, encoder_hidden_states, *args, **kwargs):
        hidden, encoder = super().forward(hidden_states, encoder_hidden_states, *args, **kwargs)
        return encoder, hidden


def test_flux2_fbcache_restarts_and_reuses_persistent_storage(cache_on_cpu):
    pytest.importorskip("diffusers.models.transformers.transformer_flux2")
    from xfuser.model_executor.cache.adapters.flux2 import Flux2FBCachedTransformerBlocks

    blocks = [_Flux2Block(), _Flux2Block()]
    cached = Flux2FBCachedTransformerBlocks(blocks, rel_l1_thresh=100.0, num_steps=3, return_hidden_states_first=False)
    root = torch.nn.ModuleList([cached])
    reference_blocks = [_Flux2Block(), _Flux2Block()]
    previous_input = None

    # Include a cold start, a different request at the same shape, and a
    # shape change. All restart the real FLUX.2 cache through the public helper.
    for value, token_count in [(1.0, 8), (2.0, 8), (3.0, 5)]:
        utils.restart_step_caches(root, num_steps=3)
        hidden = torch.full((1, token_count, 4), value)
        encoder = torch.full((1, 3, 4), value)
        expected_hidden, expected_encoder = hidden, encoder
        for block in reference_blocks:
            expected_encoder, expected_hidden = block(expected_hidden, expected_encoder)

        calls_before = blocks[-1].calls
        out_encoder, out_hidden = cached(hidden, encoder)
        assert blocks[-1].calls == calls_before + 1
        torch.testing.assert_close(out_hidden, expected_hidden)
        torch.testing.assert_close(out_encoder, expected_encoder)

        current_input = cached.cache_context.modulated_inputs
        if previous_input is not None and previous_input.shape == current_input.shape:
            # Same-shape request resets must retain the CUDA-graph-safe buffer.
            assert current_input.data_ptr() == previous_input.data_ptr()
        previous_input = current_input

        for _ in range(2):
            out_encoder, out_hidden = cached(hidden, encoder)
            torch.testing.assert_close(out_hidden, expected_hidden)
            torch.testing.assert_close(out_encoder, expected_encoder)
        assert blocks[-1].calls == calls_before + 1
