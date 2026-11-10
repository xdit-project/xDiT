"""Wan attention with --ulysses_a2a sdma --ulysses_sdma_overlap_qkv: Q/K sent before V is projected."""

import pytest

pytestmark = [pytest.mark.multi_gpu, pytest.mark.rocm]


def _setup(rank, world_size, init_method):
    """One Ulysses group of every rank, with the runtime state the SDMA path reads."""
    import torch

    from xfuser.config.args import xFuserArgs
    from xfuser.core.distributed import (
        get_runtime_state,
        init_distributed_environment,
        initialize_model_parallel,
        initialize_runtime_state,
    )

    torch.cuda.set_device(rank)
    init_distributed_environment(rank=rank, world_size=world_size, local_rank=rank, distributed_init_method=init_method)
    initialize_model_parallel(ulysses_degree=world_size)
    engine_config, _ = xFuserArgs(attention_backend="AITER", ulysses_degree=world_size).create_config()
    initialize_runtime_state(engine_config=engine_config)
    get_runtime_state().set_attention_backend("AITER")
    return torch.device("cuda", rank)


def _configure(mode, head_groups, **options):
    """Make these the runtime's Ulysses all-to-all options (--ulysses_a2a, --ulysses_sdma_*)."""
    from xfuser.config.ulysses_a2a import UlyssesA2AConfig
    from xfuser.core.distributed import get_runtime_state

    get_runtime_state().runtime_config.ulysses_a2a = UlyssesA2AConfig(mode, head_groups, **options)


def _processor_worker(rank, world_size, init_method, backend):
    from types import SimpleNamespace

    import torch
    from diffusers.models.transformers.transformer_wan import WanAttention
    from xfuser.core.distributed import get_runtime_state
    from xfuser.core.distributed.sdma import ulysses
    from xfuser.core.distributed.parallel_state import destroy_distributed_environment, destroy_model_parallel
    from xfuser.model_executor.layers import usp_sdma
    from xfuser.model_executor.models.transformers.transformer_wan import (
        xFuserWanAttnProcessor,
        xFuserWanSdmaAttnProcessor,
    )

    graph = None
    original_send = ulysses.send
    calls = 0

    def counted_send(process_group, gi, packed, tensors=None):
        # Q/K sent on their own: the Q/K-first path ran.
        nonlocal calls
        calls += tensors is not None and tuple(tensors) == (0, 1)
        return original_send(process_group, gi, packed, tensors)

    try:
        device = _setup(rank, world_size, init_method)
        get_runtime_state().set_attention_backend(backend)
        torch.manual_seed(0)
        attn = WanAttention(256, heads=40, dim_head=128, processor=xFuserWanAttnProcessor()).to(device, torch.bfloat16)
        attn.eval()
        # As the runner prepares the transformer: the SDMA processor replaces the standard one.
        _configure("sdma", 1, overlap_qkv=True)
        usp_sdma.prepare(SimpleNamespace(attn=attn), ["attn"], cpu_offload=False)
        assert type(attn.processor) is xFuserWanSdmaAttnProcessor
        generator = torch.Generator(device=device).manual_seed(rank)
        hidden = torch.randn(1, 257, 256, generator=generator, device=device, dtype=torch.bfloat16)
        angle = torch.randn(1, 257, 1, 128, generator=generator, device=device)
        rotary = angle.cos(), angle.sin()
        ulysses.send = counted_send
        compiled = torch.compile(attn, fullgraph=True)
        with torch.no_grad():
            for run in (attn, compiled):
                _configure("packed", 1)
                expected = run(hidden, rotary_emb=rotary)
                _configure("sdma", 1, overlap_qkv=True)
                before = calls
                actual = run(hidden, rotary_emb=rotary)
                assert calls == before + 1, "the Q/K-first path did not run"
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            _configure("sdma", 1, overlap_qkv=True)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured = attn(hidden, rotary_emb=rotary)
            for repeat in range(3):
                hidden.copy_(torch.randn(hidden.shape, generator=generator, device=device, dtype=hidden.dtype))
                graph.replay()
                _configure("packed", 1)
                expected = attn(hidden, rotary_emb=rotary)
                torch.testing.assert_close(captured, expected, rtol=0, atol=0)
            # The head-group count does not apply: Wan self-attention sends Q/K first in one group.
            _configure("sdma", 2, overlap_qkv=True)
            before = calls
            actual = attn(hidden, rotary_emb=rotary)
            assert calls == before + 1, "the Q/K-first path must run with --ulysses_sdma_head_groups 2"
            _configure("packed", 1)
            torch.testing.assert_close(actual, attn(hidden, rotary_emb=rotary), rtol=0, atol=0)
            # Unsupported: FP32 QK norms keep the usual path.
            attn.norm_q.to(torch.float32)
            attn.norm_k.to(torch.float32)
            _configure("sdma", 1, overlap_qkv=True)
            before = calls
            actual = attn(hidden, rotary_emb=rotary)
            assert calls == before, "unsupported settings must not send Q/K first"
            _configure("sdma", 1)
            torch.testing.assert_close(actual, attn(hidden, rotary_emb=rotary), rtol=0, atol=0)
    finally:
        graph = None
        ulysses.send = original_send
        _configure("default", 2)
        torch.compiler.reset()
        torch.cuda.synchronize()
        destroy_model_parallel()
        destroy_distributed_environment()


@pytest.mark.parametrize("backend", ["AITER_BF16", "AITER_FP8"])
def test_sdma_processor_sends_qk_first_and_matches_packed(accelerator_ranks, backend):
    """Wan self-attention through its SDMA processor: Q/K sent before V, eager, compiled and
    captured, with any head-group setting; unsupported calls keep the usual path."""
    accelerator_ranks(_processor_worker, world_size=2, timeout=900, args=(backend,))
