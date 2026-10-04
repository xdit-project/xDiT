"""LTX-Video under Ulysses and CFG parallelism must match one unsplit diffusers forward.

Ranks are spawned over Gloo on CPU. Each builds the same tiny random model and
compares the xDiT wrapper's gathered output with diffusers'
LTXVideoTransformer3DModel run unsplit. The token count is odd, so Ulysses has
to pad the sequence and keep the padded keys out of attention; prompts of
different lengths and an image-to-video per-token timestep exercise the CFG
slicing of every batch-shaped input.
"""

import queue
import time
import traceback
from types import SimpleNamespace
from unittest.mock import patch

import pytest

pytestmark = pytest.mark.gloo

_CONFIG = dict(
    in_channels=8,
    out_channels=8,
    num_attention_heads=4,
    attention_head_dim=8,
    cross_attention_dim=32,
    num_layers=2,
    caption_channels=24,
)
_FRAMES, _HEIGHT, _WIDTH = 3, 3, 3  # 27 tokens
_TEXT_LEN = 6


def _worker(rank, world_size, init_method, layout, result_queue):
    dist = None
    try:
        import torch
        import torch.distributed as dist
        from diffusers.models.transformers.transformer_ltx import LTXVideoTransformer3DModel

        from xfuser.core.attention.spec import AttentionBackendType
        from xfuser.core.distributed import parallel_state
        from xfuser.model_executor.models.transformers.transformer_ltx_video import (
            xFuserLTXVideoTransformer3DWrapper,
        )

        with patch.object(parallel_state, "set_device"):
            parallel_state.init_distributed_environment(
                backend="gloo",
                distributed_init_method=init_method,
                local_rank=rank,
                rank=rank,
                world_size=world_size,
            )
        parallel_state.initialize_model_parallel(
            backend="gloo",
            classifier_free_guidance_degree=layout["cfg"],
            sequence_parallel_degree=layout["ulysses"],
            ulysses_degree=layout["ulysses"],
        )

        torch.manual_seed(0)
        reference = LTXVideoTransformer3DModel(**_CONFIG).eval()
        wrapper = xFuserLTXVideoTransformer3DWrapper.from_config(reference.config).eval()
        wrapper.load_state_dict(reference.state_dict())

        generator = torch.Generator().manual_seed(1)
        tokens = _FRAMES * _HEIGHT * _WIDTH
        mask = torch.zeros(2, _TEXT_LEN, dtype=torch.int64)
        mask[0, :2] = 1
        mask[1, :5] = 1
        timestep = torch.full((2, tokens), 800.0)
        timestep[:, : _HEIGHT * _WIDTH] = 0.0
        inputs = dict(
            hidden_states=torch.randn(2, tokens, _CONFIG["in_channels"], generator=generator),
            encoder_hidden_states=torch.randn(2, _TEXT_LEN, _CONFIG["caption_channels"], generator=generator),
            timestep=timestep,
            encoder_attention_mask=mask,
            num_frames=_FRAMES,
            height=_HEIGHT,
            width=_WIDTH,
            rope_interpolation_scale=(8 / 25, 32, 32),
            return_dict=False,
        )

        runtime = SimpleNamespace(
            attention_backend=AttentionBackendType.SDPA,
            runtime_config=SimpleNamespace(use_spargeattn_head_balance=False),
            fp8_comms=None,
        )
        with patch("xfuser.model_executor.layers.usp.get_runtime_state", return_value=runtime), torch.no_grad():
            expected = reference(**inputs)[0]
            actual = wrapper(**inputs)[0]

        result_queue.put(("returned", rank, tuple(actual.shape), (actual - expected).abs().max().item()))
    except Exception:  # noqa: BLE001 - report arbitrary child failures to the parent
        result_queue.put(("error", rank, traceback.format_exc()))
    finally:
        if dist is not None and dist.is_initialized():
            dist.destroy_process_group()


def _run_spawned(torch, init_method, layout, timeout):
    world_size = layout["cfg"] * layout["ulysses"]
    context = torch.multiprocessing.get_context("spawn")
    result_queue = context.Queue()
    processes = [
        context.Process(target=_worker, args=(rank, world_size, init_method, layout, result_queue))
        for rank in range(world_size)
    ]
    for process in processes:
        process.start()

    deadline = time.monotonic() + timeout
    for process in processes:
        process.join(max(0.0, deadline - time.monotonic()))
    hung = [process.pid for process in processes if process.is_alive()]
    for process in processes:
        if process.is_alive():
            process.kill()
            process.join(5)

    results = []
    while len(results) < world_size:
        try:
            results.append(result_queue.get(timeout=1))
        except queue.Empty:
            break
    return world_size, hung, results


@pytest.mark.slow
@pytest.mark.parametrize(
    "layout",
    [{"cfg": 1, "ulysses": 2}, {"cfg": 2, "ulysses": 2}],
    ids=["ulysses2", "cfg2-ulysses2"],
)
def test_parallel_forward_matches_unsplit_diffusers(tmp_path, layout):
    torch = pytest.importorskip("torch")
    pytest.importorskip("diffusers")
    if not torch.distributed.is_available() or not torch.distributed.is_gloo_available():
        pytest.skip("torch.distributed with gloo is unavailable")

    world_size, hung, results = _run_spawned(torch, f"file://{tmp_path / 'ltx-video-gloo-init'}", layout, timeout=300)

    errors = [result for result in results if result[0] == "error"]
    assert not errors, errors
    assert not hung, f"workers hung: {hung}"
    assert len(results) == world_size
    tokens = _FRAMES * _HEIGHT * _WIDTH
    for _, rank, shape, max_abs_diff in results:
        assert shape == (2, tokens, _CONFIG["out_channels"]), rank
        assert max_abs_diff < 1e-5, (rank, max_abs_diff)
