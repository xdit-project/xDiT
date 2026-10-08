"""Gathering latents for the parallel VAE creates its process group once, on every rank."""

import datetime
import queue
import time
import traceback
from types import SimpleNamespace
from unittest.mock import patch

import pytest

pytestmark = [pytest.mark.gloo, pytest.mark.slow]

# Classifier-free guidance splits the batch across both ranks, so only rank 1 is
# DP-last. Rank 0 never gathers, but ``new_group`` is collective and it must take
# part in creating the group anyway.
_WORLD_SIZE = 2
_REQUESTS = 3


def _gather_worker(rank, world_size, init_method, result_queue):
    dist = None
    try:
        import torch
        import torch.distributed as dist

        from xfuser.config.config import RuntimeConfig
        from xfuser.core.distributed import parallel_state, runtime_state
        from xfuser.core.distributed.runtime_state import DiTRuntimeState
        from xfuser.model_executor.pipelines import base_pipeline

        class _Pipeline(base_pipeline.xFuserPipelineBaseWrapper):
            def __call__(self):
                pass

        cpu = torch.device("cpu")
        # Keep every rank on CPU even when accelerators are visible.
        with (
            patch.object(parallel_state, "set_device"),
            patch.object(base_pipeline, "get_device", lambda local_rank: cpu),
        ):
            parallel_state.init_distributed_environment(
                backend="gloo",
                distributed_init_method=init_method,
                local_rank=rank,
                rank=rank,
                world_size=world_size,
            )
            parallel_state.initialize_model_parallel(
                backend="gloo",
                classifier_free_guidance_degree=world_size,
                use_parallel_vae=True,
            )
            state = DiTRuntimeState.__new__(DiTRuntimeState)
            state.runtime_config = RuntimeConfig(dtype=torch.float32, use_parallel_vae=True)
            # The VAE shares the model ranks; dedicated VAE ranks are a Ray-only topology.
            state.parallel_config = SimpleNamespace(vae_parallel_size=0)
            runtime_state._RUNTIME = state

            pipeline = _Pipeline.__new__(_Pipeline)
            gathered = []
            with patch.object(dist, "new_group", wraps=dist.new_group) as new_group:
                for request_idx in range(_REQUESTS):
                    latents = torch.full((1, 4, 2, 2), float(10 * rank + request_idx))
                    latents = pipeline.gather_latents_for_vae(latents)
                    gathered.append([float(sample.unique()) for sample in latents])
                calls = new_group.call_count

            # Groups created later by all ranks must still line up.
            later = dist.new_group(list(range(world_size)), timeout=datetime.timedelta(seconds=20))
            total = torch.ones(1)
            dist.all_reduce(total, group=later)
            assert total.item() == world_size

            parallel_state.get_world_group().barrier()

        result_queue.put(("returned", rank, calls, gathered))
    except Exception:  # noqa: BLE001 - report arbitrary child failures to the parent
        result_queue.put(("error", rank, traceback.format_exc()))
    finally:
        if dist is not None and dist.is_initialized():
            dist.destroy_process_group()


def _run_spawned(torch, init_method, *, timeout):
    context = torch.multiprocessing.get_context("spawn")
    result_queue = context.Queue()
    processes = [
        context.Process(
            target=_gather_worker,
            args=(rank, _WORLD_SIZE, init_method, result_queue),
        )
        for rank in range(_WORLD_SIZE)
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
    while len(results) < _WORLD_SIZE:
        try:
            results.append(result_queue.get(timeout=1))
        except queue.Empty:
            break
    return processes, hung, results


def _requests(*ranks):
    return [[float(10 * rank + request_idx) for rank in ranks] for request_idx in range(_REQUESTS)]


def test_repeated_requests_create_the_vae_gather_group_once_on_every_rank(tmp_path):
    torch = pytest.importorskip("torch")
    if not torch.distributed.is_available() or not torch.distributed.is_gloo_available():
        pytest.skip("torch.distributed with the gloo backend is unavailable")

    processes, hung, results = _run_spawned(
        torch,
        f"file://{tmp_path / 'vae-gather-init'}",
        timeout=120,
    )

    assert not hung, f"latent gather hung; results so far: {results}"
    errors = [result for result in results if result[0] == "error"]
    assert not errors, "\n".join(result[2] for result in errors)
    assert [process.exitcode for process in processes] == [0] * _WORLD_SIZE
    # Rank 0 is not DP-last, so it keeps its own latents; rank 1 gathers only its own.
    assert {rank: (calls, gathered) for _, rank, calls, gathered in results} == {
        0: (1, _requests(0)),
        1: (1, _requests(1)),
    }
