"""Gathering final latents across data-parallel ranks reuses one process group."""

import traceback
from functools import partial
from types import SimpleNamespace
from unittest.mock import patch

import pytest

pytestmark = [pytest.mark.gloo, pytest.mark.slow]

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
            parallel_state.initialize_model_parallel(backend="gloo", data_parallel_degree=world_size)

            state = DiTRuntimeState.__new__(DiTRuntimeState)
            state.runtime_config = RuntimeConfig(dtype=torch.bfloat16)
            runtime_state._RUNTIME = state

            pipeline = _Pipeline.__new__(_Pipeline)
            gathered = []
            with patch.object(dist, "new_group", wraps=dist.new_group) as new_group:
                for request_idx in range(_REQUESTS):
                    latents = torch.full((1, 4, 2, 2), float(10 * rank + request_idx))
                    latents = pipeline.gather_broadcast_latents(latents)
                    gathered.append(
                        (
                            latents.dtype,
                            [float(sample.unique()) for sample in latents],
                        )
                    )
                new_group_calls = new_group.call_count

        result_queue.put(("returned", rank, new_group_calls, gathered))
    except Exception:  # noqa: BLE001 - report arbitrary child failures to the parent
        result_queue.put(("error", rank, traceback.format_exc()))
    finally:
        if dist is not None and dist.is_initialized():
            dist.destroy_process_group()


def test_repeated_requests_create_the_dp_last_group_once(tmp_path, run_spawned):
    torch = pytest.importorskip("torch")
    if not torch.distributed.is_available() or not torch.distributed.is_gloo_available():
        pytest.skip("torch.distributed with the gloo backend is unavailable")

    processes, hung, _, results = run_spawned(
        torch,
        _gather_worker,
        f"file://{tmp_path / 'dp-last-group-init'}",
        world_size=_WORLD_SIZE,
        timeout=120,
    )

    assert not hung, f"latent gather hung; results so far: {results}"
    errors = [result for result in results if result[0] == "error"]
    assert not errors, "\n".join(result[2] for result in errors)
    assert [process.exitcode for process in processes] == [0] * _WORLD_SIZE
    # Every rank receives both ranks' latents, in rank order, for every request.
    expected = [
        (
            torch.float32,
            [float(request_idx), float(10 + request_idx)],
        )
        for request_idx in range(_REQUESTS)
    ]
    assert {rank: (calls, gathered) for _, rank, calls, gathered in results} == {
        0: (1, expected),
        1: (1, expected),
    }


def _dtype_worker(rank, world_size, init_method, result_queue, *, pipeline_stages):
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
            if pipeline_stages:
                parallel_state.initialize_model_parallel(backend="gloo", pipeline_parallel_degree=world_size)
            else:
                parallel_state.initialize_model_parallel(backend="gloo", data_parallel_degree=world_size)

            state = DiTRuntimeState.__new__(DiTRuntimeState)
            # Neither the engine config's dtype nor the VAE's matches what the pipeline computed in.
            state.runtime_config = RuntimeConfig(dtype=torch.float16)
            runtime_state._RUNTIME = state

            pipeline = _Pipeline.__new__(_Pipeline)
            vae_dtype = torch.bfloat16 if pipeline_stages else torch.float32
            pipeline.module = SimpleNamespace(vae=SimpleNamespace(dtype=vae_dtype))
            # The last PipeFusion stage holds the final latents, earlier stages hold none.
            holds_latents = not pipeline_stages or rank == world_size - 1
            latents = torch.full((1, 4, 2, 2), 3.0, dtype=torch.float32 if pipeline_stages else torch.bfloat16)
            latents = pipeline.gather_broadcast_latents(latents if holds_latents else None)

        result_queue.put(("returned", rank, str(latents.dtype), latents.float().unique().tolist()))
    except Exception:  # noqa: BLE001 - report arbitrary child failures to the parent
        result_queue.put(("error", rank, traceback.format_exc()))
    finally:
        if dist is not None and dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.parametrize(
    ("pipeline_stages", "dtype"),
    [
        pytest.param(False, "torch.bfloat16", id="dp-pipeline-dtype-differs-from-engine-and-vae"),
        pytest.param(True, "torch.float32", id="earlier-pipeline-stage-has-no-latents-and-vae-dtype-differs"),
    ],
)
def test_every_rank_receives_the_latents_in_the_dtype_they_were_sent(tmp_path, run_spawned, pipeline_stages, dtype):
    torch = pytest.importorskip("torch")
    if not torch.distributed.is_available() or not torch.distributed.is_gloo_available():
        pytest.skip("torch.distributed with the gloo backend is unavailable")

    processes, hung, _, results = run_spawned(
        torch,
        partial(_dtype_worker, pipeline_stages=pipeline_stages),
        f"file://{tmp_path / 'dtype-init'}",
        world_size=_WORLD_SIZE,
        timeout=120,
    )

    assert not hung, f"latent broadcast hung; results so far: {results}"
    errors = [result for result in results if result[0] == "error"]
    assert not errors, "\n".join(result[2] for result in errors)
    assert [process.exitcode for process in processes] == [0] * _WORLD_SIZE
    assert sorted((rank, got_dtype, values) for _, rank, got_dtype, values in results) == [
        (0, dtype, [3.0]),
        (1, dtype, [3.0]),
    ]
