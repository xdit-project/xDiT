"""initialize_runtime_state() without an engine config on more than one rank."""

import queue
import time
import traceback

import pytest

pytestmark = pytest.mark.gloo

_WORLD_SIZE = 2


def _worker(rank, world_size, init_method, result_queue, model_parallel):
    dist = None
    try:
        from unittest.mock import patch

        import torch.distributed as dist

        from xfuser.core.distributed import get_runtime_state, initialize_runtime_state, parallel_state

        with patch.object(parallel_state, "set_device"):
            parallel_state.init_distributed_environment(
                backend="gloo",
                distributed_init_method=init_method,
                local_rank=rank,
                rank=rank,
                world_size=world_size,
            )
        if model_parallel is not None:
            parallel_state.initialize_model_parallel(backend="gloo", **model_parallel)

        initialize_runtime_state()

        config = get_runtime_state().parallel_config
        degrees = {name: getattr(config, name) for name in ("dp_degree", "cfg_degree", "ulysses_degree", "ring_degree")}
        result_queue.put(("returned", rank, degrees))
    except BaseException:
        result_queue.put(("error", rank, traceback.format_exc()))
    finally:
        try:
            from xfuser.core.distributed import parallel_state

            parallel_state.destroy_model_parallel()
        except BaseException:
            pass
        if dist is not None and dist.is_initialized():
            dist.destroy_process_group()


def _run_spawned(torch, init_method, model_parallel, *, timeout):
    context = torch.multiprocessing.get_context("spawn")
    result_queue = context.Queue()
    processes = [
        context.Process(target=_worker, args=(rank, _WORLD_SIZE, init_method, result_queue, model_parallel))
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
            process.terminate()
            process.join(5)

    results = []
    while len(results) < _WORLD_SIZE:
        try:
            results.append(result_queue.get(timeout=1))
        except queue.Empty:
            break
    return processes, hung, results


@pytest.mark.parametrize(
    "model_parallel, expected",
    [
        # Without model-parallel groups each rank runs the whole model. Sequence
        # parallelism needs yunchang's accelerator groups; see the accelerator test.
        pytest.param(None, dict(dp_degree=2, cfg_degree=1, ulysses_degree=1, ring_degree=1), id="no-groups"),
        pytest.param(
            dict(classifier_free_guidance_degree=2),
            dict(dp_degree=1, cfg_degree=2, ulysses_degree=1, ring_degree=1),
            id="cfg",
        ),
        pytest.param(
            dict(data_parallel_degree=2),
            dict(dp_degree=2, cfg_degree=1, ulysses_degree=1, ring_degree=1),
            id="data-parallel",
        ),
    ],
)
def test_runtime_state_without_engine_config_describes_the_ranks(tmp_path, model_parallel, expected):
    torch = pytest.importorskip("torch", reason="PyTorch is required for gloo test")
    if not torch.distributed.is_available() or not torch.distributed.is_gloo_available():
        pytest.skip("torch.distributed gloo backend is unavailable")

    processes, hung, results = _run_spawned(torch, f"file://{tmp_path / 'gloo-init'}", model_parallel, timeout=300)

    assert not hung, f"hung worker pids: {hung}"
    assert [process.exitcode for process in processes] == [0] * _WORLD_SIZE
    assert sorted(results, key=lambda result: result[1]) == [
        ("returned", rank, expected) for rank in range(_WORLD_SIZE)
    ]
