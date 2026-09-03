"""Distributed tests for model-replica process-group membership."""

import queue
import time
import traceback
from contextlib import nullcontext
from unittest.mock import patch

import pytest

_WORLD_SIZE = 4
_DP_DEGREE = 2
_CFG_DEGREE = 2


def _model_replica_group_worker(
    rank,
    world_size,
    init_method,
    result_queue,
    *,
    backend,
    mock_device_selection,
):
    dist = None
    try:
        import torch
        import torch.distributed as dist

        from xfuser.core.distributed import parallel_state

        device_selection = (
            patch.object(parallel_state, "set_device")
            if mock_device_selection
            else nullcontext()
        )
        with device_selection:
            parallel_state.init_distributed_environment(
                backend=backend,
                distributed_init_method=init_method,
                local_rank=rank,
                rank=rank,
                world_size=world_size,
            )
        parallel_state.initialize_model_parallel(
            backend=backend,
            classifier_free_guidance_degree=_CFG_DEGREE,
            data_parallel_degree=_DP_DEGREE,
        )

        replica = parallel_state.get_model_replica_group()
        device = (
            torch.device(f"cuda:{rank}") if backend == "nccl" else torch.device("cpu")
        )
        gathered = replica.all_gather(
            torch.tensor([rank], dtype=torch.int64, device=device)
        )

        # Only the first replica enters this extra barrier. The following world
        # barrier proves that the second replica was not required to release it.
        if rank < world_size // _DP_DEGREE:
            replica.barrier()
        parallel_state.get_world_group().barrier()

        result_queue.put(
            (
                "returned",
                rank,
                parallel_state.get_data_parallel_rank(),
                replica.ranks,
                gathered.tolist(),
            )
        )
    except Exception:  # noqa: BLE001 - report arbitrary child failures to the parent
        result_queue.put(("error", rank, traceback.format_exc()))
    finally:
        if dist is not None and dist.is_initialized():
            dist.destroy_process_group()


def _gloo_model_replica_group_worker(rank, world_size, init_method, result_queue):
    _model_replica_group_worker(
        rank,
        world_size,
        init_method,
        result_queue,
        backend="gloo",
        mock_device_selection=True,
    )


def _nccl_model_replica_group_worker(rank, world_size, init_method, result_queue):
    _model_replica_group_worker(
        rank,
        world_size,
        init_method,
        result_queue,
        backend="nccl",
        mock_device_selection=False,
    )


def _run_spawned(torch, worker, init_method, *, world_size, timeout):
    context = torch.multiprocessing.get_context("spawn")
    result_queue = context.Queue()
    processes = [
        context.Process(
            target=worker,
            args=(rank, world_size, init_method, result_queue),
        )
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
            process.terminate()
            process.join(5)

    for process in processes:
        if process.is_alive():
            process.kill()
            process.join(5)
    survivors = [process.pid for process in processes if process.is_alive()]

    results = []
    while len(results) < world_size:
        try:
            results.append(result_queue.get(timeout=1))
        except queue.Empty:
            break
    return processes, hung, survivors, results


def _require_torch_distributed():
    torch = pytest.importorskip(
        "torch", reason="PyTorch is required for distributed process-group tests"
    )
    if not torch.distributed.is_available():
        pytest.skip("torch.distributed is unavailable")
    return torch


def _real_gpu_unavailability(torch):
    reasons = []
    if not torch.cuda.is_available():
        reasons.append("CUDA-compatible GPUs are unavailable")
    visible_gpus = torch.cuda.device_count()
    if visible_gpus < _WORLD_SIZE:
        reasons.append(
            f"{_WORLD_SIZE} visible CUDA-compatible GPUs are required; found {visible_gpus}"
        )
    if not torch.distributed.is_nccl_available():
        reasons.append("NCCL/RCCL is unavailable")
    return "; ".join(reasons) if reasons else None


def _assert_replica_group_results(processes, hung, survivors, results):
    assert not survivors, f"workers survived SIGKILL: {survivors}"
    assert not hung, f"model-replica collective hung worker pids: {hung}"
    assert [process.exitcode for process in processes] == [0] * _WORLD_SIZE
    assert len(results) == _WORLD_SIZE
    assert all(result[0] == "returned" for result in results), results

    facts = {
        rank: (dp_rank, replica_ranks, gathered_ranks)
        for _, rank, dp_rank, replica_ranks, gathered_ranks in results
    }
    assert facts == {
        0: (0, [0, 1], [0, 1]),
        1: (0, [0, 1], [0, 1]),
        2: (1, [2, 3], [2, 3]),
        3: (1, [2, 3], [2, 3]),
    }


@pytest.mark.slow
def test_model_replica_group_on_real_gpus_without_hanging(tmp_path):
    torch = _require_torch_distributed()
    unavailable = _real_gpu_unavailability(torch)
    if unavailable:
        pytest.skip(
            f"real-GPU model-replica test requirements unsatisfied: {unavailable}"
        )

    processes, hung, survivors, results = _run_spawned(
        torch,
        _nccl_model_replica_group_worker,
        f"file://{tmp_path / 'model-replica-nccl-init'}",
        world_size=_WORLD_SIZE,
        timeout=60,
    )

    _assert_replica_group_results(processes, hung, survivors, results)


@pytest.mark.slow
def test_model_replica_group_on_cpu_without_hanging(tmp_path):
    torch = _require_torch_distributed()
    if _real_gpu_unavailability(torch) is None:
        pytest.skip(
            "real GPUs are available; the NCCL test covers replica-group behavior"
        )
    if not torch.distributed.is_gloo_available():
        pytest.skip("torch.distributed gloo backend is unavailable")

    processes, hung, survivors, results = _run_spawned(
        torch,
        _gloo_model_replica_group_worker,
        f"file://{tmp_path / 'model-replica-gloo-init'}",
        world_size=_WORLD_SIZE,
        timeout=30,
    )

    _assert_replica_group_results(processes, hung, survivors, results)


if __name__ == "__main__":
    import sys
    import pytest

    sys.exit(pytest.main(sys.argv))
