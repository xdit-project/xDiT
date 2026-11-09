"""Real two-rank coverage for the circular PipeFusion transport."""

import queue
import time
import traceback

import pytest

pytestmark = pytest.mark.gloo


def _worker(rank, init_method, result_queue):
    import torch
    import torch.distributed as dist

    try:
        from xfuser.core.distributed import group_coordinator
        from xfuser.model_executor.pipefusion import (
            PipeFusionAsyncCallbacks,
            PipeFusionAsyncDriver,
            PipeFusionTransport,
        )

        dist.init_process_group(
            backend="gloo",
            init_method=init_method,
            rank=rank,
            world_size=2,
        )
        group_coordinator.envs.get_device = lambda _local_rank: torch.device("cpu")
        group_coordinator.synchronize = lambda: None
        group = group_coordinator.PipelineGroupCoordinator([[0, 1]], rank, "gloo")
        group.set_config(torch.float32)
        seen = []

        def prepare(work, _timestep, received):
            if rank == 0 and received is None:
                return torch.tensor(float(work.patch_index + 1))
            return received

        def forward(work, _timestep, prepared):
            seen.append((work.step_index, work.patch_index, prepared.item()))
            return prepared + (10 if rank == 0 else 100)

        def commit(work, _timestep, output):
            if rank == 1 and work.step_index == work.num_steps - 1:
                return None
            return output

        result = PipeFusionAsyncDriver(
            transport=PipeFusionTransport(
                group,
                first_stage=rank == 0,
                num_steps=2,
                num_patches=2,
            ),
            hooks=PipeFusionAsyncCallbacks(
                prepare_patch_fn=prepare,
                forward_patch_fn=forward,
                commit_patch_fn=commit,
                finalize_fn=lambda: tuple(seen),
            ),
            advance_patch=lambda: None,
        ).run((1, 2))
        result_queue.put(("ok", rank, result))
    except Exception:  # noqa: BLE001 - child traceback is returned to the parent
        result_queue.put(("error", rank, traceback.format_exc()))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def test_circular_transport_completes_without_hanging(tmp_path):
    torch = pytest.importorskip("torch")
    if not torch.distributed.is_available() or not torch.distributed.is_gloo_available():
        pytest.skip("Gloo distributed support is unavailable")

    context = torch.multiprocessing.get_context("spawn")
    result_queue = context.Queue()
    init_method = f"file://{tmp_path / 'init'}"
    processes = [context.Process(target=_worker, args=(rank, init_method, result_queue)) for rank in range(2)]
    deadline = time.monotonic() + 180
    for process in processes:
        process.start()
    for process in processes:
        process.join(max(0, deadline - time.monotonic()))

    hung = [process.pid for process in processes if process.is_alive()]
    for process in processes:
        if process.is_alive():
            process.kill()
            process.join(5)

    results = []
    while len(results) < 2:
        try:
            results.append(result_queue.get(timeout=1))
        except queue.Empty:
            break

    assert not hung, f"PipeFusion Gloo workers hung: {hung}"
    assert [process.exitcode for process in processes] == [0, 0]
    assert all(result[0] == "ok" for result in results), results
    seen = {rank: values for _, rank, values in results}
    assert seen[0] == (
        (0, 0, 1.0),
        (0, 1, 2.0),
        (1, 0, 111.0),
        (1, 1, 112.0),
    )
    assert seen[1] == (
        (0, 0, 11.0),
        (0, 1, 12.0),
        (1, 0, 121.0),
        (1, 1, 122.0),
    )
