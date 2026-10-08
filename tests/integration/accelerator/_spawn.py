"""Run a multi-GPU test body from pytest without torchrun.

Each rank is a spawned interpreter so CUDA and HIP can initialize. The parent
surfaces the child traceback; a non-zero exit alone hides the failure.
The total timeout includes interpreter startup, imports, and the worker body.
Allow ten minutes by default: cold imports on a slow filesystem took 388 s (#817).
"""

import os
import queue
import time
import traceback

import pytest


def spawn_accelerator_ranks(worker, tmp_path, *, world_size, timeout=600, init_filename="dist-init", args=()):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or torch.cuda.device_count() < world_size:
        pytest.skip(f"requires {world_size} accelerator devices")
    if not torch.distributed.is_nccl_available():
        pytest.skip("NCCL/RCCL is unavailable")

    context = torch.multiprocessing.get_context("spawn")
    result_queue = context.Queue()
    init_method = f"file://{tmp_path / init_filename}"
    processes = [
        context.Process(
            target=_guard,
            args=(rank, world_size, init_method, worker, result_queue, args),
        )
        for rank in range(world_size)
    ]
    deadline = time.monotonic() + timeout
    for process in processes:
        process.start()
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

    errors = []
    while True:
        try:
            item = result_queue.get(timeout=1)
        except queue.Empty:
            break
        if item is not None:
            errors.append(item)

    exitcodes = [process.exitcode for process in processes]
    if hung or errors or exitcodes != [0] * world_size:
        details = []
        if hung:
            details.append(f"hung pids: {hung}")
        if exitcodes != [0] * world_size:
            details.append(f"exit codes: {exitcodes}")
        details.extend(errors)
        pytest.fail("\n".join(details))


def _guard(rank, world_size, init_method, worker, result_queue, args):
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    try:
        worker(rank, world_size, init_method, *args)
    except Exception:
        result_queue.put(traceback.format_exc())
        raise
    result_queue.put(None)
