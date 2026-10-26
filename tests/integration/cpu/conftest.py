"""Shared harness for tests that spawn one process per distributed rank."""

import queue
import time

import pytest


def _run_spawned(torch, worker, init_method, *, world_size, timeout=600):
    """Run ranks within one budget, including interpreter startup and imports.

    Cold worker imports took 388 s on a slow filesystem (#817).
    """
    context = torch.multiprocessing.get_context("spawn")
    result_queue = context.Queue()
    processes = [
        context.Process(
            target=worker,
            args=(rank, world_size, init_method, result_queue),
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
    survivors = [process.pid for process in processes if process.is_alive()]

    results = []
    while len(results) < world_size:
        try:
            results.append(result_queue.get(timeout=1))
        except queue.Empty:
            break
    return processes, hung, survivors, results


@pytest.fixture
def run_spawned():
    """Run ``worker(rank, world_size, init_method, result_queue)`` in one spawned process per rank.

    Returns ``(processes, hung, survivors, results)``: hung workers are
    terminated after ``timeout`` seconds, survivors outlived SIGKILL, and
    results are whatever the workers put on the queue.
    """
    return _run_spawned
