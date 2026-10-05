"""The HTTP engine must yield its event loop while waiting for Ray workers."""

import asyncio
import time
from pathlib import Path

import pytest


def test_http_engine_yields_while_waiting_for_real_ray_workers(tmp_path):
    ray = pytest.importorskip("ray")
    pytest.importorskip("fastapi")
    from entrypoints.launch import Engine, GenerateRequest

    class _Worker:
        def __init__(self, result):
            self.result = result

        def ready(self):
            return True

        def generate(self, request):
            gate = Path(request.prompt)
            deadline = time.monotonic() + 5
            while not gate.exists() and time.monotonic() < deadline:
                time.sleep(0.01)
            if not gate.exists():
                raise RuntimeError("HTTP event loop did not release the worker")
            return self.result

    ray.init(num_cpus=1, include_dashboard=False, object_store_memory=80 * 1024 * 1024)
    try:
        worker_class = ray.remote(num_cpus=0)(_Worker)
        workers = [worker_class.remote(None), worker_class.remote({"output": "generated.png"})]
        ray.get([worker.ready.remote() for worker in workers])
        engine = object.__new__(Engine)
        engine.workers = workers
        gate = tmp_path / "release-worker"

        async def exercise():
            task = asyncio.create_task(engine.generate(GenerateRequest(prompt=str(gate))))
            await asyncio.sleep(0)
            gate.touch()
            return await task

        assert asyncio.run(exercise()) == {"output": "generated.png"}
    finally:
        ray.shutdown()
