"""The HTTP server hands its options to every worker and request.

Ray and FastAPI are not package dependencies, so this module skips without them.
"""

import asyncio

import pytest
import torch

pytest.importorskip("ray")
pytest.importorskip("fastapi")

from entrypoints import launch  # noqa: E402


class _FakeWorker:
    def __init__(self, kwargs):
        self.kwargs = kwargs
        self.requests = []
        worker = self

        class _Generate:
            @staticmethod
            def remote(request):
                worker.requests.append(request)
                if worker.kwargs["rank"] == worker.kwargs["world_size"] - 1:
                    return {"output": request.save_disk_path}
                return None

        self.generate = _Generate


class _FakeImageGenerator:
    @staticmethod
    def remote(xfuser_args, **kwargs):
        return _FakeWorker(kwargs)


@pytest.fixture
def make_engine(monkeypatch):
    monkeypatch.setattr(launch.ray, "is_initialized", lambda: True)
    monkeypatch.setattr(launch.ray, "get", lambda refs: refs)
    monkeypatch.setattr(launch, "ImageGenerator", _FakeImageGenerator)

    def make(**overrides):
        kwargs = dict(
            world_size=2,
            xfuser_args=object(),
            dtype=torch.bfloat16,
            master_addr="10.0.0.7",
            master_port=29600,
        )
        kwargs.update(overrides)
        return launch.Engine(**kwargs)

    return make


def test_workers_receive_dtype_and_rendezvous_address(make_engine):
    engine = make_engine()

    assert [worker.kwargs for worker in engine.workers] == [
        dict(rank=rank, world_size=2, dtype=torch.bfloat16, master_addr="10.0.0.7", master_port=29600)
        for rank in range(2)
    ]


def test_server_save_path_applies_only_when_the_request_has_none(make_engine, tmp_path):
    engine = make_engine(save_disk_path=str(tmp_path))

    unset = asyncio.run(engine.generate(launch.GenerateRequest(prompt="a cat")))
    own = asyncio.run(engine.generate(launch.GenerateRequest(prompt="a cat", save_disk_path="elsewhere")))

    assert unset == {"output": str(tmp_path)}
    assert own == {"output": "elsewhere"}
    assert all(len(worker.requests) == 2 for worker in engine.workers)


def test_without_a_server_save_path_the_image_is_returned_inline(make_engine):
    engine = make_engine()

    result = asyncio.run(engine.generate(launch.GenerateRequest(prompt="a cat")))

    assert result == {"output": None}
