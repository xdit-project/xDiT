import torch

from xfuser.core.distributed.group_coordinator import PipelineGroupCoordinator


class Work:
    def __init__(self):
        self.waited = False

    def wait(self):
        self.waited = True
        return True


def test_pipeline_p2p_splits_large_payloads_into_bounded_chunks(monkeypatch):
    coordinator = object.__new__(PipelineGroupCoordinator)
    coordinator.rank_in_group = 0
    coordinator.world_size = 2
    coordinator.ranks = [0, 1]
    coordinator.device_groups = ["even", "odd"]
    coordinator.device_group = "default"
    coordinator.pipeline_chunk_bytes = {}
    coordinator.set_pipeline_tensor_chunk_bytes("wan", 8)
    calls = []
    works = []

    def record(operation):
        def invoke(tensor, **kwargs):
            calls.append((operation, tensor.numel(), kwargs))
            work = Work()
            works.append(work)
            return work

        return invoke

    monkeypatch.setattr(torch.distributed, "isend", record("send"))
    monkeypatch.setattr(torch.distributed, "irecv", record("recv"))
    tensor = torch.arange(5, dtype=torch.float32)

    send = coordinator._pipeline_isend(tensor, name="wan")
    receive = coordinator._pipeline_irecv(tensor, name="wan")

    assert [(operation, size) for operation, size, _kwargs in calls] == [
        ("send", 2),
        ("send", 2),
        ("send", 1),
        ("recv", 2),
        ("recv", 2),
        ("recv", 1),
    ]
    assert all(call[2]["group"] == "even" for call in calls[:3])
    assert all(call[2]["group"] == "odd" for call in calls[3:])
    send.wait()
    receive.wait()
    assert all(work.waited for work in works)


def test_pipeline_p2p_does_not_chunk_unconfigured_payloads(monkeypatch):
    coordinator = object.__new__(PipelineGroupCoordinator)
    coordinator.rank_in_group = 0
    coordinator.world_size = 2
    coordinator.ranks = [0, 1]
    coordinator.device_groups = ["even", "odd"]
    coordinator.device_group = "default"
    coordinator.pipeline_chunk_bytes = {"wan": 8}
    calls = []

    def record(tensor, **kwargs):
        calls.append((tensor.numel(), kwargs))
        return Work()

    monkeypatch.setattr(torch.distributed, "isend", record)
    tensor = torch.arange(5, dtype=torch.float32)

    coordinator._pipeline_isend(tensor, name="qwen")

    assert calls == [(5, {"dst": 1, "group": "even"})]


def test_pipeline_known_shapes_preallocate_varying_receive_buffers():
    coordinator = object.__new__(PipelineGroupCoordinator)
    coordinator.device = torch.device("cpu")
    coordinator.dtype = torch.float32
    coordinator.recv_shape = {}
    coordinator.send_shape = {}
    coordinator.recv_buffer = {}

    coordinator.set_pipeline_tensor_shapes(
        "hidden",
        [torch.Size((1, 4, 8)), torch.Size((1, 3, 8))],
        [torch.Size((1, 4)), torch.Size((1, 3))],
    )

    assert coordinator.recv_shape["hidden"] == {
        0: torch.Size((1, 4, 8)),
        1: torch.Size((1, 3, 8)),
    }
    assert coordinator.send_shape["hidden"] == {
        0: torch.Size((1, 4)),
        1: torch.Size((1, 3)),
    }
    assert tuple(coordinator.recv_buffer["hidden"][0].shape) == (1, 4, 8)
    assert tuple(coordinator.recv_buffer["hidden"][1].shape) == (1, 3, 8)
    assert not coordinator.recv_buffer["hidden"][0].count_nonzero()
