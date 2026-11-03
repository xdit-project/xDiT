import torch

from xfuser.core.distributed.group_coordinator import PipelineGroupCoordinator


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
