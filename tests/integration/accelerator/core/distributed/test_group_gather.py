"""Gather accepts tensor views such as CFG halves and transposes."""

import pytest

pytestmark = pytest.mark.multi_gpu


def _gather_worker(rank, world_size, init_method, layout):
    from datetime import timedelta

    import torch
    import torch.distributed as dist

    from xfuser.core.distributed.group_coordinator import GroupCoordinator

    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group(
        "nccl", rank=rank, world_size=world_size, init_method=init_method, timeout=timedelta(seconds=30)
    )
    try:
        group = GroupCoordinator([list(range(world_size))], rank, "nccl")
        base = torch.arange(256, dtype=torch.float32, device=device).reshape(2, 8, 4, 4)
        storage = base + rank * 1000
        if layout == "cfg_slice":
            value = storage.chunk(2, dim=1)[0]
        elif layout == "transpose":
            value = storage.transpose(1, 2)
        else:
            value = storage
        assert value.is_contiguous() == (layout == "contiguous")
        original_storage = storage.clone()
        original_stride = value.stride()

        def view_of(tensor):
            if layout == "cfg_slice":
                return tensor.chunk(2, dim=1)[0]
            if layout == "transpose":
                return tensor.transpose(1, 2)
            return tensor

        expected = [view_of(base + source_rank * 1000) for source_rank in range(world_size)]

        for dst in range(world_size):
            for dim in (0, -1):
                output = group.gather(value, dst=dst, dim=dim)
                if rank == dst:
                    torch.testing.assert_close(output, torch.cat(expected, dim=dim), rtol=0, atol=0)
                else:
                    assert output is None
                torch.testing.assert_close(storage, original_storage, rtol=0, atol=0)
                assert value.stride() == original_stride
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("layout", ["contiguous", "cfg_slice", "transpose"])
def test_gather_tensor_views(layout, accelerator_ranks):
    accelerator_ranks(_gather_worker, world_size=2, init_filename=f"gather-{layout}", args=(layout,))
