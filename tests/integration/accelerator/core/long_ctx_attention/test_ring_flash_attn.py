import pytest

pytestmark = [pytest.mark.multi_gpu, pytest.mark.nvidia]

_WORLD_SIZE = 2


def _create_tensors(rank, world_size, device):
    import torch
    import torch.distributed as dist

    shape = (1, 128, 4, 32)
    query = torch.randn(shape, device=device, dtype=torch.float16)
    key = torch.randn(shape, device=device, dtype=torch.float16, requires_grad=True)
    value = torch.randn(shape, device=device, dtype=torch.float16, requires_grad=True)
    dist.broadcast(query, src=0)
    dist.broadcast(key, src=0)
    dist.broadcast(value, src=0)
    local_query = query.chunk(world_size, dim=1)[rank]
    local_key = key.chunk(world_size, dim=1)[rank]
    local_value = value.chunk(world_size, dim=1)[rank]
    return query, key, value, local_query, local_key, local_value


def _ring_worker(rank, world_size, init_method, case):
    import torch
    import torch.distributed as dist
    from flash_attn import flash_attn_func

    from xfuser.core.distributed import init_distributed_environment
    from xfuser.core.long_ctx_attention.ring.ring_flash_attn import xdit_ring_flash_attn_func

    torch.cuda.set_device(rank)
    torch.manual_seed(42 + rank)
    init_distributed_environment(
        rank=rank,
        world_size=world_size,
        local_rank=rank,
        distributed_init_method=init_method,
    )
    device = torch.device(f"cuda:{rank}")
    try:
        query, key, value, local_query, local_key, local_value = _create_tensors(rank, world_size, device)
        if case == "basic":
            reference = flash_attn_func(query, key, value, dropout_p=0.0, causal=True, window_size=(-1, -1))
            reference = reference.chunk(world_size, dim=1)[rank]
            output = xdit_ring_flash_attn_func(
                q=local_query,
                k=local_key,
                v=local_value,
                dropout_p=0.0,
                causal=True,
                window_size=(-1, -1),
                group=dist.group.WORLD,
            )
            torch.testing.assert_close(reference, output, rtol=1e-3, atol=1e-3)
            assert reference.shape == output.shape
            return

        joint_query, joint_key, joint_value, _, _, _ = _create_tensors(rank, world_size, device)
        if case == "rear":
            reference_key = torch.cat([key, joint_key], dim=1)
            reference_value = torch.cat([value, joint_value], dim=1)
            joint_strategy = "rear"
        elif case == "front":
            reference_key = torch.cat([joint_key, key], dim=1)
            reference_value = torch.cat([joint_value, value], dim=1)
            joint_strategy = "front"
        else:
            raise AssertionError(f"unknown ring case {case}")

        reference = flash_attn_func(
            query,
            reference_key,
            reference_value,
            dropout_p=0.0,
            causal=False,
            window_size=(-1, -1),
        )
        reference = reference.chunk(world_size, dim=1)[rank]
        output = xdit_ring_flash_attn_func(
            q=local_query,
            k=local_key,
            v=local_value,
            dropout_p=0.0,
            causal=False,
            window_size=(-1, -1),
            joint_tensor_key=joint_key,
            joint_tensor_value=joint_value,
            joint_strategy=joint_strategy,
        )
        torch.testing.assert_close(reference, output, rtol=1e-3, atol=1e-3)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.parametrize("case", ["basic", "rear", "front"])
def test_xdit_ring_flash_attn_func(case, accelerator_ranks):
    pytest.importorskip("flash_attn")
    accelerator_ranks(_ring_worker, world_size=_WORLD_SIZE, init_filename=f"ring-{case}", args=(case,))
