import pytest

pytestmark = [pytest.mark.multi_gpu, pytest.mark.nvidia]

_WORLD_SIZE = 4


def _create_tensors(rank, world_size, device, dtype):
    import torch
    import torch.distributed as dist

    shape = (1, 128, 4, 32)
    query = torch.randn(shape, device=device, dtype=dtype)
    key = torch.randn(shape, device=device, dtype=dtype)
    value = torch.randn(shape, device=device, dtype=dtype)
    dist.broadcast(query, src=0)
    dist.broadcast(key, src=0)
    dist.broadcast(value, src=0)
    return (
        query,
        key,
        value,
        query.chunk(world_size, dim=1)[rank],
        key.chunk(world_size, dim=1)[rank],
        value.chunk(world_size, dim=1)[rank],
    )


def _xfuser_worker(rank, world_size, init_method, case):
    import torch
    from flash_attn import flash_attn_func
    from yunchang.kernels import AttnType

    from xfuser.core.distributed import init_distributed_environment, initialize_model_parallel
    from xfuser.core.distributed.parallel_state import destroy_distributed_environment, destroy_model_parallel
    from xfuser.core.long_ctx_attention import xFuserLongContextAttention
    from xfuser.envs import PACKAGES_CHECKER

    if not PACKAGES_CHECKER.get_packages_info()["has_flash_attn"] or not hasattr(AttnType, "FA"):
        raise AssertionError("flash attention backend is not available")

    ring_degree = world_size // 2
    ulysses_degree = 2
    torch.cuda.set_device(rank)
    init_distributed_environment(
        rank=rank,
        world_size=world_size,
        local_rank=rank,
        distributed_init_method=init_method,
    )
    initialize_model_parallel(ring_degree=ring_degree, ulysses_degree=ulysses_degree)
    device = torch.device(f"cuda:{rank}")
    try:
        for dtype in (torch.float16, torch.bfloat16):
            torch.manual_seed(42 + rank)
            query, key, value, local_query, local_key, local_value = _create_tensors(rank, world_size, device, dtype)
            layer = xFuserLongContextAttention(
                scatter_idx=2,
                gather_idx=1,
                ring_impl_type="basic",
                attn_type=AttnType.FA,
                use_kv_cache=False,
            ).to(device=device, dtype=dtype)
            assert layer.ring_pg.size() == ring_degree
            assert layer.ulysses_pg.size() == ulysses_degree

            if case == "layer":
                reference = flash_attn_func(query, key, value, dropout_p=0.0, window_size=(-1, -1))
                reference = reference.chunk(world_size, dim=1)[rank]
                output = layer(
                    attn=None,
                    query=local_query,
                    key=local_key,
                    value=local_value,
                    dropout_p=0.0,
                    window_size=(-1, -1),
                )
                assert torch.max(torch.abs(output - reference)) < 1e-2
            elif case == "rear":
                joint_query, joint_key, joint_value, _, _, _ = _create_tensors(rank, world_size, device, dtype)
                reference = flash_attn_func(
                    torch.cat([query, joint_query], dim=1),
                    torch.cat([key, joint_key], dim=1),
                    torch.cat([value, joint_value], dim=1),
                    dropout_p=0.0,
                    window_size=(-1, -1),
                )
                base = reference[:, : query.shape[1]]
                joint = reference[:, query.shape[1] :]
                reference = torch.cat([base.chunk(world_size, dim=1)[rank], joint], dim=1)
                output = layer(
                    attn=None,
                    query=local_query,
                    key=local_key,
                    value=local_value,
                    dropout_p=0.0,
                    window_size=(-1, -1),
                    joint_tensor_query=joint_query,
                    joint_tensor_key=joint_key,
                    joint_tensor_value=joint_value,
                    joint_strategy="rear",
                )
            else:
                raise AssertionError(f"unknown xfuser attention case {case}")

            torch.testing.assert_close(reference, output, rtol=1e-2, atol=1e-2)
    finally:
        destroy_model_parallel()
        destroy_distributed_environment()


@pytest.mark.parametrize("case", ["layer", "rear"])
def test_xfuser_long_context_attention(case, accelerator_ranks):
    pytest.importorskip("flash_attn")
    accelerator_ranks(_xfuser_worker, world_size=_WORLD_SIZE, init_filename=f"xfuser-{case}", args=(case,))
