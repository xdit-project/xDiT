"""xFuserLongContextAttention runs, and matches dense attention, without flash-attn."""

import pytest

pytestmark = pytest.mark.multi_gpu

_SHAPE = (1, 64, 4, 32)  # (batch, sequence, heads, head dim)


def _worker(rank, world_size, init_method, ulysses, ring):
    import torch
    import torch.distributed as dist
    import torch.nn.functional as F

    from xfuser.core.distributed import init_distributed_environment, initialize_model_parallel
    from xfuser.core.distributed.parallel_state import destroy_distributed_environment, destroy_model_parallel
    from xfuser import envs
    from xfuser.core.long_ctx_attention import xFuserLongContextAttention

    # As if flash-attn (and FlashAttention-3) were not installed. Each rank is its own process.
    envs.PACKAGES_CHECKER.get_packages_info().update(has_flash_attn=False, has_flash_attn_3=False)

    torch.cuda.set_device(rank)
    init_distributed_environment(rank=rank, world_size=world_size, local_rank=rank, distributed_init_method=init_method)
    initialize_model_parallel(ring_degree=ring, ulysses_degree=ulysses)
    device = torch.device("cuda", rank)
    try:
        torch.manual_seed(0)
        query, key, value = (torch.randn(_SHAPE, device=device, dtype=torch.bfloat16) for _ in range(3))
        for tensor in (query, key, value):
            dist.broadcast(tensor, src=0)
        reference = F.scaled_dot_product_attention(
            query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2)
        ).transpose(1, 2)

        layer = xFuserLongContextAttention()
        output = layer(
            None,
            query.chunk(world_size, dim=1)[rank],
            key.chunk(world_size, dim=1)[rank],
            value.chunk(world_size, dim=1)[rank],
        )
        torch.testing.assert_close(output, reference.chunk(world_size, dim=1)[rank], rtol=2e-2, atol=2e-2)
    finally:
        destroy_model_parallel()
        destroy_distributed_environment()


@pytest.mark.parametrize("ulysses, ring", [(2, 1), (1, 2)], ids=["ulysses2", "ring2"])
def test_default_attention_without_flash_attn_matches_dense(accelerator_ranks, ulysses, ring):
    accelerator_ranks(_worker, world_size=ulysses * ring, args=(ulysses, ring))
