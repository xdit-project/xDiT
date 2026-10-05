import pytest

pytestmark = pytest.mark.multi_gpu

_WORLD_SIZE = 2


def _feedforward_worker(rank, world_size, init_method):
    import torch
    import torch.distributed as dist
    from diffusers.models.attention import FeedForward

    from xfuser.core.distributed import init_distributed_environment, initialize_model_parallel
    from xfuser.core.distributed.parallel_state import destroy_distributed_environment, destroy_model_parallel
    from xfuser.model_executor.layers.feedforward import xFuserFeedForwardWrapper

    torch.cuda.set_device(rank)
    init_distributed_environment(
        rank=rank,
        world_size=world_size,
        local_rank=rank,
        distributed_init_method=init_method,
    )
    initialize_model_parallel(tensor_parallel_degree=world_size)
    try:
        torch.manual_seed(0)
        inputs = torch.ones(1, 20, device=rank)
        dist.broadcast(inputs, src=0)

        torch.manual_seed(0)
        reference = FeedForward(20, 5, bias=True, activation_fn="geglu").to(rank)
        for param in reference.parameters():
            dist.broadcast(param.data, src=0)

        expected = reference(inputs)
        wrapped = xFuserFeedForwardWrapper(reference)
        actual = wrapped(inputs)
        assert torch.allclose(expected, actual, atol=1e-2)
    finally:
        destroy_model_parallel()
        destroy_distributed_environment()


def test_feedforward_matches_diffusers(accelerator_ranks):
    accelerator_ranks(_feedforward_worker, world_size=_WORLD_SIZE)
