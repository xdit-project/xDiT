"""initialize_runtime_state() without an engine config under sequence parallelism."""

import pytest

pytestmark = pytest.mark.multi_gpu


def _worker(rank, world_size, init_method, ulysses, ring):
    import torch

    from xfuser.core.distributed import (
        get_runtime_state,
        init_distributed_environment,
        initialize_model_parallel,
        initialize_runtime_state,
    )
    from xfuser.core.distributed.parallel_state import destroy_distributed_environment, destroy_model_parallel

    torch.cuda.set_device(rank)
    init_distributed_environment(rank=rank, world_size=world_size, local_rank=rank, distributed_init_method=init_method)
    initialize_model_parallel(ulysses_degree=ulysses, ring_degree=ring)
    try:
        initialize_runtime_state()
        config = get_runtime_state().parallel_config
        assert (config.dp_degree, config.cfg_degree, config.ulysses_degree, config.ring_degree) == (1, 1, ulysses, ring)
    finally:
        destroy_model_parallel()
        destroy_distributed_environment()


@pytest.mark.parametrize("ulysses, ring", [(2, 1), (1, 2)], ids=["ulysses2", "ring2"])
def test_runtime_state_without_engine_config_takes_the_sequence_parallel_degrees(accelerator_ranks, ulysses, ring):
    accelerator_ranks(_worker, world_size=ulysses * ring, args=(ulysses, ring))
