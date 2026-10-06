"""The SDPA kernels that serve ring attention return one log-sum-exp per query.

Ring attention merges each step's output with its log-sum-exp row by row, so a
kernel whose log-sum-exp is longer than the query breaks the merge.
"""

import pytest
import torch
import torch.nn.functional as F

from xfuser.config.args import xFuserArgs
from xfuser.core.attention.backends.sdpa import kernel
from xfuser.core.attention.spec import AttnCall
from xfuser.core.distributed import (
    get_runtime_state,
    init_distributed_environment,
    initialize_model_parallel,
    initialize_runtime_state,
)
from xfuser.core.distributed.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
)
from xfuser.model_executor.layers.usp import USP


def _qkv(seq_len, dtype, device, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return [torch.randn(1, 2, seq_len, 64, generator=generator).to(device=device, dtype=dtype) for _ in range(3)]


def _kernel_available(backend, query, key, value):
    can_use = getattr(torch.backends.cuda, f"can_use_{backend}_attention", None)
    if can_use is None:
        return False
    try:
        params = torch.backends.cuda.SDPAParams(query, key, value, None, 0.0, False, False)
    except TypeError:  # PyTorch before 2.5 has no enable_gqa argument
        params = torch.backends.cuda.SDPAParams(query, key, value, None, 0.0, False)
    return can_use(params)


@pytest.mark.parametrize("seq_len", [15, 32, 33])
@pytest.mark.parametrize(
    "kernel_fn, backend, dtype",
    [
        pytest.param(kernel.sdpa_efficient, "efficient", torch.float32, id="efficient-fp32"),
        pytest.param(kernel.sdpa_efficient, "efficient", torch.bfloat16, id="efficient-bf16"),
        pytest.param(kernel.sdpa_flash, "flash", torch.bfloat16, id="flash-bf16"),
        pytest.param(kernel.cudnn, "cudnn", torch.bfloat16, id="cudnn-bf16"),
    ],
)
def test_ring_capable_sdpa_kernels_return_one_lse_per_query(kernel_fn, backend, dtype, seq_len):
    if not torch.cuda.is_available():
        pytest.skip("requires an accelerator")
    query, key, value = _qkv(seq_len, dtype, "cuda")
    if not _kernel_available(backend, query, key, value):
        pytest.skip(f"PyTorch's {backend} attention kernel is unavailable here")
    _, lse = kernel_fn(query, key, value, AttnCall())

    scores = query.float() @ key.float().transpose(-1, -2) * query.shape[-1] ** -0.5
    expected = torch.logsumexp(scores, dim=-1)
    assert lse.shape == expected.shape
    torch.testing.assert_close(lse.float(), expected, rtol=2e-2, atol=2e-2)


def _ring_worker(rank, world_size, init_method, local_seq_len):
    torch.cuda.set_device(rank)
    init_distributed_environment(
        rank=rank,
        world_size=world_size,
        local_rank=rank,
        distributed_init_method=init_method,
    )
    initialize_model_parallel(ring_degree=world_size, ulysses_degree=1)
    try:
        args = xFuserArgs()
        args.ulysses_degree = 1
        args.ring_degree = world_size
        engine_config, _ = args.create_config()
        initialize_runtime_state(engine_config=engine_config)
        get_runtime_state().set_attention_backend("SDPA_EFFICIENT")

        device = torch.device("cuda", rank)
        query, key, value = _qkv(world_size * local_seq_len, torch.float32, device)
        expected = F.scaled_dot_product_attention(query, key, value)

        def shard(tensor):
            return tensor.chunk(world_size, dim=2)[rank]

        with torch.no_grad():
            actual = USP(shard(query), shard(key), shard(value))
        torch.testing.assert_close(actual, shard(expected), rtol=1e-4, atol=1e-4)
    finally:
        destroy_model_parallel()
        destroy_distributed_environment()


@pytest.mark.multi_gpu
@pytest.mark.parametrize("local_seq_len", [15, 32])
def test_ring_attention_on_the_efficient_kernel_matches_full_attention(accelerator_ranks, local_seq_len):
    accelerator_ranks(_ring_worker, world_size=2, args=(local_seq_len,))
