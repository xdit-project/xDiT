import torch
import torch.distributed as dist
import pytest


from diffusers.models.attention_processor import (
    Attention,
    HunyuanAttnProcessor2_0,
)
from xfuser.model_executor.layers.attention_processor import (
    xFuserHunyuanAttnProcessor2_0,
)

from xfuser.core.cache_manager.cache_manager import get_cache_manager
from xfuser.core.distributed import (
    get_runtime_state,
    init_distributed_environment,
    initialize_model_parallel,
    initialize_runtime_state,
)
from xfuser.core.distributed.parallel_state import destroy_distributed_environment, destroy_model_parallel
from xfuser.config.args import xFuserArgs


def _attention_worker(rank, world_size, init_method, attn_type):
    """Initialize sequence parallelism, then compare one attention processor."""
    torch.cuda.set_device(rank)
    init_distributed_environment(
        rank=rank,
        world_size=world_size,
        local_rank=rank,
        distributed_init_method=init_method,
    )
    initialize_model_parallel(ring_degree=1, ulysses_degree=world_size)
    try:
        args = xFuserArgs()
        args.ulysses_degree = world_size
        args.ring_degree = 1
        engine_config, _ = args.create_config()
        initialize_runtime_state(engine_config=engine_config)
        # Match the diffusers processor, which uses PyTorch SDPA.
        get_runtime_state().set_attention_backend("SDPA")
        run_attn_test(rank, world_size, attn_type)
    finally:
        destroy_model_parallel()
        destroy_distributed_environment()


def run_attn_test(rank, world_size, attn_type: str):
    """Example test function to run on each process."""
    print(f"run_attn_test {attn_type}")
    dim = 32
    sequence_len = 16
    num_attention_heads = 8
    hidden_dim = dim // num_attention_heads
    qk_norm = True
    dtype = torch.bfloat16
    print(f"world_size: {world_size}, rank: {rank}")

    _type_dict = {
        "HunyuanDiT": (HunyuanAttnProcessor2_0(), xFuserHunyuanAttnProcessor2_0()),
    }
    processor, parallel_processor = _type_dict[attn_type]

    torch.manual_seed(0)

    attn1 = (
        Attention(
            query_dim=dim,
            cross_attention_dim=None,
            dim_head=hidden_dim,
            heads=num_attention_heads,
            qk_norm="layer_norm" if qk_norm else None,
            eps=1e-6,
            bias=True,
            processor=processor,
        )
        .cuda(rank)
        .to(dtype)
    )
    attn2 = (
        Attention(
            query_dim=dim,
            cross_attention_dim=None,
            dim_head=hidden_dim,
            heads=num_attention_heads,
            qk_norm="layer_norm" if qk_norm else None,
            eps=1e-6,
            bias=True,
            processor=parallel_processor,
        )
        .cuda(rank)
        .to(dtype)
    )

    for p1, p2 in zip(attn1.parameters(), attn2.parameters()):
        p2.data.copy_(p1.data)

    get_cache_manager().register_cache_entry(attn2, layer_type="attn", cache_type="sequence_parallel_attn_cache")

    # prepare inputs
    norm_hidden_states = torch.randn(1, sequence_len, dim, dtype=dtype).cuda(rank)
    dist.broadcast(norm_hidden_states, 0)

    use_image_rotary_emb = False

    # prepare rope parameters
    cos = torch.randn(sequence_len, hidden_dim, dtype=dtype).cuda(rank)
    dist.broadcast(cos, 0)
    sin = torch.randn(sequence_len, hidden_dim, dtype=dtype).cuda(rank)
    dist.broadcast(sin, 0)
    image_rotary_emb = cos, sin
    cos_shard = torch.chunk(cos, world_size, dim=0)[rank]
    sin_shard = torch.chunk(sin, world_size, dim=0)[rank]
    image_rotary_emb_shard = cos_shard, sin_shard

    attn_output1 = attn1(
        norm_hidden_states,
        # encoder_hidden_states=None,
        image_rotary_emb=image_rotary_emb if use_image_rotary_emb else None,
    )

    attn_output1_shard = torch.chunk(attn_output1, world_size, dim=1)[rank]

    norm_hidden_states_shard = torch.chunk(norm_hidden_states, world_size, dim=1)[rank]

    attn_output2_shard = attn2(
        norm_hidden_states_shard,
        # encoder_hidden_states=None,
        image_rotary_emb=image_rotary_emb_shard if use_image_rotary_emb else None,
    )

    # check if the outputs are close
    # print(attn_output1_shard - attn_output2_shard)
    assert torch.allclose(attn_output1_shard, attn_output2_shard, atol=1e-2), "Outputs are not close"


@pytest.mark.multi_gpu
@pytest.mark.parametrize("attn_type", ["HunyuanDiT"])
def test_multi_process(attn_type, accelerator_ranks):
    accelerator_ranks(
        _attention_worker,
        world_size=4,
        init_filename="attention-processor-init",
        args=(attn_type,),
    )
