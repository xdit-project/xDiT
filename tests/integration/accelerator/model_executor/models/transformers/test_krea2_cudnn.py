"""Krea-2 on cuDNN attention matches the stock transformer with padded prompts.

Prompts of different lengths are padded, and the padded text tokens must not
serve as attention keys. cuDNN excludes them through the key-padding mask.
"""

import pytest
import torch

pytestmark = pytest.mark.nvidia

_CONFIG = dict(
    in_channels=16,
    num_layers=2,
    attention_head_dim=32,
    num_attention_heads=4,
    num_key_value_heads=2,
    intermediate_size=64,
    timestep_embed_dim=32,
    text_hidden_dim=32,
    num_text_layers=2,
    text_num_attention_heads=2,
    text_num_key_value_heads=2,
    text_intermediate_size=64,
    num_layerwise_text_blocks=1,
    num_refiner_text_blocks=1,
    axes_dims_rope=(8, 12, 12),
)
_TEXT, _HEIGHT, _WIDTH = 7, 4, 5


def _inputs(device):
    generator = torch.Generator().manual_seed(0)
    text_mask = torch.zeros(2, _TEXT, dtype=torch.bool)
    text_mask[0, :3] = True
    text_mask[1, :] = True
    image_ids = torch.stack(
        torch.meshgrid(torch.zeros(1), torch.arange(float(_HEIGHT)), torch.arange(float(_WIDTH)), indexing="ij"), dim=-1
    ).reshape(-1, 3)
    position_ids = torch.cat([torch.zeros(_TEXT, 3), image_ids])
    return dict(
        hidden_states=torch.randn(2, _HEIGHT * _WIDTH, 16, generator=generator).to(device, torch.bfloat16),
        encoder_hidden_states=torch.randn(2, _TEXT, 2, 32, generator=generator).to(device, torch.bfloat16),
        timestep=torch.tensor([0.7, 0.7], device=device),
        position_ids=position_ids.to(device),
        encoder_attention_mask=text_mask.to(device),
        return_dict=False,
    )


def _worker(rank, world_size, init_method):
    from diffusers.models.transformers.transformer_krea2 import Krea2Transformer2DModel

    from xfuser.config.args import xFuserArgs
    from xfuser.core.distributed import (
        get_runtime_state,
        init_distributed_environment,
        initialize_model_parallel,
        initialize_runtime_state,
    )
    from xfuser.core.distributed.parallel_state import destroy_distributed_environment, destroy_model_parallel
    from xfuser.model_executor.models.transformers.transformer_krea2 import xFuserKrea2Transformer2DWrapper

    torch.cuda.set_device(rank)
    init_distributed_environment(rank=rank, world_size=world_size, local_rank=rank, distributed_init_method=init_method)
    initialize_model_parallel(ulysses_degree=world_size)
    try:
        args = xFuserArgs()
        args.ulysses_degree = world_size
        args.attention_backend = "CUDNN"
        engine_config, _ = args.create_config()
        initialize_runtime_state(engine_config=engine_config)
        assert get_runtime_state().attention_backend.name == "CUDNN"

        device = torch.device("cuda", rank)
        torch.manual_seed(0)
        reference = Krea2Transformer2DModel(**_CONFIG).to(device, torch.bfloat16).eval()
        parallel = xFuserKrea2Transformer2DWrapper(**_CONFIG)
        parallel.load_state_dict(reference.state_dict())
        parallel = parallel.to(device, torch.bfloat16).eval()

        inputs = _inputs(device)
        with torch.no_grad():
            expected = reference(**inputs)[0]
            actual = parallel(**inputs)[0]
        # On B200 the two match bit for bit, single device and Ulysses 2. Attending to
        # sample 0's padded text keys would move its output by up to ~2e-2, so the
        # tolerance stays an order of magnitude below that.
        torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-3)
    finally:
        destroy_model_parallel()
        destroy_distributed_environment()


@pytest.mark.parametrize(
    "world_size",
    [1, pytest.param(2, marks=pytest.mark.multi_gpu)],
    ids=["single-device", "ulysses2"],
)
def test_krea2_on_cudnn_matches_the_stock_transformer(accelerator_ranks, world_size):
    accelerator_ranks(_worker, world_size=world_size)
