"""FLUX.3 Ulysses parity against stock diffusers on a tiny random flow-map transformer."""

import pytest

pytest.importorskip(
    "diffusers.models.transformers.transformer_flux3",
    reason="installed diffusers does not include FLUX.3",
)

pytestmark = pytest.mark.multi_gpu

NUM_HEADS = 24
# 7 text tokens, a 3x5 image and two references (2x3 and 1x7): every stream and both
# joint sequences (22 and 35 tokens) need SP padding at degrees 2, 4 and 8.
TEXT_LEN = 7
IMAGE_GRID = (3, 5)
REFERENCE_GRIDS = ((2, 3), (1, 7))


def _tiny_model(device):
    import torch
    from diffusers.models.transformers.transformer_flux3 import Flux3Transformer2DModel

    torch.manual_seed(0)
    return (
        Flux3Transformer2DModel(
            in_channels=8,
            cond_in_channels=8,
            num_layers=2,
            num_single_layers=2,
            attention_head_dim=16,
            num_attention_heads=NUM_HEADS,
            joint_attention_dim=32,
            timestep_channels=32,
            axes_dims_rope=(4, 4, 4, 4),
            flow_map=True,
        )
        .to(device)
        .eval()
    )


def _ids(height, width, l_coord):
    import torch

    grid = torch.meshgrid(
        torch.arange(1), torch.arange(height), torch.arange(width), torch.tensor([l_coord]), indexing="ij"
    )
    return torch.stack(grid, dim=-1).reshape(-1, 4).float()


def _inputs(device, with_references):
    import torch

    generator = torch.Generator().manual_seed(1)
    image_len = IMAGE_GRID[0] * IMAGE_GRID[1]
    inputs = {
        "hidden_states": torch.randn(1, image_len, 8, generator=generator),
        "encoder_hidden_states": torch.randn(1, TEXT_LEN, 32, generator=generator),
        "timestep": torch.tensor([0.7]),
        "target_timestep": torch.tensor([0.5]),
        "img_ids": _ids(*IMAGE_GRID, 0),
        "txt_ids": torch.stack([torch.zeros(TEXT_LEN)] * 3 + [torch.arange(TEXT_LEN).float()], dim=-1),
    }
    if with_references:
        reference_len = sum(h * w for h, w in REFERENCE_GRIDS)
        inputs["image_cond_hidden_states"] = torch.randn(1, reference_len, 8, generator=generator)
        inputs["image_cond_ids"] = torch.cat(
            [_ids(h, w, -(1 + index)) for index, (h, w) in enumerate(REFERENCE_GRIDS)], dim=0
        )
    return {name: value.to(device) for name, value in inputs.items()}


def _ulysses_worker(rank, world_size, init_method):
    import torch

    from xfuser.config.args import xFuserArgs
    from xfuser.core.distributed import (
        get_runtime_state,
        init_distributed_environment,
        initialize_model_parallel,
        initialize_runtime_state,
    )
    from xfuser.core.distributed.parallel_state import destroy_distributed_environment, destroy_model_parallel
    from xfuser.model_executor.models.transformers.transformer_flux3 import xFuserFlux3Transformer2DWrapper

    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    init_distributed_environment(
        rank=rank,
        world_size=world_size,
        local_rank=rank,
        distributed_init_method=init_method,
    )
    initialize_model_parallel(ulysses_degree=world_size, ring_degree=1)
    try:
        engine_config, _ = xFuserArgs(attention_backend="SDPA", ulysses_degree=world_size).create_config()
        initialize_runtime_state(engine_config=engine_config)
        get_runtime_state().set_attention_backend("SDPA")
        stock = _tiny_model(device)
        wrapper = xFuserFlux3Transformer2DWrapper.from_config(stock.config).to(device).eval()
        wrapper.load_state_dict(stock.state_dict())

        for with_references in (False, True):
            inputs = _inputs(device, with_references)
            with torch.no_grad():
                reference = stock(**inputs).sample
                actual = wrapper(**inputs).sample
            torch.testing.assert_close(
                actual,
                reference,
                rtol=1e-4,
                atol=1e-4,
                msg=lambda m: f"Ulysses {world_size}, references={with_references}: {m}",
            )
    finally:
        destroy_model_parallel()
        destroy_distributed_environment()


@pytest.mark.parametrize("world_size", [2, 4, 8])
def test_ulysses_matches_single_gpu_diffusers(accelerator_ranks, world_size):
    accelerator_ranks(_ulysses_worker, world_size=world_size, timeout=300)
