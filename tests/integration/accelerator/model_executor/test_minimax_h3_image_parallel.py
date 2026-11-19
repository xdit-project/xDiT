"""MiniMax-H3-Image Ulysses and parallel-VAE parity against stock diffusers on tiny random models."""

import pytest

pytest.importorskip(
    "diffusers.models.transformers.transformer_minimax_h3_image",
    reason="installed diffusers does not include MiniMax-H3-Image",
)

pytestmark = pytest.mark.multi_gpu

BACKEND = "SDPA"
NUM_HEADS = 8
NUM_PDD_HEADS = 3
IMAGE_GRID = (6, 8)
# Packed lengths: 7 + 48 = 55 and 6 + 12 + 48 = 66 are padded to 64 and 128 rows;
# 16 + 48 = 64 already aligns, so no row is padded.
LAYOUTS = {
    "t2i-padded": (7, 0),
    "t2i-aligned": (16, 0),
    "edit": (6, 12),
}


def _tiny_config():
    return {
        "num_attention_heads": NUM_HEADS,
        "attention_head_dim": 16,
        "hidden_size": 32,
        "num_layers": 2,
        "num_refiner_layers": 1,
        "ffn_dim": 48,
        "in_channels": 4,
        "patch_size": (1, 2, 2),
        "text_dim": 8,
        "freq_dim": 8,
        "time_embed_hidden_dim": 32,
        "time_embed_dim": 16,
        "rope_freq_dim": 2,
        "num_pdd_heads": NUM_PDD_HEADS,
    }


def _inputs(device, text_tokens, reference_tokens, pdd_step_index):
    import torch

    image_tokens = IMAGE_GRID[0] * IMAGE_GRID[1]
    sequence_length = text_tokens + reference_tokens + image_tokens
    token_tags = torch.zeros(sequence_length, dtype=torch.long)
    token_tags[:text_tokens] = 1
    timestep_indices = torch.zeros(sequence_length, dtype=torch.long)
    timestep_indices[text_tokens : text_tokens + reference_tokens] = 1
    grid = torch.stack(
        torch.meshgrid(torch.arange(float(IMAGE_GRID[0])), torch.arange(float(IMAGE_GRID[1])), indexing="ij"), -1
    ).flatten(0, 1)
    position_ids = torch.zeros(sequence_length, 3)
    position_ids[:text_tokens, 0] = torch.arange(text_tokens, dtype=torch.float32)
    position_ids[text_tokens : text_tokens + reference_tokens, 0] = text_tokens
    position_ids[text_tokens : text_tokens + reference_tokens, 1:] = grid[:reference_tokens]
    position_ids[text_tokens + reference_tokens :, 0] = text_tokens + 1
    position_ids[text_tokens + reference_tokens :, 1:] = grid

    generator = torch.Generator().manual_seed(1)
    inputs = {
        "hidden_states": torch.randn(1, reference_tokens + image_tokens, 16, generator=generator),
        "encoder_hidden_states": torch.randn(1, text_tokens, 8, generator=generator),
        "timestep": torch.tensor([0.25, 0.999]) if reference_tokens else torch.tensor([0.25]),
        "timestep_indices": timestep_indices,
        "token_tags": token_tags,
        "position_ids": position_ids,
        "image_indices": torch.arange(text_tokens, sequence_length),
        "text_indices": torch.arange(text_tokens),
    }
    inputs = {name: value.to(device) for name, value in inputs.items()}
    inputs["pdd_step_index"] = pdd_step_index
    return inputs


def _setup(rank, world_size, init_method, ulysses_degree, transformer=None):
    """Join the process group; with ``transformer``, set up the runtime state a
    runner gives a pipeline holding it, which counts transformer forwards."""
    from types import SimpleNamespace

    import torch

    from xfuser.config.args import xFuserArgs
    from xfuser.core.distributed import (
        get_runtime_state,
        init_distributed_environment,
        initialize_model_parallel,
        initialize_runtime_state,
    )

    torch.cuda.set_device(rank)
    init_distributed_environment(
        rank=rank,
        world_size=world_size,
        local_rank=rank,
        distributed_init_method=init_method,
    )
    initialize_model_parallel(ulysses_degree=ulysses_degree)
    engine_config, _ = xFuserArgs(attention_backend=BACKEND, ulysses_degree=ulysses_degree).create_config()
    pipeline = None if transformer is None else SimpleNamespace(transformer=transformer)
    initialize_runtime_state(pipeline, engine_config=engine_config)
    get_runtime_state().set_attention_backend(BACKEND)
    return torch.device(f"cuda:{rank}")


def _ulysses_worker(rank, world_size, init_method):
    import torch
    from diffusers import MiniMaxH3ImageTransformer2DModel

    from xfuser.core.distributed.parallel_state import destroy_distributed_environment, destroy_model_parallel
    from xfuser.model_executor.models.transformers.transformer_minimax_h3_image import (
        xFuserMiniMaxH3ImageTransformer2DWrapper,
    )

    try:
        torch.manual_seed(0)
        stock = MiniMaxH3ImageTransformer2DModel(**_tiny_config()).eval()
        device = _setup(rank, world_size, init_method, ulysses_degree=world_size, transformer=stock)
        stock = stock.to(device)
        wrapper = xFuserMiniMaxH3ImageTransformer2DWrapper.from_config(stock.config).to(device).eval()
        wrapper.load_state_dict(stock.state_dict())
        wrapper.fuse_qkv_projections()

        for layout, (text_tokens, reference_tokens) in LAYOUTS.items():
            for step in range(NUM_PDD_HEADS):
                inputs = _inputs(device, text_tokens, reference_tokens, step)
                with torch.no_grad():
                    reference = stock(**inputs).sample
                    actual = wrapper(**inputs).sample
                torch.testing.assert_close(
                    actual,
                    reference,
                    rtol=1e-4,
                    atol=1e-4,
                    msg=lambda m: f"Ulysses {world_size}, {layout} layout, step {step}: {m}",
                )
    finally:
        destroy_model_parallel()
        destroy_distributed_environment()


@pytest.mark.parametrize("ulysses_degree", [2, 4, 8])
def test_ulysses_matches_single_gpu_diffusers(accelerator_ranks, ulysses_degree):
    accelerator_ranks(_ulysses_worker, world_size=ulysses_degree, timeout=300)


def _fsdp_worker(rank, world_size, init_method, fsdp2):
    import torch
    from diffusers import MiniMaxH3ImageTransformer2DModel

    from xfuser.core.distributed.parallel_state import destroy_distributed_environment, destroy_model_parallel
    from xfuser.core.distributed.sharding import shard_component
    from xfuser.model_executor.models.transformers.transformer_minimax_h3_image import (
        xFuserMiniMaxH3ImageTransformer2DWrapper,
    )

    try:
        torch.manual_seed(0)
        stock = MiniMaxH3ImageTransformer2DModel(**_tiny_config()).eval()
        device = _setup(rank, world_size, init_method, ulysses_degree=world_size, transformer=stock)
        stock = stock.to(device)
        wrapper = xFuserMiniMaxH3ImageTransformer2DWrapper.from_config(stock.config).eval()
        wrapper.load_state_dict(stock.state_dict())
        # memory_efficient_init shards with FSDP2; the default is FSDP1.
        wrapper = shard_component(
            wrapper, wrap_attrs=["transformer_blocks"], device_id=device, memory_efficient_init=fsdp2
        )
        # The runner compiles a sharded transformer block by block.
        blocks = wrapper.transformer_blocks
        for index in range(len(blocks)):
            blocks[index] = torch.compile(blocks[index])

        for step in range(NUM_PDD_HEADS):
            inputs = _inputs(device, *LAYOUTS["edit"], step)
            with torch.no_grad():
                reference = stock(**inputs).sample
                actual = wrapper(**inputs).sample
            torch.testing.assert_close(
                actual, reference, rtol=1e-4, atol=1e-4, msg=lambda m: f"FSDP2={fsdp2}, step {step}: {m}"
            )
    finally:
        destroy_model_parallel()
        destroy_distributed_environment()


@pytest.mark.parametrize("fsdp2", [False, True], ids=["fsdp1", "fsdp2"])
def test_sharded_wrapper_matches_single_gpu_diffusers(accelerator_ranks, fsdp2):
    accelerator_ranks(_fsdp_worker, world_size=2, timeout=300, args=(fsdp2,))


def _tiny_vae(device):
    import torch
    from diffusers import AutoencoderKLMiniMaxH3Image

    torch.manual_seed(0)
    vae = (
        AutoencoderKLMiniMaxH3Image(
            latent_channels=4,
            block_out_channels=(8, 8, 8),
            layers_per_block=1,
            spatial_downsample_factors=(2, 2, 1),
            temporal_downsample_factors=(1, 1, 1),
            norm_num_groups=4,
            decoder_num_layers=2,
            decoder_num_attention_heads=2,
            decoder_attention_head_dim=8,
            decoder_num_register_tokens=2,
            latents_mean=[0.0] * 4,
            latents_std=[1.0] * 4,
        )
        .to(device)
        .eval()
    )
    vae.enable_tiling(16, 16, 8, 8)
    return vae


def _vae_worker(rank, world_size, init_method):
    import torch

    from xfuser.core.distributed import get_world_group
    from xfuser.core.distributed.parallel_state import destroy_distributed_environment, destroy_model_parallel
    from xfuser.model_executor.models.runner_models import minimax_h3_image

    try:
        device = _setup(rank, world_size, init_method, ulysses_degree=world_size)
        vae = _tiny_vae(device)
        generator = torch.Generator().manual_seed(2)
        # A 4x6 grid of tiles, more than any rank count; and a 1x2 grid, fewer than
        # most, which leaves some ranks without a tile to decode.
        for latent_shape in ((1, 4, 10, 14), (1, 4, 4, 6)):
            latents = torch.randn(latent_shape, generator=generator).to(device)
            with torch.no_grad():
                if hasattr(vae, "_xfuser_original_decode"):
                    vae._decode = vae._xfuser_original_decode
                    del vae._xfuser_original_decode
                reference = vae.decode(latents).sample
                minimax_h3_image.install_minimax_h3_image_tiled_decode(vae, get_world_group())
                actual = vae.decode(latents).sample
            torch.testing.assert_close(
                actual,
                reference,
                rtol=1e-5,
                atol=1e-5,
                msg=lambda m: f"{world_size} ranks, latents {latent_shape}: {m}",
            )
    finally:
        destroy_model_parallel()
        destroy_distributed_environment()


@pytest.mark.parametrize("world_size", [2, 4])
def test_parallel_tiled_decode_matches_the_vae_loop(accelerator_ranks, world_size):
    accelerator_ranks(_vae_worker, world_size=world_size, timeout=300)
