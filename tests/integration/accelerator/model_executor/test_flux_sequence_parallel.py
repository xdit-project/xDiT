"""FLUX.1 under sequence parallelism matches the single-device transformer for supported token counts/configurations."""

import pytest
import torch

from diffusers import FluxTransformer2DModel

from xfuser.config.args import xFuserArgs
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
from xfuser.model_executor.models.transformers.transformer_flux import (
    xFuserFlux1Transformer2DWrapper,
)

_CONFIG = dict(
    patch_size=1,
    in_channels=8,
    num_layers=1,
    num_single_layers=1,
    attention_head_dim=16,
    num_attention_heads=6,
    joint_attention_dim=32,
    pooled_projection_dim=16,
    guidance_embeds=False,
    axes_dims_rope=(4, 6, 6),
)


def _inputs(image_tokens, text_tokens, device):
    generator = torch.Generator().manual_seed(0)

    def randn(*shape):
        return torch.randn(*shape, generator=generator).to(device)

    img_ids = torch.zeros(image_tokens, 3)
    img_ids[:, 1] = torch.arange(image_tokens) // 4
    img_ids[:, 2] = torch.arange(image_tokens) % 4
    return dict(
        hidden_states=randn(1, image_tokens, _CONFIG["in_channels"]),
        encoder_hidden_states=randn(1, text_tokens, _CONFIG["joint_attention_dim"]),
        pooled_projections=randn(1, _CONFIG["pooled_projection_dim"]),
        timestep=torch.tensor([0.5], device=device),
        img_ids=img_ids.to(device),
        txt_ids=torch.zeros(text_tokens, 3, device=device),
        return_dict=False,
    )


def _flux_worker(rank, world_size, init_method, ulysses, ring, image_tokens, text_tokens, error, pipeline_patches=1):
    torch.cuda.set_device(rank)
    init_distributed_environment(
        rank=rank,
        world_size=world_size,
        local_rank=rank,
        distributed_init_method=init_method,
    )
    initialize_model_parallel(ring_degree=ring, ulysses_degree=ulysses)
    try:
        args = xFuserArgs()
        args.ulysses_degree = ulysses
        args.ring_degree = ring
        engine_config, _ = args.create_config()
        initialize_runtime_state(engine_config=engine_config)
        # SDPA proper has no ring path; its memory-efficient kernel does.
        get_runtime_state().set_attention_backend("SDPA_EFFICIENT" if ring > 1 else "SDPA")
        get_runtime_state().max_condition_sequence_length = text_tokens
        get_runtime_state().num_pipeline_patch = pipeline_patches

        device = torch.device("cuda", rank)
        torch.manual_seed(0)
        reference = FluxTransformer2DModel(**_CONFIG).to(device).eval()
        parallel = xFuserFlux1Transformer2DWrapper(**_CONFIG).to(device).eval()
        parallel.load_state_dict(reference.state_dict())

        inputs = _inputs(image_tokens, text_tokens, device)
        with torch.no_grad():
            if error is not None:
                with pytest.raises(error):
                    parallel(**inputs)
                return
            expected = reference(**inputs)[0]
            actual = parallel(**inputs)[0]
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)
    finally:
        destroy_model_parallel()
        destroy_distributed_environment()


@pytest.mark.multi_gpu
@pytest.mark.parametrize(
    "ulysses, ring, image_tokens, text_tokens",
    [
        pytest.param(2, 1, 16, 8, id="both-divisible"),
        pytest.param(2, 1, 15, 8, id="padded-image"),
        pytest.param(2, 1, 16, 7, id="replicated-text"),
        pytest.param(3, 1, 16, 8, id="padded-image-replicated-text"),
        # Ring cases keep each rank's query length a multiple of 32, which the
        # memory-efficient kernel needs to merge ring steps.
        pytest.param(1, 2, 34, 15, id="ring-replicated-text"),
        pytest.param(2, 2, 40, 6, id="ulysses-ring-replicated-text"),
    ],
)
def test_flux_sequence_parallel_matches_single_device(accelerator_ranks, ulysses, ring, image_tokens, text_tokens):
    accelerator_ranks(
        _flux_worker,
        world_size=ulysses * ring,
        args=(ulysses, ring, image_tokens, text_tokens, None),
    )


@pytest.mark.multi_gpu
def test_flux_ring_rejects_image_tokens_it_cannot_shard(accelerator_ranks):
    accelerator_ranks(
        _flux_worker,
        world_size=2,
        args=(1, 2, 15, 8, NotImplementedError),
    )


@pytest.mark.multi_gpu
def test_flux_pipefusion_rejects_padded_image_tokens(accelerator_ranks):
    accelerator_ranks(
        _flux_worker,
        world_size=2,
        args=(2, 1, 15, 8, NotImplementedError, 2),
    )
