"""FLUX.2 under PipeFusion must keep one stale-KV cache entry per attention layer.

Each stage runs its share of a tiny FLUX.2 transformer: first a full-image pass, as
PipeFusion's warmup does, then patch by patch. Every patch sees the same inputs as the
full pass, so the cached KV of the other patches is exactly the fresh KV. The patch
outputs must therefore reproduce diffusers' own forward over the whole image.
"""

import copy

import pytest
import torch
import torch.distributed as dist

# The FLUX.2 PipeFusion pipeline needs a diffusers release with both FLUX.2 pipelines.
pipeline_flux2 = pytest.importorskip("xfuser.model_executor.pipelines.pipeline_flux2")

from diffusers import (  # noqa: E402
    FlowMatchEulerDiscreteScheduler,
    Flux2Pipeline,
    Flux2Transformer2DModel,
)

from xfuser.config.args import xFuserArgs  # noqa: E402
from xfuser.core.distributed import (  # noqa: E402
    get_runtime_state,
    init_distributed_environment,
    is_pipeline_first_stage,
    is_pipeline_last_stage,
)
from xfuser.core.distributed.parallel_state import (  # noqa: E402
    destroy_distributed_environment,
    destroy_model_parallel,
)

# A FLUX.2 pipeline without a VAE packs latents 16 pixels per token, so 64x64 is a 4x4
# token grid, and two pipeline patches of two token rows each.
HEIGHT = WIDTH = 64
IMAGE_TOKENS = (HEIGHT // 16) * (WIDTH // 16)
TEXT_TOKENS = 5
IN_CHANNELS = 8
JOINT_DIM = 12
INNER_DIM = 2 * 16


def _tiny_transformer():
    torch.manual_seed(0)
    return Flux2Transformer2DModel(
        in_channels=IN_CHANNELS,
        num_layers=2,
        num_single_layers=2,
        attention_head_dim=16,
        num_attention_heads=2,
        joint_attention_dim=JOINT_DIM,
        timestep_guidance_channels=32,
        axes_dims_rope=(4, 4, 4, 4),
    ).eval()


def _inputs(device):
    generator = torch.Generator().manual_seed(1)
    rows, cols = torch.meshgrid(torch.arange(HEIGHT // 16), torch.arange(WIDTH // 16), indexing="ij")
    img_ids = torch.zeros(IMAGE_TOKENS, 4)
    img_ids[:, 1] = rows.flatten()
    img_ids[:, 2] = cols.flatten()
    txt_ids = torch.zeros(TEXT_TOKENS, 4)
    txt_ids[:, 3] = torch.arange(TEXT_TOKENS)
    inputs = {
        "hidden_states": torch.randn(1, IMAGE_TOKENS, IN_CHANNELS, generator=generator),
        "encoder_hidden_states": torch.randn(1, TEXT_TOKENS, JOINT_DIM, generator=generator),
        "timestep": torch.tensor([0.4]),
        "guidance": torch.tensor([3.5]),
        "img_ids": img_ids,
        "txt_ids": txt_ids,
    }
    return {name: tensor.to(device) for name, tensor in inputs.items()}


def _stage_forward(transformer, inputs, token_slice, device):
    """Run this rank's stage on one image region, passing activations rank 0 -> 1."""
    image_tokens = token_slice.stop - token_slice.start
    hidden_states = inputs["hidden_states"][:, token_slice]
    encoder_hidden_states = inputs["encoder_hidden_states"]
    if not is_pipeline_first_stage():
        hidden_states = torch.empty(1, image_tokens, INNER_DIM, device=device)
        encoder_hidden_states = torch.empty(1, TEXT_TOKENS, INNER_DIM, device=device)
        dist.broadcast(hidden_states, src=0)
        dist.broadcast(encoder_hidden_states, src=0)

    ((hidden_states, encoder_hidden_states),) = transformer(
        hidden_states=hidden_states,
        encoder_hidden_states=encoder_hidden_states,
        timestep=inputs["timestep"],
        guidance=inputs["guidance"],
        img_ids=inputs["img_ids"][token_slice],
        txt_ids=inputs["txt_ids"],
        return_dict=False,
    )

    if is_pipeline_first_stage():
        dist.broadcast(hidden_states.contiguous(), src=0)
        dist.broadcast(encoder_hidden_states.contiguous(), src=0)
    return hidden_states


def _check_stages_against_diffusers(world_size, device):
    args = xFuserArgs()
    args.pipefusion_parallel_degree = world_size
    args.num_pipeline_patch = 2
    engine_config, _ = args.create_config()

    transformer = _tiny_transformer().to(device)
    reference = copy.deepcopy(transformer)
    pipe = pipeline_flux2.xFuserFlux2Pipeline(
        Flux2Pipeline(
            scheduler=FlowMatchEulerDiscreteScheduler(),
            vae=None,
            text_encoder=None,
            tokenizer=None,
            transformer=transformer,
        ),
        engine_config,
    )
    state = get_runtime_state()
    state.set_attention_backend("SDPA")
    state.set_input_parameters(
        height=HEIGHT,
        width=WIDTH,
        batch_size=1,
        num_inference_steps=2,
        max_condition_sequence_length=TEXT_TOKENS,
        split_text_embed_in_sp=False,
    )
    assert state.num_pipeline_patch == 2

    inputs = _inputs(device)
    with torch.no_grad():
        expected = reference(**inputs, return_dict=False)[0]

        state.set_patched_mode(patch_mode=False)
        full = _stage_forward(pipe.transformer, inputs, slice(0, IMAGE_TOKENS), device)

        state.set_patched_mode(patch_mode=True)
        bounds = state.pp_patches_token_start_idx_local
        patch_rounds = []
        for _ in range(2):
            patches = []
            for patch_idx in range(state.num_pipeline_patch):
                token_slice = slice(bounds[patch_idx], bounds[patch_idx + 1])
                patches.append(_stage_forward(pipe.transformer, inputs, token_slice, device))
                state.next_patch()
            patch_rounds.append(torch.cat(patches, dim=1))

    if is_pipeline_last_stage():
        torch.testing.assert_close(full, expected, rtol=1e-4, atol=1e-4)
        for patched in patch_rounds:
            torch.testing.assert_close(patched, expected, rtol=1e-4, atol=1e-4)


def _pipefusion_worker(rank, world_size, init_method):
    torch.cuda.set_device(rank)
    init_distributed_environment(
        rank=rank,
        world_size=world_size,
        local_rank=rank,
        distributed_init_method=init_method,
    )
    # Tear down only after a clean run. When one rank fails, the other is still
    # waiting for it in a collective, and tearing down would hang instead of letting
    # this rank report its error.
    _check_stages_against_diffusers(world_size, torch.device("cuda", rank))
    destroy_model_parallel()
    destroy_distributed_environment()


@pytest.mark.multi_gpu
def test_flux2_pipefusion_patches_match_full_forward(accelerator_ranks):
    accelerator_ranks(
        _pipefusion_worker,
        world_size=2,
        init_filename="flux2-pipefusion-init",
    )
