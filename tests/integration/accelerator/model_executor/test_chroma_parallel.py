"""Chroma under Ulysses and CFG parallelism matches stock diffusers.

Each rank also runs the unmodified diffusers model on the full problem, so the
reference needs no communication. The transformer case covers text and image
lengths that do and do not split evenly across ranks (the uneven case exercises the
sequence-parallel padding that the attention bias must hide). The pipeline case runs
the guidance branches on separate CFG ranks and compares the final latents.
Tiny random weights; nothing is downloaded.
"""

import pytest

pytestmark = pytest.mark.multi_gpu

_TINY_CONFIG = dict(
    patch_size=1,
    in_channels=8,
    num_layers=2,
    num_single_layers=2,
    attention_head_dim=16,
    num_attention_heads=4,
    joint_attention_dim=32,
    axes_dims_rope=(4, 6, 6),
    approximator_num_channels=16,
    approximator_hidden_dim=32,
    approximator_layers=1,
)


def _models(device):
    import torch
    from diffusers import ChromaTransformer2DModel

    from xfuser.model_executor.models.transformers.transformer_chroma import (
        xFuserChromaTransformer2DWrapper,
    )

    torch.manual_seed(0)
    reference = ChromaTransformer2DModel(**_TINY_CONFIG).eval()
    wrapper = xFuserChromaTransformer2DWrapper.from_config(reference.config).eval()
    wrapper.load_state_dict(reference.state_dict())
    return reference.to(device), wrapper.to(device)


def _transformer_inputs(device, num_txt, height, width):
    import torch

    generator = torch.Generator().manual_seed(1)
    img_ids = torch.zeros(height, width, 3)
    img_ids[..., 1] = torch.arange(height)[:, None]
    img_ids[..., 2] = torch.arange(width)[None, :]
    text_mask = (torch.arange(num_txt)[None] <= 2).float()
    inputs = dict(
        hidden_states=torch.randn(1, height * width, 8, generator=generator),
        encoder_hidden_states=torch.randn(1, num_txt, 32, generator=generator),
        timestep=torch.tensor([0.6]),
        img_ids=img_ids.reshape(-1, 3),
        txt_ids=torch.zeros(num_txt, 3),
        attention_mask=torch.cat([text_mask, torch.ones(1, height * width, dtype=torch.bool)], dim=1),
    )
    return {name: tensor.to(device) for name, tensor in inputs.items()}


def _check_transformer(device):
    import torch

    reference, wrapper = _models(device)
    # Divisible by every degree used here, then lengths that need padding.
    for num_txt, height, width in [(8, 4, 4), (7, 3, 5)]:
        inputs = _transformer_inputs(device, num_txt, height, width)
        with torch.no_grad():
            expected = reference(**inputs, return_dict=False)[0]
            actual = wrapper(**inputs, return_dict=False)[0]
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)


def _check_pipeline(device):
    import torch
    from diffusers import ChromaPipeline, FlowMatchEulerDiscreteScheduler

    from xfuser.model_executor.pipelines.pipeline_chroma import xFuserChromaPipeline

    reference, wrapper = _models(device)
    generator = torch.Generator().manual_seed(3)
    prompt_embeds = torch.randn(1, 6, 32, generator=generator).to(device)
    negative_prompt_embeds = torch.randn(1, 6, 32, generator=generator).to(device)
    call = dict(
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        # Different valid lengths, so a swapped branch would be caught.
        prompt_attention_mask=(torch.arange(6, device=device) <= 3).float()[None],
        negative_prompt_attention_mask=(torch.arange(6, device=device) <= 1).float()[None],
        height=64,
        width=64,
        num_inference_steps=3,
        guidance_scale=4.0,
        output_type="latent",
    )

    def run(pipeline_cls, transformer):
        pipe = pipeline_cls(
            scheduler=FlowMatchEulerDiscreteScheduler(),
            vae=None,
            text_encoder=None,
            tokenizer=None,
            transformer=transformer,
        )
        pipe.set_progress_bar_config(disable=True)
        return pipe(**call, generator=torch.Generator(device=device).manual_seed(5)).images

    expected = run(ChromaPipeline, reference)
    actual = run(xFuserChromaPipeline, wrapper)
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)


def _chroma_worker(rank, world_size, init_method, ulysses_degree, cfg_degree, case):
    import torch

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

    torch.cuda.set_device(rank)
    init_distributed_environment(
        rank=rank,
        world_size=world_size,
        local_rank=rank,
        distributed_init_method=init_method,
    )
    initialize_model_parallel(
        ulysses_degree=ulysses_degree,
        classifier_free_guidance_degree=cfg_degree,
    )
    try:
        args = xFuserArgs(
            attention_backend="SDPA",
            ulysses_degree=ulysses_degree,
            use_cfg_parallel=cfg_degree == 2,
        )
        engine_config, _ = args.create_config()
        initialize_runtime_state(engine_config=engine_config)
        get_runtime_state().set_attention_backend("SDPA")
        device = torch.device(f"cuda:{rank}")
        if case == "transformer":
            _check_transformer(device)
        else:
            _check_pipeline(device)
    finally:
        destroy_model_parallel()
        destroy_distributed_environment()


@pytest.mark.parametrize("ulysses_degree", [2, 4])
def test_ulysses_transformer_matches_diffusers(ulysses_degree, accelerator_ranks):
    accelerator_ranks(
        _chroma_worker,
        world_size=ulysses_degree,
        init_filename=f"chroma-ulysses-{ulysses_degree}",
        args=(ulysses_degree, 1, "transformer"),
    )


@pytest.mark.parametrize("ulysses_degree", [1, 2])
def test_cfg_parallel_pipeline_matches_diffusers(ulysses_degree, accelerator_ranks):
    accelerator_ranks(
        _chroma_worker,
        world_size=2 * ulysses_degree,
        init_filename=f"chroma-cfg-u{ulysses_degree}",
        args=(ulysses_degree, 2, "pipeline"),
    )
