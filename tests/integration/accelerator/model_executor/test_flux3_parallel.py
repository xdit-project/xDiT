"""FLUX.3 sequence- and CFG-parallel parity against stock diffusers on tiny random transformers."""

import pytest

pytest.importorskip(
    "diffusers.models.transformers.transformer_flux3",
    reason="installed diffusers does not include FLUX.3",
)

pytestmark = pytest.mark.multi_gpu

# SDPA refuses ring attention; the efficient kernel serves Ulysses and ring alike.
BACKEND = "SDPA_EFFICIENT"
NUM_HEADS = 24
# 7 text tokens, a 3x5 image and two references (2x3 and 1x7, 13 tokens): every stream
# needs SP padding at degrees 2, 4 and 8. The joint sequence needs it at 4 and 8 without
# references (22 tokens) and at every degree with them (35 tokens).
TEXT_LEN = 7
IMAGE_GRID = (3, 5)
REFERENCE_GRIDS = ((2, 3), (1, 7))
# Shapes that divide by 2, 4 and 8, so no stream or joint sequence is padded.
EVEN_SHAPES = (16, (4, 8), ((2, 4), (2, 4)))
LAYOUTS = {"padded": (TEXT_LEN, IMAGE_GRID, REFERENCE_GRIDS), "even": EVEN_SHAPES}


def _tiny_model(device, flow_map=True):
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
            flow_map=flow_map,
        )
        .to(device)
        .eval()
    )


def _txt_ids(length):
    import torch

    return torch.stack([torch.zeros(length)] * 3 + [torch.arange(length).float()], dim=-1)


def _ids(height, width, l_coord):
    import torch

    grid = torch.meshgrid(
        torch.arange(1), torch.arange(height), torch.arange(width), torch.tensor([l_coord]), indexing="ij"
    )
    return torch.stack(grid, dim=-1).reshape(-1, 4).float()


def _inputs(device, layout, with_references):
    import torch

    text_len, image_grid, reference_grids = LAYOUTS[layout]
    generator = torch.Generator().manual_seed(1)
    image_len = image_grid[0] * image_grid[1]
    inputs = {
        "hidden_states": torch.randn(1, image_len, 8, generator=generator),
        "encoder_hidden_states": torch.randn(1, text_len, 32, generator=generator),
        "timestep": torch.tensor([0.7]),
        "target_timestep": torch.tensor([0.5]),
        "img_ids": _ids(*image_grid, 0),
        "txt_ids": _txt_ids(text_len),
    }
    if with_references:
        reference_len = sum(h * w for h, w in reference_grids)
        inputs["image_cond_hidden_states"] = torch.randn(1, reference_len, 8, generator=generator)
        inputs["image_cond_ids"] = torch.cat(
            [_ids(h, w, -(1 + index)) for index, (h, w) in enumerate(reference_grids)], dim=0
        )
    return {name: value.to(device) for name, value in inputs.items()}


def _setup(rank, world_size, init_method, ulysses_degree, ring_degree=1, cfg_degree=1):
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
    initialize_model_parallel(
        classifier_free_guidance_degree=cfg_degree, ulysses_degree=ulysses_degree, ring_degree=ring_degree
    )
    engine_config, _ = xFuserArgs(
        attention_backend=BACKEND,
        ulysses_degree=ulysses_degree,
        ring_degree=ring_degree,
        use_cfg_parallel=cfg_degree > 1,
    ).create_config()
    initialize_runtime_state(engine_config=engine_config)
    get_runtime_state().set_attention_backend(BACKEND)
    return torch.device(f"cuda:{rank}")


def _wrapper_of(stock, device):
    from xfuser.model_executor.models.transformers.transformer_flux3 import xFuserFlux3Transformer2DWrapper

    wrapper = xFuserFlux3Transformer2DWrapper.from_config(stock.config).to(device).eval()
    wrapper.load_state_dict(stock.state_dict())
    return wrapper


def _sp_worker(rank, world_size, init_method, ulysses_degree, ring_degree):
    import torch

    from xfuser.core.distributed.parallel_state import destroy_distributed_environment, destroy_model_parallel

    try:
        device = _setup(rank, world_size, init_method, ulysses_degree, ring_degree)
        stock = _tiny_model(device)
        wrapper = _wrapper_of(stock, device)

        for layout in LAYOUTS:
            for with_references in (False, True):
                inputs = _inputs(device, layout, with_references)
                with torch.no_grad():
                    reference = stock(**inputs).sample
                # The text stream either runs whole on every rank or is sharded.
                for replicate_text in (True, False):
                    wrapper.replicate_text_max_tokens = 512 if replicate_text else 0
                    with torch.no_grad():
                        actual = wrapper(**inputs).sample
                    torch.testing.assert_close(
                        actual,
                        reference,
                        rtol=1e-4,
                        atol=1e-4,
                        msg=lambda m: (
                            f"Ulysses {ulysses_degree} x ring {ring_degree}, {layout} layout, "
                            f"references={with_references}, replicated text={replicate_text}: {m}"
                        ),
                    )
    finally:
        destroy_model_parallel()
        destroy_distributed_environment()


@pytest.mark.parametrize(
    ("ulysses_degree", "ring_degree"),
    [(2, 1), (4, 1), (8, 1), (1, 2), (2, 2), (2, 4), (4, 2)],
)
def test_sequence_parallel_matches_single_gpu_diffusers(accelerator_ranks, ulysses_degree, ring_degree):
    accelerator_ranks(
        _sp_worker,
        world_size=ulysses_degree * ring_degree,
        timeout=300,
        args=(ulysses_degree, ring_degree),
    )


def _cfg_worker(rank, world_size, init_method):
    import torch
    from diffusers.guiders import ClassifierFreeGuidance
    from diffusers.modular_pipelines.flux3.denoise import Flux3LoopDenoiser
    from diffusers.modular_pipelines.modular_pipeline import BlockState

    from xfuser.core.distributed.parallel_state import destroy_distributed_environment, destroy_model_parallel
    from xfuser.model_executor.pipelines.guiders import xFuserCFGParallelGuidance

    try:
        device = _setup(rank, world_size, init_method, ulysses_degree=world_size // 2, cfg_degree=2)
        stock = _tiny_model(device, flow_map=False)
        wrapper = _wrapper_of(stock, device)

        # The branches differ in text length, as a prompt and an empty negative prompt do,
        # so they cannot simply be batched.
        generator = torch.Generator().manual_seed(1)
        negative_len = TEXT_LEN + 2
        fields = {
            "prompt_embeds": torch.randn(1, TEXT_LEN, 32, generator=generator),
            "negative_prompt_embeds": torch.randn(1, negative_len, 32, generator=generator),
            "txt_ids": _txt_ids(TEXT_LEN),
            "negative_txt_ids": _txt_ids(negative_len),
            "img_ids": _ids(*IMAGE_GRID, 0),
        }
        fields = {name: value.to(device) for name, value in fields.items()}
        latents = torch.randn(1, IMAGE_GRID[0] * IMAGE_GRID[1], 8, generator=generator).to(device)

        def noise_pred(transformer, guider):
            # The stock denoise step, as in the pipeline: only the guider differs.
            components = type("Components", (), {})()
            components.transformer = transformer
            components.guider = guider
            state = BlockState(latents=latents, denoiser_input_fields=dict(fields), num_inference_steps=2)
            _, state = Flux3LoopDenoiser()(components, state, i=0, t=torch.tensor(700.0, device=device))
            return state.noise_pred

        with torch.no_grad():
            reference = noise_pred(stock, ClassifierFreeGuidance(guidance_scale=3.5))
            actual = noise_pred(wrapper, xFuserCFGParallelGuidance(guidance_scale=3.5))
        torch.testing.assert_close(actual, reference, rtol=1e-4, atol=1e-4)
    finally:
        destroy_model_parallel()
        destroy_distributed_environment()


@pytest.mark.parametrize("world_size", [2, 4])
def test_cfg_parallel_matches_single_gpu_diffusers(accelerator_ranks, world_size):
    accelerator_ranks(_cfg_worker, world_size=world_size, timeout=300)
