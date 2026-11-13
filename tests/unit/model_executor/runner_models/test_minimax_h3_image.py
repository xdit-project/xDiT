import pytest
import torch

pytest.importorskip(
    "diffusers.models.transformers.transformer_minimax_h3_image",
    reason="installed diffusers does not include MiniMax-H3-Image",
)


def _tiny_config():
    return {
        "num_attention_heads": 2,
        "attention_head_dim": 16,
        "hidden_size": 24,
        "num_layers": 2,
        "num_refiner_layers": 2,
        "ffn_dim": 32,
        "in_channels": 4,
        "patch_size": (1, 2, 2),
        "text_dim": 8,
        "freq_dim": 8,
        "time_embed_hidden_dim": 24,
        "time_embed_dim": 16,
        "rope_freq_dim": 2,
        "num_pdd_heads": 3,
    }


def _tiny_inputs(text_tokens=5, reference_tokens=0, image_tokens=48, pdd_step_index=0):
    """A packed [text | reference rows | generated rows] layout, as the pipeline builds it."""
    sequence_length = text_tokens + reference_tokens + image_tokens
    text_indices = torch.arange(text_tokens)
    image_indices = torch.arange(text_tokens, sequence_length)

    token_tags = torch.zeros(sequence_length, dtype=torch.long)
    token_tags[text_indices] = 1
    # An edit holds its reference rows at their own timestep, the second one.
    timestep_indices = torch.zeros(sequence_length, dtype=torch.long)
    timestep_indices[text_tokens : text_tokens + reference_tokens] = 1
    timestep = torch.tensor([0.25, 0.999]) if reference_tokens else torch.tensor([0.25])

    position_ids = torch.zeros(sequence_length, 3)
    position_ids[:text_tokens, 0] = torch.arange(text_tokens, dtype=torch.float32)
    grid = torch.stack(torch.meshgrid(torch.arange(6.0), torch.arange(8.0), indexing="ij"), -1).flatten(0, 1)
    position_ids[sequence_length - image_tokens :, 1:] = grid[:image_tokens]
    position_ids[sequence_length - image_tokens :, 0] = text_tokens + 1
    if reference_tokens:
        position_ids[text_tokens : text_tokens + reference_tokens, 0] = text_tokens
        position_ids[text_tokens : text_tokens + reference_tokens, 1:] = grid[:reference_tokens]

    generator = torch.Generator().manual_seed(0)
    return {
        "hidden_states": torch.randn(1, reference_tokens + image_tokens, 16, generator=generator),
        "encoder_hidden_states": torch.randn(1, text_tokens, 8, generator=generator),
        "timestep": timestep,
        "timestep_indices": timestep_indices,
        "token_tags": token_tags,
        "position_ids": position_ids,
        "image_indices": image_indices,
        "text_indices": text_indices,
        "pdd_step_index": pdd_step_index,
    }


@pytest.fixture
def single_rank(monkeypatch):
    """Run the wrapper as one rank of SDPA attention, without a process group."""
    from xfuser.core.attention.spec import AttentionBackendType
    from xfuser.model_executor.models.transformers import transformer_minimax_h3, transformer_minimax_h3_image

    class _RuntimeState:
        attention_backend = AttentionBackendType.SDPA

        def has_attention_schedule(self):
            return False

        def increment_step_counter(self):
            pass

    runtime_state = _RuntimeState()
    for module in (transformer_minimax_h3, transformer_minimax_h3_image):
        monkeypatch.setattr(module, "get_ulysses_parallel_world_size", lambda: 1)
        monkeypatch.setattr(module, "get_runtime_state", lambda: runtime_state)
    monkeypatch.setattr(transformer_minimax_h3_image, "get_ulysses_parallel_rank", lambda: 0)
    monkeypatch.setattr("xfuser.core.distributed.get_runtime_state", lambda: runtime_state)
    monkeypatch.setattr("xfuser.model_executor.layers.usp.get_runtime_state", lambda: runtime_state)


def _tiny_models():
    from diffusers import MiniMaxH3ImageTransformer2DModel

    from xfuser.core.attention.spec import AttentionBackendType
    from xfuser.model_executor.models.transformers.transformer_minimax_h3_image import (
        xFuserMiniMaxH3ImageTransformer2DWrapper,
    )

    torch.manual_seed(0)
    base = MiniMaxH3ImageTransformer2DModel(**_tiny_config()).eval()
    wrapped = xFuserMiniMaxH3ImageTransformer2DWrapper(
        **_tiny_config(), attention_backend=AttentionBackendType.SDPA
    ).eval()
    wrapped.load_state_dict(base.state_dict())
    return base, wrapped


# 53 and 66 rows do not align to 64, so the wrapper pads them; 64 needs no pad.
@pytest.mark.parametrize(
    "layout",
    [
        {"text_tokens": 5},
        {"text_tokens": 16},
        {"text_tokens": 6, "reference_tokens": 12},
    ],
    ids=["t2i-padded", "t2i-aligned", "edit"],
)
def test_wrapper_matches_diffusers_on_one_rank(single_rank, layout):
    base, wrapped = _tiny_models()

    for step in range(_tiny_config()["num_pdd_heads"]):
        inputs = _tiny_inputs(**layout, pdd_step_index=step)
        with torch.no_grad():
            expected = base(**inputs).sample
            torch.testing.assert_close(wrapped(**inputs).sample, expected)
    wrapped.fuse_qkv_projections()
    with torch.no_grad():
        torch.testing.assert_close(wrapped(**inputs).sample, expected)


def test_wrapper_projects_each_timestep_modulation_once(single_rank):
    """The AdaLN tables depend on the timestep alone, so a request repeating the grid reuses them."""
    base, wrapped = _tiny_models()
    projections = []
    wrapped.time_embedder.register_forward_hook(lambda *args: projections.append(None))

    with torch.no_grad():
        for _ in range(2):
            for step in range(_tiny_config()["num_pdd_heads"]):
                inputs = _tiny_inputs(pdd_step_index=step)
                inputs["timestep"] = torch.tensor([1.0 - 0.25 * step])
                expected = base(**inputs).sample
                torch.testing.assert_close(wrapped(**inputs).sample, expected, rtol=0, atol=0)

    assert len(projections) == _tiny_config()["num_pdd_heads"]


def test_wrapper_projects_the_modulation_under_autograd(single_rank):
    _, wrapped = _tiny_models()
    inputs = _tiny_inputs()
    with torch.no_grad():
        wrapped(**inputs)

    wrapped(**inputs).sample.sum().backward()

    assert wrapped.transformer_blocks[0].adaln_proj.linear.weight.grad is not None
    assert wrapped.norm_out.linear.weight.grad is not None


def test_wrapper_keeps_the_checkpoint_config_through_from_config(single_rank):
    from xfuser.model_executor.models.transformers.transformer_minimax_h3_image import (
        xFuserMiniMaxH3ImageTransformer2DWrapper,
    )

    base, _ = _tiny_models()
    rebuilt = xFuserMiniMaxH3ImageTransformer2DWrapper.from_config(base.config)

    assert len(rebuilt.pdd_proj_out) == _tiny_config()["num_pdd_heads"]
    for key, value in base.config.items():
        if not key.startswith("_"):
            assert rebuilt.config[key] == value, key


def test_compiled_body_serves_every_step_with_one_graph(single_rank):
    """The output head index is a Python int; it must stay out of the compiled region."""
    from torch._dynamo.testing import CompileCounter

    _, wrapped = _tiny_models()
    counter = CompileCounter()
    torch._dynamo.reset()
    wrapped._forward_body = torch.compile(wrapped._forward_body, backend=counter, fullgraph=True)

    with torch.no_grad():
        for step in range(_tiny_config()["num_pdd_heads"]):
            wrapped(**_tiny_inputs(pdd_step_index=step))
    torch._dynamo.reset()

    assert counter.frame_count == 1


def test_runner_is_registered():
    from xfuser.model_executor.models.runner_models.base_model import MODEL_REGISTRY

    assert "MiniMaxAI/MiniMax-H3-Image" in MODEL_REGISTRY
    assert "MiniMax-H3-Image" in MODEL_REGISTRY


@pytest.fixture
def runner_env(monkeypatch):
    # Runner construction logs through runner_utils.log, which reads these.
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")


@pytest.mark.parametrize("ulysses_degree", [1, 2, 4, 8])
def test_runner_accepts_ulysses_degrees_that_divide_heads_and_alignment(runner_env, ulysses_degree):
    from xfuser import xFuserArgs
    from xfuser.model_executor.models.runner_models.minimax_h3_image import xFuserMiniMaxH3ImageModel

    xFuserMiniMaxH3ImageModel(
        xFuserArgs(model="MiniMax-H3-Image", ulysses_degree=ulysses_degree, text_encoder_tp_degree=ulysses_degree)
    )


@pytest.mark.parametrize(
    "options, match",
    [
        ({"ulysses_degree": 7}, "Ulysses degree"),
        ({"ulysses_degree": 2, "text_encoder_tp_degree": 4}, "text_encoder_tp_degree"),
        ({"use_cfg_parallel": True}, "cfg"),
        ({"use_parallel_vae": True, "vae_parallel_size": 2}, "vae_parallel_size"),
        ({"ulysses_degree": 2, "enable_model_cpu_offload": True}, "single-GPU"),
        ({"enable_sequential_cpu_offload": True}, "enable_model_cpu_offload"),
        ({"batch_size": 2}, "batching"),
    ],
)
def test_runner_refuses_what_it_cannot_run(runner_env, options, match):
    from xfuser import xFuserArgs
    from xfuser.model_executor.models.runner_models.minimax_h3_image import xFuserMiniMaxH3ImageModel

    with pytest.raises(ValueError, match=match):
        xFuserMiniMaxH3ImageModel(xFuserArgs(model="MiniMax-H3-Image", **options))


@pytest.mark.parametrize(
    "args, match",
    [
        ({"negative_prompt": "blurry"}, "negative prompt"),
        ({"num_inference_steps": 8}, "6-step"),
        ({"prompt": ["a", "b"]}, "one prompt"),
    ],
)
def test_runner_refuses_requests_off_the_distilled_contract(runner_env, args, match):
    from xfuser import xFuserArgs
    from xfuser.model_executor.models.runner_models.minimax_h3_image import xFuserMiniMaxH3ImageModel

    model = xFuserMiniMaxH3ImageModel(xFuserArgs(model="MiniMax-H3-Image"))
    request = {"prompt": "a cat", "height": 2048, "width": 2048, "num_inference_steps": 6, **args}
    with pytest.raises(ValueError, match=match):
        model._validate_args(request)


def _tiny_vae():
    from diffusers import AutoencoderKLMiniMaxH3Image

    torch.manual_seed(0)
    vae = AutoencoderKLMiniMaxH3Image(
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
    ).eval()
    # 4x compression: 16-pixel tiles with an 8-pixel overlap cut a 40x56 image
    # into a 4x6 grid whose overlaps are not all equal.
    vae.enable_tiling(16, 16, 8, 8)
    return vae


def test_batched_tiled_decode_matches_the_vae_loop():
    from xfuser.model_executor.models.runner_models import minimax_h3_image

    vae = _tiny_vae()
    latents = torch.randn(1, 4, 10, 14, generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        expected = vae.decode(latents).sample
        minimax_h3_image.install_minimax_h3_image_tiled_decode(vae)
        actual = vae.decode(latents).sample

    assert actual.shape == expected.shape == (1, 3, 40, 56)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


def test_tiled_decode_leaves_untiled_decode_alone():
    from xfuser.model_executor.models.runner_models.minimax_h3_image import install_minimax_h3_image_tiled_decode

    vae = _tiny_vae()
    vae.disable_tiling()
    latents = torch.randn(1, 4, 6, 6, generator=torch.Generator().manual_seed(2))
    with torch.no_grad():
        expected = vae.decode(latents).sample
        install_minimax_h3_image_tiled_decode(vae)
        torch.testing.assert_close(vae.decode(latents).sample, expected, rtol=0, atol=0)


class _OpaqueWrapper(torch.nn.Module):
    """Wraps a module the way FSDP does: the inner one as ``module``, forward(*args, **kwargs)."""

    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


def test_the_denoise_step_sees_the_layout_fields_of_a_wrapped_transformer(runner_env, single_rank):
    """The denoise step passes only the fields the transformer's forward names; behind
    FSDP's wrapper it would pass none of token_tags, position_ids and the row indices."""
    import inspect
    from types import SimpleNamespace

    from xfuser import xFuserArgs
    from xfuser.model_executor.models.runner_models.minimax_h3_image import xFuserMiniMaxH3ImageModel

    model = xFuserMiniMaxH3ImageModel(xFuserArgs(model="MiniMax-H3-Image"))
    _, wrapped = _tiny_models()
    model.pipe = SimpleNamespace(transformer=_OpaqueWrapper(wrapped))

    model._expose_transformer_signature()

    transformer = model.pipe.transformer
    fields = {"token_tags", "position_ids", "image_indices", "text_indices", "pdd_step_index"}
    assert fields <= set(inspect.signature(transformer.forward).parameters)
    inputs = _tiny_inputs()
    with torch.no_grad():
        torch.testing.assert_close(transformer(**inputs).sample, wrapped(**inputs).sample)
