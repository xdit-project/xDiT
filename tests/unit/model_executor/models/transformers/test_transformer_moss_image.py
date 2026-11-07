"""CPU checks for the MOSS-Image DiT schedule, RoPE, and per-sample text length."""

import pytest
import torch

pytest.importorskip("diffusers")

from xfuser.model_executor.models.transformers.transformer_moss_image import (  # noqa: E402
    MOSSImageScheduler,
    MOSSImageTransformer,
    validate_skip_layers,
)
from xfuser.model_executor.pipelines.pipeline_moss_image import (  # noqa: E402
    combine_moss_guidance,
    skip_window_active,
    xFuserMOSSImagePipeline,
)


def _tiny():
    torch.manual_seed(0)
    return MOSSImageTransformer(
        dim=64,
        ffn_dim=128,
        num_heads=4,
        num_layers=2,
        text_dim=32,
        freq_dim=16,
        latent_channels=8,
    )


def test_eight_step_schedule_snaps_onto_the_training_grid():
    scheduler = MOSSImageScheduler()
    scheduler.set_timesteps(8)
    expected = [
        1.0,
        0.9679999947547913,
        0.9259999990463257,
        0.8709999918937683,
        0.7910000085830688,
        0.6710000038146973,
        0.46300002932548523,
        0.024000000208616257,
    ]
    assert scheduler.sigmas[:-1].tolist() == pytest.approx(expected)


def test_shift_changes_the_schedule_and_bad_steps_are_refused():
    low = MOSSImageScheduler()
    high = MOSSImageScheduler()
    high.register_to_config(shift=5.0)
    low.register_to_config(shift=1.0)
    low.set_timesteps(8)
    high.set_timesteps(8)
    assert not torch.equal(low.sigmas, high.sigmas)
    with pytest.raises(ValueError, match="positive"):
        MOSSImageScheduler().set_timesteps(0)


def test_skip_layers_must_be_in_range():
    assert validate_skip_layers(None, 4) == frozenset()
    assert validate_skip_layers((0, 3), 4) == frozenset({0, 3})
    with pytest.raises(ValueError, match="uncond_skip_layers"):
        validate_skip_layers((4,), 4, "uncond_skip_layers")


def test_rope_frequencies_stay_fp32_when_the_module_is_cast():
    rope = _tiny().rope
    rope.to(dtype=torch.bfloat16)
    assert rope.bands.dtype == torch.float32
    table = rope(height=2, width=2, text_length=3, device=torch.device("cpu"))
    assert table.shape == (1, 7, 16)
    assert table.dtype == torch.float32


def test_masked_text_tokens_do_not_change_the_sample():
    model = _tiny().eval()
    hidden = torch.randn(1, 8, 2, 2)
    text = torch.randn(1, 4, 32)
    timestep = torch.tensor(500.0)
    with torch.no_grad():
        short = model(hidden, timestep, text[:, :2], return_dict=False)[0]
        mask = torch.tensor([[1, 1, 0, 0]], dtype=torch.bool)
        padded = model(hidden, timestep, text, encoder_attention_mask=mask, return_dict=False)[0]
    assert short.shape == (1, 8, 2, 2)
    assert torch.allclose(short, padded)


def test_skipping_a_block_changes_the_prediction():
    model = _tiny().eval()
    hidden = torch.randn(1, 8, 2, 2)
    text = torch.randn(1, 2, 32)
    timestep = torch.tensor(500.0)
    with torch.no_grad():
        full = model(hidden, timestep, text, return_dict=False)[0]
        skipped = model(hidden, timestep, text, skip_layers=(0,), return_dict=False)[0]
    assert not torch.allclose(full, skipped)


def test_recommended_skip_window_leaves_the_first_step_dense():
    assert skip_window_active(0, 50, 0.01, 1.0) is False
    assert skip_window_active(1, 50, 0.01, 1.0) is True
    assert skip_window_active(0, 50, 0.0, 1.0) is True


def test_guidance_mixes_the_two_predictions_in_fp32():
    conditional = torch.tensor([2.0])
    uncond = torch.tensor([1.0])
    skipped = torch.tensor([0.5])
    mixed = combine_moss_guidance(conditional, uncond, 3.0, skipped, 0.0, do_cfg=True, slg_active=False)
    assert mixed.item() == pytest.approx(4.0)
    corrected = combine_moss_guidance(conditional, uncond, 3.0, skipped, 2.0, do_cfg=True, slg_active=True)
    assert corrected.item() == pytest.approx(7.0)


def test_pipeline_skips_the_negative_block_only_inside_the_window():
    transformer = _tiny().eval()
    seen = []
    original = transformer.forward

    def wrapped(*args, **kwargs):
        seen.append(kwargs.get("skip_layers"))
        return original(*args, **kwargs)

    transformer.forward = wrapped
    pipe = xFuserMOSSImagePipeline(transformer, MOSSImageScheduler())
    embeds = torch.randn(1, 3, 32)
    mask = torch.ones(1, 3, dtype=torch.bool)
    output = pipe(
        prompt_embeds=embeds,
        prompt_attention_mask=mask,
        negative_prompt_embeds=embeds,
        negative_prompt_attention_mask=mask,
        height=32,
        width=32,
        num_inference_steps=2,
        guidance_scale=3.0,
        uncond_skip_layers=(0,),
        slg_start=0.01,
        slg_stop=1.0,
        output_type="latent",
        generator=torch.Generator().manual_seed(0),
    )
    assert seen == [None, None, None, frozenset({0})]
    assert output.images.shape == (1, 8, 2, 2)
    assert output.images.dtype == torch.float32


def test_pipeline_adds_the_positive_skip_pass_when_requested():
    transformer = _tiny().eval()
    seen = []
    original = transformer.forward

    def wrapped(*args, **kwargs):
        seen.append(kwargs.get("skip_layers"))
        return original(*args, **kwargs)

    transformer.forward = wrapped
    pipe = xFuserMOSSImagePipeline(transformer, MOSSImageScheduler())
    embeds = torch.randn(1, 2, 32)
    mask = torch.ones(1, 2, dtype=torch.bool)
    pipe(
        prompt_embeds=embeds,
        prompt_attention_mask=mask,
        negative_prompt_embeds=embeds,
        negative_prompt_attention_mask=mask,
        height=32,
        width=32,
        num_inference_steps=1,
        guidance_scale=3.0,
        uncond_skip_layers=(0,),
        slg_layers=(1,),
        slg_scale=2.0,
        slg_start=0.0,
        output_type="latent",
    )
    assert seen == [None, frozenset({0}), frozenset({1})]


def test_pipeline_rejects_a_resolution_the_vae_cannot_pack():
    pipe = xFuserMOSSImagePipeline(_tiny(), MOSSImageScheduler())
    embeds = torch.randn(1, 2, 32)
    with pytest.raises(ValueError, match="multiples of 16"):
        pipe(prompt_embeds=embeds, height=24, width=32, output_type="latent")
