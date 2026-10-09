"""HunyuanDiT's last stage sends the predicted noise to the scheduler."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("diffusers")

from xfuser.model_executor.pipelines import pipeline_hunyuandit  # noqa: E402
from xfuser.model_executor.pipelines.pipeline_hunyuandit import xFuserHunyuanDiTPipeline  # noqa: E402


@pytest.mark.parametrize("first_stage", [True, False], ids=["single_stage", "downstream_stage"])
@pytest.mark.parametrize("guidance_scale, expected_offset", [(0.0, 0), (1.0, 0), (4.5, 72)])
def test_last_stage_returns_noise_prediction(monkeypatch, first_stage, guidance_scale, expected_offset):
    monkeypatch.setattr(pipeline_hunyuandit, "is_pipeline_first_stage", lambda: first_stage)
    monkeypatch.setattr(pipeline_hunyuandit, "is_pipeline_last_stage", lambda: True)
    monkeypatch.setattr(pipeline_hunyuandit, "get_classifier_free_guidance_world_size", lambda: 1)

    # The third-party transformer returns noise and learned variance channels.
    # Keep both distinct from the input so returning the input or variance fails.
    prediction_batch = 2 if guidance_scale > 1 else 1
    noise = torch.arange(16 * prediction_batch, dtype=torch.float32).reshape(prediction_batch, 4, 2, 2)
    variance = torch.full_like(noise, 1000)
    transformer = Mock(return_value=(torch.cat((noise, variance), dim=1),))
    scheduler = SimpleNamespace(scale_model_input=lambda latents, timestep: latents * 2)
    wrapper = object.__new__(xFuserHunyuanDiTPipeline)
    wrapper.module = SimpleNamespace(transformer=transformer, scheduler=scheduler)
    wrapper._guidance_scale = guidance_scale

    # A downstream stage already receives the expanded activation batch.
    latents = torch.full((1 if first_stage else prediction_batch, 4, 2, 2), -4.0)

    actual = wrapper._backbone_forward(
        latents=latents,
        prompt_embeds=torch.zeros(1, 2, 8),
        prompt_attention_mask=torch.ones(1, 2),
        prompt_embeds_2=torch.zeros(1, 2, 8),
        prompt_attention_mask_2=torch.ones(1, 2),
        add_time_ids=torch.zeros(1, 6),
        style=torch.zeros(1, dtype=torch.long),
        image_rotary_emb=(torch.zeros(1, 8), torch.zeros(1, 8)),
        t=torch.tensor(1),
        device=torch.device("cpu"),
        guidance_scale=guidance_scale,
        guidance_rescale=0.0,
        skips=None if first_stage else torch.zeros(3, prediction_batch, 4, 8),
    )

    # The unconditional prediction is 0..15 and the text prediction 16..31;
    # at guidance 4.5, each noise element therefore increases by 72.
    expected = torch.arange(16, dtype=torch.float32).reshape(1, 4, 2, 2) + expected_offset
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
