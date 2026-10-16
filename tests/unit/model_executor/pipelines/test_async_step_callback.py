"""PipeFusion rejects callbacks it cannot apply consistently across stages."""

from types import SimpleNamespace

import pytest
import torch

from xfuser.model_executor.pipelines import base_pipeline
from xfuser.model_executor.pipelines.base_pipeline import xFuserPipelineBaseWrapper
from xfuser.model_executor.pipelines.pipeline_flux import xFuserFluxPipeline
from xfuser.model_executor.pipelines.pipeline_flux2 import xFuserFlux2Pipeline
from xfuser.model_executor.pipelines.pipeline_stable_diffusion_3 import (
    xFuserStableDiffusion3Pipeline,
)


@pytest.mark.parametrize(
    ("pipeline_cls", "kwargs"),
    [
        (
            xFuserFluxPipeline,
            {
                "latents": None,
                "prompt_embeds": None,
                "pooled_prompt_embeds": None,
                "text_ids": None,
                "latent_image_ids": None,
                "guidance": None,
            },
        ),
        (
            xFuserFlux2Pipeline,
            {
                "latents": None,
                "prompt_embeds": None,
                "text_ids": None,
                "latent_image_ids": None,
                "guidance": None,
            },
        ),
        (
            xFuserStableDiffusion3Pipeline,
            {
                "latents": None,
                "prompt_embeds": None,
                "pooled_prompt_embeds": None,
            },
        ),
    ],
)
def test_async_pipefusion_rejects_step_callbacks_before_communication(
    pipeline_cls,
    kwargs,
):
    pipeline = object.__new__(pipeline_cls)

    with pytest.raises(NotImplementedError, match="cannot be applied consistently"):
        pipeline._async_pipeline(
            **kwargs,
            timesteps=[1],
            num_warmup_steps=0,
            progress_bar=None,
            callback_on_step_end=lambda *_args: None,
        )


def test_async_callback_is_rejected_before_warmup_selection(monkeypatch):
    monkeypatch.setattr(
        base_pipeline,
        "get_pipeline_parallel_world_size",
        lambda: 4,
    )

    pipeline = SimpleNamespace(
        _validate_pipefusion_async_callback=(xFuserPipelineBaseWrapper._validate_pipefusion_async_callback)
    )
    with pytest.raises(NotImplementedError, match="cannot be applied consistently"):
        xFuserPipelineBaseWrapper._pipefusion_async_enabled(
            pipeline,
            num_timesteps=4,
            pipeline_warmup_steps=1,
            callback_on_step_end=lambda *_args: None,
        )


@pytest.mark.parametrize("first_stage", [True, False])
def test_sync_latent_handoff_broadcasts_from_last_stage(monkeypatch, first_stage):
    events = []
    latents = torch.ones(1, 2) if not first_stage else torch.ones(1, 3)
    metadata = ((1, 2), torch.float32)

    def broadcast_object(value, src):
        events.append(("metadata", value, src))
        return metadata

    def broadcast(value, src):
        events.append(("tensor", tuple(value.shape), src))
        return value

    group = SimpleNamespace(
        broadcast_object=broadcast_object,
        broadcast=broadcast,
    )
    monkeypatch.setattr(base_pipeline, "get_pipeline_parallel_world_size", lambda: 2)
    monkeypatch.setattr(base_pipeline, "get_pp_group", lambda: group)
    monkeypatch.setattr(base_pipeline, "is_pipeline_first_stage", lambda: first_stage)
    monkeypatch.setattr(base_pipeline, "is_pipeline_last_stage", lambda: not first_stage)

    result = xFuserPipelineBaseWrapper._pipefusion_handoff_sync_latents(None, latents, name="handoff")

    expected_metadata = None if first_stage else metadata
    assert events == [("metadata", expected_metadata, 1), ("tensor", (1, 2), 1)]
    assert result.shape == (1, 2)
