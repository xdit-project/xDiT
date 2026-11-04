"""The FLUX.2 warmup must run with the guidance scale the request asks for.

`prepare_run` warms the pipeline up before the timed call. It left the guidance
scale at `__call__`'s default of 4.0, so a klein base checkpoint, which refuses
guidance above 1 in the parallel loop, failed in warmup even when the request
asked for guidance 1.
"""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from xfuser.config import InputConfig
from xfuser.model_executor.pipelines import base_pipeline, pipeline_flux2
from xfuser.model_executor.pipelines.pipeline_flux2 import xFuserFlux2PipelineBase
from xfuser.model_executor.pipelines.pipeline_flux2_klein import xFuserFlux2KleinPipeline


@pytest.mark.parametrize("guidance_scale", [1.0, 4.5])
def test_warmup_uses_the_requested_guidance_scale(monkeypatch, guidance_scale):
    runtime_state = SimpleNamespace(runtime_config=SimpleNamespace(warmup_steps=2))
    monkeypatch.setattr(pipeline_flux2, "get_runtime_state", lambda: runtime_state)
    # The warmup seeds a generator; this test has no accelerator.
    monkeypatch.setattr(torch, "Generator", lambda device=None: Mock())
    pipeline = xFuserFlux2KleinPipeline.__new__(xFuserFlux2KleinPipeline)
    pipeline.__call__ = Mock()

    pipeline.prepare_run(InputConfig(height=256, width=256, guidance_scale=guidance_scale))

    pipeline.__call__.assert_called_once()
    assert pipeline.__call__.call_args.kwargs.get("guidance_scale") == guidance_scale


def _parallel_return_pipeline(monkeypatch, *, owns_parallel_vae_output):
    """A PipeFusion call that stops before the VAE and disagrees the two last-group checks.

    The distributed helper says this rank is not the last pipeline stage. The wrapper
    method says whether this rank owns the parallel-VAE output.
    """
    state = SimpleNamespace(
        runtime_config=SimpleNamespace(warmup_steps=5, use_parallel_vae=True),
        parallel_config=SimpleNamespace(vae_parallel_size=0),
        set_input_parameters=lambda **_kwargs: None,
    )
    monkeypatch.setattr(pipeline_flux2, "get_runtime_state", lambda: state)
    monkeypatch.setattr(pipeline_flux2, "get_pipeline_parallel_world_size", lambda: 2)
    monkeypatch.setattr(pipeline_flux2, "is_dp_last_group", lambda: not owns_parallel_vae_output)
    monkeypatch.setattr(pipeline_flux2, "retrieve_timesteps", lambda *_args, **_kwargs: (torch.zeros(1), 1))
    for name in (
        "get_classifier_free_guidance_world_size",
        "get_sequence_parallel_world_size",
        "get_tensor_model_parallel_world_size",
    ):
        monkeypatch.setattr(base_pipeline, name, lambda: 1)
    monkeypatch.setattr(base_pipeline, "get_pipeline_parallel_world_size", lambda: 2)
    monkeypatch.setattr(base_pipeline, "get_fast_attn_enable", lambda: False)
    monkeypatch.setattr(base_pipeline, "get_runtime_state", lambda: state)

    pipeline = xFuserFlux2PipelineBase.__new__(xFuserFlux2PipelineBase)
    pipeline.prompts_split_by_caller = True
    pipeline.config = SimpleNamespace(is_distilled=True)
    pipeline._execution_device = "cpu"
    pipeline.transformer = SimpleNamespace(config=SimpleNamespace(in_channels=16))
    pipeline.scheduler = SimpleNamespace(config=SimpleNamespace(use_flow_sigmas=False), order=1)
    pipeline.check_inputs = lambda **_kwargs: None
    pipeline.encode_prompt = lambda **_kwargs: (torch.zeros(1, 2, 4), torch.zeros(1, 2, 3))
    pipeline.prepare_latents = lambda **_kwargs: (torch.zeros(1, 4, 4), torch.zeros(1, 4, 3))
    pipeline._prepare_reference_images = lambda image, height, width: (None, height, width)
    pipeline._sync_pipeline = lambda **_kwargs: torch.zeros(1, 4, 4)
    pipeline.progress_bar = lambda total: nullcontext()
    pipeline.maybe_free_model_hooks = lambda: None
    pipeline.is_dp_last_group = lambda: owns_parallel_vae_output
    return pipeline


@pytest.mark.parametrize("owns_parallel_vae_output", [True, False])
def test_parallel_vae_returns_the_image_from_the_wrapper_owner(monkeypatch, owns_parallel_vae_output):
    pipeline = _parallel_return_pipeline(monkeypatch, owns_parallel_vae_output=owns_parallel_vae_output)

    result = pipeline(prompt="a cat", height=64, width=64, num_inference_steps=1, output_type="latent")

    if owns_parallel_vae_output:
        assert result.images.shape == (1, 4, 4)
    else:
        assert result is None
