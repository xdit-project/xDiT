"""The FLUX.2 warmup must run with the guidance scale the request asks for.

`prepare_run` warms the pipeline up before the timed call. It left the guidance
scale at `__call__`'s default of 4.0, so a klein base checkpoint, which refuses
guidance above 1 in the parallel loop, failed in warmup even when the request
asked for guidance 1.
"""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from xfuser.config import InputConfig
from xfuser.model_executor.pipelines import pipeline_flux2
from xfuser.model_executor.pipelines.pipeline_flux2_klein import xFuserFlux2KleinPipeline


@pytest.mark.parametrize("guidance_scale", [1.0, 4.5])
def test_warmup_uses_the_requested_guidance_scale(monkeypatch, guidance_scale):
    runtime_state = SimpleNamespace(runtime_config=SimpleNamespace(warmup_steps=2))
    monkeypatch.setattr(pipeline_flux2, "get_runtime_state", lambda: runtime_state)
    # The warmup seeds a CUDA generator; this test has no accelerator.
    monkeypatch.setattr(torch, "Generator", lambda device=None: Mock())
    pipeline = xFuserFlux2KleinPipeline.__new__(xFuserFlux2KleinPipeline)
    pipeline.__call__ = Mock()

    pipeline.prepare_run(InputConfig(height=256, width=256, guidance_scale=guidance_scale))

    pipeline.__call__.assert_called_once()
    assert pipeline.__call__.call_args.kwargs.get("guidance_scale") == guidance_scale
