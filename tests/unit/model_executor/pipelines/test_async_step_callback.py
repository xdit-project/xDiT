"""PipeFusion's async loop reports each denoising step to callback_on_step_end once (#558)."""

from unittest.mock import patch

import pytest
import torch

from xfuser.model_executor.pipelines import base_pipeline
from xfuser.model_executor.pipelines.base_pipeline import xFuserPipelineBaseWrapper


def _step_end(pipe, callback, tensor_inputs, patch_latents, *, last_stage=True, **step_tensors):
    with patch.object(base_pipeline, "is_pipeline_last_stage", return_value=last_stage):
        xFuserPipelineBaseWrapper._async_pipeline_step_end(
            pipe, callback, tensor_inputs, 7, torch.tensor(500), patch_latents, patch_dim=2, step_tensors=step_tensors
        )


def test_callback_sees_the_whole_step_on_the_last_stage():
    pipe = object()
    patches = [torch.randn(1, 4, 2, 8), torch.randn(1, 4, 3, 8)]
    prompt_embeds = torch.randn(1, 5, 16)
    calls = []

    def callback(p, step, t, kwargs):
        calls.append((p, step, int(t), kwargs))
        return kwargs

    _step_end(pipe, callback, ["latents", "prompt_embeds"], patches, prompt_embeds=prompt_embeds)

    assert len(calls) == 1
    p, step, t, kwargs = calls[0]
    assert (p, step, t) == (pipe, 7, 500)
    torch.testing.assert_close(kwargs["latents"], torch.cat(patches, dim=2))
    assert kwargs["prompt_embeds"] is prompt_embeds


def test_callback_is_not_run_on_earlier_stages():
    calls = []
    _step_end(object(), lambda *a: calls.append(a), ["latents"], [torch.zeros(1)], last_stage=False)
    assert calls == []


def test_replacing_latents_is_reported_as_ignored():
    patches = [torch.zeros(1, 4, 2, 8)]

    def callback(p, step, t, kwargs):
        return {"latents": kwargs["latents"] + 1}

    with patch.object(base_pipeline.logger, "warning") as warning:
        _step_end(object(), callback, ["latents"], patches)

    warning.assert_called_once()
    assert "latents" in warning.call_args.args[0]
    torch.testing.assert_close(patches[0], torch.zeros(1, 4, 2, 8))


def test_requesting_a_tensor_the_loop_does_not_provide_is_an_error():
    with pytest.raises(ValueError, match="negative_prompt_embeds"):
        _step_end(object(), lambda *a: {}, ["latents", "negative_prompt_embeds"], [torch.zeros(1, 4, 2, 8)])
