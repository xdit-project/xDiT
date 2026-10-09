"""The PipeFusion FLUX.2 wrappers must leave the text-encoder layers to the bound pipeline.

dev and klein stack different hidden-state layers of their text encoders, and each
diffusers pipeline's ``encode_prompt`` defaults to its own. The shared ``__call__`` used to
pass dev's layers to klein as well.
"""

from types import SimpleNamespace
from unittest import mock

import pytest

pipeline_flux2 = pytest.importorskip("xfuser.model_executor.pipelines.pipeline_flux2")


class _Encoded(Exception):
    pass


def _encode_prompt_kwargs(monkeypatch, **call_kwargs):
    """The keyword arguments ``__call__`` hands the bound pipeline's ``encode_prompt``."""
    runtime_state = SimpleNamespace(set_input_parameters=lambda **kwargs: None)
    monkeypatch.setattr(pipeline_flux2, "get_runtime_state", lambda: runtime_state)
    monkeypatch.setattr(pipeline_flux2, "get_pipeline_parallel_world_size", lambda: 1)
    monkeypatch.setattr(pipeline_flux2.xFuserFlux2PipelineBase, "use_naive_forward", lambda self: False)
    wrapper = object.__new__(pipeline_flux2.xFuserFlux2PipelineBase)
    wrapper.module = mock.Mock(encode_prompt=mock.Mock(side_effect=_Encoded))

    with pytest.raises(_Encoded):
        wrapper(prompt="a lighthouse at dusk", height=64, width=64, **call_kwargs)
    return wrapper.module.encode_prompt.call_args.kwargs


def test_the_bound_pipelines_default_layers_are_kept(monkeypatch):
    assert "text_encoder_out_layers" not in _encode_prompt_kwargs(monkeypatch)


def test_requested_layers_are_passed_on(monkeypatch):
    kwargs = _encode_prompt_kwargs(monkeypatch, text_encoder_out_layers=(9, 18, 27))
    assert kwargs["text_encoder_out_layers"] == (9, 18, 27)
