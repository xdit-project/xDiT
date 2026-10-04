from types import SimpleNamespace

import pytest

from xfuser.config.config import InputConfig
from xfuser.core.distributed import runtime_state as runtime_state_module
from xfuser.core.distributed.runtime_state import DiTRuntimeState
from xfuser.model_executor.models.runner_models import flux
from xfuser.model_executor.models.runner_models.flux import (
    xFuserFluxKontextModel,
    xFuserFluxModel,
)


def _runtime_state(monkeypatch, sp_degree: int) -> DiTRuntimeState:
    """A fresh FLUX.1 runtime state (16 pixels per token row) on ``sp_degree``
    Ulysses ranks, as the runner sees it before its first request."""
    monkeypatch.setattr(runtime_state_module, "get_sequence_parallel_world_size", lambda: sp_degree)
    monkeypatch.setattr(runtime_state_module, "get_sequence_parallel_rank", lambda: 0)
    monkeypatch.setattr(
        runtime_state_module,
        "get_pp_group",
        lambda: SimpleNamespace(reset_buffer=lambda: None, set_config=lambda **_: None),
    )
    state = DiTRuntimeState.__new__(DiTRuntimeState)
    state.parallel_config = SimpleNamespace(pp_config=SimpleNamespace(num_pipeline_patch=1))
    state.runtime_config = SimpleNamespace(warmup_steps=0, dtype=None)
    state.input_config = InputConfig()
    state.num_pipeline_patch = 1
    state.ready = False
    state.split_latents_by_rows = True
    state.vae_scale_factor = 16
    state.backbone_patch_size = 1
    monkeypatch.setattr(flux, "get_runtime_state", lambda: state)
    monkeypatch.setattr(flux, "get_pipeline_parallel_world_size", lambda: 1)
    return state


def _runner(model_cls):
    model = object.__new__(model_cls)
    model.config = SimpleNamespace(batch_size=None)
    model.pipe = lambda **kwargs: SimpleNamespace(images=["image"])
    model._make_generator = lambda seed: None
    return model


def _input_args(size: int) -> dict:
    return {
        "height": size,
        "width": size,
        "prompt": "a cat",
        "image": None,
        "max_area": None,
        "num_inference_steps": 4,
        "guidance_scale": 3.5,
        "max_sequence_length": 512,
        "seed": 42,
    }


@pytest.mark.parametrize("model_cls", [xFuserFluxModel, xFuserFluxKontextModel])
@pytest.mark.parametrize(
    ("ulysses_degree", "size"),
    [
        (3, 1056),  # 66 token rows: divisible by 3, unlike the 1024 default
        (2, 1040),  # 65 token rows: the transformer pads the token sequence
    ],
)
def test_ulysses_runs_record_the_requested_size(monkeypatch, model_cls, ulysses_degree, size):
    state = _runtime_state(monkeypatch, sp_degree=ulysses_degree)

    output = _runner(model_cls)._run_pipe(_input_args(size))

    assert output.images == ["image"]
    assert (state.input_config.height, state.input_config.width) == (size, size)
