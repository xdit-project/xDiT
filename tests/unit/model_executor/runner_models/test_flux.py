from types import SimpleNamespace

import pytest

from xfuser import xFuserArgs
from xfuser.model_executor.models.runner_models import flux
from xfuser.model_executor.models.runner_models.flux import (
    xFuserFluxKontextModel,
    xFuserFluxModel,
)


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
def test_ulysses_runs_record_the_requested_size(monkeypatch, flux_runtime_state, model_cls, ulysses_degree, size):
    state = flux_runtime_state(sp_degree=ulysses_degree)
    monkeypatch.setattr(flux, "get_runtime_state", lambda: state)
    monkeypatch.setattr(flux, "get_pipeline_parallel_world_size", lambda: 1)

    output = _runner(model_cls)._run_pipe(_input_args(size))

    assert output.images == ["image"]
    assert (state.input_config.height, state.input_config.width) == (size, size)


@pytest.fixture
def single_process(monkeypatch):
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")


@pytest.mark.parametrize(
    ("model_cls", "model"),
    [
        (flux.xFuserFlux2Model, "FLUX.2-dev"),
        (flux.xFuserFlux2Klein9BModel, "FLUX.2-klein-9B"),
        (flux.xFuserFlux2Klein4BModel, "FLUX.2-klein-4B"),
    ],
)
def test_flux2_pipefusion_refuses_parallel_vae_at_construction(single_process, tmp_path, model_cls, model):
    # The FLUX.2 PipeFusion pipeline decodes on its last stage only, so a decoder
    # sharded across every stage would wait forever for the other stages.
    args = {"model": model, "pipefusion_parallel_degree": 2, "output_directory": str(tmp_path)}
    model_cls(xFuserArgs(**args))

    with pytest.raises(ValueError, match="--use_parallel_vae is not supported with --pipefusion_parallel_degree"):
        model_cls(xFuserArgs(**args, use_parallel_vae=True))
