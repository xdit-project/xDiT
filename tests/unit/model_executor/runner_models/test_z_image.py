"""CPU tests for what the Z-Image runners hand their diffusers pipeline."""

from types import SimpleNamespace
from unittest import mock

import pytest

from xfuser import xFuserArgs
from xfuser.model_executor.models.runner_models.base_model import MODEL_REGISTRY
from xfuser.model_executor.models.runner_models import z_image  # noqa: F401  registers


@pytest.fixture(autouse=True)
def _single_process(monkeypatch):
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")


def _args(model, **overrides):
    # Keys the CLI always passes; unset flags are None.
    args = {"prompt": "a cat", "dataset_path": None, "seed": 0, "height": 64, "width": 64, "num_inference_steps": 2}
    args |= {"negative_prompt": None, "guidance_scale": None}
    return model.preprocess_args({**args, **overrides})


@pytest.mark.parametrize("name", ["Z-Image", "Z-Image-Turbo"])
def test_negative_prompt_reaches_the_pipeline(name):
    model = MODEL_REGISTRY[name](xFuserArgs(model=name))
    model.pipe = mock.MagicMock(return_value=SimpleNamespace(images=["image"]), _execution_device="cpu")

    model._run_pipe(_args(model, negative_prompt="blurry", guidance_scale=2.0))

    assert model.pipe.call_args.kwargs["negative_prompt"] == "blurry"


@pytest.mark.parametrize(
    ("name", "overrides"),
    [("Z-Image", {"guidance_scale": 0.0}), ("Z-Image-Turbo", {})],
)
def test_negative_prompt_is_refused_when_guidance_is_off(name, overrides):
    model = MODEL_REGISTRY[name](xFuserArgs(model=name))

    with pytest.raises(ValueError, match="--negative_prompt unless --guidance_scale is above 0"):
        _args(model, negative_prompt="blurry", **overrides)
    _args(model, **overrides)
