"""CPU tests for how the Wan2.1-VACE runner hands prompts to its diffusers pipeline."""

import numpy as np
import pytest
from PIL import Image

from xfuser.model_executor.models.runner_models.wan import xFuserWan21VACEModel


class _FakeVACEPipe:
    """Records calls and, like diffusers' WanVACEPipeline, takes one prompt string per call."""

    _execution_device = "cpu"

    def __init__(self):
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        if not isinstance(kwargs["prompt"], str):
            raise ValueError("Passing a list of prompts is not yet supported.")
        return type("Output", (), {"frames": np.zeros((1, 3, 4, 4, 3), dtype=np.float32)})()


@pytest.fixture
def model():
    model = object.__new__(xFuserWan21VACEModel)
    model.pipe = _FakeVACEPipe()
    return model


@pytest.fixture
def cli_args(tmp_path, monkeypatch):
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    paths = []
    for name in ("first.png", "last.png"):
        Image.new("RGB", (4, 4)).save(tmp_path / name)
        paths.append(str(tmp_path / name))
    return {
        "height": 4,
        "width": 4,
        "num_frames": 3,
        "num_inference_steps": 2,
        "seed": 42,
        "input_images": paths,
        "dataset_path": None,
    }


def test_single_cli_prompt_reaches_the_pipe_as_a_string(model, cli_args):
    args = model.preprocess_args({**cli_args, "prompt": ["a red fox in the snow"]})
    model._run_pipe(args)

    assert [call["prompt"] for call in model.pipe.calls] == ["a red fox in the snow"]


def test_several_prompts_are_rejected_with_a_clear_error(model, cli_args):
    with pytest.raises(ValueError, match="one prompt per run"):
        model.preprocess_args({**cli_args, "prompt": ["a fox", "an owl"]})
