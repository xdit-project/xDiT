from types import SimpleNamespace
from unittest import mock

import pytest

from xfuser import xFuserArgs
from xfuser.model_executor.models.runner_models import cosmos3  # noqa: F401  registers
from xfuser.model_executor.models.runner_models.base_model import MODEL_REGISTRY


@pytest.fixture
def model(monkeypatch):
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    model = MODEL_REGISTRY["Cosmos3-Nano"](xFuserArgs(model="Cosmos3-Nano"))
    model.pipe = mock.MagicMock(return_value=SimpleNamespace(video=["video"]), _execution_device="cpu")
    return model


def _input_args(**overrides):
    args = {
        "prompt": "a cat",
        "negative_prompt": "blurry",
        "height": 64,
        "width": 64,
        "num_frames": 5,
        "num_inference_steps": 2,
        "guidance_scale": 6.0,
        "seed": 0,
    }
    return {**args, **overrides}


@pytest.mark.parametrize(
    "overrides",
    [{"prompt": ["a cat", "a dog"]}, {"negative_prompt": ["blurry", "dark"]}],
)
def test_more_than_one_prompt_per_call_is_rejected(model, overrides):
    with pytest.raises(ValueError, match="runs one .*prompt per pipeline call, got 2.*--batch_size 1"):
        model._run_pipe(_input_args(**overrides))

    model.pipe.assert_not_called()


def test_single_prompt_list_reaches_the_pipe_as_a_string(model):
    model._run_pipe(_input_args(prompt=["a cat"], negative_prompt=["blurry"]))

    assert model.pipe.call_args.kwargs["prompt"] == "a cat"
    assert model.pipe.call_args.kwargs["negative_prompt"] == "blurry"


def test_batch_size_one_hands_each_call_its_own_prompt_and_negative_prompt(model):
    model.config.batch_size = 1
    model._run_timed_pipe = lambda args: (model._run_pipe(args), 0.0)  # skip the distributed timing wrapper

    model._run_pipe_batched(_input_args(prompt=["a cat", "a dog"], negative_prompt=["blurry", "dark"]))

    calls = [call.kwargs for call in model.pipe.call_args_list]
    assert [(c["prompt"], c["negative_prompt"]) for c in calls] == [("a cat", "blurry"), ("a dog", "dark")]
