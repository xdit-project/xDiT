import json
from types import SimpleNamespace

import pytest

from xfuser.config import xFuserArgs
from xfuser.model_executor.models.runner_models import base_model


@pytest.fixture(autouse=True)
def single_process(monkeypatch):
    """The runner logs and saves from the last rank, read from the launcher's environment."""
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")


class _StubModel(base_model.xFuserModel):
    settings = base_model.ModelSettings(model_name="test/model")

    def _load_model(self):
        raise NotImplementedError

    def _run_pipe(self, input_args):
        raise NotImplementedError


def test_outputs_land_in_an_output_directory_that_did_not_exist(tmp_path):
    output_directory = tmp_path / "benchmark_results" / "run"
    model = _StubModel(xFuserArgs(model="test/model", output_directory=str(output_directory)))

    model.save_timings([1.5])

    assert json.loads((output_directory / "timings.json").read_text()) == [1.5]


@pytest.mark.parametrize("dp_rank", [0, 1])
def test_one_prompt_is_shared_by_every_data_parallel_group(monkeypatch, dp_rank):
    monkeypatch.setattr(base_model, "get_data_parallel_world_size", lambda: 2)
    monkeypatch.setattr(base_model, "get_data_parallel_rank", lambda: dp_rank)
    model = object.__new__(_StubModel)
    model.config = SimpleNamespace(data_parallel_degree=2)

    # --prompt "a cat" parses to a one-element list.
    split_args = model._split_prompts_for_dp({"prompt": ["a cat"], "negative_prompt": None})

    assert split_args["prompt"] == ["a cat"]


def test_too_few_prompts_for_the_data_parallel_groups_are_rejected(monkeypatch):
    monkeypatch.setattr(base_model, "get_data_parallel_world_size", lambda: 3)
    monkeypatch.setattr(base_model, "get_data_parallel_rank", lambda: 0)
    model = object.__new__(_StubModel)
    model.config = SimpleNamespace(data_parallel_degree=3)

    with pytest.raises(ValueError, match="less than data_parallel_world_size"):
        model._split_prompts_for_dp({"prompt": ["a cat", "a dog"], "negative_prompt": None})


def _model_class(**attributes):
    from xfuser.model_executor.models.runner_models.base_model import xFuserModel

    def _load_model(self):
        raise AssertionError("refused before loading")

    def _run_pipe(self, input_args):
        raise AssertionError("refused before running")

    return type("ConfigOnlyModel", (xFuserModel,), {"_load_model": _load_model, "_run_pipe": _run_pipe, **attributes})


def test_video_model_without_fps_is_refused_at_config_time():
    from xfuser import xFuserArgs
    from xfuser.model_executor.models.runner_models.base_model import ModelSettings

    model_class = _model_class(settings=ModelSettings(model_name="video-without-fps", model_output_type="video"))

    with pytest.raises(ValueError, match="produces video output but fps is not set"):
        model_class(xFuserArgs(model="video-without-fps"))


def test_ulysses_degree_that_does_not_divide_heads_is_refused_at_config_time():
    from xfuser import xFuserArgs
    from xfuser.model_executor.models.runner_models.base_model import ModelCapabilities, ModelSettings

    model_class = _model_class(
        attention_heads=6,
        capabilities=ModelCapabilities(ulysses_degree=True, ring_degree=True),
        settings=ModelSettings(model_name="six-heads"),
    )

    model_class(xFuserArgs(model="six-heads", ulysses_degree=3))
    with pytest.raises(ValueError, match=r"six-heads has 6 attention heads.*\(1, 2, 3, 6\); got 4\..*--ring_degree"):
        model_class(xFuserArgs(model="six-heads", ulysses_degree=4))
