import pytest


@pytest.fixture(autouse=True)
def _single_process(monkeypatch):
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")


def _model_class(**attributes):
    from xfuser.model_executor.models.runner_models.base_model import xFuserModel

    def _load_model(self):
        raise AssertionError("refused before loading")

    def _run_pipe(self, input_args):
        raise AssertionError("refused before running")

    return type("ConfigOnlyModel", (xFuserModel,), {"_load_model": _load_model, "_run_pipe": _run_pipe, **attributes})


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
