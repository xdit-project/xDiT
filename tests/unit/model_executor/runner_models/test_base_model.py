import json

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
