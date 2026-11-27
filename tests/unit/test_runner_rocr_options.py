import pytest

from xfuser import runner

FLUSH = "HSA_ENABLE_SDMA_HDP_FLUSH"


@pytest.mark.parametrize(
    ("config", "environment", "expected"),
    [
        ({"ulysses_a2a": "sdma"}, {}, "0"),
        # The environment wins.
        ({"ulysses_a2a": "sdma"}, {FLUSH: "1"}, "1"),
        ({"ulysses_a2a": "default"}, {}, None),
        ({}, {}, None),
    ],
)
def test_the_sdma_transport_starts_rocr_without_the_hdp_flush(monkeypatch, config, environment, expected):
    monkeypatch.setattr(runner.os, "environ", dict(environment))
    runner._set_rocr_options(config)
    assert runner.os.environ.get(FLUSH) == expected


def test_rocr_options_are_set_before_the_runtime_loads(monkeypatch):
    monkeypatch.setattr(runner.os, "environ", {})
    seen = []

    def load_runtime():
        seen.append(runner.os.environ.get(FLUSH))
        raise RuntimeError("stop here")

    monkeypatch.setattr(runner, "_model_registry", load_runtime)
    with pytest.raises(RuntimeError, match="stop here"):
        runner.xFuserModelRunner({"ulysses_a2a": "sdma"})
    assert seen == ["0"]
