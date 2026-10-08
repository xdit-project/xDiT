"""Runner contract for the Wan2.1 text-to-video 1.3B checkpoint.

Everything here builds the runner model from a parsed config only; nothing loads
weights or touches the network.
"""

import pytest

# transformer/config.json of the 1.3B checkpoint.
WAN21_T2V_1_3B_HEADS = 12


@pytest.fixture(autouse=True)
def _single_process(monkeypatch):
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")


def _build(model_name="Wan2.1-T2V-1.3B", **config):
    from xfuser import xFuserArgs
    from xfuser.model_executor.models.runner_models.base_model import MODEL_REGISTRY

    return MODEL_REGISTRY[model_name](xFuserArgs(model=model_name, **config))


@pytest.mark.parametrize(
    "parallel",
    [
        {"ulysses_degree": 2},
        {"ulysses_degree": 3},
        {"ulysses_degree": 4},
        {"ring_degree": 2},
        {"ulysses_degree": 4, "ring_degree": 2},
    ],
)
def test_wan21_t2v_1_3b_accepts_ulysses_degrees_that_divide_its_heads(parallel):
    _build(**parallel)


def test_wan21_t2v_1_3b_rejects_ulysses_degree_that_splits_a_head():
    with pytest.raises(ValueError, match=rf"{WAN21_T2V_1_3B_HEADS} attention heads.*got 8.*--ring_degree"):
        _build(ulysses_degree=8)


def test_wan21_t2v_14b_keeps_accepting_ulysses_degree_8():
    _build("Wan2.1-T2V", ulysses_degree=8)
