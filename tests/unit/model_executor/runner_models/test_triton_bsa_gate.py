"""TRITON_BSA is refused by models that publish no token grid for it.

The backend runs dense on every call that carries no grid, so a model that never
publishes one would run dense while the user asked for sparse attention.
"""

import pytest

pytest.importorskip("diffusers")


@pytest.fixture(autouse=True)
def _single_process(monkeypatch):
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")


def _build(model_name, **config):
    from xfuser import xFuserArgs
    from xfuser.model_executor.models.runner_models.base_model import MODEL_REGISTRY

    return MODEL_REGISTRY[model_name](xFuserArgs(model=model_name, **config))


@pytest.mark.parametrize(
    "config",
    [
        {"attention_backend": "TRITON_BSA"},
        {"attention_backend": "AITER", "cross_attention_backend": "TRITON_BSA"},
    ],
)
def test_a_model_without_a_token_grid_refuses_block_sparse_attention(config):
    with pytest.raises(ValueError, match="does not support TRITON_BSA"):
        _build("Wan2.1-T2V-1.3B", **config)


def test_prism_accepts_block_sparse_attention():
    _build("Prism", attention_backend="TRITON_BSA", cross_attention_backend="AITER")
