"""Runner contract for Chroma1-HD: which parallel layouts it accepts and refuses.

Chroma attends with a dense text bias, so the runner must refuse layouts that would
silently drop it (ring attention, mask-unaware backends) and Ulysses degrees that do
not divide its 24 heads, before anything is downloaded.
"""

import pytest

pytest.importorskip("diffusers")

from xfuser import xFuserArgs  # noqa: E402
from xfuser.model_executor.models.runner_models.chroma import xFuserChromaModel  # noqa: E402


@pytest.fixture(autouse=True)
def _single_process(monkeypatch):
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")


def _model(**config):
    return xFuserChromaModel(xFuserArgs(model="Chroma1-HD", **config))


@pytest.mark.parametrize("ulysses_degree", [1, 2, 3, 4, 6, 8, 12, 24])
def test_ulysses_degrees_that_divide_the_heads_are_accepted(ulysses_degree):
    _model(ulysses_degree=ulysses_degree, use_cfg_parallel=True)


@pytest.mark.parametrize("ulysses_degree", [5, 16])
def test_ulysses_degree_must_divide_the_24_heads(ulysses_degree):
    with pytest.raises(ValueError, match="24 attention heads.*--ulysses_degree must divide 24"):
        _model(ulysses_degree=ulysses_degree)


@pytest.mark.parametrize(
    "config",
    [{"ring_degree": 2}, {"pipefusion_parallel_degree": 2}],
)
def test_unsupported_parallelism_is_refused(config):
    with pytest.raises(ValueError, match="does not support"):
        _model(**config)


def test_backends_that_drop_the_text_bias_are_refused():
    with pytest.raises(ValueError, match="additive bias that only SDPA applies"):
        _model(attention_backend="FLASH")


def test_sdpa_is_used_when_no_backend_is_named():
    config = xFuserArgs(model="Chroma1-HD")
    xFuserChromaModel(config)
    assert config.attention_backend == "SDPA"
