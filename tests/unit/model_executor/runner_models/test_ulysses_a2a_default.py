"""Runners that default the Ulysses all-to-all to SDMA on gfx950."""

import pytest

FLUX3_BASE = "black-forest-labs/flux-3-image-klein-base"


@pytest.fixture
def build(monkeypatch):
    from xfuser import xFuserArgs
    from xfuser.core.attention import requirements
    from xfuser.model_executor.models.runner_models.base_model import MODEL_REGISTRY

    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "8")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "8")

    def _build(name, arch="gfx950:sramecc+:xnack-", **cli):
        monkeypatch.setattr(requirements, "device_arch", lambda: arch)
        cli.setdefault("ulysses_degree", 8)
        config = xFuserArgs(model=name, **cli)
        MODEL_REGISTRY[name](config)
        return config

    return _build


def test_flux3_klein_base_defaults_to_sdma_on_gfx950(build):
    assert build(FLUX3_BASE).ulysses_a2a == "sdma"


def test_an_explicit_choice_is_kept(build):
    assert build(FLUX3_BASE, ulysses_a2a="default").ulysses_a2a == "default"


@pytest.mark.parametrize(
    "cli",
    [
        {"arch": "gfx942:sramecc+:xnack-"},
        {"ulysses_degree": 1},
    ],
)
def test_the_default_all_to_all_stays_elsewhere(build, cli):
    assert build(FLUX3_BASE, **cli).ulysses_a2a is None


def test_more_than_one_node_keeps_the_default_all_to_all(build, monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "16")
    assert build(FLUX3_BASE).ulysses_a2a is None


def test_other_models_keep_the_default_all_to_all(build):
    assert build("Tongyi-MAI/Z-Image").ulysses_a2a is None
