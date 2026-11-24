"""Runners that default to AITER MHA v4 BF16 with split KV where AITER can run it."""

import pytest

from xfuser.core.attention.spec import AttentionBackendType


@pytest.fixture
def build(monkeypatch):
    from xfuser import xFuserArgs
    from xfuser.core.attention import requirements
    from xfuser.model_executor.models.runner_models import base_model
    from xfuser.model_executor.models.runner_models.base_model import MODEL_REGISTRY

    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")

    def _build(name, mha_v4, arch="gfx950:sramecc+:xnack-", **cli):
        monkeypatch.setattr(requirements, "device_arch", lambda: arch)
        monkeypatch.setattr(base_model.attention_registry, "available", lambda backend: None if mha_v4 else "absent")
        monkeypatch.setattr(base_model, "_MHA_V4_KV_SPLITS", requirements.ALWAYS if mha_v4 else requirements.NEVER)
        config = xFuserArgs(model=name, **cli)
        MODEL_REGISTRY[name](config)
        return config

    return _build


@pytest.mark.parametrize(
    ("name", "splits"),
    [
        ("Qwen/Qwen-Image-Edit", "auto"),
        ("Qwen/Qwen-Image-2512", "auto"),
        ("black-forest-labs/FLUX.1-Kontext-dev", "auto"),
    ],
)
def test_default_is_mha_v4_with_the_model_split(build, name, splits):
    config = build(name, mha_v4=True)
    assert config.attention_backend == AttentionBackendType.AITER_BF16.name
    assert config.mha_v4_kv_splits == splits


def test_explicit_choices_are_kept(build):
    config = build("Qwen/Qwen-Image-Edit", mha_v4=True, attention_backend="AITER", mha_v4_kv_splits="2")
    assert config.attention_backend == "AITER"
    assert config.mha_v4_kv_splits == 2

    config = build("Qwen/Qwen-Image-Edit", mha_v4=True, mha_v4_kv_splits="8")
    assert config.attention_backend == AttentionBackendType.AITER_BF16.name
    assert config.mha_v4_kv_splits == 8

    config = build("Qwen/Qwen-Image-Edit", mha_v4=True, mha_v4_kv_splits="1")
    assert config.attention_backend == AttentionBackendType.AITER_BF16.name
    assert config.mha_v4_kv_splits == 1


def test_without_split_kv_the_global_default_stays(build):
    config = build("Qwen/Qwen-Image-Edit", mha_v4=False)
    assert config.attention_backend is None
    assert config.mha_v4_kv_splits is None


@pytest.mark.parametrize(
    "name", ["Qwen/Qwen-Image-Edit", "Qwen/Qwen-Image-2512", "black-forest-labs/FLUX.1-Kontext-dev"]
)
def test_gfx942_keeps_the_global_default(build, name):
    config = build(name, mha_v4=True, arch="gfx942:sramecc+:xnack-")
    assert config.attention_backend is None
    assert config.mha_v4_kv_splits is None
