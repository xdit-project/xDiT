"""Offline coverage for the Cache-DiT adapter package boundary."""

import dataclasses
import sys
from types import SimpleNamespace
from types import ModuleType

import pytest
import torch

from xfuser.model_executor.cache.adapters.cache_dit import apply_cache_dit_cache
from xfuser.model_executor.cache.adapters.cache_dit import apply, config, context
from xfuser.model_executor.cache.presets import CacheDitAdapterConfig, DBCachePreset


@dataclasses.dataclass
class _FakeDBCacheConfig:
    Fn_compute_blocks: int = 0
    Bn_compute_blocks: int = 0
    residual_diff_threshold: float = 0.0
    max_warmup_steps: int = 0
    max_cached_steps: int = 0
    num_inference_steps: int = 0
    enable_separate_cfg: bool = False


class _FakeForwardPattern:
    DIT_BLOCK = "dit-block"


class _FakeBlockAdapter:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def test_lazy_import_explains_missing_optional_dependency(monkeypatch):
    monkeypatch.setitem(sys.modules, "cache_dit", None)

    with pytest.raises(ImportError, match=r"cache-dit is required for --cache_method dbcache"):
        config._import_cache_dit()


def test_build_config_merges_preset_and_json_overrides(monkeypatch):
    monkeypatch.setattr(config, "_build_calibrator_config", lambda *_: "calibrator")
    monkeypatch.setattr(config, "_build_scm_mask", lambda *_: None)
    monkeypatch.setattr(config, "_resolve_enable_separate_cfg", lambda requested: requested)

    db_config, calibrator = config._build_config(
        num_steps=12,
        preset_kwargs=DBCachePreset(
            Fn_compute_blocks=2,
            residual_diff_threshold=0.1,
            max_warmup_steps=3,
            enable_separate_cfg=True,
        ),
        cache_config_json='{"residual_diff_threshold": 0.2, "max_cached_steps": 7}',
        enable_separate_cfg=False,
        DBCacheConfig=_FakeDBCacheConfig,
    )

    assert db_config == _FakeDBCacheConfig(
        Fn_compute_blocks=2,
        residual_diff_threshold=0.2,
        max_warmup_steps=3,
        max_cached_steps=7,
        num_inference_steps=12,
        enable_separate_cfg=True,
    )
    assert calibrator == "calibrator"


def test_single_transformer_application_builds_configured_adapter(monkeypatch):
    calls = []

    def enable_cache(adapter, **kwargs):
        calls.append((adapter, kwargs))

    monkeypatch.setattr(
        apply,
        "_import_cache_dit",
        lambda: (enable_cache, _FakeDBCacheConfig, _FakeBlockAdapter, _FakeForwardPattern),
    )
    monkeypatch.setattr(apply, "_install_sp_can_cache_sync", lambda: None)
    monkeypatch.setattr(apply, "_is_parallelized_flag", lambda: False)
    monkeypatch.setattr(apply, "_is_rank0", lambda: False)
    monkeypatch.setattr(config, "_build_calibrator_config", lambda *_: None)
    monkeypatch.setattr(config, "_build_scm_mask", lambda *_: None)
    monkeypatch.setattr(config, "_resolve_enable_separate_cfg", lambda requested: requested)

    transformer = SimpleNamespace(blocks=torch.nn.ModuleList([torch.nn.Identity()]))
    adapter_config = CacheDitAdapterConfig(blocks=(("blocks", "DIT_BLOCK"),))

    assert (
        apply_cache_dit_cache(
            transformer,
            num_steps=8,
            pipe=SimpleNamespace(),
            preset_kwargs=DBCachePreset(Fn_compute_blocks=1, scm_policy=None),
            adapter_config=adapter_config,
        )
        is transformer
    )

    adapter, kwargs = calls.pop()
    assert transformer._is_parallelized is False
    assert adapter.kwargs["blocks"] is transformer.blocks
    assert adapter.kwargs["forward_pattern"] == "dit-block"
    assert kwargs == {
        "cache_config": _FakeDBCacheConfig(
            Fn_compute_blocks=1,
            residual_diff_threshold=0.08,
            max_warmup_steps=8,
            max_cached_steps=-1,
            num_inference_steps=8,
            enable_separate_cfg=False,
        )
    }


def test_context_sync_is_a_noop_without_distributed_initialization(monkeypatch):
    monkeypatch.setattr(context, "_SP_SYNC_PATCHED", False)
    monkeypatch.setattr(context.dist, "is_available", lambda: False)

    context._install_sp_can_cache_sync()

    assert context._SP_SYNC_PATCHED is False


def test_context_sync_installs_once_when_distributed(monkeypatch):
    class FakeCachedContextManager:
        def can_cache(self):
            return False

    cache_manager = ModuleType("cache_dit.caching.cache_contexts.cache_manager")
    cache_manager.CachedContextManager = FakeCachedContextManager
    monkeypatch.setitem(sys.modules, "cache_dit", ModuleType("cache_dit"))
    monkeypatch.setitem(sys.modules, "cache_dit.caching", ModuleType("cache_dit.caching"))
    monkeypatch.setitem(
        sys.modules,
        "cache_dit.caching.cache_contexts",
        ModuleType("cache_dit.caching.cache_contexts"),
    )
    monkeypatch.setitem(sys.modules, cache_manager.__name__, cache_manager)
    monkeypatch.setattr(context, "_SP_SYNC_PATCHED", False)
    monkeypatch.setattr(context.dist, "is_available", lambda: True)
    monkeypatch.setattr(context.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(context.dist, "get_world_size", lambda: 2)

    context._install_sp_can_cache_sync()
    installed_can_cache = FakeCachedContextManager.can_cache
    context._install_sp_can_cache_sync()

    assert context._SP_SYNC_PATCHED is True
    assert FakeCachedContextManager.can_cache is installed_can_cache
