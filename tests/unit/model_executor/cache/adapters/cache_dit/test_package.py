"""Offline coverage for the Cache-DiT adapter package boundary."""

import dataclasses
import sys
from types import SimpleNamespace

import pytest
import torch

from xfuser.core import distributed
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
    steps_computation_mask: object = None
    steps_computation_policy: str = "dynamic"


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


def test_build_config_keeps_pipefusion_decision_sharing_out_of_cache_dit(monkeypatch):
    monkeypatch.setattr(config, "_build_calibrator_config", lambda *_: None)
    monkeypatch.setattr(config, "_build_scm_mask", lambda *_: None)
    monkeypatch.setattr(config, "_resolve_enable_separate_cfg", lambda requested: requested)

    db_config, _ = config._build_config(
        num_steps=12,
        preset_kwargs=DBCachePreset(
            share_pipefusion_decisions=True,
        ),
        cache_config_json=None,
        enable_separate_cfg=False,
        DBCacheConfig=_FakeDBCacheConfig,
    )

    assert db_config._xdit_share_pipefusion_decisions is True


@pytest.mark.parametrize(
    ("num_steps", "expected_cache_steps"),
    [
        (25, list(range(8, 23, 2))),
        (28, list(range(8, 26, 2))),
    ],
)
def test_alternating_scm_uses_warmup_and_final_corrections(
    num_steps,
    expected_cache_steps,
):
    mask = config._build_alternating_mask(
        num_steps,
        max_warmup_steps=8,
    )

    assert [index for index, value in enumerate(mask) if value == 0] == (expected_cache_steps)
    assert all(mask[index] == 1 for index in range(8))
    assert mask[-2:] == [1, 1]
    assert "00" not in "".join(map(str, mask))


def test_build_config_applies_alternating_static_policy(monkeypatch):
    monkeypatch.setattr(config, "_build_calibrator_config", lambda *_: None)
    monkeypatch.setattr(
        config,
        "_resolve_enable_separate_cfg",
        lambda requested: requested,
    )

    db_config, _ = config._build_config(
        num_steps=25,
        preset_kwargs=DBCachePreset(
            max_warmup_steps=8,
            scm_policy="alternating",
        ),
        cache_config_json='{"steps_computation_policy": "static"}',
        enable_separate_cfg=False,
        DBCacheConfig=_FakeDBCacheConfig,
    )

    assert db_config.steps_computation_policy == "static"
    assert [index for index, value in enumerate(db_config.steps_computation_mask) if value == 0] == list(
        range(8, 23, 2)
    )


def test_single_transformer_application_builds_configured_adapter(monkeypatch):
    calls = []

    def enable_cache(adapter, **kwargs):
        calls.append((adapter, kwargs))

    monkeypatch.setattr(
        apply,
        "_import_cache_dit",
        lambda: (enable_cache, _FakeDBCacheConfig, _FakeBlockAdapter, _FakeForwardPattern),
    )
    monkeypatch.setattr(apply, "is_parallelized_flag", lambda: False)
    monkeypatch.setattr(apply, "install_cache_decision_sync", lambda _transformer: None)
    monkeypatch.setattr(distributed, "get_pipeline_parallel_world_size", lambda: 1)
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
            preset_kwargs=DBCachePreset(
                Fn_compute_blocks=4,
                scm_policy=None,
                share_pipefusion_decisions=True,
            ),
            adapter_config=adapter_config,
        )
        is transformer
    )

    adapter, kwargs = calls.pop()
    assert transformer._is_parallelized is False
    assert transformer._xdit_share_pipefusion_decisions is True
    assert adapter.kwargs["blocks"] is transformer.blocks
    assert adapter.kwargs["forward_pattern"] == "dit-block"
    assert kwargs == {
        "cache_config": _FakeDBCacheConfig(
            Fn_compute_blocks=4,
            residual_diff_threshold=0.08,
            max_warmup_steps=8,
            max_cached_steps=-1,
            num_inference_steps=8,
            enable_separate_cfg=False,
        )
    }


def test_multi_transformer_application_rejects_pipefusion(monkeypatch):
    monkeypatch.setattr(distributed, "get_pipeline_parallel_world_size", lambda: 2)

    with pytest.raises(ValueError, match="does not yet support multi-transformer"):
        apply.apply_cache_dit_cache_multi(
            SimpleNamespace(),
            8,
            adapter_configs=[],
            presets=[],
        )


def test_static_pipefusion_disables_calibrator_and_installs_fn_bypass(
    monkeypatch,
):
    calls = []
    installed = []

    def enable_cache(_adapter, **kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(
        apply,
        "_import_cache_dit",
        lambda: (
            enable_cache,
            _FakeDBCacheConfig,
            _FakeBlockAdapter,
            _FakeForwardPattern,
        ),
    )
    monkeypatch.setattr(apply, "is_parallelized_flag", lambda: False)
    monkeypatch.setattr(apply, "install_cache_decision_sync", lambda _transformer: None)
    monkeypatch.setattr(apply, "install_pipefusion_patch_contexts", lambda _transformer: None)
    monkeypatch.setattr(
        apply,
        "install_pipefusion_static_fn_bypass",
        lambda transformer, cache_config: installed.append((transformer, cache_config)) or True,
    )
    monkeypatch.setattr(distributed, "get_pipeline_parallel_world_size", lambda: 2)
    monkeypatch.setattr(apply, "_is_rank0", lambda: False)
    monkeypatch.setattr(config, "_build_calibrator_config", lambda *_: "calibrator")
    monkeypatch.setattr(config, "_build_scm_mask", lambda *_: [1, 0, 1])
    monkeypatch.setattr(
        config,
        "_resolve_enable_separate_cfg",
        lambda requested: requested,
    )
    transformer = SimpleNamespace(blocks=torch.nn.ModuleList([torch.nn.Identity(), torch.nn.Identity()]))

    apply_cache_dit_cache(
        transformer,
        num_steps=3,
        pipe=SimpleNamespace(),
        preset_kwargs=DBCachePreset(Fn_compute_blocks=2),
        cache_config='{"steps_computation_policy": "static"}',
        adapter_config=CacheDitAdapterConfig(blocks=(("blocks", "DIT_BLOCK"),)),
    )

    assert len(installed) == 1
    assert installed[0][0] is transformer
    assert "calibrator_config" not in calls[0]


def test_context_sync_installs_once_and_is_local_for_pipefusion(monkeypatch):
    class _Manager:
        def __init__(self):
            self.calls = 0
            self.parallelized = None

        def can_cache(self, *, parallelized=False):
            self.calls += 1
            self.parallelized = parallelized
            return True

    manager = _Manager()
    transformer = SimpleNamespace(_context_manager=manager)
    monkeypatch.setattr(context.dist, "is_available", lambda: True)
    monkeypatch.setattr(context.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(context.dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(distributed, "get_pipeline_parallel_world_size", lambda: 2)
    monkeypatch.setattr(distributed, "get_sequence_parallel_world_size", lambda: 1)
    monkeypatch.setattr(
        distributed,
        "get_world_group",
        lambda: pytest.fail("PP-only decisions must not use the world group"),
    )

    context.install_cache_decision_sync(transformer)
    installed_can_cache = manager.can_cache
    context.install_cache_decision_sync(transformer)

    assert manager.can_cache is installed_can_cache
    assert manager.can_cache(parallelized=True) is True
    assert manager.calls == 1
    assert manager.parallelized is False


def test_static_pipefusion_bypasses_fn_blocks():
    class _Manager:
        def Fn_compute_blocks(self):
            return 2

    manager = _Manager()
    transformer = SimpleNamespace(_context_manager=manager)
    cache_config = SimpleNamespace(
        steps_computation_policy="static",
        steps_computation_mask=[1, 0, 1],
    )

    assert (
        context.install_pipefusion_static_fn_bypass(
            transformer,
            cache_config,
        )
        is True
    )
    assert manager.Fn_compute_blocks() == 0
    assert (
        context.install_pipefusion_static_fn_bypass(
            transformer,
            cache_config,
        )
        is True
    )


def test_dynamic_pipefusion_preserves_fn_blocks():
    class _Manager:
        def Fn_compute_blocks(self):
            return 2

    manager = _Manager()
    transformer = SimpleNamespace(_context_manager=manager)
    cache_config = SimpleNamespace(
        steps_computation_policy="dynamic",
        steps_computation_mask=[1, 0, 1],
    )

    assert (
        context.install_pipefusion_static_fn_bypass(
            transformer,
            cache_config,
        )
        is False
    )
    assert manager.Fn_compute_blocks() == 2


def test_context_sync_is_not_installed_for_single_rank(monkeypatch):
    class _Manager:
        def can_cache(self):
            return True

    manager = _Manager()
    original = manager.can_cache
    transformer = SimpleNamespace(_context_manager=manager)
    monkeypatch.setattr(context.dist, "is_available", lambda: True)
    monkeypatch.setattr(context.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(context.dist, "get_world_size", lambda: 1)

    context.install_cache_decision_sync(transformer)

    assert manager.can_cache == original
    assert not hasattr(manager, "_xdit_cache_decision_sync_installed")


def test_pipefusion_context_sync_uses_only_stage_sequence_group(monkeypatch):
    class _Manager:
        def can_cache(self, *, parallelized=False):
            assert parallelized is False
            return True

    class _Group:
        device_group = object()

        def __init__(self):
            self.broadcasts = 0

        def broadcast(self, decision, src):
            assert decision.item() == 1
            assert src == 0
            self.broadcasts += 1

    group = _Group()
    barriers = []
    transformer = SimpleNamespace(_context_manager=_Manager())
    monkeypatch.setattr(context.dist, "is_available", lambda: True)
    monkeypatch.setattr(context.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(context.dist, "get_world_size", lambda: 4)
    monkeypatch.setattr(
        context.dist,
        "barrier",
        lambda *, group: barriers.append(group),
    )
    monkeypatch.setattr(torch.cuda, "current_device", lambda: "cpu")
    monkeypatch.setattr(distributed, "get_pipeline_parallel_world_size", lambda: 2)
    monkeypatch.setattr(distributed, "get_sequence_parallel_world_size", lambda: 2)
    monkeypatch.setattr(distributed, "get_sp_group", lambda: group)
    monkeypatch.setattr(
        distributed,
        "get_world_group",
        lambda: pytest.fail("PP × SP decisions must not use the world group"),
    )

    context.install_cache_decision_sync(transformer)

    assert transformer._context_manager.can_cache(parallelized=True) is True
    assert group.broadcasts == 1
    assert barriers == [group.device_group]


def test_pipefusion_can_share_patch_zero_decision(monkeypatch):
    class _Manager:
        def __init__(self):
            self.calls = 0
            self.context = SimpleNamespace(name="blocks:pipefusion_patch_0")

        def can_cache(self, *_args, **_kwargs):
            self.calls += 1
            return True

        def get_context(self):
            return self.context

        def get_current_step(self):
            return 0

    state = SimpleNamespace(
        patch_mode=True,
        pipeline_patch_idx=0,
    )
    monkeypatch.setattr(distributed, "get_pipeline_parallel_world_size", lambda: 2)
    monkeypatch.setattr(distributed, "get_sequence_parallel_world_size", lambda: 1)
    monkeypatch.setattr(distributed, "get_runtime_state", lambda: state)
    monkeypatch.setattr(context.dist, "is_available", lambda: True)
    monkeypatch.setattr(context.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(context.dist, "get_world_size", lambda: 2)

    manager = _Manager()
    transformer = SimpleNamespace(
        _context_manager=manager,
        _xdit_share_pipefusion_decisions=True,
    )
    context.install_cache_decision_sync(transformer)

    assert manager.can_cache(torch.ones(1), prefix="blocks_Fn") is True
    assert manager.can_cache(torch.ones(1), prefix="blocks_Bn") is True
    state.pipeline_patch_idx = 1
    manager.context.name = "blocks:pipefusion_patch_1"
    assert manager.can_cache(torch.ones(1), prefix="blocks_Fn") is True
    assert manager.can_cache(torch.ones(1), prefix="blocks_Bn") is True
    assert manager.calls == 2


def test_pipefusion_leader_decisions_keep_cfg_phases_separate(monkeypatch):
    class _Manager:
        def __init__(self):
            self.calls = 0
            self.cfg_phase = False
            self.context = SimpleNamespace(name="blocks:pipefusion_patch_0")

        def can_cache(self, *_args, **_kwargs):
            result = self.calls == 0
            self.calls += 1
            return result

        def get_context(self):
            return self.context

        def get_current_step(self):
            return 3

        def is_separate_cfg_step(self):
            return self.cfg_phase

    state = SimpleNamespace(
        patch_mode=True,
        pipeline_patch_idx=0,
    )
    monkeypatch.setattr(distributed, "get_pipeline_parallel_world_size", lambda: 2)
    monkeypatch.setattr(distributed, "get_sequence_parallel_world_size", lambda: 1)
    monkeypatch.setattr(distributed, "get_runtime_state", lambda: state)
    monkeypatch.setattr(context.dist, "is_available", lambda: True)
    monkeypatch.setattr(context.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(context.dist, "get_world_size", lambda: 2)

    manager = _Manager()
    transformer = SimpleNamespace(
        _context_manager=manager,
        _xdit_share_pipefusion_decisions=True,
    )
    context.install_cache_decision_sync(transformer)

    assert manager.can_cache(torch.ones(1), prefix="blocks_Fn") is True
    manager.cfg_phase = True
    assert manager.can_cache(torch.ones(1), prefix="blocks_Fn") is False
    state.pipeline_patch_idx = 1
    manager.context.name = "blocks:pipefusion_patch_1"
    manager.cfg_phase = False
    assert manager.can_cache(torch.ones(1), prefix="blocks_Fn") is True
    manager.cfg_phase = True
    assert manager.can_cache(torch.ones(1), prefix="blocks_Fn") is False
    assert manager.calls == 2


def test_pipefusion_patch_contexts_follow_patch_lifecycle(monkeypatch):
    class _Manager:
        def __init__(self):
            self.reset_names = []
            self.selected_name = None

        def reset_context(self, name, *args, **kwargs):
            self.reset_names.append(name)
            return name

        def set_context(self, name, *args, **kwargs):
            self.selected_name = name
            return name

    state = SimpleNamespace(
        patch_mode=True,
        pipeline_patch_idx=1,
        num_pipeline_patch=2,
        parallel_config=SimpleNamespace(pp_config=SimpleNamespace(num_pipeline_patch=3)),
    )
    monkeypatch.setattr(distributed, "get_runtime_state", lambda: state)
    manager = _Manager()
    manager._xdit_pipefusion_leader_decisions = {
        ("stale",): True,
    }
    transformer = SimpleNamespace(
        _context_manager=manager,
        _context_names=["blocks"],
    )

    context.install_pipefusion_patch_contexts(transformer)
    context.install_pipefusion_patch_contexts(transformer)
    manager.reset_context("blocks")
    manager.set_context("blocks")

    assert manager.reset_names == [
        "blocks",
        "blocks:pipefusion_patch_0",
        "blocks:pipefusion_patch_1",
        "blocks:pipefusion_patch_2",
    ]
    assert manager.selected_name == "blocks:pipefusion_patch_1"
    assert manager._xdit_pipefusion_leader_decisions == {}
