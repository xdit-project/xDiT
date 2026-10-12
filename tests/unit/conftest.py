from types import SimpleNamespace

import pytest

from xfuser.config.config import InputConfig
from xfuser.core.distributed import runtime_state as runtime_state_module
from xfuser.core.distributed.runtime_state import DiTRuntimeState


@pytest.fixture
def flux_runtime_state(monkeypatch):
    """Build a fresh DiT runtime state with FLUX.1 geometry (16 pixels per
    token row) on ``sp_degree`` sequence-parallel ranks, without a process
    group, as a runner sees it before its first request."""

    def make(sp_degree: int) -> DiTRuntimeState:
        monkeypatch.setattr(runtime_state_module, "get_sequence_parallel_world_size", lambda: sp_degree)
        monkeypatch.setattr(runtime_state_module, "get_sequence_parallel_rank", lambda: 0)
        monkeypatch.setattr(
            runtime_state_module,
            "get_pp_group",
            lambda: SimpleNamespace(reset_buffer=lambda: None, set_config=lambda **_: None),
        )
        state = DiTRuntimeState.__new__(DiTRuntimeState)
        state.parallel_config = SimpleNamespace(pp_config=SimpleNamespace(num_pipeline_patch=1))
        state.runtime_config = SimpleNamespace(warmup_steps=0, dtype=None)
        state.input_config = InputConfig()
        state.num_pipeline_patch = 1
        state.ready = False
        state.split_latents_by_rows = True
        state.vae_scale_factor = 16
        state.backbone_patch_size = 1
        return state

    return make
