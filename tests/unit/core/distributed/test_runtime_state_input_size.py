from types import SimpleNamespace

import pytest

from xfuser.config.config import InputConfig
from xfuser.core.distributed import runtime_state as runtime_state_module
from xfuser.core.distributed.runtime_state import DiTRuntimeState


def _flux_like_state(monkeypatch, sp_degree: int) -> DiTRuntimeState:
    """A DiT runtime state with FLUX.1 geometry (16 pixels per token row) on
    ``sp_degree`` sequence-parallel ranks, without a process group."""
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


def test_row_split_keeps_rejecting_heights_the_ranks_cannot_share(monkeypatch):
    state = _flux_like_state(monkeypatch, sp_degree=2)

    with pytest.raises(ValueError, match="not divisible by the number of sequence parallel devices"):
        state.set_input_parameters(height=1040, width=1040, batch_size=1, num_inference_steps=4)


def test_token_sharding_callers_accept_heights_the_ranks_cannot_share_by_row(monkeypatch):
    state = _flux_like_state(monkeypatch, sp_degree=2)

    state.set_input_parameters(
        height=1040,
        width=1040,
        batch_size=1,
        num_inference_steps=4,
        split_latents_by_rows=False,
    )

    assert (state.input_config.height, state.input_config.width) == (1040, 1040)
    assert state.num_pipeline_patch == 1
    assert state.pp_patches_height is None


def test_switching_to_row_split_revalidates_an_unchanged_size(monkeypatch):
    state = _flux_like_state(monkeypatch, sp_degree=2)
    state.set_input_parameters(height=1040, width=1040, batch_size=1, split_latents_by_rows=False)

    with pytest.raises(ValueError, match="not divisible"):
        state.set_input_parameters(height=1040, width=1040, batch_size=1, split_latents_by_rows=True)


def test_row_split_metadata_follows_the_requested_height(monkeypatch):
    state = _flux_like_state(monkeypatch, sp_degree=3)

    state.set_input_parameters(height=1056, width=1056, batch_size=1, split_latents_by_rows=False)

    # 66 token rows over 3 ranks.
    assert state.pp_patches_height == [22]
    assert state.pp_patches_token_num == [22 * 66]
