from types import SimpleNamespace

import pytest
import torch

from xfuser.model_executor.pipefusion.compile import (
    PipeFusionCompileCapture,
    PipeFusionRuntimeSnapshot,
)


class _Recorder(torch.nn.Module):
    def __init__(self, state):
        super().__init__()
        self.state = state
        self.observed = []

    def forward(self, hidden_states, *, scale=1):
        self.observed.append(
            (
                self.state.patch_mode,
                self.state.pipeline_patch_idx,
                tuple(hidden_states.shape),
            )
        )
        return hidden_states * scale


class _ShapeOnlyRecorder(_Recorder):
    def __init__(self, state):
        super().__init__(state)
        self.real_calls = 0
        self.capture_calls = 0

    def forward(self, hidden_states, *, scale=1):
        self.real_calls += 1
        return super().forward(hidden_states, scale=scale)

    def pipefusion_compile_capture_forward(self, hidden_states, *, scale=1):
        self.capture_calls += 1
        return torch.zeros_like(hidden_states)


def _state():
    return SimpleNamespace(
        patch_mode=False,
        pipeline_patch_idx=0,
        pp_patches_height=[2, 2],
        pp_patches_start_idx_local=[0, 2, 4],
        pp_patches_start_end_idx_global=[[0, 2], [2, 4]],
        pp_patches_token_num=[4, 4],
        pp_patches_token_start_idx_local=[0, 4, 8],
        pp_patches_token_start_end_idx_global=[[0, 4], [4, 8]],
    )


def test_capture_deduplicates_signatures_and_replays_runtime_state():
    state = _state()
    component = _Recorder(state)
    capture = PipeFusionCompileCapture({"transformer": component}, state)

    with capture.hooks():
        component(torch.ones(1, 8, 4), scale=2)
        component(torch.ones(1, 8, 4), scale=2)
        state.patch_mode = True
        state.pipeline_patch_idx = 1
        component(torch.ones(1, 4, 4), scale=2)

    assert len(capture.calls) == 2
    assert all(call.args[0].device.type == "cpu" for call in capture.calls)

    component.observed.clear()
    state.patch_mode = False
    state.pipeline_patch_idx = 0
    capture.replay(torch.device("cpu"))

    assert component.observed == [
        (False, 0, (1, 8, 4)),
        (True, 1, (1, 4, 4)),
    ]
    assert state.patch_mode is False
    assert state.pipeline_patch_idx == 0


def test_capture_distinguishes_dispatch_and_stride_variants():
    state = _state()
    state.attention_backend = "low"
    state.use_high_precision_gemm = False
    component = _Recorder(state)
    capture = PipeFusionCompileCapture({"transformer": component}, state)
    contiguous = torch.ones(1, 8, 4)
    transposed = torch.ones(1, 4, 8).transpose(1, 2)

    with capture.hooks():
        component(contiguous)
        component(transposed)
        state.attention_backend = "high"
        state.use_high_precision_gemm = True
        component(contiguous)

    assert len(capture.calls) == 3


def test_capture_removes_hooks_after_error():
    state = _state()
    component = _Recorder(state)
    capture = PipeFusionCompileCapture({"transformer": component}, state)

    with pytest.raises(RuntimeError, match="stop"):
        with capture.hooks():
            component(torch.ones(1, 8, 4))
            raise RuntimeError("stop")

    component(torch.ones(1, 2, 4))
    assert len(capture.calls) == 1
    assert not hasattr(state, "_xdit_compile_capture_bypass_attention")


def test_capture_scopes_attention_bypass():
    state = _state()
    component = _Recorder(state)
    capture = PipeFusionCompileCapture({"transformer": component}, state)

    assert not hasattr(state, "_xdit_compile_capture_bypass_attention")
    with capture.hooks():
        assert state._xdit_compile_capture_bypass_attention is True
    assert not hasattr(state, "_xdit_compile_capture_bypass_attention")

    state._xdit_compile_capture_bypass_attention = False
    with capture.hooks():
        assert state._xdit_compile_capture_bypass_attention is True
    assert state._xdit_compile_capture_bypass_attention is False


def test_capture_uses_and_restores_shape_only_stage_forward():
    state = _state()
    component = _ShapeOnlyRecorder(state)
    capture = PipeFusionCompileCapture({"transformer": component}, state)

    with capture.hooks():
        output = component(torch.ones(1, 8, 4), scale=2)

    assert len(capture.calls) == 1
    assert component.capture_calls == 1
    assert component.real_calls == 0
    torch.testing.assert_close(output, torch.zeros_like(output))

    component(torch.ones(1, 8, 4), scale=2)
    assert component.real_calls == 1


def test_runtime_snapshot_restores_temporary_layout():
    state = _state()
    state.patch_mode = True
    state.pipeline_patch_idx = 1
    state.pp_patches_token_num = [5, 3]
    state._xdit_pipefusion_layout_fingerprint = ("reference", 5, 3)
    snapshot = PipeFusionRuntimeSnapshot.capture(state)

    state.patch_mode = False
    state.pipeline_patch_idx = 0
    state.pp_patches_token_num = [4, 4]
    del state._xdit_pipefusion_layout_fingerprint

    with snapshot.install(state):
        assert state.patch_mode is True
        assert state.pipeline_patch_idx == 1
        assert state.pp_patches_token_num == [5, 3]
        assert state._xdit_pipefusion_layout_fingerprint == ("reference", 5, 3)

    assert state.patch_mode is False
    assert state.pipeline_patch_idx == 0
    assert state.pp_patches_token_num == [4, 4]
    assert not hasattr(state, "_xdit_pipefusion_layout_fingerprint")
