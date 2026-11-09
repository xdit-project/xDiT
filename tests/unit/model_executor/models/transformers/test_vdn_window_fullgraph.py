import pytest
import torch

from xfuser.model_executor.models.transformers.vdn_minimax_h3 import window_softmax
from xfuser.model_executor.models.transformers.vdn_minimax_h3.config import (
    VDNHybridAttentionArchConfig,
)
from xfuser.model_executor.models.transformers.vdn_minimax_h3.linear_branch import (
    VDNH3Layout,
)


def _case(device, dtype, heads, head_dim):
    # [text 0:3 | video 3:83 (20 frames x 4 rows) | audio 83:85 | pad 85:128]
    layout = VDNH3Layout(
        seq_len=128,
        used=85,
        text_len=3,
        video_start=3,
        num_frames=20,
        tokens_per_frame=4,
        frame_height=2,
        frame_width=2,
    )
    plan = window_softmax.DecomposedPlan(layout, VDNHybridAttentionArchConfig(), torch.device(device))
    torch.manual_seed(0)
    q, k, v = (torch.randn(layout.seq_len, heads, head_dim, device=device, dtype=dtype) for _ in range(3))
    gate = torch.rand(layout.seq_len, heads, device=device, dtype=dtype)
    return layout, plan, q, k, v, gate


def _assert_window_compiles_fullgraph(layout, plan, q, k, v, gate):
    def window(q, k, v, gate):
        return window_softmax.windowed_softmax(
            q,
            k,
            v,
            plan=plan,
            layout=layout,
            scale=q.shape[-1] ** -0.5,
            softmax_gate=gate,
        )

    def dense(q, k, v, gate):
        return window_softmax.dense_softmax(q, k, v, used=layout.used, scale=q.shape[-1] ** -0.5, softmax_gate=gate)

    for fn in (window, dense):
        torch._dynamo.reset()
        compiled = torch.compile(fn, fullgraph=True, backend="eager")
        torch.testing.assert_close(compiled(q, k, v, gate), fn(q, k, v, gate))


@pytest.fixture
def sdpa_backend(monkeypatch):
    monkeypatch.setattr(window_softmax, "_WINDOW_BACKEND", "sdpa")
    window_softmax.prepare_window_backend()


@pytest.mark.parametrize("capture_scalar_outputs", [False, True])
def test_sdpa_window_compiles_fullgraph(sdpa_backend, capture_scalar_outputs):
    with torch._dynamo.config.patch(capture_scalar_outputs=capture_scalar_outputs):
        _assert_window_compiles_fullgraph(*_case("cpu", torch.float32, 2, 8))


def test_sdpa_window_matches_reference_softmax(sdpa_backend):
    layout, plan, q, k, v, gate = _case("cpu", torch.float32, 2, 8)
    out = window_softmax.windowed_softmax(q, k, v, plan=plan, layout=layout, scale=8**-0.5, softmax_gate=gate)
    used = layout.used
    ref = (
        torch.nn.functional.scaled_dot_product_attention(
            q[plan.dense_q].transpose(0, 1).unsqueeze(0),
            k[:used].transpose(0, 1).unsqueeze(0),
            v[:used].transpose(0, 1).unsqueeze(0),
            scale=8**-0.5,
        )
        .squeeze(0)
        .transpose(0, 1)
    )
    torch.testing.assert_close(out[plan.dense_q], ref * gate[plan.dense_q].unsqueeze(-1))
    assert torch.count_nonzero(out[used:]) == 0


def test_aiter_triton_falls_back_to_sdpa_before_compile(monkeypatch):
    def _missing():
        raise ImportError("no aiter")

    monkeypatch.setattr(window_softmax, "_WINDOW_BACKEND", "aiter")
    monkeypatch.setattr(window_softmax, "_aiter_triton_varlen_func", _missing)
    monkeypatch.setattr(window_softmax, "_AITER_TRITON_OP", None)
    window_softmax.prepare_window_backend()
    assert window_softmax._WINDOW_BACKEND == "sdpa"
