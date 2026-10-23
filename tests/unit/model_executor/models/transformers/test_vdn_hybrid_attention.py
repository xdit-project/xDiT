# SPDX-License-Identifier: Apache-2.0
"""CPU behavioural tests for the VDN-H3 hybrid attention internals.

These exercise the three pieces that have a reference to check against without a
GPU: the window-softmax decomposition (vs a dense masked softmax), the chunked
boundary scans (vs the full linear-branch scan), and the transform-config parser.
They import only the leaf submodules, so no AITER / device is needed.
"""

from __future__ import annotations

import math

import pytest
import torch

from xfuser.model_executor.models.transformers.vdn_minimax_h3.config import (
    VDNHybridAttentionArchConfig,
)
from xfuser.model_executor.models.transformers.vdn_minimax_h3.linear_branch import (
    VDNH3Layout,
    run_boundary_scans,
    run_scans,
)
from xfuser.model_executor.models.transformers.vdn_minimax_h3.window_softmax import (
    DecomposedPlan,
    set_window_backend,
    window_mask_frames,
    windowed_softmax,
)


# --------------------------------------------------------------------------
# (a) window softmax vs a dense masked-softmax reference
# --------------------------------------------------------------------------


def _toy_layout(*, text_len: int, num_frames: int, gh: int, gw: int, trailing: int = 0, pad: int = 0) -> VDNH3Layout:
    tpf = gh * gw
    used = text_len + num_frames * tpf + trailing
    return VDNH3Layout(
        seq_len=used + pad,
        used=used,
        text_len=text_len,
        video_start=text_len,
        num_frames=num_frames,
        tokens_per_frame=tpf,
        frame_height=gh,
        frame_width=gw,
    )


def _reference_window(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    layout: VDNH3Layout,
    hybrid: VDNHybridAttentionArchConfig,
    scale: float,
    softmax_gate: torch.Tensor | None,
) -> torch.Tensor:
    """Independent dense masked softmax over the ``used`` rows, with the attend
    mask built straight from ``window_mask_frames`` + the layout."""
    used = layout.used
    bounds, dense_rows, dense_cols = window_mask_frames(hybrid, layout.num_frames)
    global_rows: list[int] = []
    for a, b in layout.global_ranges:
        global_rows += list(range(a, b))

    mask = torch.zeros(used, used, dtype=torch.bool)
    dense_q = set(global_rows)
    for f in dense_rows:
        s, e = layout.frame_rows(f)
        dense_q |= set(range(s, e))
    for r in dense_q:
        mask[r, :used] = True
    for f in range(layout.num_frames):
        if f in dense_rows:
            continue
        lo, hi = bounds[f]
        kv_rows = set(global_rows)
        for kf in sorted(set(range(lo, hi + 1)) | dense_cols):
            s, e = layout.frame_rows(kf)
            kv_rows |= set(range(s, e))
        s, e = layout.frame_rows(f)
        kv = sorted(kv_rows)
        for r in range(s, e):
            mask[r, kv] = True

    qd = q[:used].permute(1, 0, 2).double()  # [H, used, d]
    kd = k[:used].permute(1, 0, 2).double()
    vd = v[:used].permute(1, 0, 2).double()
    scores = (qd @ kd.transpose(-1, -2)) * scale
    scores = scores.masked_fill(~mask.unsqueeze(0), float("-inf"))
    out = torch.softmax(scores, dim=-1) @ vd  # [H, used, d]
    out = out.permute(1, 0, 2)  # [used, H, d]
    if softmax_gate is not None:
        out = out * softmax_gate[:used].double().unsqueeze(-1)
    return out


@pytest.mark.parametrize("with_gate", [False, True])
@pytest.mark.parametrize("max_gather_rows", [200_000, 8])
def test_windowed_softmax_matches_dense_masked_reference(with_gate: bool, max_gather_rows: int):
    set_window_backend("sdpa")
    torch.manual_seed(0)
    heads, dim = 2, 8
    # chunk=1, radius=0 -> each frame attends only itself; anchors 0 and F-1 dense
    # both ways. A trailing "audio" block exercises the second global range.
    layout = _toy_layout(text_len=3, num_frames=5, gh=2, gw=2, trailing=2, pad=4)
    hybrid = VDNHybridAttentionArchConfig(chunk=1, radius=0, anchor_frames="both")
    assert not hybrid.full_cover(layout.num_frames)

    q = torch.randn(layout.seq_len, heads, dim)
    k = torch.randn(layout.seq_len, heads, dim)
    v = torch.randn(layout.seq_len, heads, dim)
    gate = torch.rand(layout.seq_len, heads) if with_gate else None
    scale = 1.0 / math.sqrt(dim)

    plan = DecomposedPlan(layout, hybrid, torch.device("cpu"), max_gather_rows=max_gather_rows)
    out = windowed_softmax(q, k, v, plan=plan, layout=layout, scale=scale, softmax_gate=gate)
    ref = _reference_window(q, k, v, layout=layout, hybrid=hybrid, scale=scale, softmax_gate=gate)

    assert torch.allclose(out[: layout.used].double(), ref, atol=1e-5, rtol=1e-5)
    # padding rows past ``used`` must be zeroed
    assert torch.count_nonzero(out[layout.used :]) == 0


def test_windowed_softmax_wider_radius_covers_neighbour_frames():
    set_window_backend("sdpa")
    torch.manual_seed(1)
    heads, dim = 2, 8
    layout = _toy_layout(text_len=2, num_frames=6, gh=2, gw=3)
    hybrid = VDNHybridAttentionArchConfig(chunk=2, radius=1, anchor_frames="none")
    q = torch.randn(layout.seq_len, heads, dim)
    k = torch.randn(layout.seq_len, heads, dim)
    v = torch.randn(layout.seq_len, heads, dim)
    scale = 1.0 / math.sqrt(dim)
    plan = DecomposedPlan(layout, hybrid, torch.device("cpu"))
    out = windowed_softmax(q, k, v, plan=plan, layout=layout, scale=scale, softmax_gate=None)
    ref = _reference_window(q, k, v, layout=layout, hybrid=hybrid, scale=scale, softmax_gate=None)
    assert torch.allclose(out[: layout.used].double(), ref, atol=1e-5, rtol=1e-5)


# --------------------------------------------------------------------------
# (b) chunked boundary scans vs the full linear-branch scan
# --------------------------------------------------------------------------


def _boundary_frame_indices(num_frames: int, chunk: int, offset: int) -> tuple[list[int], list[int]]:
    """Frames the chunked gather reads: prefix at each chunk's last frame, suffix
    at each chunk's first frame, on the ``offset`` grid."""
    if chunk <= 1:
        allf = list(range(num_frames))
        return allf, allf
    padded = offset + num_frames
    num_chunks = -(-padded // chunk)
    ends = [min((c + 1) * chunk - 1, padded - 1) - offset for c in range(num_chunks)]
    starts = [c * chunk - offset for c in range(num_chunks)]
    return [f for f in ends if 0 <= f < num_frames], [f for f in starts if 0 <= f < num_frames]


@pytest.mark.parametrize("chunk,offset", [(1, 0), (3, 0), (3, 2), (4, 1)])
@pytest.mark.parametrize("with_text_state", [False, True])
def test_boundary_scans_match_full_scan_at_boundaries(chunk: int, offset: int, with_text_state: bool):
    torch.manual_seed(0)
    num_frames, heads, dk, dv = 7, 2, 4, 4
    eye = torch.eye(dk, dtype=torch.float64)
    transitions = torch.randn(num_frames, heads, dk, dk, dtype=torch.float64) * 0.1 + eye
    injections = torch.randn(num_frames, heads, dv, dk, dtype=torch.float64) * 0.1
    text_state = torch.randn(heads, dv, dk, dtype=torch.float64) * 0.1 if with_text_state else None

    prefix_full, suffix_full = run_scans(transitions, injections, text_state)
    prefix_b, suffix_b = run_boundary_scans(transitions, injections, text_state, chunk=chunk, frame_offset=offset)

    ends, starts = _boundary_frame_indices(num_frames, chunk, offset)
    assert ends and starts
    for f in ends:
        assert torch.allclose(prefix_b[f], prefix_full[f], atol=1e-6)
    for f in starts:
        assert torch.allclose(suffix_b[f], suffix_full[f], atol=1e-6)


# --------------------------------------------------------------------------
# (c) transform-config parsing
# --------------------------------------------------------------------------


def test_from_transform_config_parses_fields_and_short_conv_shapes():
    base = {
        "softmax_attention": {"chunk": 5, "radius": 1},
        "linear_attention": {
            "delta_rule": "vdn_solve",
            "linear_head_dim": 128,
            "bridge": "alpha",
            "a_fp32": True,
            "enable_text_state": True,
            "short_conv": {"targets": ["k", "v"]},
        },
        "anchor_frames": "both",
        "enable_softmax_gate": True,
    }
    cfg = VDNHybridAttentionArchConfig.from_transform_config(base)
    assert (cfg.chunk, cfg.radius, cfg.anchor_frames) == (5, 1, "both")
    assert cfg.linear_head_dim == 128 and cfg.delta_rule == "vdn_solve"
    assert cfg.short_conv == ("k", "v") and cfg.enable_text_state is True

    # short_conv may also arrive as a bare list; the parser normalises both.
    flat_sc = {**base, "linear_attention": {**base["linear_attention"], "short_conv": ["q", "k"]}}
    assert VDNHybridAttentionArchConfig.from_transform_config(flat_sc).short_conv == ("q", "k")

    # the wrapper unwraps a {"config": ...} envelope before calling the parser;
    # parsing the inner object yields the same config either way.
    envelope = {"config": base}
    inner = envelope["config"] if "config" in envelope else envelope
    assert VDNHybridAttentionArchConfig.from_transform_config(inner) == cfg


def test_window_bounds_and_full_cover_follow_the_parsed_geometry():
    # radius large enough that every frame's window spans the clip -> dense/full cover
    wide = VDNHybridAttentionArchConfig(chunk=1, radius=10, anchor_frames="none")
    assert wide.full_cover(5)
    # a narrow window is not full cover and clamps at the clip edges
    narrow = VDNHybridAttentionArchConfig(chunk=1, radius=0, anchor_frames="none")
    assert not narrow.full_cover(5)
    assert narrow.window_bounds(3) == [(0, 0), (1, 1), (2, 2)]


def test_from_transform_config_rejects_unknown_delta_rule():
    bad = {
        "softmax_attention": {"chunk": 5, "radius": 1},
        "linear_attention": {"delta_rule": "nope", "linear_head_dim": 128, "short_conv": {"targets": []}},
        "anchor_frames": "both",
    }
    with pytest.raises(ValueError, match="delta_rule"):
        VDNHybridAttentionArchConfig.from_transform_config(bad)
