# SPDX-License-Identifier: Apache-2.0
#
# Ported from OpenVDN / Video DeltaNet (VDN-H3): Apache-2.0, Copyright 2026 the
# VDN authors. https://github.com/OpenVDN/vdn-minimax-h3 (commit e262cb5)
#   source: src/models/softmax_attention/window.py, decomposed.py
# Cross-referenced with SGLang's hybrid_window_attn_h3 diffusion integration
# (https://github.com/sgl-project/sglang, Apache-2.0).
"""VDN-H3 window softmax on the packed MiniMax-H3 layout.

An exact softmax over a chunk-aligned frame window: frame t belongs to chunk
t // chunk and attends to chunks [c - radius, c + radius]; frames 0 and F-1 are
dense anchors; text/condition/audio rows are dense both ways; padding rows sit
outside every mask; a per-(token, head) sigmoid gate scales the output.

The window runs as a union of dense varlen attention calls: the dense-query rows
against all keys, then per-chunk gathered ``[globals | window | anchors]`` K/V.
This mirrors SGLang's ``hybrid_window_attn_h3`` decomposition. The varlen executor
is selected from ``--attention_backend`` (SDPA -> per-segment SDPA, correct on ROCm
and CUDA; AITER -> one AITER CK/ASM ``flash_attn_varlen_func`` call per varlen group;
AITER_FP8 -> AITER per-tensor FP8 varlen with Hadamard-rotated Q/K). The
``--vdn_window_backend`` flag overrides it directly with one of sdpa / aiter /
aiter_ck / aiter_fp8 (``aiter`` is AITER's Triton varlen bf16 kernel, SDPA fallback).
"""

from __future__ import annotations

import functools
import logging

import torch
import torch.nn.functional as F

from xfuser.model_executor.models.transformers.vdn_minimax_h3.config import (
    VDNHybridAttentionArchConfig,
)
from xfuser.model_executor.models.transformers.vdn_minimax_h3.linear_branch import (
    VDNH3Layout,
)

logger = logging.getLogger(__name__)

_WINDOW_BACKENDS = ("sdpa", "aiter", "aiter_ck", "aiter_fp8")
_WINDOW_BACKEND = "sdpa"  # set by set_window_backend() at load, before modules attach
_AITER_VARLEN = None  # resolved lazily; set to False after a failed import
_AITER_FP8_VARLEN = None  # (xfuser::aiter_fp8_varlen, hadamard) set by prepare_window_backend
_AITER_CK_OP = None
_AITER_TRITON_OP = None
_SDPA_OP = None
_BACKEND_LOGGED = False


# --------------------------------------------------------------------------
# Layout recovery from the transformer's packed metadata
# --------------------------------------------------------------------------


def build_layout(
    *,
    position_ids: torch.Tensor,
    text_indices: torch.Tensor,
    audio_indices: torch.Tensor,
    video_indices: torch.Tensor,
    used: int,
    seq_len: int,
) -> VDNH3Layout:
    """Recover the VDN layout from the xDiT packed metadata.

    ``used`` is the real (pre-pad) row count; ``seq_len`` is the padded length the
    blocks operate on. The video rows must be contiguous (they are, for t2va /
    fl2va). The video grid is read from ``position_ids`` values (T, H, W)."""
    video_indices = video_indices.to(torch.long)
    text_indices = text_indices.to(torch.long)
    if int(video_indices.numel()) == 0:
        raise ValueError("VDN layout: no video rows in the packed sequence")
    video_start = int(video_indices[0].item())
    num_video = int(video_indices.numel())
    if int(video_indices[-1].item()) - video_start + 1 != num_video:
        raise ValueError("VDN layout: video rows are not contiguous")
    text_len = int(text_indices.numel())
    expected_text = torch.arange(text_len, device=text_indices.device)
    if text_len and not torch.equal(text_indices.sort().values, expected_text):
        raise ValueError(
            "VDN layout expects the text rows at the head of the packed sequence "
            "[0, text_len); got a different placement."
        )
    video_pos = position_ids.index_select(0, video_indices)
    frames = int(torch.unique(video_pos[:, 0]).numel())
    grid_h = int(torch.unique(video_pos[:, 1]).numel())
    grid_w = int(torch.unique(video_pos[:, 2]).numel())
    tokens_per_frame = grid_h * grid_w
    if frames * tokens_per_frame != num_video:
        raise ValueError(
            f"VDN layout: recovered grid T={frames} H={grid_h} W={grid_w} "
            f"({frames * tokens_per_frame} rows) != {num_video} video rows"
        )
    return VDNH3Layout(
        seq_len=int(seq_len),
        used=int(used),
        text_len=text_len,
        video_start=video_start,
        num_frames=frames,
        tokens_per_frame=tokens_per_frame,
        frame_height=grid_h,
        frame_width=grid_w,
    )


# --------------------------------------------------------------------------
# Window mask geometry (mirrors SGLang window_mask_frames)
# --------------------------------------------------------------------------


def window_mask_frames(
    hybrid: VDNHybridAttentionArchConfig, num_frames: int
) -> tuple[list[tuple[int, int]], set[int], set[int]]:
    """(clamped per-frame window bounds, dense-ROW frames, dense-COLUMN frames)."""
    bounds = [(max(lo, 0), min(hi, num_frames - 1)) for lo, hi in hybrid.window_bounds(num_frames)]
    anchors = {0, num_frames - 1} if hybrid.anchor_frames != "none" else set()
    dense_rows = anchors if hybrid.anchor_frames in ("rows", "both") else set()
    dense_cols = anchors if hybrid.anchor_frames in ("columns", "both") else set()
    return bounds, dense_rows, dense_cols


def _merge_ranges(ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for a, b in sorted(ranges):
        if out and out[-1][1] >= a:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def _cat_ranges(ranges: list[tuple[int, int]], *, device: torch.device) -> torch.Tensor:
    if not ranges:
        return torch.empty(0, dtype=torch.long, device=device)
    return torch.cat([torch.arange(a, b, device=device, dtype=torch.long) for a, b in ranges])


def _chunk_groups(raw_bounds: list[tuple[int, int]], dense_rows: set[int]) -> list[list[int]]:
    # consecutive window frames with identical bounds share one varlen segment
    groups: list[list[int]] = []
    for f in range(len(raw_bounds)):
        if f in dense_rows:
            continue
        if groups and raw_bounds[groups[-1][-1]] == raw_bounds[f] and groups[-1][-1] == f - 1:
            groups[-1].append(f)
        else:
            groups.append([f])
    return groups


class _ChunkGroup:
    __slots__ = ("frames", "query_rows", "kv_rows")

    def __init__(self, frames, query_rows, kv_rows):
        self.frames = frames
        self.query_rows = query_rows
        self.kv_rows = kv_rows


class _WindowPass:
    __slots__ = (
        "query_rows",
        "query_slice",
        "kv_rows",
        "cu_q",
        "cu_k",
        "max_q",
        "max_k",
    )

    def __init__(self, query_rows, query_slice, kv_rows, cu_q, cu_k, max_q, max_k):
        self.query_rows = query_rows
        self.query_slice = query_slice
        self.kv_rows = kv_rows
        self.cu_q = cu_q
        self.cu_k = cu_k
        self.max_q = max_q
        self.max_k = max_k


def _window_pass(layout: VDNH3Layout, groups: list[_ChunkGroup], device: torch.device) -> _WindowPass:
    query_lens = [int(group.query_rows.numel()) for group in groups]
    kv_lens = [int(group.kv_rows.numel()) for group in groups]
    frames = [frame for group in groups for frame in group.frames]
    contiguous = frames == list(range(frames[0], frames[0] + len(frames)))
    zero = torch.zeros(1, dtype=torch.long)
    return _WindowPass(
        query_rows=torch.cat([group.query_rows for group in groups]),
        query_slice=((layout.frame_rows(frames[0])[0], layout.frame_rows(frames[-1])[1]) if contiguous else None),
        kv_rows=torch.cat([group.kv_rows for group in groups]),
        cu_q=torch.cat([zero, torch.tensor(query_lens).cumsum(0)]).to(device, torch.int32),
        cu_k=torch.cat([zero, torch.tensor(kv_lens).cumsum(0)]).to(device, torch.int32),
        max_q=max(query_lens),
        max_k=max(kv_lens),
    )


def _window_passes(
    layout: VDNH3Layout,
    groups: list[_ChunkGroup],
    max_gather_rows: int,
    device: torch.device,
) -> list[_WindowPass]:
    passes: list[_WindowPass] = []
    current: list[_ChunkGroup] = []
    current_rows = 0
    for group in groups:
        rows = int(group.kv_rows.numel())
        if current and current_rows + rows > max_gather_rows:
            passes.append(_window_pass(layout, current, device))
            current, current_rows = [], 0
        current.append(group)
        current_rows += rows
    if current:
        passes.append(_window_pass(layout, current, device))
    return passes


class DecomposedPlan:
    """Query-row groups with identical kept key sets, as dense varlen calls."""

    __slots__ = ("dense_q", "dense_cu_q", "dense_cu_k", "passes")

    def __init__(
        self,
        layout: VDNH3Layout,
        hybrid: VDNHybridAttentionArchConfig,
        device: torch.device,
        max_gather_rows: int = 200_000,
    ) -> None:
        used, num_frames = layout.used, layout.num_frames
        bounds, dense_rows, dense_cols = window_mask_frames(hybrid, num_frames)
        rows = functools.partial(_cat_ranges, device=device)
        dense_ranges = _merge_ranges(layout.global_ranges + [layout.frame_rows(f) for f in sorted(dense_rows)])
        self.dense_q = rows(dense_ranges)
        self.dense_cu_q = torch.tensor([0, int(self.dense_q.numel())], dtype=torch.int32, device=device)
        self.dense_cu_k = torch.tensor([0, used], dtype=torch.int32, device=device)
        groups = []
        for frames in _chunk_groups(hybrid.window_bounds(num_frames), dense_rows):
            lo, hi = bounds[frames[0]]
            kv_frames = sorted(set(range(lo, hi + 1)) | dense_cols)
            groups.append(
                _ChunkGroup(
                    frames=frames,
                    query_rows=rows(_merge_ranges([layout.frame_rows(f) for f in frames])),
                    kv_rows=rows(_merge_ranges(layout.global_ranges + [layout.frame_rows(f) for f in kv_frames])),
                )
            )
        self.passes = _window_passes(layout, groups, max_gather_rows, device)
        window_rows = sum(int(p.query_rows.numel()) for p in self.passes)
        covered = int(self.dense_q.numel()) + window_rows
        if covered != used:
            raise ValueError(f"window decomposition covers {covered} of {used} packed rows")


# --------------------------------------------------------------------------
# Varlen executors
# --------------------------------------------------------------------------


def _ensure_aiter_ck_op():
    """Register one opaque CK varlen op so torch.compile does not trace AITER.

    Called at module attach, before the transformer blocks are compiled.
    """
    global _AITER_CK_OP
    if _AITER_CK_OP is not None:
        return _AITER_CK_OP
    import inspect

    import aiter

    params = inspect.signature(aiter.flash_attn_varlen_func).parameters
    how_v3 = 2 if "how_v3_bf16_cvt" in params else None

    def _vdn_aiter_ck_varlen(q, k, v, cu_q, cu_k, max_q, max_k, scale):
        kwargs = {
            "dropout_p": 0.0,
            "softmax_scale": float(scale),
            "causal": False,
            "return_lse": False,
            "return_attn_probs": False,
        }
        if how_v3 is not None:
            kwargs["how_v3_bf16_cvt"] = how_v3
        out = aiter.flash_attn_varlen_func(
            q.contiguous(),
            k.contiguous(),
            v.contiguous(),
            cu_q.to(device=q.device, dtype=torch.int32).contiguous(),
            cu_k.to(device=k.device, dtype=torch.int32).contiguous(),
            max_seqlen_q=int(max_q),
            max_seqlen_k=int(max_k),
            **kwargs,
        )
        return out[0] if isinstance(out, tuple) else out

    def _vdn_aiter_ck_varlen_fake(q, k, v, cu_q, cu_k, max_q, max_k, scale):
        return torch.empty_like(q)

    # This module uses postponed annotations. The op schema needs real types.
    _annotations = {
        "q": torch.Tensor,
        "k": torch.Tensor,
        "v": torch.Tensor,
        "cu_q": torch.Tensor,
        "cu_k": torch.Tensor,
        "max_q": int,
        "max_k": int,
        "scale": float,
        "return": torch.Tensor,
    }
    _vdn_aiter_ck_varlen.__annotations__ = _annotations
    _vdn_aiter_ck_varlen_fake.__annotations__ = _annotations
    op = torch.library.custom_op("xfuser::vdn_aiter_ck_varlen", mutates_args=())(_vdn_aiter_ck_varlen)
    op.register_fake(_vdn_aiter_ck_varlen_fake)

    _AITER_CK_OP = op
    return _AITER_CK_OP


def set_window_backend(name: str) -> None:
    """Select the window varlen kernel (one of ``_WINDOW_BACKENDS``).

    Called once per load, before the VDN modules are attached. The runner derives
    ``name`` from ``--attention_backend`` or the ``--vdn_window_backend`` override.
    """
    global _WINDOW_BACKEND
    backend = name.lower()
    if backend == "ck":
        backend = "aiter_ck"
    if backend not in _WINDOW_BACKENDS:
        raise ValueError(f"VDN window backend {name!r} is not one of {', '.join(_WINDOW_BACKENDS)}.")
    _WINDOW_BACKEND = backend


def _ensure_sdpa_op():
    """Register the per-segment SDPA loop as one opaque op.

    The loop reads the segment boundaries on the host (``tolist``), which
    ``fullgraph=True`` cannot trace. Called at module attach, before the
    transformer blocks are compiled.
    """
    global _SDPA_OP
    if _SDPA_OP is not None:
        return _SDPA_OP

    def _vdn_sdpa_varlen(q, k, v, cu_q, cu_k, scale):
        out = torch.empty_like(q)
        cq = cu_q.tolist()
        ck = cu_k.tolist()
        for i in range(len(cq) - 1):
            q0, q1 = cq[i], cq[i + 1]
            k0, k1 = ck[i], ck[i + 1]
            if q1 <= q0:
                continue
            qi = q[q0:q1].transpose(0, 1).unsqueeze(0)  # [1, H, Lq, d]
            ki = k[k0:k1].transpose(0, 1).unsqueeze(0)
            vi = v[k0:k1].transpose(0, 1).unsqueeze(0)
            oi = F.scaled_dot_product_attention(qi, ki, vi, dropout_p=0.0, is_causal=False, scale=float(scale))
            out[q0:q1] = oi.squeeze(0).transpose(0, 1)
        return out

    def _vdn_sdpa_varlen_fake(q, k, v, cu_q, cu_k, scale):
        return torch.empty_like(q)

    # This module uses postponed annotations. The op schema needs real types.
    _annotations = {
        "q": torch.Tensor,
        "k": torch.Tensor,
        "v": torch.Tensor,
        "cu_q": torch.Tensor,
        "cu_k": torch.Tensor,
        "scale": float,
        "return": torch.Tensor,
    }
    _vdn_sdpa_varlen.__annotations__ = _annotations
    _vdn_sdpa_varlen_fake.__annotations__ = _annotations
    op = torch.library.custom_op("xfuser::vdn_sdpa_varlen", mutates_args=())(_vdn_sdpa_varlen)
    op.register_fake(_vdn_sdpa_varlen_fake)

    _SDPA_OP = op
    return _SDPA_OP


def prepare_window_backend() -> None:
    """Resolve the backend chosen by ``set_window_backend`` before the transformer
    blocks are compiled.

    The blocks compile with ``fullgraph=True``, so the import fallback and the op
    registration happen here, not in the traced ``_varlen``.
    """
    global _WINDOW_BACKEND, _BACKEND_LOGGED
    if _WINDOW_BACKEND == "aiter":
        try:
            _ensure_aiter_triton_op()
        except (ImportError, AttributeError) as exc:
            logger.warning(
                "AITER Triton varlen unavailable for VDN-H3 window softmax (%s); falling back to SDPA.",
                exc,
            )
            _WINDOW_BACKEND = "sdpa"
    if _WINDOW_BACKEND == "aiter_ck":
        _ensure_aiter_ck_op()
        message = "AITER CK varlen (one flash_attn_varlen_func call per group)."
    elif _WINDOW_BACKEND == "aiter":
        message = "AITER Triton varlen (xfuser::vdn_aiter_triton_varlen)."
    elif _WINDOW_BACKEND == "aiter_fp8":
        _ensure_aiter_fp8_parts()
        message = "AITER FP8 varlen (per-tensor, Hadamard-rotated Q/K)."
    elif _WINDOW_BACKEND == "sdpa":
        _ensure_sdpa_op()
        message = "per-segment SDPA (xfuser::vdn_sdpa_varlen)."
    else:
        return
    if _BACKEND_LOGGED:
        return
    _BACKEND_LOGGED = True
    rank0 = True
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        rank0 = torch.distributed.get_rank() == 0
    if rank0:
        logger.info("VDN-H3 window softmax: %s", message)


def _aiter_ck_varlen(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    cu_q: torch.Tensor,
    cu_k: torch.Tensor,
    max_q: int,
    max_k: int,
    scale: float,
) -> torch.Tensor:
    op = _ensure_aiter_ck_op()
    return op(q, k, v, cu_q, cu_k, int(max_q), int(max_k), float(scale))


def _aiter_triton_varlen_func():
    global _AITER_VARLEN
    if _AITER_VARLEN is None:
        import importlib

        _AITER_VARLEN = importlib.import_module("aiter.ops.triton.attention.mha").flash_attn_varlen_func
    return _AITER_VARLEN


def _ensure_aiter_triton_op():
    """Register the AITER Triton varlen call as one opaque op.

    AITER resolves its kernel config from disk on each call, which
    ``fullgraph=True`` cannot trace.
    """
    global _AITER_TRITON_OP
    if _AITER_TRITON_OP is not None:
        return _AITER_TRITON_OP
    varlen = _aiter_triton_varlen_func()

    def _vdn_aiter_triton_varlen(q, k, v, cu_q, cu_k, max_q, max_k, scale):
        out = varlen(
            q=q.contiguous(),
            k=k.contiguous(),
            v=v.contiguous(),
            cu_seqlens_q=cu_q.to(device=q.device, dtype=torch.int32).contiguous(),
            cu_seqlens_k=cu_k.to(device=k.device, dtype=torch.int32).contiguous(),
            max_seqlen_q=int(max_q),
            max_seqlen_k=int(max_k),
            softmax_scale=float(scale),
            causal=False,
        )
        return out[0] if isinstance(out, tuple) else out

    def _vdn_aiter_triton_varlen_fake(q, k, v, cu_q, cu_k, max_q, max_k, scale):
        return torch.empty_like(q)

    # This module uses postponed annotations. The op schema needs real types.
    _annotations = {
        "q": torch.Tensor,
        "k": torch.Tensor,
        "v": torch.Tensor,
        "cu_q": torch.Tensor,
        "cu_k": torch.Tensor,
        "max_q": int,
        "max_k": int,
        "scale": float,
        "return": torch.Tensor,
    }
    _vdn_aiter_triton_varlen.__annotations__ = _annotations
    _vdn_aiter_triton_varlen_fake.__annotations__ = _annotations
    op = torch.library.custom_op("xfuser::vdn_aiter_triton_varlen", mutates_args=())(_vdn_aiter_triton_varlen)
    op.register_fake(_vdn_aiter_triton_varlen_fake)

    _AITER_TRITON_OP = op
    return _AITER_TRITON_OP


def _sdpa_varlen(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    cu_q: torch.Tensor,
    cu_k: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Per-segment SDPA over a varlen batch. q [Tq, H, d], k/v [Tk, H, d].

    ``cu_q`` / ``cu_k`` are the cumulative segment boundaries (int32). Returns
    [Tq, H, d]."""
    return _ensure_sdpa_op()(q, k, v, cu_q, cu_k, float(scale))


def _ensure_aiter_fp8_parts():
    """Import the AITER_FP8 backend's varlen op and Hadamard helpers.

    ``hadamard.prepare`` resolves the vendor symbol through importlib and warms
    the 128-wide matrix; Dynamo cannot trace either, so this runs before compile.
    """
    global _AITER_FP8_VARLEN
    if _AITER_FP8_VARLEN is None:
        from xfuser.core.attention.backends.aiter_fp8.kernel import _varlen_op
        from xfuser.core.attention.numerics import hadamard

        hadamard.prepare()
        _AITER_FP8_VARLEN = (_varlen_op, hadamard)
    return _AITER_FP8_VARLEN


def _aiter_fp8_varlen(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    cu_q: torch.Tensor,
    cu_k: torch.Tensor,
    max_q: int,
    max_k: int,
    scale: float,
) -> torch.Tensor:
    """Per-tensor FP8 varlen attention. q [Tq, H, d], k/v [Tk, H, d] -> [Tq, H, d].

    Q and K are Hadamard-rotated before quantization, matching
    ``AttentionBackendType.AITER_FP8``. The quant + kernel live in the
    ``xfuser::aiter_fp8_varlen`` custom op, so torch.compile treats them as one
    opaque node.
    """
    kernel, hadamard = _ensure_aiter_fp8_parts()
    q_rot, k_rot = hadamard.rotate_qk(q, k)
    out = kernel(
        q_rot,
        k_rot,
        v.contiguous(),
        cu_q.to(device=q.device, dtype=torch.int32).contiguous(),
        cu_k.to(device=k.device, dtype=torch.int32).contiguous(),
        int(max_q),
        int(max_k),
        float(scale),
        False,
    )
    if isinstance(out, tuple):
        out = out[0]
    if out.dtype != q.dtype:
        out = out.to(dtype=q.dtype)
    return out


def _varlen(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    cu_q: torch.Tensor,
    cu_k: torch.Tensor,
    max_q: int,
    max_k: int,
    scale: float,
) -> torch.Tensor:
    if _WINDOW_BACKEND == "aiter_fp8":
        return _aiter_fp8_varlen(
            q,
            k,
            v,
            cu_q=cu_q,
            cu_k=cu_k,
            max_q=max_q,
            max_k=max_k,
            scale=scale,
        )
    if _WINDOW_BACKEND == "aiter_ck":
        return _aiter_ck_varlen(
            q,
            k,
            v,
            cu_q=cu_q,
            cu_k=cu_k,
            max_q=max_q,
            max_k=max_k,
            scale=scale,
        )
    if _WINDOW_BACKEND == "aiter":
        return _ensure_aiter_triton_op()(q, k, v, cu_q, cu_k, int(max_q), int(max_k), float(scale))
    return _sdpa_varlen(q, k, v, cu_q=cu_q, cu_k=cu_k, scale=scale)


# --------------------------------------------------------------------------
# Apply
# --------------------------------------------------------------------------


def windowed_softmax(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    plan: DecomposedPlan,
    layout: VDNH3Layout,
    scale: float,
    softmax_gate: torch.Tensor | None,
) -> torch.Tensor:
    """query/key/value: [S, H, d] packed rows (post-norm, post-RoPE for q/k) ->
    [S, H, d]; ``softmax_gate`` [S, H] scales the output per (row, head). Rows at
    and past ``used`` (padding) are zeroed."""
    used = layout.used
    out = torch.empty_like(query)
    key_used = key[:used].contiguous()
    value_used = value[:used].contiguous()
    if plan.dense_q.numel():
        out[plan.dense_q] = _varlen(
            torch.index_select(query, 0, plan.dense_q),
            key_used,
            value_used,
            cu_q=plan.dense_cu_q,
            cu_k=plan.dense_cu_k,
            max_q=int(plan.dense_q.numel()),
            max_k=used,
            scale=scale,
        )
    for window in plan.passes:
        keys = torch.index_select(key_used, 0, window.kv_rows)
        values = torch.index_select(value_used, 0, window.kv_rows)
        if window.query_slice is not None:
            start, stop = window.query_slice
            out[start:stop] = _varlen(
                query[start:stop],
                keys,
                values,
                cu_q=window.cu_q,
                cu_k=window.cu_k,
                max_q=window.max_q,
                max_k=window.max_k,
                scale=scale,
            )
        else:
            out[window.query_rows] = _varlen(
                torch.index_select(query, 0, window.query_rows),
                keys,
                values,
                cu_q=window.cu_q,
                cu_k=window.cu_k,
                max_q=window.max_q,
                max_k=window.max_k,
                scale=scale,
            )
        del keys, values
    if softmax_gate is not None:
        out.mul_(softmax_gate.to(out.dtype).unsqueeze(-1))
    if used < out.shape[0]:
        out[used:].zero_()
    return out


def dense_softmax(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    used: int,
    scale: float,
    softmax_gate: torch.Tensor | None,
) -> torch.Tensor:
    """Full (non-windowed) softmax over the ``used`` rows; the ``full_cover`` path
    where the linear branch is off. query/key/value [S, H, d] -> [S, H, d]."""
    out = torch.empty_like(query)
    cu = torch.tensor([0, used], dtype=torch.int32, device=query.device)
    out[:used] = _varlen(
        query[:used].contiguous(),
        key[:used].contiguous(),
        value[:used].contiguous(),
        cu_q=cu,
        cu_k=cu,
        max_q=used,
        max_k=used,
        scale=scale,
    )
    if softmax_gate is not None:
        out[:used].mul_(softmax_gate[:used].to(out.dtype).unsqueeze(-1))
    if used < out.shape[0]:
        out[used:].zero_()
    return out


__all__ = [
    "DecomposedPlan",
    "build_layout",
    "windowed_softmax",
    "dense_softmax",
    "window_mask_frames",
]
