# SPDX-License-Identifier: Apache-2.0
#
# Ported from OpenVDN / Video DeltaNet (VDN-H3): Apache-2.0, Copyright 2026 the
# VDN authors. https://github.com/OpenVDN/vdn-minimax-h3 (commit e262cb5)
#   source: src/models/linear_attention/{branch,delta_rule,scan,features,layers}.py
# Cross-referenced with SGLang's hybrid_window_attn_h3 diffusion integration
# (https://github.com/sgl-project/sglang, Apache-2.0).
"""VDN-H3 (Video DeltaNet MiniMax-H3) linear attention branch, eager single-GPU.

Reimplementation of OpenVDN's ``BidirectionalLinearBranch`` for xDiT. The branch
summarises everything the chunked window softmax cannot see, for every video
token, in five steps:

    0. text state      the prompt rows written once into a zero state; both
                       directional scans start from half of it
    1. features        SiLU (+ separable 5x5 spatial / 5-tap temporal depthwise
                       conv on k, v), L2-normalised q/k, NoPE
    2. frame stats     A = K^T diag(beta) K (fp32), B = V^T diag(beta) K per frame
    3. two scans       Video Delta rule S_t = (S_{t-1} diag(alpha_t) + B_t)(I + A_t)^-1
                       forward and reverse over frames
    4. boundary gather prefix[lo-1] + suffix[hi+1] decayed to frame t by
                       prod alpha over the window (the exact complement of the
                       softmax window; ends read the text state)
    5. readout         q . S -> RMSNorm(head_dim) -> low-rank sigmoid gate

Inference-only and eager (no fused Triton/CUDA kernels; the ROCm path). Parameter
names follow VDN one level below ``attn.linear_attention`` so the overlay's 800
branch tensors load unchanged. No tensor/sequence parallelism:
``local_heads == num_attention_heads``.
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

_BF16 = torch.bfloat16
_FP32 = torch.float32

# Each directional scan starts from TEXT_STATE_SCALE * S_text. Baked into the
# trained checkpoints, not a knob (see VDN BidirectionalLinearBranch).
TEXT_STATE_SCALE = 0.5
SHORT_CONV_KERNEL = 5


# --------------------------------------------------------------------------
# Packed-sequence geometry
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class VDNH3Layout:
    """Where the modalities sit in the packed H3 sequence.

    xDiT packs t2va as ``[text | audio | video | pad]``; ``used`` is the count of
    real (non-pad) rows. Text, condition and audio rows are "global" for the
    softmax branch (dense both ways); only the text rows seed the linear branch's
    state. Rows at and past ``used`` are padding and sit outside every mask.
    """

    seq_len: int
    used: int
    text_len: int
    video_start: int
    num_frames: int
    tokens_per_frame: int
    frame_height: int
    frame_width: int

    def __post_init__(self) -> None:
        if self.frame_height * self.frame_width != self.tokens_per_frame:
            raise ValueError(
                f"frame grid {self.frame_height}x{self.frame_width} != {self.tokens_per_frame} tokens per frame"
            )
        if self.video_end > self.used or self.used > self.seq_len:
            raise ValueError(
                f"video rows [{self.video_start}, {self.video_end}) exceed used "
                f"rows {self.used} (seq_len {self.seq_len})"
            )
        if self.text_len > self.video_start:
            raise ValueError("text rows must precede the video rows")

    @property
    def video_end(self) -> int:
        return self.video_start + self.num_frames * self.tokens_per_frame

    @property
    def frame_size(self) -> tuple[int, int]:
        return self.frame_height, self.frame_width

    @property
    def global_ranges(self) -> list[tuple[int, int]]:
        """Non-video, non-padding row ranges (text, condition, audio)."""
        return [(start, stop) for start, stop in ((0, self.video_start), (self.video_end, self.used)) if start < stop]

    def frame_rows(self, frame: int) -> tuple[int, int]:
        start = self.video_start + frame * self.tokens_per_frame
        return start, start + self.tokens_per_frame


# --------------------------------------------------------------------------
# The algorithm (eager, inference-only)
# --------------------------------------------------------------------------


def _temporal_shift(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    # depthwise 5-tap conv over frames, zero padded, symmetric; x [F, S, C], w [C, 5]
    k = SHORT_CONV_KERNEL
    pad = k // 2
    xp = F.pad(x, (0, 0, 0, 0, pad, pad))
    out = None
    for dt in range(k):
        part = xp[dt : dt + x.shape[0]] * w[:, dt].view(1, 1, -1)
        out = part if out is None else out + part
    return out


def _activate(tokens: torch.Tensor, l2norm: bool) -> torch.Tensor:
    x = F.silu(tokens)
    return F.normalize(x, dim=-1, eps=1e-6).to(x.dtype) if l2norm else x


def linear_features(
    tokens: torch.Tensor,
    *,
    proj: str,
    conv: "VDNShortConv | None",
    num_frames: int | None,
    frame_size: tuple[int, int] | None,
    frame_major: bool = False,
) -> torch.Tensor:
    """[N, H, d] raw projection -> [N, H, d] branch features:
    [short conv ->] SiLU [-> L2 norm for q, k]. ``frame_major`` returns
    [F, H, S, d] instead (the readout's bmm layout)."""
    l2norm = proj != "v"
    n_heads, head_dim = tokens.shape[-2], tokens.shape[-1]
    if frame_major and (num_frames is None or frame_size is None):
        raise ValueError("frame_major needs the (frames, height, width) grid")
    if conv is not None and proj in conv.targets:
        if frame_size is None or num_frames is None:
            raise ValueError("the short conv needs the (frames, height, width) grid")
        x, w_tm = conv.spatial(proj, tokens, num_frames, frame_size)
        out = _activate(_temporal_shift(x, w_tm).reshape(-1, n_heads, head_dim), l2norm)
    else:
        out = _activate(tokens, l2norm)
    if frame_major:
        per_frame = frame_size[0] * frame_size[1]
        return out.view(num_frames, per_frame, n_heads, head_dim).permute(0, 2, 1, 3)
    return out


def frame_statistics(
    kf: torch.Tensor,
    vf: torch.Tensor,
    beta: torch.Tensor,
    *,
    a_fp32: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """kf, vf [F, H, S, d], beta [F, H, S] -> A [F, H, dk, dk] fp32 symmetric,
    B [F, H, dv, dk] fp32. A is inverted downstream, so it needs fp32; B enters
    the state linearly."""
    kf = kf.contiguous()
    vf_b = (vf * beta.unsqueeze(-1).to(vf.dtype)).contiguous()
    if a_fp32:
        kf32 = kf.float()
        scaled32 = (kf32 * beta.unsqueeze(-1).float()).contiguous()
        # TF32 keeps I + A well conditioned where bf16 does not; scoped here.
        prev = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = True
        try:
            A = torch.matmul(scaled32.transpose(-1, -2), kf32)
        finally:
            torch.backends.cuda.matmul.allow_tf32 = prev
    else:
        A = torch.matmul((kf * beta.unsqueeze(-1).to(kf.dtype)).contiguous().transpose(-1, -2), kf).float()
    A = 0.5 * (A + A.transpose(-1, -2))
    B = torch.matmul(vf_b.transpose(-1, -2), kf).float()
    return A, B


def delta_factor_apply(
    rule: str,
    alpha: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    *,
    tokens_per_frame: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One frame's statistics -> (transition [F,H,dk,dk], injection [F,H,dv,dk]) fp32.

    vdn_solve  (the released checkpoints): S' = (S diag(alpha) + B)(I + A)^-1,
               exact Cholesky inverse.
    sana_scaled: S' = (S diag(alpha))(I - c^2 A) + c B, c = 1/sqrt(S).
    vdn_scaled: S' = (S diag(alpha) + c B)(I + c^2 A)^-1.
    """
    A32, B32 = A.float(), B.float()
    eye = torch.eye(A32.shape[-1], device=A32.device, dtype=_FP32).expand_as(A32)
    if rule == "sana_scaled":
        inv_tokens = 1.0 / tokens_per_frame
        transition = alpha.unsqueeze(-1) * (eye - inv_tokens * A32)
        injection = math.sqrt(inv_tokens) * B32
        return transition, injection
    if rule == "vdn_scaled":
        inv_tokens = 1.0 / tokens_per_frame
        A32 = A32 * inv_tokens
        B32 = B32 * math.sqrt(inv_tokens)
    elif rule != "vdn_solve":
        raise ValueError(f"unknown delta rule {rule!r}")
    chol = torch.linalg.cholesky(A32 + eye)
    # (I+A)^-1 = L^-T L^-1: a batched trsm at 128x128 is far slower than the GEMM
    linv = torch.linalg.solve_triangular(chol, eye, upper=False, left=True)
    inv = linv.transpose(-1, -2) @ linv
    transition = alpha.unsqueeze(-1) * inv
    injection = B32 @ inv
    return transition, injection


def run_scans(
    transitions: torch.Tensor,
    injections: torch.Tensor,
    text_state: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """prefix[t] = frames 0..t, suffix[t] = frames t..F-1 (both fp32
    [F, H, dv, dk]); both start from ``text_state`` (or zero)."""
    num_frames = transitions.shape[0]
    start = torch.zeros_like(injections[0]) if text_state is None else text_state.to(injections.dtype)
    prefix = torch.empty_like(injections)
    suffix = torch.empty_like(injections)
    state = start
    for frame in range(num_frames):
        torch.baddbmm(injections[frame], state, transitions[frame], out=prefix[frame])
        state = prefix[frame]
    state = start
    for frame in range(num_frames - 1, -1, -1):
        torch.baddbmm(injections[frame], state, transitions[frame], out=suffix[frame])
        state = suffix[frame]
    return prefix, suffix


def _compose_chunk(
    transitions: torch.Tensor, injections: torch.Tensor, reverse: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    # fold each chunk's frames into one affine map S -> S @ M + C, batched over chunks
    order = list(range(transitions.shape[0]))
    if reverse:
        order.reverse()
    chunks, heads = transitions.shape[1], transitions.shape[2]
    dk, dv = transitions.shape[-1], injections.shape[-2]
    folded_t = transitions[order[0]]
    folded_b = injections[order[0]]
    for j in order[1:]:
        step_t = transitions[j].view(chunks * heads, dk, dk)
        folded_b = torch.baddbmm(
            injections[j].view(chunks * heads, dv, dk),
            folded_b.view(chunks * heads, dv, dk),
            step_t,
        ).view(chunks, heads, dv, dk)
        folded_t = torch.bmm(folded_t.view(chunks * heads, dk, dk), step_t).view(chunks, heads, dk, dk)
    return folded_t, folded_b


@functools.lru_cache(maxsize=64)
def _boundary_frames(
    num_frames: int, chunk: int, frame_offset: int, device: str
) -> tuple[int, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # the gather reads prefix at chunk ends and suffix at chunk starts, on the offset grid
    padded = frame_offset + num_frames
    num_chunks = -(-padded // chunk)
    ends = [min((c + 1) * chunk - 1, padded - 1) - frame_offset for c in range(num_chunks)]
    starts = [c * chunk - frame_offset for c in range(num_chunks)]
    dev = torch.device(device)
    ends = [(f, c) for c, f in enumerate(ends) if f >= 0]
    starts = [(f, c) for c, f in enumerate(starts) if f >= 0]
    return (
        num_chunks,
        torch.tensor([f for f, _ in ends], device=dev),
        torch.tensor([c for _, c in ends], device=dev),
        torch.tensor([f for f, _ in starts], device=dev),
        torch.tensor([c for _, c in starts], device=dev),
    )


def run_boundary_scans(
    transitions: torch.Tensor,
    injections: torch.Tensor,
    text_state: torch.Tensor | None,
    *,
    chunk: int,
    frame_offset: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``run_scans`` restricted to what the chunked gather reads: prefix at each
    chunk's last frame, suffix at each chunk's first frame, zero elsewhere."""
    if chunk <= 1:
        return run_scans(transitions, injections, text_state)
    num_frames, heads, dv, dk = injections.shape
    num_chunks, ends, end_chunks, starts, start_chunks = _boundary_frames(
        num_frames, chunk, frame_offset, str(injections.device)
    )
    # identity / zero padding fills the leading offset and the partial last chunk
    lead, tail = frame_offset, num_chunks * chunk - frame_offset - num_frames
    eye = torch.eye(dk, device=transitions.device, dtype=transitions.dtype)
    transitions = torch.cat([eye.expand(lead, heads, dk, dk), transitions, eye.expand(tail, heads, dk, dk)])
    injections = torch.cat(
        [
            injections.new_zeros(lead, heads, dv, dk),
            injections,
            injections.new_zeros(tail, heads, dv, dk),
        ]
    )
    # frame-major so each composition step reads contiguous operands
    by_frame_t = transitions.view(num_chunks, chunk, heads, dk, dk).transpose(0, 1).contiguous()
    by_frame_b = injections.view(num_chunks, chunk, heads, dv, dk).transpose(0, 1).contiguous()
    start = (
        torch.zeros(heads, dv, dk, dtype=injections.dtype, device=injections.device)
        if text_state is None
        else text_state.to(injections.dtype)
    )
    fwd_t, fwd_b = _compose_chunk(by_frame_t, by_frame_b, reverse=False)
    rev_t, rev_b = _compose_chunk(by_frame_t, by_frame_b, reverse=True)
    chunk_t = torch.stack([fwd_t, rev_t.flip(0)], dim=1)  # [C, 2, H, dk, dk]
    boundary = torch.stack([fwd_b, rev_b.flip(0)], dim=1)  # [C, 2, H, dv, dk]
    flat = boundary.view(num_chunks, 2 * heads, dv, dk)
    state = torch.stack([start, start], dim=0).view(2 * heads, dv, dk)
    for c in range(num_chunks):
        flat[c].baddbmm_(state, chunk_t[c].view(2 * heads, dk, dk))
        state = flat[c]
    prefix = torch.zeros(num_frames, heads, dv, dk, dtype=injections.dtype, device=injections.device)
    suffix = torch.zeros_like(prefix)
    prefix.index_copy_(0, ends, boundary[end_chunks, 0])
    suffix.index_copy_(0, starts, boundary[num_chunks - 1 - start_chunks, 1])
    return prefix, suffix


@functools.lru_cache(maxsize=64)
def _gather_indices(bounds: tuple[tuple[int, int], ...], num_frames: int, device: str) -> tuple[torch.Tensor, ...]:
    dev = torch.device(device)
    last_before = torch.tensor([lo for lo, _ in bounds], device=dev) - 1
    first_after = torch.tensor([hi for _, hi in bounds], device=dev) + 1
    return (
        last_before,
        first_after,
        last_before.clamp(min=0),
        first_after.clamp(max=num_frames - 1),
        last_before >= 0,
        first_after < num_frames,
        torch.arange(num_frames, device=dev),
    )


def gather_linear_state(
    prefix: torch.Tensor,
    suffix: torch.Tensor,
    alpha: torch.Tensor,
    bounds: list[tuple[int, int]],
    *,
    bridge: str,
    text_state: torch.Tensor | None,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """Everything OUTSIDE the softmax window of frame t, decayed to t:
    prefix[lo-1] * prod_{u=lo..t} alpha_u + suffix[hi+1] * prod_{u=t..hi} alpha_u.
    Out-of-range sides read the text state (the scans' virtual start) when one was
    given, else contribute nothing. -> [F, H, dv, dk] in ``out_dtype``."""
    num_frames = prefix.shape[0]
    (
        last_before,
        first_after,
        before_idx,
        after_idx,
        has_before,
        has_after,
        frames,
    ) = _gather_indices(tuple(bounds), num_frames, str(prefix.device))
    state_before = prefix[before_idx]
    state_after = suffix[after_idx]
    if text_state is not None:
        ts = text_state.to(state_before.dtype)
        state_before = torch.where(has_before.view(-1, 1, 1, 1), state_before, ts)
        state_after = torch.where(has_after.view(-1, 1, 1, 1), state_after, ts)
    if bridge == "alpha":
        log_alpha = torch.log(alpha.clamp_min(1e-12))
        log_prefix = torch.cat([torch.zeros_like(log_alpha[:1]), log_alpha.cumsum(0)])
        bridge_before = (last_before + 1).clamp(min=0)
        bridge_after = first_after.clamp(max=num_frames)
        alpha_from_before = torch.exp(log_prefix[frames + 1] - log_prefix[bridge_before])
        alpha_from_after = torch.exp(log_prefix[bridge_after] - log_prefix[frames])
        # alpha is per KEY channel: broadcast over dv, not dk
        state_before = state_before * alpha_from_before.unsqueeze(2)
        state_after = state_after * alpha_from_after.unsqueeze(2)
    elif bridge != "none":
        raise ValueError(f"unknown bridge {bridge!r}")
    if text_state is not None:
        out = state_before + state_after
    else:
        out = state_before * has_before.view(-1, 1, 1, 1) + state_after * has_after.view(-1, 1, 1, 1)
    return out.to(out_dtype)


def linear_epilogue(readout: torch.Tensor, norm_weight: torch.Tensor, gate: torch.Tensor, eps: float) -> torch.Tensor:
    """readout [F, H, S, dv] -> RMSNorm over dv -> * gate [F*S, H, dv] -> [F*S, H*dv]."""
    ms = torch.linalg.vector_norm(readout, dim=-1, keepdim=True, dtype=_FP32).pow(2) / (readout.shape[-1])
    normed = readout * torch.rsqrt(ms + eps).to(readout.dtype) * norm_weight.to(readout.dtype)
    frames, heads, per_frame, dim = normed.shape
    rows = frames * per_frame
    return normed.permute(0, 2, 1, 3).reshape(rows, heads * dim) * gate.reshape(rows, heads * dim)


# --------------------------------------------------------------------------
# Submodules (parameter names match the overlay's branch keys)
# --------------------------------------------------------------------------


class VDNFrameAlpha(nn.Module):
    """alpha_t = exp(-exp(A_log) * softplus(up(down(frame_mean)) + dt_bias)),
    per frame / head / key channel, in fp32 (KDA's double-exponential gate)."""

    def __init__(self, hidden_size: int, heads: int, head_dim: int) -> None:
        super().__init__()
        self.heads, self.head_dim = heads, head_dim
        self.down = nn.Linear(hidden_size, head_dim, bias=False, dtype=_BF16)
        self.up = nn.Linear(head_dim, heads * head_dim, bias=False, dtype=_BF16)
        # fp32: the scan multiplies alpha over ~100 frames, so bf16 error compounds
        self.A_log = nn.Parameter(torch.empty(heads, dtype=_FP32), requires_grad=False)
        self.dt_bias = nn.Parameter(torch.empty(heads * head_dim, dtype=_FP32), requires_grad=False)

    def forward(self, frame_mean: torch.Tensor) -> torch.Tensor:
        """frame_mean [F, hidden] fp32 -> alpha [F, H, d] fp32."""
        delta = F.linear(frame_mean.float(), self.down.weight.float())
        delta = F.linear(delta, self.up.weight.float()) + self.dt_bias.float()
        scale = torch.exp(self.A_log.float())[:, None]
        delta = delta.view(-1, self.heads, self.head_dim)
        return torch.exp(-scale * F.softplus(delta))


class VDNOutputGate(nn.Module):
    """Low-rank sigmoid gate: sigmoid(up(down(x))) -> [T, H, d]."""

    def __init__(self, hidden_size: int, heads: int, head_dim: int) -> None:
        super().__init__()
        self.heads, self.head_dim = heads, head_dim
        self.down = nn.Linear(hidden_size, head_dim, bias=False, dtype=_BF16)
        self.up = nn.Linear(head_dim, heads * head_dim, bias=True, dtype=_BF16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up_gate(self.down(x))

    def up_gate(self, hidden: torch.Tensor) -> torch.Tensor:
        gate = self.up(hidden)
        return torch.sigmoid(gate).view(-1, self.heads, self.head_dim)


class VDNSoftmaxGate(nn.Module):
    """Per-(token, head) sigmoid gate on the softmax branch output."""

    def __init__(self, hidden_size: int, heads: int) -> None:
        super().__init__()
        self.up = nn.Linear(hidden_size, heads, bias=True, dtype=_BF16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.up(x))


class VDNShortConv(nn.Module):
    """Separable depthwise short conv (5x5 spatial per frame, then 5 taps across
    frames) on the projections named in ``targets``; channels are head-major."""

    def __init__(self, channels: int, targets: tuple[str, ...]) -> None:
        super().__init__()
        self.targets = tuple(targets)
        k = SHORT_CONV_KERNEL
        for name in self.targets:
            setattr(
                self,
                f"{name}_sp",
                nn.ParameterDict(
                    {
                        "weight": nn.Parameter(
                            torch.empty(channels, 1, k, k, dtype=_BF16),
                            requires_grad=False,
                        )
                    }
                ),
            )
            setattr(
                self,
                f"{name}_tm",
                nn.ParameterDict(
                    {
                        "weight": nn.Parameter(
                            torch.empty(channels, 1, k, dtype=_BF16),
                            requires_grad=False,
                        )
                    }
                ),
            )

    def spatial(
        self,
        proj: str,
        tokens: torch.Tensor,
        num_frames: int,
        frame_size: tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The 5x5 depthwise half on [F*S, H, d] tokens -> ([F, S, C], w_tm [C, 5])."""
        n_heads, head_dim = tokens.shape[-2], tokens.shape[-1]
        grid_h, grid_w = frame_size
        channels = n_heads * head_dim
        w_sp = getattr(self, f"{proj}_sp")["weight"]
        w_tm = getattr(self, f"{proj}_tm")["weight"]
        volume = tokens.reshape(num_frames, grid_h, grid_w, channels).permute(0, 3, 1, 2)
        volume = F.conv2d(volume, w_sp, padding=SHORT_CONV_KERNEL // 2, groups=channels)
        x = volume.permute(0, 2, 3, 1).reshape(num_frames, grid_h * grid_w, channels)
        return x, w_tm.squeeze(1).to(x.dtype)


def _branch_norm(dim: int, eps: float = 1e-6) -> nn.RMSNorm:
    # weight holder only; the arithmetic runs in the epilogue (fp32 second moment)
    return nn.RMSNorm(dim, eps=eps, dtype=_BF16)


# --------------------------------------------------------------------------
# The module
# --------------------------------------------------------------------------


class MiniMaxH3VDNLinearBranch(nn.Module):
    """VDN's BidirectionalLinearBranch, single-GPU (all heads local)."""

    def __init__(
        self,
        hybrid: "VDNHybridAttentionArchConfig",
        *,
        hidden_size: int,
        num_attention_heads: int,
        attention_head_dim: int,
    ) -> None:
        super().__init__()
        if hybrid.linear_head_dim != attention_head_dim:
            raise ValueError(
                f"hybrid_attention.linear_head_dim={hybrid.linear_head_dim} != attention_head_dim={attention_head_dim}"
            )
        self.hybrid = hybrid
        self.heads = num_attention_heads
        self.head_dim = attention_head_dim
        channels = num_attention_heads * self.head_dim
        self.short_conv = VDNShortConv(channels, hybrid.short_conv) if hybrid.short_conv else None
        self.alpha = VDNFrameAlpha(hidden_size, num_attention_heads, self.head_dim)
        self.beta_proj = nn.Linear(hidden_size, num_attention_heads, bias=False, dtype=_BF16)
        self.output_gate = VDNOutputGate(hidden_size, num_attention_heads, self.head_dim)
        self.norm = _branch_norm(self.head_dim)

    # ---- pieces the attention module computes on the row shard ----

    def beta(self, x: torch.Tensor) -> torch.Tensor:
        """x [T, hidden] -> beta [T, H] (sigmoid)."""
        return torch.sigmoid(self.beta_proj(x))

    # ---- the text state ----

    def text_statistics(
        self,
        text_k_raw: torch.Tensor,
        text_v_raw: torch.Tensor,
        text_beta: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        """(A [1, H, dk, dk], B [1, H, dv, dk]) fp32 of the prompt rows, no conv."""
        length = text_k_raw.shape[0]
        heads, head_dim = text_k_raw.shape[1], self.head_dim
        key = linear_features(text_k_raw, proj="k", conv=None, num_frames=None, frame_size=None)
        value = linear_features(text_v_raw, proj="v", conv=None, num_frames=None, frame_size=None)
        key = key.view(1, length, heads, head_dim).permute(0, 2, 1, 3)
        value = value.view(1, length, heads, head_dim).permute(0, 2, 1, 3)
        beta = text_beta.view(1, length, heads).permute(0, 2, 1)
        A, B = frame_statistics(key, value, beta, a_fp32=self.hybrid.a_fp32)
        return A, B, length

    def text_state(
        self,
        text_k_raw: torch.Tensor,
        text_v_raw: torch.Tensor,
        text_beta: torch.Tensor,
    ) -> torch.Tensor:
        """S_text [H, dv, dk] fp32: the prompt written into a zero state as one
        delta-rule chunk, scaled by TEXT_STATE_SCALE."""
        A, B, length = self.text_statistics(text_k_raw, text_v_raw, text_beta)
        heads = A.shape[1]
        ones = torch.ones(1, heads, self.head_dim, device=A.device, dtype=_FP32)
        _, injection = delta_factor_apply(self.hybrid.delta_rule, ones, A, B, tokens_per_frame=length)
        return TEXT_STATE_SCALE * injection[0]

    # ---- the branch ----

    def forward(
        self,
        *,
        q_raw: torch.Tensor,
        k_raw: torch.Tensor,
        v_raw: torch.Tensor,
        beta: torch.Tensor,
        gate: torch.Tensor,
        frame_mean: torch.Tensor,
        layout: VDNH3Layout,
        text_k_raw: torch.Tensor | None = None,
        text_v_raw: torch.Tensor | None = None,
        text_beta: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Linear readout of the video rows, [V, H * d] in q's dtype."""
        hybrid = self.hybrid
        num_frames, per_frame = layout.num_frames, layout.tokens_per_frame
        bounds = hybrid.window_bounds(num_frames)
        text_state = None
        text_stats = None
        if hybrid.enable_text_state:
            if text_k_raw is None or text_v_raw is None or text_beta is None:
                raise ValueError("enable_text_state needs the prompt rows' k/v/beta")
            if text_k_raw.shape[0] > 0:
                if hybrid.delta_rule == "vdn_solve":
                    A_text, B_text, _ = self.text_statistics(text_k_raw, text_v_raw, text_beta)
                    text_stats = (A_text, B_text)
                else:
                    text_state = self.text_state(text_k_raw, text_v_raw, text_beta)

        skip_ends = hybrid.anchor_frames == "both"
        n_heads = q_raw.shape[1]
        if not skip_ends:
            return self._readout(
                q_raw,
                k_raw,
                v_raw,
                beta,
                gate,
                frame_mean,
                num_frames,
                per_frame,
                bounds,
                layout.frame_size,
                text_state,
                text_stats=text_stats,
            )
        out = q_raw.new_empty(num_frames * per_frame, n_heads * self.head_dim)
        if num_frames <= 2:
            return out.zero_()
        inner = slice(per_frame, (num_frames - 1) * per_frame)
        readout = self._readout(
            q_raw[inner],
            k_raw[inner],
            v_raw[inner],
            beta[inner],
            gate[inner],
            frame_mean[1:-1],
            num_frames - 2,
            per_frame,
            [(lo - 1, hi - 1) for lo, hi in bounds[1 : num_frames - 1]],
            layout.frame_size,
            text_state,
            frame_offset=1,
            text_stats=text_stats,
        )
        out[:per_frame].zero_()
        out[(num_frames - 1) * per_frame :].zero_()
        out[inner] = readout
        return out

    def _readout(
        self,
        q_raw: torch.Tensor,
        k_raw: torch.Tensor,
        v_raw: torch.Tensor,
        beta: torch.Tensor,
        gate: torch.Tensor,
        frame_mean: torch.Tensor,
        num_frames: int,
        per_frame: int,
        bounds: list[tuple[int, int]],
        frame_size: tuple[int, int],
        text_state: torch.Tensor | None,
        frame_offset: int = 0,
        text_stats: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        n_heads, head_dim = q_raw.shape[1], self.head_dim
        shape = (num_frames, per_frame, n_heads, head_dim)
        features = functools.partial(
            linear_features,
            conv=self.short_conv,
            num_frames=num_frames,
            frame_size=frame_size,
        )
        query_by_frame = features(q_raw, proj="q", frame_major=True)
        key = features(k_raw, proj="k")
        value = features(v_raw, proj="v")
        key_by_frame = key.view(shape).permute(0, 2, 1, 3)
        value_by_frame = value.view(shape).permute(0, 2, 1, 3)
        beta_by_frame = beta.view(num_frames, per_frame, n_heads).permute(0, 2, 1)
        A, B = frame_statistics(key_by_frame, value_by_frame, beta_by_frame, a_fp32=self.hybrid.a_fp32)
        alpha = self.alpha(frame_mean)
        if text_stats is not None:
            # the prompt leads as a virtual frame; alpha 1 since its old state is zero
            A = torch.cat([text_stats[0], A])
            B = torch.cat([text_stats[1], B])
            alpha_all = torch.cat([alpha.new_ones((1,) + alpha.shape[1:]), alpha])
        else:
            alpha_all = alpha
        transitions, injections = delta_factor_apply(
            self.hybrid.delta_rule, alpha_all, A, B, tokens_per_frame=per_frame
        )
        if text_stats is not None:
            text_state = TEXT_STATE_SCALE * injections[0]
            transitions, injections = transitions[1:], injections[1:]
        prefix, suffix = run_boundary_scans(
            transitions,
            injections,
            text_state,
            chunk=self.hybrid.chunk,
            frame_offset=frame_offset,
        )
        del transitions, injections
        linear_state = gather_linear_state(
            prefix,
            suffix,
            alpha,
            bounds,
            bridge=self.hybrid.bridge,
            text_state=text_state,
            out_dtype=q_raw.dtype,
        )
        del prefix, suffix
        readout = torch.matmul(query_by_frame, linear_state.transpose(-1, -2))
        return linear_epilogue(readout, self.norm.weight, gate, self.norm.eps)


# Late import to avoid a cycle at module import time.
from xfuser.model_executor.models.transformers.vdn_minimax_h3.config import (  # noqa: E402
    VDNHybridAttentionArchConfig,
)

__all__ = [
    "MiniMaxH3VDNLinearBranch",
    "VDNH3Layout",
    "VDNFrameAlpha",
    "VDNOutputGate",
    "VDNSoftmaxGate",
    "VDNShortConv",
    "TEXT_STATE_SCALE",
]
