# SPDX-License-Identifier: Apache-2.0
#
# Ported from OpenVDN / Video DeltaNet (VDN-H3): Apache-2.0, Copyright 2026 the
# VDN authors. https://github.com/OpenVDN/vdn-minimax-h3 (commit e262cb5)
#   source: src/models/{hybrid_attention,attention_gates}.py,
#           src/models/softmax_attention/dense_processor.py
# Cross-referenced with SGLang's hybrid_window_attn_h3 diffusion integration
# (https://github.com/sgl-project/sglang, Apache-2.0).
"""VDN-H3 hybrid attention for xDiT: window softmax + Video-Delta linear branch.

The diffusers ``MiniMaxH3Attention`` keeps its dense q/k/v/out projections; VDN
adds four submodules whose names match the overlay's branch checkpoint keys:

    attn.linear_attention   MiniMaxH3VDNLinearBranch (alpha, beta_proj,
                            output_gate, short_conv, norm)
    attn.softmax_gate       VDNSoftmaxGate (per-(token, head) sigmoid gate)
    attn.to_out_linear      Linear(heads*linear_head_dim -> hidden), the branch's
                            own output projection (added to the video rows only)

The processor computes, from the block's normed-modulated hidden states x:

    window branch : softmax over the chunk-aligned frame window on norm+RoPE(q,k),
                    gated per (row, head) by softmax_gate(x), out-projected by
                    to_out[0]
    linear branch : Video-Delta readout over the pre-norm q/k/v of the video rows
                    (text rows seed the state), out-projected by to_out_linear,
                    added onto the video rows only

Batch size 1. Single-GPU and head-sharded Ulysses are both supported (the
processor all-to-alls q/k/v into a per-head shard). The per-request geometry
(layout + window decomposition) is primed once per denoise by the transformer
wrapper and stashed on ``attn._vdn_meta``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

from diffusers.models.transformers.transformer_minimax_h3 import _apply_rotary_emb

from xfuser.core.distributed import (
    get_sp_group,
    get_ulysses_parallel_rank,
    get_ulysses_parallel_world_size,
)
from xfuser.model_executor.models.transformers.vdn_minimax_h3.config import (
    VDNHybridAttentionArchConfig,
)
from xfuser.model_executor.layers.usp import (
    _ft_c_input_all_to_all,
    _ft_c_output_all_to_all,
)
from xfuser.model_executor.models.transformers.vdn_minimax_h3.linear_branch import (
    MiniMaxH3VDNLinearBranch,
    VDNFrameAlpha,
    VDNH3Layout,
    VDNShortConv,
    VDNSoftmaxGate,
    _BF16,
)
from xfuser.model_executor.models.transformers.vdn_minimax_h3.window_softmax import (
    DecomposedPlan,
    build_layout,
    dense_softmax,
    prepare_window_backend,
    windowed_softmax,
)


@dataclass
class VDNRequestMeta:
    """Per-denoise geometry, primed once by the wrapper and shared by every block."""

    layout: VDNH3Layout
    plan: DecomposedPlan | None  # None when the window covers the full sequence
    full_cover: bool


def build_vdn_meta(
    hybrid: VDNHybridAttentionArchConfig,
    *,
    position_ids: torch.Tensor,
    text_indices: torch.Tensor,
    audio_indices: torch.Tensor,
    video_indices: torch.Tensor,
    padded_length: int,
) -> VDNRequestMeta:
    used = int(position_ids.shape[0])
    layout = build_layout(
        position_ids=position_ids,
        text_indices=text_indices,
        audio_indices=audio_indices,
        video_indices=video_indices,
        used=used,
        seq_len=int(padded_length),
    )
    full_cover = hybrid.full_cover(layout.num_frames)
    plan = None if full_cover else DecomposedPlan(layout, hybrid, position_ids.device)
    return VDNRequestMeta(layout=layout, plan=plan, full_cover=full_cover)


def attach_vdn_modules(
    attn: nn.Module,
    hybrid: VDNHybridAttentionArchConfig,
    *,
    hidden_size: int,
    num_attention_heads: int,
    attention_head_dim: int,
) -> None:
    """Add the VDN branch submodules to a diffusers ``MiniMaxH3Attention`` so the
    overlay's ``attn.{linear_attention,softmax_gate,to_out_linear}.*`` keys load."""
    linear_channels = num_attention_heads * hybrid.linear_head_dim
    attn.linear_attention = MiniMaxH3VDNLinearBranch(
        hybrid,
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        attention_head_dim=attention_head_dim,
    )
    attn.softmax_gate = VDNSoftmaxGate(hidden_size, num_attention_heads) if hybrid.enable_softmax_gate else None
    attn.to_out_linear = nn.Linear(linear_channels, hidden_size, bias=False, dtype=_BF16)
    attn._vdn_meta = None
    attn.set_processor(VDNMiniMaxH3AttnProcessor(hybrid))
    prepare_window_backend()


# --------------------------------------------------------------------------
# Head-sharded (Ulysses) helpers
# --------------------------------------------------------------------------
#
# Under Ulysses SP each rank owns a contiguous head range and the FULL sequence
# (via all-to-all). The window softmax needs no per-head parameters. The linear
# branch's forward only touches head-dependent parameters through ``short_conv``
# and ``alpha`` (``beta``/``output_gate``/``softmax_gate`` are computed on the
# sequence shard with all heads BEFORE the all-to-all, and ``norm`` is over the
# head dim, so it is head-independent). We therefore build a per-rank branch that
# reuses the validated forward with head-sliced ``short_conv`` and ``alpha``.


def _slice_alpha(alpha: VDNFrameAlpha, h0: int, h1: int, d: int) -> VDNFrameAlpha:
    local = VDNFrameAlpha.__new__(VDNFrameAlpha)
    nn.Module.__init__(local)
    local.heads = h1 - h0
    local.head_dim = d
    local.down = alpha.down  # hidden -> head_dim, head-independent (shared)
    up = nn.Linear(
        d,
        (h1 - h0) * d,
        bias=False,
        dtype=alpha.up.weight.dtype,
        device=alpha.up.weight.device,
    )
    up.weight = nn.Parameter(alpha.up.weight[h0 * d : h1 * d].clone(), requires_grad=False)
    local.up = up
    local.A_log = nn.Parameter(alpha.A_log[h0:h1].clone(), requires_grad=False)
    local.dt_bias = nn.Parameter(alpha.dt_bias[h0 * d : h1 * d].clone(), requires_grad=False)
    return local


def _slice_short_conv(sc: VDNShortConv | None, h0: int, h1: int, d: int) -> VDNShortConv | None:
    if sc is None:
        return None
    local = VDNShortConv.__new__(VDNShortConv)
    nn.Module.__init__(local)
    local.targets = sc.targets
    c0, c1 = h0 * d, h1 * d  # channels are head-major
    for name in sc.targets:
        sp = getattr(sc, f"{name}_sp")["weight"][c0:c1].clone()
        tm = getattr(sc, f"{name}_tm")["weight"][c0:c1].clone()
        setattr(
            local,
            f"{name}_sp",
            nn.ParameterDict({"weight": nn.Parameter(sp, requires_grad=False)}),
        )
        setattr(
            local,
            f"{name}_tm",
            nn.ParameterDict({"weight": nn.Parameter(tm, requires_grad=False)}),
        )
    return local


def _build_local_branch(branch: MiniMaxH3VDNLinearBranch, h0: int, h1: int, d: int) -> MiniMaxH3VDNLinearBranch:
    """A branch whose forward runs on the rank's head slice [h0, h1)."""
    local = MiniMaxH3VDNLinearBranch.__new__(MiniMaxH3VDNLinearBranch)
    nn.Module.__init__(local)
    local.hybrid = branch.hybrid
    local.heads = h1 - h0
    local.head_dim = d
    local.norm = branch.norm  # RMSNorm(head_dim): head-independent (shared)
    local.short_conv = _slice_short_conv(branch.short_conv, h0, h1, d)
    local.alpha = _slice_alpha(branch.alpha, h0, h1, d)
    # forward() never reads these, but keep references for module completeness.
    local.beta_proj = branch.beta_proj
    local.output_gate = branch.output_gate
    return local


def ensure_ulysses_local_branch(attn: nn.Module) -> MiniMaxH3VDNLinearBranch | None:
    """Build (once) and return this rank's head-sliced linear branch.

    It creates ``nn.Parameter``s, which Dynamo cannot trace, so the transformer's
    eager pre-hook calls this before the compiled forward ever reaches it.
    """
    world = get_ulysses_parallel_world_size()
    if world <= 1:
        return None
    local = getattr(attn, "_vdn_local_branch", None)
    if local is None:
        h_local = attn.heads // world
        h0 = get_ulysses_parallel_rank() * h_local
        local = _build_local_branch(attn.linear_attention, h0, h0 + h_local, attn.linear_attention.head_dim)
        attn._vdn_local_branch = local
    return local


def _seq_to_head_a2a(t: torch.Tensor) -> torch.Tensor:
    """[S_local, H, d] -> [S_full, H_local, d] (scatter heads, gather sequence)."""
    t = t.permute(1, 0, 2).unsqueeze(0).contiguous()  # [1, H, S_local, d]
    t = _ft_c_input_all_to_all(t)  # [1, H_local, S_full, d]
    return t[0].permute(1, 0, 2).contiguous()  # [S_full, H_local, d]


def _head_to_seq_a2a(t: torch.Tensor) -> torch.Tensor:
    """[S_full, H_local, d] -> [S_local, H, d] (gather heads, scatter sequence)."""
    t = t.permute(1, 0, 2).unsqueeze(0).contiguous()  # [1, H_local, S_full, d]
    t = _ft_c_output_all_to_all(t)  # [1, H, S_local, d]
    return t[0].permute(1, 0, 2).contiguous()  # [S_local, H, d]


class VDNMiniMaxH3AttnProcessor:
    """Hybrid attention: window softmax + Video-Delta linear branch.

    Single-GPU and head-sharded (Ulysses) both supported; batch size 1.
    """

    def __init__(self, hybrid: VDNHybridAttentionArchConfig) -> None:
        self.hybrid = hybrid

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if attention_mask is not None:
            raise ValueError(
                "VDN-H3 attention expects padding via the varlen metadata primed "
                "by the transformer wrapper, not an attention_mask."
            )
        meta: VDNRequestMeta | None = getattr(attn, "_vdn_meta", None)
        if meta is None:
            raise RuntimeError(
                "VDN attention was called without primed geometry; the VDN "
                "transformer wrapper must set attn._vdn_meta before forward."
            )
        if hidden_states.shape[0] != 1:
            raise NotImplementedError(f"VDN-H3 port handles batch size 1, got {hidden_states.shape[0]}.")

        world = get_ulysses_parallel_world_size()
        if world <= 1:
            return self._compute(attn, hidden_states, rotary_emb, meta)
        return self._compute_ulysses(attn, hidden_states, rotary_emb, meta, world)

    def _video_frame_mean(self, x_local: torch.Tensor, rank: int, s_local: int, layout: "VDNH3Layout") -> torch.Tensor:
        """Per-frame mean of x over the video rows, reduced across the SP group.

        The frame mean is a reduction over the full hidden dim, so every rank needs
        the whole frame even though it only holds a sequence shard. Each rank sums
        the video rows it owns into the frame they belong to; an all-reduce then
        completes the sum. The per-frame token count is globally fixed
        (``tokens_per_frame``), so the mean is the reduced sum divided by it.

        The video rows are contiguous, so the shard's share of them is a host-int
        slice: no device reads, which keeps the block loop inside one compiled frame.
        """
        channels = x_local.shape[1]
        num_frames, tpf = layout.num_frames, layout.tokens_per_frame
        sums = x_local.new_zeros(num_frames, channels, dtype=torch.float32)
        if num_frames > 0:
            # The rank owns the contiguous global rows [g0, g0 + s_local); its
            # video rows are therefore [max(g0, video_start), min(end, video_end)).
            g0 = rank * s_local
            lo = max(g0, layout.video_start)
            hi = min(g0 + s_local, layout.video_end)
            if lo < hi:
                rows = x_local[lo - g0 : hi - g0].float()
                frame = (torch.arange(lo, hi, device=x_local.device) - layout.video_start) // tpf
                sums.index_add_(0, frame, rows)
        sums = get_sp_group().all_reduce(sums)
        return sums / tpf

    def _compute_ulysses(
        self,
        attn,
        hidden_states: torch.Tensor,
        rotary_emb: tuple[torch.Tensor, torch.Tensor] | None,
        meta: "VDNRequestMeta",
        world: int,
    ) -> torch.Tensor:
        """Head-sharded Ulysses hybrid attention.

        Each rank projects its sequence shard with ALL heads, computes the
        per-(token, head) gates locally, then all-to-alls the per-head tensors into
        [full sequence, local heads]; QK-norm and RoPE are applied after the
        all-to-all. Both branches run on the local head slice, so
        the window softmax and the Video-Delta scan FLOPs are divided across ranks.
        The per-head outputs are all-to-all'd back to [local sequence, all heads] and
        the shared output projections finish on the sequence shard. ``frame_mean``
        (a full-hidden reduction) is completed with a cheap [F, C] all-reduce.
        """
        layout = meta.layout
        heads = attn.heads
        head_dim = (
            attn.to_q.out_features // heads if not attn.fused_projections else attn.to_qkv.out_features // (3 * heads)
        )
        scale = 1.0 / math.sqrt(head_dim)
        rank = get_ulysses_parallel_rank()
        h_local = heads // world

        x = hidden_states[0]  # [S_local, C]
        s_local = x.shape[0]

        # ---- projections on the sequence shard, all heads --------------------
        if attn.fused_projections:
            q, k, v = attn.to_qkv(hidden_states).chunk(3, dim=-1)
        else:
            q = attn.to_q(hidden_states)
            k = attn.to_k(hidden_states)
            v = attn.to_v(hidden_states)
        q = q.unflatten(-1, (heads, -1))  # [1, S_local, H, d]
        k = k.unflatten(-1, (heads, -1))
        v = v.unflatten(-1, (heads, -1))

        # per-(token, head) gates, computed with all heads on the shard ---------
        softmax_gate = attn.softmax_gate(x) if attn.softmax_gate is not None else None
        branch = attn.linear_attention
        need_linear = (not meta.full_cover) and layout.num_frames > 0
        if need_linear:
            beta = branch.beta(x)  # [S_local, H]
            gate = branch.output_gate(x)  # [S_local, H, d]
            frame_mean = self._video_frame_mean(x, rank, s_local, layout)  # [F, C]

        # ---- scatter heads / gather sequence ---------------------------------
        # Raw q/k cross once and feed both branches. QK-norm is per head and RoPE
        # per token, so the window branch applies them after the all-to-all;
        # rotary_emb covers the full padded sequence (_rope_full_sequence).
        q_raw = _seq_to_head_a2a(q[0])  # [S_full, H_local, d]
        k_raw = _seq_to_head_a2a(k[0])
        v_sm = _seq_to_head_a2a(v[0])
        q_sm = attn.norm_q(q_raw)
        k_sm = attn.norm_k(k_raw)
        if rotary_emb is not None:
            q_sm = _apply_rotary_emb(q_sm.unsqueeze(0), *rotary_emb)[0]
            k_sm = _apply_rotary_emb(k_sm.unsqueeze(0), *rotary_emb)[0]
        if softmax_gate is not None:
            softmax_gate = _seq_to_head_a2a(softmax_gate.unsqueeze(-1)).squeeze(-1)

        # ---- window (softmax) branch, local heads over the full sequence ------
        if meta.full_cover:
            window = dense_softmax(
                q_sm,
                k_sm,
                v_sm,
                used=layout.used,
                scale=scale,
                softmax_gate=softmax_gate,
            )
        else:
            window = windowed_softmax(
                q_sm,
                k_sm,
                v_sm,
                plan=meta.plan,
                layout=layout,
                scale=scale,
                softmax_gate=softmax_gate,
            )  # [S_full, H_local, d]

        # gather heads / scatter sequence, then the shared output projection ----
        window = _head_to_seq_a2a(window)  # [S_local, H, d]
        out = attn.to_out[0](window.reshape(s_local, heads * head_dim))

        # ---- linear (Video-Delta) branch, local heads over the video rows -----
        if need_linear:
            v_raw = v_sm  # v is shared between the branches
            beta = _seq_to_head_a2a(beta.unsqueeze(-1)).squeeze(-1)  # [S_full, H_local]
            gate = _seq_to_head_a2a(gate)  # [S_full, H_local, d]

            local_branch = ensure_ulysses_local_branch(attn)

            video = slice(layout.video_start, layout.video_end)
            text = slice(0, layout.text_len)
            text_k = k_raw[text] if branch.hybrid.enable_text_state else None
            text_v = v_raw[text] if branch.hybrid.enable_text_state else None
            text_beta = beta[text] if branch.hybrid.enable_text_state else None
            linear = local_branch(
                q_raw=q_raw[video],
                k_raw=k_raw[video],
                v_raw=v_raw[video],
                beta=beta[video],
                gate=gate[video],
                frame_mean=frame_mean,
                layout=layout,
                text_k_raw=text_k,
                text_v_raw=text_v,
                text_beta=text_beta,
            )  # [V_full, H_local * d]

            # Scatter the video-row readout back into a full-sequence buffer so the
            # inverse all-to-all sees a sequence length divisible by the world size;
            # non-video rows stay zero and project to zero (to_out_linear has no bias).
            v_full = layout.seq_len
            linear_full = q_raw.new_zeros(v_full, h_local, head_dim)
            linear_full[video] = linear.view(-1, h_local, head_dim).to(linear_full.dtype)
            linear_full = _head_to_seq_a2a(linear_full)  # [S_local, H, d]
            out = out + attn.to_out_linear(linear_full.reshape(s_local, heads * head_dim).to(out.dtype))

        out = attn.to_out[1](out)  # dropout (identity at inference)
        return out.unsqueeze(0)

    def _compute(
        self,
        attn,
        hidden_states: torch.Tensor,
        rotary_emb: tuple[torch.Tensor, torch.Tensor] | None,
        meta: "VDNRequestMeta",
    ) -> torch.Tensor:
        """The hybrid attention over the FULL packed sequence (all heads)."""
        layout = meta.layout
        heads = attn.heads
        head_dim = (
            attn.to_q.out_features // heads if not attn.fused_projections else attn.to_qkv.out_features // (3 * heads)
        )
        scale = 1.0 / math.sqrt(head_dim)

        x = hidden_states[0]  # [S, C]

        # ---- raw projections (shared by both branches) --------------------
        if attn.fused_projections:
            q, k, v = attn.to_qkv(hidden_states).chunk(3, dim=-1)
        else:
            q = attn.to_q(hidden_states)
            k = attn.to_k(hidden_states)
            v = attn.to_v(hidden_states)
        q = q.unflatten(-1, (heads, -1))  # [1, S, H, d]
        k = k.unflatten(-1, (heads, -1))
        v = v.unflatten(-1, (heads, -1))

        # ---- window (softmax) branch --------------------------------------
        q_sm = attn.norm_q(q)
        k_sm = attn.norm_k(k)
        if rotary_emb is not None:
            q_sm = _apply_rotary_emb(q_sm, *rotary_emb)
            k_sm = _apply_rotary_emb(k_sm, *rotary_emb)
        q_sm = q_sm[0]  # [S, H, d]
        k_sm = k_sm[0]
        v_sm = v[0]
        softmax_gate = attn.softmax_gate(x) if attn.softmax_gate is not None else None
        if meta.full_cover:
            window = dense_softmax(
                q_sm,
                k_sm,
                v_sm,
                used=layout.used,
                scale=scale,
                softmax_gate=softmax_gate,
            )
        else:
            window = windowed_softmax(
                q_sm,
                k_sm,
                v_sm,
                plan=meta.plan,
                layout=layout,
                scale=scale,
                softmax_gate=softmax_gate,
            )
        out = attn.to_out[0](window.reshape(layout.seq_len, heads * head_dim))

        # ---- linear (Video-Delta) branch on the video rows ----------------
        if not meta.full_cover and layout.num_frames > 0:
            branch = attn.linear_attention
            video = slice(layout.video_start, layout.video_end)
            text = slice(0, layout.text_len)
            q_raw = q[0]
            k_raw = k[0]
            v_raw = v[0]
            beta = branch.beta(x)  # [S, H]
            gate = branch.output_gate(x)  # [S, H, d]
            frame_mean = x[video].view(layout.num_frames, layout.tokens_per_frame, -1).mean(dim=1, dtype=torch.float32)
            text_k = k_raw[text] if branch.hybrid.enable_text_state else None
            text_v = v_raw[text] if branch.hybrid.enable_text_state else None
            text_beta = beta[text] if branch.hybrid.enable_text_state else None
            linear = branch(
                q_raw=q_raw[video],
                k_raw=k_raw[video],
                v_raw=v_raw[video],
                beta=beta[video],
                gate=gate[video],
                frame_mean=frame_mean,
                layout=layout,
                text_k_raw=text_k,
                text_v_raw=text_v,
                text_beta=text_beta,
            )  # [V, H*d]
            out[video] = out[video] + attn.to_out_linear(linear.to(out.dtype))

        out = attn.to_out[1](out)  # dropout (identity at inference)
        return out.unsqueeze(0)


__all__ = [
    "VDNRequestMeta",
    "VDNMiniMaxH3AttnProcessor",
    "attach_vdn_modules",
    "build_vdn_meta",
    "ensure_ulysses_local_branch",
]
