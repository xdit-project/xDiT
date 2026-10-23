# SPDX-License-Identifier: Apache-2.0
#
# Ported from OpenVDN / Video DeltaNet (VDN-H3): Apache-2.0, Copyright 2026 the
# VDN authors. https://github.com/OpenVDN/vdn-minimax-h3 (commit e262cb5)
#   source: src/models/hybrid_transform.py
# Cross-referenced with SGLang's hybrid_window_attn_h3 diffusion integration
# (https://github.com/sgl-project/sglang, Apache-2.0).
"""VDN-H3 transformer wrapper for xDiT: base MiniMax-H3 + hybrid attention.

Subclasses ``xFuserMiniMaxH3Transformer3DWrapper``. Beyond the dense H3 stack it:

* accepts the ``hybrid_attention`` config block (present in the materialized VDN
  ``transformer/config.json``) and builds a ``VDNHybridAttentionArchConfig``;
* attaches the VDN branch submodules to every transformer block's attention so
  the overlay's 800 ``attn.{linear_attention,softmax_gate,to_out_linear}.*``
  keys load, and swaps in the VDN hybrid attention processor;
* primes the per-request geometry (packed layout + window decomposition) once
  per denoise step in a forward pre-hook, reading ``position_ids`` / index
  tensors (which need device reads and so must run eager, outside any compiled
  region), and stashes it on each ``attn._vdn_meta``.

Supports Ulysses sequence parallelism: the attention processor all-to-alls
q/k/v into a per-head shard, so each block computes on ``local_heads`` of the
full padded sequence.
"""

from __future__ import annotations

from typing import Any

import torch

from xfuser.model_executor.models.transformers.transformer_minimax_h3 import (
    MINIMAX_H3_PACKED_SEQUENCE_ALIGNMENT,
    xFuserMiniMaxH3Transformer3DWrapper,
)
from xfuser.model_executor.models.transformers.vdn_minimax_h3.attention import (
    attach_vdn_modules,
    build_vdn_meta,
    ensure_ulysses_local_branch,
)
from xfuser.model_executor.models.transformers.vdn_minimax_h3.config import (
    VDNHybridAttentionArchConfig,
)


def _padded_length(sequence_length: int) -> int:
    align = MINIMAX_H3_PACKED_SEQUENCE_ALIGNMENT
    return (sequence_length + align - 1) // align * align


class xFuserMiniMaxH3VDNTransformer3DWrapper(xFuserMiniMaxH3Transformer3DWrapper):
    """MiniMax-H3 with VDN hybrid attention (window softmax + linear branch)."""

    _rope_full_sequence = True

    def __init__(
        self,
        *,
        hybrid_attention: dict[str, Any] | None = None,
        num_attention_heads: int = 56,
        attention_head_dim: int = 128,
        hidden_size: int = 5376,
        num_layers: int = 50,
        num_refiner_layers: int = 2,
        ffn_dim: int = 14336,
        in_channels: int = 24,
        audio_in_channels: int = 32,
        patch_size: tuple[int, int, int] = (1, 2, 2),
        text_dim: int = 5120,
        freq_dim: int = 256,
        time_embed_hidden_dim: int = 5376,
        time_embed_dim: int = 2688,
        rope_freq_dim: int = 16,
        rope_theta: float = 10000.0,
        norm_eps: float = 1e-5,
        qk_norm_eps: float = 1e-5,
        final_norm_eps: float = 1e-5,
        attention_backend=None,
        enable_fasth3_vsa: bool = False,
    ) -> None:
        if hybrid_attention is None:
            raise ValueError(
                "VDN-H3 transformer needs the `hybrid_attention` config block; "
                "point from_pretrained at a materialized VDN transformer dir."
            )
        if enable_fasth3_vsa:
            raise ValueError("VDN-H3 and FastH3 VSA are mutually exclusive.")
        super().__init__(
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_refiner_layers=num_refiner_layers,
            ffn_dim=ffn_dim,
            in_channels=in_channels,
            audio_in_channels=audio_in_channels,
            patch_size=patch_size,
            text_dim=text_dim,
            freq_dim=freq_dim,
            time_embed_hidden_dim=time_embed_hidden_dim,
            time_embed_dim=time_embed_dim,
            rope_freq_dim=rope_freq_dim,
            rope_theta=rope_theta,
            norm_eps=norm_eps,
            qk_norm_eps=qk_norm_eps,
            final_norm_eps=final_norm_eps,
            attention_backend=attention_backend,
            enable_fasth3_vsa=False,
        )
        self.register_to_config(hybrid_attention=hybrid_attention)
        self.hybrid = VDNHybridAttentionArchConfig.from_transform_config(
            hybrid_attention["config"] if "config" in hybrid_attention else hybrid_attention
        )
        for block in self.transformer_blocks:
            attach_vdn_modules(
                block.attn,
                self.hybrid,
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                attention_head_dim=attention_head_dim,
            )
        self._vdn_meta_key: tuple | None = None
        self._vdn_meta = None
        self.register_forward_pre_hook(type(self)._prime_vdn_meta_hook, with_kwargs=True)

    # ---- geometry priming ------------------------------------------------

    def prime_vdn_meta(
        self,
        position_ids: torch.Tensor,
        video_indices: torch.Tensor,
        audio_indices: torch.Tensor,
        text_indices: torch.Tensor,
    ) -> None:
        """Recover the packed layout + window decomposition; cache by geometry.

        Reads ``position_ids`` values (device sync), so it must run eager. The
        VDN xDiT port runs the transformer eager; there is no compiled-region
        priming wrapper in this pass."""
        sequence_length = int(position_ids.shape[0])
        key = (
            int(text_indices.numel()),
            int(audio_indices.numel()),
            sequence_length,
            tuple(position_ids[-1].tolist()),
        )
        if self._vdn_meta_key != key:
            self._vdn_meta = build_vdn_meta(
                self.hybrid,
                position_ids=position_ids,
                text_indices=text_indices,
                audio_indices=audio_indices,
                video_indices=video_indices,
                padded_length=_padded_length(sequence_length),
            )
            self._vdn_meta_key = key
        for block in self.transformer_blocks:
            block.attn._vdn_meta = self._vdn_meta
            ensure_ulysses_local_branch(block.attn)

    @staticmethod
    def _prime_vdn_meta_hook(module, args, kwargs) -> None:
        bound = _bind_forward_layout(module, args, kwargs)
        module.prime_vdn_meta(
            bound["position_ids"],
            bound["video_indices"],
            bound["audio_indices"],
            bound["text_indices"],
        )


_FORWARD_LAYOUT_ARGS = (
    "hidden_states",
    "audio_hidden_states",
    "encoder_hidden_states",
    "timestep",
    "timestep_indices",
    "token_tags",
    "position_ids",
    "video_indices",
    "audio_indices",
    "text_indices",
)


def _bind_forward_layout(module, args, kwargs) -> dict[str, torch.Tensor]:
    """Resolve the layout tensors whether they arrived positionally or by name."""
    bound: dict[str, torch.Tensor] = {}
    for index, name in enumerate(_FORWARD_LAYOUT_ARGS):
        if name in kwargs:
            bound[name] = kwargs[name]
        elif index < len(args):
            bound[name] = args[index]
    missing = [n for n in ("position_ids", "video_indices", "audio_indices", "text_indices") if n not in bound]
    if missing:
        raise RuntimeError(f"VDN meta priming missing forward inputs: {missing}")
    return bound


__all__ = ["xFuserMiniMaxH3VDNTransformer3DWrapper"]
