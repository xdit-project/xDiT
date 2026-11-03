# SPDX-License-Identifier: Apache-2.0
#
# Ported from OpenVDN / Video DeltaNet (VDN-H3): Apache-2.0, Copyright 2026 the
# VDN authors. https://github.com/OpenVDN/vdn-minimax-h3 (commit e262cb5)
#   source: src/models/ (hybrid_attention, hybrid_transform, attention_gates,
#           linear_attention/, softmax_attention/)
# Cross-referenced with SGLang's hybrid_window_attn_h3 diffusion integration
# (https://github.com/sgl-project/sglang, Apache-2.0).
"""VDN-H3 (Video DeltaNet MiniMax-H3) hybrid attention for xDiT.

This package ports OpenVDN/vdn-minimax-h3's hybrid attention (a chunk-aligned
frame-window softmax plus a bidirectional Video-Delta linear branch) onto xDiT's
MiniMax-H3 transformer. Head-sharded Ulysses sequence parallelism is supported:
each rank's attention processor works on ``local_heads`` of the full sequence.

The submodule/parameter names match the VDN overlay checkpoint's linear-branch
keys (``transformer_blocks.N.attn.{linear_attention.*, softmax_gate.*,
to_out_linear.weight}``) so the 800 branch tensors load with no remapping.
"""

from xfuser.model_executor.models.transformers.vdn_minimax_h3.config import (
    VDNHybridAttentionArchConfig,
)
from xfuser.model_executor.models.transformers.vdn_minimax_h3.attention import (
    VDNMiniMaxH3AttnProcessor,
    attach_vdn_modules,
    build_vdn_meta,
)
from xfuser.model_executor.models.transformers.vdn_minimax_h3.transformer import (
    xFuserMiniMaxH3VDNTransformer3DWrapper,
)

__all__ = [
    "VDNHybridAttentionArchConfig",
    "VDNMiniMaxH3AttnProcessor",
    "attach_vdn_modules",
    "build_vdn_meta",
    "xFuserMiniMaxH3VDNTransformer3DWrapper",
]
