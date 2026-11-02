# SPDX-License-Identifier: Apache-2.0
"""NA3D flash attention processor for the LTX-2.5 VAE diffusion decoder."""

from __future__ import annotations

import weakref

import torch

from xfuser.model_executor.layers.ltx2.na3d_eager_attn import (
    LTX2VideoVaeEagerSdpaAttnProcessor,
)

# Architectures the AITER na3d_flash kernel accepts (asserted by its launcher).
_SUPPORTED_ARCHS = ("gfx942", "gfx950")


class LTX2VideoVaeMfmaAttnProcessor:
    """Flash-NA3D attention processor for the LTX-2.5 diffusion decoder.

    Drop-in replacement for LTX2VideoVaeEagerSdpaAttnProcessor. Delegates to
    AITER's Triton flash-NA3D kernel (ROCm gfx950 / MI350X, gfx942 / MI300X)
    with a fused QKV GEMM to reduce memory bandwidth. Grids outside the
    kernel's supported domain (e.g. narrow decoder tiles) use the tiled SDPA
    processor.

    Construction raises ``ImportError`` when the installed AITER lacks
    ``na3d_flash`` and ``RuntimeError`` on an unsupported device, so callers
    can fall back to another processor.
    """

    def __init__(self):
        from aiter.ops.triton.attention.na3d_flash import na3d_flash_attn

        if not torch.cuda.is_available():
            raise RuntimeError("AITER flash-NA3D requires a ROCm device.")
        arch = torch.cuda.get_device_properties(torch.cuda.current_device()).gcnArchName.split(":")[0]
        if arch not in _SUPPORTED_ARCHS:
            raise RuntimeError(f"AITER flash-NA3D supports {_SUPPORTED_ARCHS}; got {arch}.")
        self._na3d_flash_attn = na3d_flash_attn
        self._fallback = LTX2VideoVaeEagerSdpaAttnProcessor()
        self._fused_qkv: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()

    @staticmethod
    def _kernel_supports(attn, hidden_states: torch.Tensor) -> bool:
        """Whether the grid is inside the domain the AITER launcher asserts."""
        width = hidden_states.shape[3]
        kernel_w = attn.kernel_size[2]
        head_dim = attn.head_dim
        return (
            hidden_states.dtype == torch.bfloat16
            and width >= 16
            and kernel_w <= 33
            and (kernel_w <= 17 or width >= 32)
            and head_dim >= 16
            and head_dim & (head_dim - 1) == 0
        )

    def _get_fused_qkv(self, attn) -> torch.nn.Linear:
        if attn not in self._fused_qkv:
            W_cat = torch.cat([attn.to_q.weight.data, attn.to_k.weight.data, attn.to_v.weight.data], dim=0)
            b_cat = torch.cat([attn.to_q.bias.data, attn.to_k.bias.data, attn.to_v.bias.data], dim=0)
            out_f, in_f = W_cat.shape
            fused = torch.nn.Linear(in_f, out_f, bias=True, device=W_cat.device, dtype=W_cat.dtype)
            fused.weight = torch.nn.Parameter(W_cat, requires_grad=False)
            fused.bias = torch.nn.Parameter(b_cat, requires_grad=False)
            self._fused_qkv[attn] = fused
        return self._fused_qkv[attn]

    def _project_qkv_fused(self, attn, hidden_states: torch.Tensor):
        B, T, H, W, C = hidden_states.shape
        shape = (B, T, H, W, attn.heads, attn.head_dim)
        fused = self._get_fused_qkv(attn)
        qkv = fused(hidden_states)
        q_raw, k_raw, v_raw = qkv.chunk(3, dim=-1)
        query = attn.norm_q(q_raw.view(shape))
        key = attn.norm_k(k_raw.view(shape))
        query = query * attn.scale
        return attn.rope(query), attn.rope(key), v_raw.view(shape)

    def __call__(self, attn, hidden_states, block_mask=None):
        if not self._kernel_supports(attn, hidden_states):
            return self._fallback(attn, hidden_states, block_mask)
        B, T, H, W, _ = hidden_states.shape
        q, k, v = self._project_qkv_fused(attn, hidden_states)
        out = self._na3d_flash_attn(q, k, v, kernel_size=tuple(attn.kernel_size))
        out = out.reshape(B, T, H, W, attn.heads * attn.head_dim)
        return attn.to_out[0](out)
