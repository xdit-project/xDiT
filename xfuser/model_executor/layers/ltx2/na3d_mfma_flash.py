# SPDX-License-Identifier: Apache-2.0
"""NA3D flash attention processor for the LTX-2.5 VAE diffusion decoder."""
from __future__ import annotations
import weakref
import torch


class LTX2VideoVaeMfmaAttnProcessor:
    """Flash-NA3D attention processor for the LTX-2.5 diffusion decoder.

    Drop-in replacement for LTX2VideoVaeEagerSdpaAttnProcessor.  Uses AITER's
    BF16-MFMA flash kernel (gfx950 / MI350X, gfx942 / MI300X, B200) and
    includes a fused QKV GEMM to reduce memory bandwidth.
    """

    def __init__(self):
        self._fused_qkv: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()

    def _get_fused_qkv(self, attn) -> torch.nn.Linear:
        if attn not in self._fused_qkv:
            W_cat = torch.cat([attn.to_q.weight.data,
                               attn.to_k.weight.data,
                               attn.to_v.weight.data], dim=0)
            b_cat = torch.cat([attn.to_q.bias.data,
                               attn.to_k.bias.data,
                               attn.to_v.bias.data], dim=0)
            out_f, in_f = W_cat.shape
            fused = torch.nn.Linear(in_f, out_f, bias=True,
                                    device=W_cat.device, dtype=W_cat.dtype)
            fused.weight = torch.nn.Parameter(W_cat, requires_grad=False)
            fused.bias   = torch.nn.Parameter(b_cat, requires_grad=False)
            self._fused_qkv[attn] = fused
        return self._fused_qkv[attn]

    def _project_qkv_fused(self, attn, hidden_states: torch.Tensor):
        B, T, H, W, C = hidden_states.shape
        shape = (B, T, H, W, attn.heads, attn.head_dim)
        fused = self._get_fused_qkv(attn)
        qkv   = fused(hidden_states)
        q_raw, k_raw, v_raw = qkv.chunk(3, dim=-1)
        query = attn.norm_q(q_raw.view(shape))
        key   = attn.norm_k(k_raw.view(shape))
        query = query * attn.scale
        return attn.rope(query), attn.rope(key), v_raw.view(shape)

    def __call__(self, attn, hidden_states, block_mask=None):
        B, T, H, W, _ = hidden_states.shape
        q, k, v = self._project_qkv_fused(attn, hidden_states)
        from aiter.ops.triton.attention.na3d_flash import na3d_flash_attn
        out = na3d_flash_attn(q, k, v, kernel_size=tuple(attn.kernel_size))
        out = out.reshape(B, T, H, W, attn.heads * attn.head_dim)
        return attn.to_out[0](out)
