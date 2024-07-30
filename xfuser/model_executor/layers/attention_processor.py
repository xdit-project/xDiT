from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F
from diffusers.models.attention import Attention

from xfuser.distributed import (
    get_sequence_parallel_world_size,
    get_sequence_parallel_rank,
    get_sp_group,
)
from xfuser.distributed.runtime_state import get_runtime_state
from xfuser.model_executor.layers import xFuserLayerBaseWrapper
from xfuser.model_executor.layers import xFuserLayerWrappersRegister
from xfuser.logger import init_logger
from xfuser.envs import PACKAGES_CHECKER

logger = init_logger(__name__)

env_info = PACKAGES_CHECKER.get_packages_info()
HAS_LONG_CTX_ATTN = env_info["has_long_ctx_attn"]
HAS_FLASH_ATTN = env_info["has_flash_attn"]


class xFuserAttentionBaseWrapper(xFuserLayerBaseWrapper):
    def __init__(
        self,
        attn: Attention,
    ):
        super().__init__(module=attn)
        self.use_long_ctx_attn_kvcache = True
        if HAS_LONG_CTX_ATTN and get_sequence_parallel_world_size() > 1:
            from yunchang import UlyssesAttention
            from xfuser.modules.long_context_attention import xFuserLongContextAttention

            if HAS_FLASH_ATTN:
                # self.hybrid_seq_parallel_attn = LongContextAttention()
                self.hybrid_seq_parallel_attn = xFuserLongContextAttention(
                    use_kv_cache=self.use_long_ctx_attn_kvcache
                )
            else:
                self.hybrid_seq_parallel_attn = UlyssesAttention(use_fa=False)

        to_k = self.module.to_k
        to_v = self.module.to_v
        assert isinstance(to_k, nn.Linear)
        assert isinstance(to_v, nn.Linear)
        assert (to_k.bias is None) == (to_v.bias is None)
        assert to_k.weight.shape == to_v.weight.shape

        in_size, out_size = to_k.in_features, to_k.out_features
        to_kv = nn.Linear(
            in_size,
            out_size * 2,
            bias=to_k.bias is not None,
            device=to_k.weight.device,
            dtype=to_k.weight.dtype,
        )
        to_kv.weight.data[:out_size].copy_(to_k.weight.data)
        to_kv.weight.data[out_size:].copy_(to_v.weight.data)

        if to_k.bias is not None:
            assert to_v.bias is not None
            to_kv.bias.data[:out_size].copy_(to_k.bias.data)
            to_kv.bias.data[out_size:].copy_(to_v.bias.data)

        self.to_kv = to_kv


# TODO(Eigensystem): implement xFuserAttentionWrapper to replace this
#!WARNING: ONLY AVAILABLE FOR PIXART_TRANSFORMER_2D, TAKE ALL ATTENTION MODULES AS INPUT
@xFuserLayerWrappersRegister.register(Attention)
class xFuserSelfAttentionWrapper(xFuserAttentionBaseWrapper):
    def __init__(
        self,
        attn: Attention,
    ):
        super().__init__(attn=attn)

    def _forward(
        self,
        hidden_states: torch.FloatTensor,
        scale: float = 1.0,
    ):
        assert isinstance(self.module, Attention)

        residual = hidden_states

        batch_size, sequence_length, _ = hidden_states.shape

        query = self.module.to_q(hidden_states)

        encoder_hidden_states = hidden_states
        kv = self.to_kv(encoder_hidden_states)

        if (
            HAS_FLASH_ATTN 
            and get_sequence_parallel_world_size() > 1
            and self.use_long_ctx_attn_kvcache 
        ):
            key, value = torch.chunk(kv, 2, dim=-1)
            inner_dim = key.shape[-1]
            head_dim = inner_dim // self.module.heads
        else:
            # the distributed sparse attention from xfuser
            if get_runtime_state().num_pipeline_patch == 1:
                local_kv = kv
            else:
                if not get_runtime_state().patch_mode:
                    local_kv = kv
                else:
                    local_kv = self.activation_cache
                    token_start_idx = sum(get_runtime_state().pp_patches_token_num[:get_runtime_state().pipeline_patch_idx])
                    token_end_idx = sum(get_runtime_state().pp_patches_token_num[:get_runtime_state().pipeline_patch_idx+1])
                    local_kv[:, token_start_idx:token_end_idx, :] = kv

                self.activation_cache = local_kv

            key, value = torch.split(local_kv, local_kv.shape[-1] // 2, dim=-1)
            inner_dim = key.shape[-1]
            head_dim = inner_dim // self.module.heads

        if HAS_LONG_CTX_ATTN and get_sequence_parallel_world_size() > 1:
            query = query.view(batch_size, -1, self.module.heads, head_dim)
            key = key.view(batch_size, -1, self.module.heads, head_dim)
            value = value.view(batch_size, -1, self.module.heads, head_dim)
            hidden_states = self.hybrid_seq_parallel_attn(
                query, key, value, dropout_p=0.0, causal=False
            )
            hidden_states = hidden_states.reshape(
                batch_size, -1, self.module.heads * head_dim
            )

        else:
            if HAS_FLASH_ATTN:
                from flash_attn import flash_attn_func

                query = query.view(batch_size, -1, self.module.heads, head_dim)
                key = key.view(batch_size, -1, self.module.heads, head_dim)
                value = value.view(batch_size, -1, self.module.heads, head_dim)
                hidden_states = flash_attn_func(
                    query, key, value, dropout_p=0.0, causal=False
                )
                hidden_states = hidden_states.reshape(
                    batch_size, -1, self.module.heads * head_dim
                )

            else:
                query = query.view(
                    batch_size, -1, self.module.heads, head_dim
                ).transpose(1, 2)
                key = key.view(batch_size, -1, self.module.heads, head_dim).transpose(
                    1, 2
                )
                value = value.view(
                    batch_size, -1, self.module.heads, head_dim
                ).transpose(1, 2)

                # the output of sdp = (batch, num_heads, seq_len, head_dim)
                # TODO: add support for self.module.scale when we move to Torch 2.1
                hidden_states = F.scaled_dot_product_attention(
                    query, key, value, dropout_p=0.0, is_causal=False
                )
                hidden_states = hidden_states.transpose(1, 2).reshape(
                    batch_size, -1, self.module.heads * head_dim
                )

        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = self.module.to_out[0](hidden_states)
        # dropout
        hidden_states = self.module.to_out[1](hidden_states)

        if self.module.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / self.module.rescale_output_factor

        return hidden_states

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        scale: float = 1.0,
        *args,
        **kwargs,
    ) -> torch.FloatTensor:
        output = self._forward(hidden_states, scale=scale)
        return output
