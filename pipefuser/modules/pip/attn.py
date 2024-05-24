import torch

# from diffusers.models.attention import Attention
from pipefuser.models.diffusers import Attention

from diffusers.utils import USE_PEFT_BACKEND
from torch import distributed as dist
from torch import nn
from torch.nn import functional as F

from pipefuser.modules.base_module import BaseModule
from pipefuser.utils import DistriConfig
from pipefuser.logger import init_logger

logger = init_logger(__name__)


HAS_FLASH_ATTN = False
from typing import Optional

HAS_LONG_CTX_ATTN = False
try:
    from yunchang import (
        ring_flash_attn_func,
        UlyssesAttention,
        LongContextAttention,
        LongContextAttentionQKVPacked,
    )

    HAS_LONG_CTX_ATTN = True
except ImportError:
    logger.warning("ring flash attn not found")


class DistriAttentionPiP(BaseModule):
    def __init__(self, module: Attention, distri_config: DistriConfig):
        super(DistriAttentionPiP, self).__init__(module, distri_config)

        to_k = module.to_k
        to_v = module.to_v
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

        self.batch_idx = 0


class DistriCrossAttentionPiP(DistriAttentionPiP):
    def __init__(self, module: Attention, distri_config: DistriConfig):
        super(DistriCrossAttentionPiP, self).__init__(module, distri_config)
        self.kv_cache = None

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        scale: float = 1.0,
        *args,
        **kwargs,
    ):
        assert encoder_hidden_states is not None
        recompute_kv = self.counter == 0

        attn = self.module
        assert isinstance(attn, Attention)

        residual = hidden_states

        batch_size, sequence_length, _ = (
            hidden_states.shape
            if encoder_hidden_states is None
            else encoder_hidden_states.shape
        )

        # args = () if USE_PEFT_BACKEND else (scale,)
        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        if recompute_kv or self.kv_cache is None:
            kv = self.to_kv(encoder_hidden_states)
            self.kv_cache = kv
        else:
            kv = self.kv_cache
        key, value = torch.split(kv, kv.shape[-1] // 2, dim=-1)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(
            batch_size, -1, attn.heads * head_dim
        )
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states, *args)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        self.counter += 1

        return hidden_states


class DistriSelfAttentionPiP(DistriAttentionPiP):
    def __init__(self, module: Attention, distri_config: DistriConfig):
        super(DistriSelfAttentionPiP, self).__init__(module, distri_config)

        if HAS_LONG_CTX_ATTN and distri_config.use_seq_parallel_attn:
            self.hybrid_seq_parallel_attn = LongContextAttention()
            # self.hybrid_seq_parallel_attn = ring_flash_attn_func

    def _forward(self, hidden_states: torch.FloatTensor, scale: float = 1.0):
        attn = self.module
        distri_config = self.distri_config
        assert isinstance(attn, Attention)

        residual = hidden_states

        batch_size, sequence_length, _ = hidden_states.shape

        query = attn.to_q(hidden_states)

        encoder_hidden_states = hidden_states
        kv = self.to_kv(encoder_hidden_states)

        # the distributed sparse attention from pipefuser
        if distri_config.pp_num_patch == 1:
            full_kv = kv
        else:
            if (
                distri_config.mode == "full_sync"
                or self.counter <= distri_config.warmup_steps
            ):
                full_kv = kv
            else:
                full_kv = self.buffer_list
                # _, c, _ = full_kv.shape
                _, c, _ = kv.shape
                # assert c % distri_config.pp_num_patch == 0
                full_kv[:, c * self.batch_idx : c * (self.batch_idx + 1), :] = kv

            self.buffer_list = full_kv

        # naive attn
        key, value = torch.split(full_kv, full_kv.shape[-1] // 2, dim=-1)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(
            batch_size, -1, attn.heads * head_dim
        )
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        scale: float = 1.0,
        *args,
        **kwargs,
    ) -> torch.FloatTensor:
        distri_config = self.distri_config
        output = self._forward(hidden_states, scale=scale)

        if (
            distri_config.mode == "full_sync"
            or self.counter <= distri_config.warmup_steps
        ):
            self.counter += 1
        else:
            self.batch_idx += 1
            if self.batch_idx == distri_config.pp_num_patch:
                self.counter += 1
                self.batch_idx = 0
        return output
