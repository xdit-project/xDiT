import torch

from diffusers.models.attention import Attention
from torch import nn
from torch.nn import functional as F

from pipefuser.refactor.config.config import ParallelConfig, RuntimeConfig
from pipefuser.refactor.layers.base_layer import PipeFuserLayerBaseWrapper
from pipefuser.refactor.layers.register import PipeFuserLayerWrappersRegister
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
    
class PipeFuserAttentionBaseWrapper(PipeFuserLayerBaseWrapper):
    def __init__(
        self, 
        attn: Attention,
        parallel_config: ParallelConfig,
        runtime_config: RuntimeConfig,
    ):
        super().__init__(
            module=attn,
            parallel_config=parallel_config,
            runtime_config=runtime_config,
        )

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

        self.batch_idx = 0

#TODO(Eigensystem): implement PipeFuserAttentionWrapper to replace this
#!WARNING: ONLY AVAILABLE FOR PIX_ART_ALPHA, TAKE ALL ATTENTION MODULES AS INPUT
@PipeFuserLayerWrappersRegister.register(Attention)
class PipeFuserSelfAttentionWrapper(PipeFuserAttentionBaseWrapper):
    def __init__(
        self, 
        attn: Attention, 
        parallel_config: ParallelConfig,
        runtime_config: RuntimeConfig,
    ):
        super().__init__(
            attn=attn,
            parallel_config=parallel_config,
            runtime_config=runtime_config,
        )

    def _forward(
        self, 
        hidden_states: torch.FloatTensor, 
        scale: float = 1.0,
        use_async: bool = False,
    ):
        assert isinstance(self.module, Attention)

        residual = hidden_states

        batch_size, sequence_length, _ = hidden_states.shape

        query = self.module.to_q(hidden_states)

        encoder_hidden_states = hidden_states
        kv = self.to_kv(encoder_hidden_states)

        # the distributed sparse attention from pipefuser
        if self.num_pipeline_patch == 1:
            full_kv = kv
        else:
            if not use_async:
                full_kv = kv
            else:
                full_kv = self.activation_cache
                # _, c, _ = full_kv.shape
                _, c, _ = kv.shape
                # assert c % distri_config.pp_num_patch == 0
                full_kv[:, c * self.batch_idx : c * (self.batch_idx + 1), :] = kv

            self.activation_cache = full_kv

        # naive attn
        key, value = torch.split(full_kv, full_kv.shape[-1] // 2, dim=-1)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // self.module.heads

        query = query.view(batch_size, -1, self.module.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, self.module.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.module.heads, head_dim).transpose(1, 2)

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
        use_async: bool = False,
        *args,
        **kwargs,
    ) -> torch.FloatTensor:
        output = self._forward(hidden_states, scale=scale, use_async=use_async)

        # if self.in_warmup_stage():
        #     self.round_step()
        # else:
        #     self.patch_step()
        return output