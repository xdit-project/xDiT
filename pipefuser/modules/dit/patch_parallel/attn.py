import torch

# from diffusers.models.attention import Attention
from diffusers.models.attention import Attention

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

try:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(device)
    if "Turing" in gpu_name or "Tesla" in gpu_name or "T4" in gpu_name:
        HAS_FLASH_ATTN = False
    else:
        from flash_attn import flash_attn_func

        HAS_FLASH_ATTN = True
except ImportError:
    HAS_FLASH_ATTN = False


class DistriAttentionPP(BaseModule):
    def __init__(self, module: Attention, distri_config: DistriConfig):
        super(DistriAttentionPP, self).__init__(module, distri_config)

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


class DistriCrossAttentionPP(DistriAttentionPP):
    def __init__(self, module: Attention, distri_config: DistriConfig):
        super(DistriCrossAttentionPP, self).__init__(module, distri_config)
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


class DistriSelfAttentionPP(DistriAttentionPP):
    def __init__(self, module: Attention, distri_config: DistriConfig):
        super(DistriSelfAttentionPP, self).__init__(module, distri_config)

        if HAS_LONG_CTX_ATTN and distri_config.use_seq_parallel_attn:
            # if gpu is Turing or Tesla
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            gpu_name = torch.cuda.get_device_name(device)
            if "Turing" in gpu_name or "Tesla" in gpu_name or "T4" in gpu_name:
                self.hybrid_seq_parallel_attn = UlyssesAttention(use_fa=False)
                logger.warning(
                    f"Turing and Tesla GPU dose not support flash_attn, we can only UlyssesAttention instead."
                )
            else:
                self.hybrid_seq_parallel_attn = LongContextAttention()

    def _forward(self, hidden_states: torch.FloatTensor, scale: float = 1.0):
        attn = self.module
        distri_config = self.distri_config
        assert isinstance(attn, Attention)

        residual = hidden_states

        batch_size, sequence_length, _ = hidden_states.shape

        # args = () if USE_PEFT_BACKEND else (scale,)
        query = attn.to_q(hidden_states)

        encoder_hidden_states = hidden_states
        kv = self.to_kv(encoder_hidden_states)

        use_seq_parallel_attn = self.distri_config.use_seq_parallel_attn

        if use_seq_parallel_attn:
            # the distributed sparse attention using ring-attention.
            key, value = torch.split(kv, kv.shape[-1] // 2, dim=-1)
            inner_dim = key.shape[-1]
            head_dim = inner_dim // attn.heads

            query = query.view(batch_size, -1, attn.heads, head_dim)
            key = key.view(batch_size, -1, attn.heads, head_dim)
            value = value.view(batch_size, -1, attn.heads, head_dim)

            hidden_states = self.hybrid_seq_parallel_attn(
                query, key, value, dropout_p=0.0, causal=False
            )

            hidden_states = hidden_states.reshape(
                batch_size, -1, attn.heads * head_dim
            ).to(query.dtype)
        else:
            # the distributed sparse attention from pipefuser
            if distri_config.n_device_per_batch == 1:
                full_kv = kv
            else:
                if self.buffer_list is None:  # buffer not created
                    full_kv = torch.cat(
                        [kv for _ in range(distri_config.n_device_per_batch)], dim=1
                    )
                elif (
                    distri_config.mode == "full_sync"
                    or self.counter <= distri_config.warmup_steps
                ):
                    dist.all_gather(
                        self.buffer_list,
                        kv,
                        group=distri_config.local_batch_parallel_group,
                        async_op=False,
                    )
                    full_kv = torch.cat(self.buffer_list, dim=1)
                else:
                    new_buffer_list = [buffer for buffer in self.buffer_list]
                    new_buffer_list[distri_config.split_idx()] = kv
                    full_kv = torch.cat(new_buffer_list, dim=1)
                    if distri_config.mode != "no_sync" and (
                        distri_config.num_inference_steps is None
                        or self.counter < distri_config.num_inference_steps - 1
                    ):
                        self.comm_manager.enqueue(self.idx, kv)

            if HAS_FLASH_ATTN:
                # flash attn
                key, value = torch.split(full_kv, full_kv.shape[-1] // 2, dim=-1)
                inner_dim = key.shape[-1]
                head_dim = inner_dim // attn.heads

                query = query.view(batch_size, -1, attn.heads, head_dim)
                key = key.view(batch_size, -1, attn.heads, head_dim)
                value = value.view(batch_size, -1, attn.heads, head_dim)

                hidden_states = flash_attn_func(
                    query, key, value, dropout_p=0.0, causal=False
                )
                hidden_states = hidden_states.reshape(
                    batch_size, -1, attn.heads * head_dim
                ).to(query.dtype)
            else:
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

        if not distri_config.use_seq_parallel_attn:
            # async preallocates memo buffer
            if (
                self.comm_manager is not None
                and self.comm_manager.handles is not None
                and self.idx is not None
            ):
                if self.comm_manager.handles[self.idx] is not None:
                    self.comm_manager.handles[self.idx].wait()
                    self.comm_manager.handles[self.idx] = None

            b, l, c = hidden_states.shape
            if distri_config.n_device_per_batch > 1 and self.buffer_list is None:
                if self.comm_manager.buffer_list is None:
                    self.idx = self.comm_manager.register_tensor(
                        shape=(b, l, self.to_kv.out_features),
                        torch_dtype=hidden_states.dtype,
                        layer_type="attn",
                    )
                else:
                    self.buffer_list = self.comm_manager.get_buffer_list(self.idx)

        output = self._forward(hidden_states, scale=scale)

        self.counter += 1
        return output
