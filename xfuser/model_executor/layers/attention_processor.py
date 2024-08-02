import inspect
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F
from diffusers.utils import deprecate
from diffusers.models.attention import Attention
from diffusers.models.attention_processor import AttnProcessor2_0, JointAttnProcessor2_0

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
        attention: Attention,
    ):
        super().__init__(module=attention)
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

class xFuserAttentionProcessorRegister:
    _XFUSER_ATTENTION_PROCESSOR_MAPPING = {}

    @classmethod
    def register(cls, origin_processor_class):
        def decorator(xfuser_processor):
            if not issubclass(xfuser_processor, origin_processor_class):
                raise ValueError(
                    f"{xfuser_processor.__class__.__name__} is not a subclass of origin class {origin_processor_class.__class__.__name__}"
                )
            cls._XFUSER_ATTENTION_PROCESSOR_MAPPING[origin_processor_class] = xfuser_processor
            return xfuser_processor
        return decorator

    @classmethod
    def get_processor(cls, processor):
        for origin_processor_class, xfuser_processor in cls._XFUSER_ATTENTION_PROCESSOR_MAPPING.items():
            if isinstance(processor, origin_processor_class):
                return xfuser_processor
        raise ValueError(f"Attention Processor class {processor.__class__.__name__} is not supported by xFuser")


@xFuserLayerWrappersRegister.register(Attention)
class xFuserAttentionWrapper(xFuserAttentionBaseWrapper):
    def __init__(
        self,
        attention: Attention,
    ):
        super().__init__(attention=attention)
        self.processor = xFuserAttentionProcessorRegister.get_processor(attention.processor)()

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **cross_attention_kwargs,
    ) -> torch.Tensor:
        r"""
        The forward method of the `Attention` class.

        Args:
            hidden_states (`torch.Tensor`):
                The hidden states of the query.
            encoder_hidden_states (`torch.Tensor`, *optional*):
                The hidden states of the encoder.
            attention_mask (`torch.Tensor`, *optional*):
                The attention mask to use. If `None`, no mask is applied.
            **cross_attention_kwargs:
                Additional keyword arguments to pass along to the cross attention.

        Returns:
            `torch.Tensor`: The output of the attention layer.
        """
        # The `Attention` class can call different attention processors / attention functions
        # here we simply pass along all tensors to the selected processor class
        # For standard processors that are defined here, `**cross_attention_kwargs` is empty
        attn_parameters = set(inspect.signature(self.processor.__call__).parameters.keys())
        quiet_attn_parameters = {"ip_adapter_masks"}
        unused_kwargs = [
            k for k, _ in cross_attention_kwargs.items() if k not in attn_parameters and k not in quiet_attn_parameters
        ]
        if len(unused_kwargs) > 0:
            logger.warning(
                f"cross_attention_kwargs {unused_kwargs} are not expected by {self.processor.__class__.__name__} and will be ignored."
            )
        cross_attention_kwargs = {k: w for k, w in cross_attention_kwargs.items() if k in attn_parameters}

        return self.processor(
            self,
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            **cross_attention_kwargs,
        )

@xFuserAttentionProcessorRegister.register(AttnProcessor2_0)
class xFuserAttnProcessor2_0(AttnProcessor2_0):
    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ):
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
            deprecate("scale", "1.0.0", deprecation_message)

        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        kv = attn.to_kv(encoder_hidden_states)

#! ---------------------------------------- KV CACHE ----------------------------------------
        if (
            HAS_FLASH_ATTN 
            and get_sequence_parallel_world_size() > 1
            and attn.use_long_ctx_attn_kvcache 
        ):
            key, value = torch.chunk(kv, 2, dim=-1)
            inner_dim = key.shape[-1]
            head_dim = inner_dim // attn.heads
        else:
            # the distributed sparse attention from xfuser
            if get_runtime_state().num_pipeline_patch == 1:
                local_kv = kv
            else:
                if not get_runtime_state().patch_mode:
                    local_kv = kv
                else:
                    local_kv = attn.activation_cache
                    token_start_idx = sum(get_runtime_state().pp_patches_token_num[:get_runtime_state().pipeline_patch_idx])
                    token_end_idx = sum(get_runtime_state().pp_patches_token_num[:get_runtime_state().pipeline_patch_idx+1])
                    local_kv[:, token_start_idx:token_end_idx, :] = kv

                attn.activation_cache = local_kv

            key, value = torch.split(local_kv, local_kv.shape[-1] // 2, dim=-1)
            inner_dim = key.shape[-1]
            head_dim = inner_dim // attn.heads
#! ---------------------------------------- KV CACHE ----------------------------------------

#! ---------------------------------------- ATTENTION ----------------------------------------
        if HAS_LONG_CTX_ATTN and get_sequence_parallel_world_size() > 1:
            query = query.view(batch_size, -1, attn.heads, head_dim)
            key = key.view(batch_size, -1, attn.heads, head_dim)
            value = value.view(batch_size, -1, attn.heads, head_dim)
            hidden_states = attn.hybrid_seq_parallel_attn(
                query, key, value, dropout_p=0.0, causal=False
            )
            hidden_states = hidden_states.reshape(batch_size, -1, attn.heads * head_dim)

        else:
            if HAS_FLASH_ATTN:
                from flash_attn import flash_attn_func

                query = query.view(batch_size, -1, attn.heads, head_dim)
                key = key.view(batch_size, -1, attn.heads, head_dim)
                value = value.view(batch_size, -1, attn.heads, head_dim)
                hidden_states = flash_attn_func(
                    query, key, value, dropout_p=0.0, causal=False
                )
                hidden_states = hidden_states.reshape(batch_size, -1, attn.heads * head_dim)

            else:
                query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
                key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
                value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

                # the output of sdp = (batch, num_heads, seq_len, head_dim)
                # TODO: add support for attn.module.scale when we move to Torch 2.1
                hidden_states = F.scaled_dot_product_attention(
                    query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
                )

                hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)

        #! ORIGIN
        # query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        # value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # # TODO: add support for attn.scale when we move to Torch 2.1
        # hidden_states = F.scaled_dot_product_attention(
        #     query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        # )

        # hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
#! ---------------------------------------- ATTENTION ----------------------------------------

        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states

@xFuserAttentionProcessorRegister.register(JointAttnProcessor2_0)
class xFuserJointAttnProcessor2_0(JointAttnProcessor2_0):
    """Attention processor used typically in processing the SD3-like self-attention projections."""

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        *args,
        **kwargs,
    ) -> torch.FloatTensor:
        residual = hidden_states

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
        context_input_ndim = encoder_hidden_states.ndim
        if context_input_ndim == 4:
            batch_size, channel, height, width = encoder_hidden_states.shape
            encoder_hidden_states = encoder_hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size = encoder_hidden_states.shape[0]

        # `sample` projections.
        query = attn.to_q(hidden_states)
        # kv = attn.to_kv(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        kv = torch.cat([key, value], dim=-1)

#! ---------------------------------------- KV CACHE ----------------------------------------
        # if use sp, use the kvcache inside long_context_attention
        if (
            HAS_FLASH_ATTN 
            and get_sequence_parallel_world_size() > 1
            and attn.use_long_ctx_attn_kvcache 
        ):
            key, value = torch.chunk(kv, 2, dim=-1)
        else:
            if get_runtime_state().num_pipeline_patch == 1:
                local_kv = kv
            else:
                if not get_runtime_state().patch_mode:
                    local_kv = kv
                else:
                    local_kv = attn.activation_cache
                    token_start_idx = sum(get_runtime_state().pp_patches_token_num[:get_runtime_state().pipeline_patch_idx])
                    token_end_idx = sum(get_runtime_state().pp_patches_token_num[:get_runtime_state().pipeline_patch_idx+1])
                    local_kv[:, token_start_idx:token_end_idx, :] = kv

                attn.activation_cache = local_kv

            key, value = torch.split(local_kv, local_kv.shape[-1] // 2, dim=-1)
#! ---------------------------------------- KV CACHE ----------------------------------------

        # `context` projections.
        if get_runtime_state().pipeline_patch_idx == 0:
            encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
        encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
        encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

#! ---------------------------------------- ATTENTION ----------------------------------------
        if HAS_LONG_CTX_ATTN and get_sequence_parallel_world_size() > 1:
            query = query.view(batch_size, -1, attn.heads, head_dim)
            key = key.view(batch_size, -1, attn.heads, head_dim)
            value = value.view(batch_size, -1, attn.heads, head_dim)
            if get_runtime_state().pipeline_patch_idx == 0:
                encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(batch_size, -1, attn.heads, head_dim)
            encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(batch_size, -1, attn.heads, head_dim)
            encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(batch_size, -1, attn.heads, head_dim)
            hidden_states = attn.hybrid_seq_parallel_attn(
                query, key, value, dropout_p=0.0, causal=False,
                joint_tensor_query=encoder_hidden_states_query_proj
                if get_runtime_state().pipeline_patch_idx == 0
                else None,
                joint_tensor_key=encoder_hidden_states_key_proj,
                joint_tensor_value=encoder_hidden_states_value_proj,
            )
            hidden_states = hidden_states.reshape(batch_size, -1, attn.heads * head_dim)

        else:
            if get_runtime_state().pipeline_patch_idx == 0:
                query = torch.cat([query, encoder_hidden_states_query_proj], dim=1)
            key = torch.cat([key, encoder_hidden_states_key_proj], dim=1)
            value = torch.cat([value, encoder_hidden_states_value_proj], dim=1)

            if HAS_FLASH_ATTN:
                from flash_attn import flash_attn_func

                query = query.view(batch_size, -1, attn.heads, head_dim)
                key = key.view(batch_size, -1, attn.heads, head_dim)
                value = value.view(batch_size, -1, attn.heads, head_dim)
                hidden_states = flash_attn_func(
                    query, key, value, dropout_p=0.0, causal=False
                )
                hidden_states = hidden_states.reshape(batch_size, -1, attn.heads * head_dim)

            else:
                query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
                key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
                value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

                # the output of sdp = (batch, num_heads, seq_len, head_dim)
                # TODO: add support for attn.module.scale when we move to Torch 2.1
                hidden_states = F.scaled_dot_product_attention(
                    query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
                )

                hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)

        #! ORIGIN
        # query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        # key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        # value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # hidden_states = hidden_states = F.scaled_dot_product_attention(
        #     query, key, value, dropout_p=0.0, is_causal=False
        # )
        # hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
#! ---------------------------------------- ATTENTION ----------------------------------------
        hidden_states = hidden_states.to(query.dtype)

        # Split the attention outputs.
        if get_runtime_state().pipeline_patch_idx == 0:
            hidden_states, encoder_hidden_states = (
                hidden_states[:, : residual.shape[1]],
                hidden_states[:, residual.shape[1] :],
            )

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)
        if not attn.context_pre_only:
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
        if context_input_ndim == 4:
            encoder_hidden_states = encoder_hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        return hidden_states, encoder_hidden_states