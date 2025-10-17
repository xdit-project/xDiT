import inspect
from typing import Optional

import torch
from torch import nn
import torch.distributed
from torch.nn import functional as F
from diffusers.utils import deprecate
from diffusers.models.attention import Attention
from diffusers.models.transformers.transformer_wan import WanAttention
from diffusers.models.transformers.sana_transformer import SanaAttnProcessor2_0
from diffusers.models.attention_dispatch import dispatch_attention_fn

from diffusers.models.attention_processor import (
    AttnProcessor2_0,
    JointAttnProcessor2_0,
    HunyuanAttnProcessor2_0,
    CogVideoXAttnProcessor2_0,
    SanaLinearAttnProcessor2_0,
)

from diffusers.models.transformers.transformer_wan import WanAttnProcessor

try:
    from diffusers.models.transformers.transformer_hunyuan_video import (
        HunyuanVideoAttnProcessor2_0,
    )
except ImportError:
    HunyuanVideoAttnProcessor2_0 = None

from diffusers.models.embeddings import apply_rotary_emb

from xfuser.core.distributed import (
    get_sequence_parallel_world_size,
    get_pipeline_parallel_world_size,
    get_ulysses_parallel_world_size,
)
from xfuser.core.fast_attention import (
    xFuserFastAttention,
    get_fast_attn_enable,
)

from xfuser.core.cache_manager.cache_manager import get_cache_manager
from xfuser.core.distributed.runtime_state import get_runtime_state
from xfuser.model_executor.layers import xFuserLayerBaseWrapper
from xfuser.model_executor.layers import xFuserLayerWrappersRegister
from xfuser.logger import init_logger
from xfuser.envs import PACKAGES_CHECKER

if torch.__version__ >= "2.5.0":
    from xfuser.model_executor.layers.usp import USP
else:
    from xfuser.model_executor.layers.usp_legacy import USP

logger = init_logger(__name__)

env_info = PACKAGES_CHECKER.get_packages_info()
HAS_AITER = env_info["has_aiter"]
HAS_LONG_CTX_ATTN = env_info["has_long_ctx_attn"]
HAS_FLASH_ATTN = env_info["has_flash_attn"]

if HAS_LONG_CTX_ATTN:
    from yunchang.kernels import AttnType


def is_v100():
    if not torch.cuda.is_available():
        return False
    device_name = torch.cuda.get_device_name(torch.cuda.current_device())
    return "V100" in device_name


def torch_compile_disable_if_v100(func):
    if is_v100():
        return torch.compiler.disable(func)
    return func


def set_hybrid_seq_parallel_attn(self, use_long_ctx_attn_kvcache):
    """
    Initialize hybrid sequence-parallel attention based on available backend.
    """
    if HAS_LONG_CTX_ATTN and get_sequence_parallel_world_size() > 1:
        from xfuser.core.long_ctx_attention import (
            xFuserLongContextAttention,
        )
        from yunchang.kernels import AttnType

        if HAS_AITER:
            assert 'AITER' in AttnType.__members__, f"AttnType.AITER not implemented in yunchang version: {yunchang.__version__}. Upgrade to latest version from source."
            self.hybrid_seq_parallel_attn = xFuserLongContextAttention(
                    use_kv_cache=self.use_long_ctx_attn_kvcache,
                    attn_type=AttnType.AITER,
            )
        elif HAS_FLASH_ATTN:
            self.hybrid_seq_parallel_attn = xFuserLongContextAttention(
                    use_kv_cache=self.use_long_ctx_attn_kvcache,
                    attn_type=AttnType.FA,
            )
        else:
            self.hybrid_seq_parallel_attn = xFuserLongContextAttention(
                    use_kv_cache=self.use_long_ctx_attn_kvcache,
                    attn_type=AttnType.TORCH,
            )
    else:
        self.hybrid_seq_parallel_attn = None


class xFuserAttentionBaseWrapper(xFuserLayerBaseWrapper):
    def __init__(
        self,
        attention: Attention,
    ):
        super().__init__(module=attention)

        to_k = self.module.to_k
        to_v = self.module.to_v
        assert isinstance(to_k, nn.Linear)
        assert isinstance(to_v, nn.Linear)
        assert (to_k.bias is None) == (to_v.bias is None)
        assert to_k.weight.shape == to_v.weight.shape


class xFuserAttentionProcessorRegister:
    _XFUSER_ATTENTION_PROCESSOR_MAPPING = {}

    @classmethod
    def register(cls, origin_processor_class):
        def decorator(xfuser_processor):
            if not issubclass(xfuser_processor, origin_processor_class):
                raise ValueError(
                    f"{xfuser_processor.__class__.__name__} is not a subclass of origin class {origin_processor_class.__class__.__name__}"
                )
            cls._XFUSER_ATTENTION_PROCESSOR_MAPPING[origin_processor_class] = (
                xfuser_processor
            )
            return xfuser_processor

        return decorator

    @classmethod
    def get_processor(cls, processor):
        for (
            origin_processor_class,
            xfuser_processor,
        ) in cls._XFUSER_ATTENTION_PROCESSOR_MAPPING.items():
            if isinstance(processor, origin_processor_class):
                return xfuser_processor
        raise ValueError(
            f"Attention Processor class {processor.__class__.__name__} is not supported by xFuser"
        )

@xFuserLayerWrappersRegister.register(Attention)
class xFuserAttentionWrapper(xFuserAttentionBaseWrapper):
    def __init__(
        self,
        attention: Attention,
        latte_temporal_attention: bool = False,
    ):
        super().__init__(attention=attention)
        self.processor = xFuserAttentionProcessorRegister.get_processor(
            attention.processor
        )()
        self.latte_temporal_attention = latte_temporal_attention

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
        attn_parameters = set(
            inspect.signature(self.processor.__call__).parameters.keys()
        )
        quiet_attn_parameters = {"ip_adapter_masks"}
        unused_kwargs = [
            k
            for k, _ in cross_attention_kwargs.items()
            if k not in attn_parameters and k not in quiet_attn_parameters
        ]
        if len(unused_kwargs) > 0:
            logger.warning(
                f"cross_attention_kwargs {unused_kwargs} are not expected by {self.processor.__class__.__name__} and will be ignored."
            )
        cross_attention_kwargs = {
            k: w for k, w in cross_attention_kwargs.items() if k in attn_parameters
        }

        return self.processor(
            self,
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            latte_temporal_attention=self.latte_temporal_attention,
            **cross_attention_kwargs,
        )


@xFuserAttentionProcessorRegister.register(AttnProcessor2_0)
class xFuserAttnProcessor2_0(AttnProcessor2_0):
    def __init__(self):
        super().__init__()
        use_long_ctx_attn_kvcache = True
        self.use_long_ctx_attn_kvcache = (
            HAS_LONG_CTX_ATTN
            and use_long_ctx_attn_kvcache
            and get_sequence_parallel_world_size() > 1
        )
        set_hybrid_seq_parallel_attn(self, self.use_long_ctx_attn_kvcache)

        if get_fast_attn_enable():
            self.fast_attn = xFuserFastAttention()

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        latte_temporal_attention: Optional[bool] = False,
        *args,
        **kwargs,
    ):
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
            deprecate("scale", "1.0.0", deprecation_message)

        #! ---------------------------------------- Fast Attention ----------------------------------------
        if get_fast_attn_enable():
            return self.fast_attn(
                attn,
                hidden_states,
                encoder_hidden_states,
                attention_mask,
                temb,
                *args,
                **kwargs,
            )
        #! ---------------------------------------- Fast Attention ----------------------------------------

        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(
                batch_size, channel, height * width
            ).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape
            if encoder_hidden_states is None
            else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(
                attention_mask, sequence_length, batch_size
            )
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(
                batch_size, attn.heads, -1, attention_mask.shape[-1]
            )

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(
                1, 2
            )

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(
                encoder_hidden_states
            )

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        #! ---------------------------------------- KV CACHE ----------------------------------------
        if not self.use_long_ctx_attn_kvcache:
            key, value = get_cache_manager().update_and_get_kv_cache(
                new_kv=[key, value],
                layer=attn,
                slice_dim=2,
                layer_type="attn",
            )
        #! ---------------------------------------- KV CACHE ----------------------------------------

        #! ---------------------------------------- ATTENTION ----------------------------------------
        if (
            HAS_LONG_CTX_ATTN
            and get_sequence_parallel_world_size() > 1
            and not latte_temporal_attention
        ):
            query = query.transpose(1, 2)
            key = key.transpose(1, 2)
            value = value.transpose(1, 2)
            hidden_states = self.hybrid_seq_parallel_attn(
                attn,
                query,
                key,
                value,
                dropout_p=0.0,
                causal=False,
                joint_strategy="none",
            )
            hidden_states = hidden_states.reshape(batch_size, -1, attn.heads * head_dim)

        else:
            if HAS_FLASH_ATTN:
                from flash_attn import flash_attn_func

                query = query.transpose(1, 2)
                key = key.transpose(1, 2)
                value = value.transpose(1, 2)
                hidden_states = flash_attn_func(
                    query, key, value, dropout_p=0.0, causal=False
                )
                hidden_states = hidden_states.reshape(
                    batch_size, -1, attn.heads * head_dim
                )

            else:
                # the output of sdp = (batch, num_heads, seq_len, head_dim)
                # TODO: add support for attn.module.scale when we move to Torch 2.1
                hidden_states = F.scaled_dot_product_attention(
                    query,
                    key,
                    value,
                    attn_mask=attention_mask,
                    dropout_p=0.0,
                    is_causal=False,
                )

                hidden_states = hidden_states.transpose(1, 2).reshape(
                    batch_size, -1, attn.heads * head_dim
                )

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
    def __init__(self):
        super().__init__()
        use_long_ctx_attn_kvcache = True
        self.use_long_ctx_attn_kvcache = (
            HAS_LONG_CTX_ATTN
            and use_long_ctx_attn_kvcache
            and get_sequence_parallel_world_size() > 1
        )
        set_hybrid_seq_parallel_attn(self, self.use_long_ctx_attn_kvcache)

        if get_fast_attn_enable():
            self.fast_attn = xFuserFastAttention()

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
        batch_size = hidden_states.shape[0]

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(
                batch_size, channel, height * width
            ).transpose(1, 2)

        if encoder_hidden_states is not None:
            context_input_ndim = encoder_hidden_states.ndim
            if context_input_ndim == 4:
                batch_size, channel, height, width = encoder_hidden_states.shape
                encoder_hidden_states = encoder_hidden_states.view(
                    batch_size, channel, height * width
                ).transpose(1, 2)

        # `sample` projections.
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        # `context` projections.
        if encoder_hidden_states is not None:
            encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
            encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
            encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)
            encoder_hidden_states_query_proj = (
                encoder_hidden_states_query_proj.view(
                    batch_size, -1, attn.heads, head_dim
                )
            )
            encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
                batch_size, -1, attn.heads, head_dim
            )
            encoder_hidden_states_value_proj = (
                encoder_hidden_states_value_proj.view(
                    batch_size, -1, attn.heads, head_dim
                )
            )
            if attn.norm_added_q is not None:
                encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj)
            if attn.norm_added_k is not None:
                encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj)
        else:
            encoder_hidden_states_query_proj = None
            encoder_hidden_states_key_proj = None
            encoder_hidden_states_value_proj = None

        #! ---------------------------------------- KV CACHE ----------------------------------------
        if not self.use_long_ctx_attn_kvcache:
            key, value = get_cache_manager().update_and_get_kv_cache(
                new_kv=[key, value],
                layer=attn,
                slice_dim=1,
                layer_type="attn",
            )
        #! ---------------------------------------- KV CACHE ----------------------------------------

        query = query.view(batch_size, -1, attn.heads, head_dim)
        key = key.view(batch_size, -1, attn.heads, head_dim)
        value = value.view(batch_size, -1, attn.heads, head_dim)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        #! ---------------------------------------- ATTENTION ----------------------------------------
        if HAS_LONG_CTX_ATTN and get_sequence_parallel_world_size() > 1:
            if encoder_hidden_states is not None:
                if get_runtime_state().split_text_embed_in_sp:
                    query = torch.cat([query, encoder_hidden_states_query_proj], dim=1)
                    key = torch.cat([key, encoder_hidden_states_key_proj], dim=1)
                    value = torch.cat([value, encoder_hidden_states_value_proj], dim=1)

                    encoder_hidden_states_query_proj = None
                    encoder_hidden_states_key_proj = None
                    encoder_hidden_states_value_proj = None
                else:
                    encoder_hidden_states_query_proj = (
                        encoder_hidden_states_query_proj.view(
                            batch_size, -1, attn.heads, head_dim
                        )
                    )
                    encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
                        batch_size, -1, attn.heads, head_dim
                    )
                    encoder_hidden_states_value_proj = (
                        encoder_hidden_states_value_proj.view(
                            batch_size, -1, attn.heads, head_dim
                        )
                    )

            hidden_states = self.hybrid_seq_parallel_attn(
                attn,
                query,
                key,
                value,
                dropout_p=0.0,
                causal=False,
                joint_tensor_query=encoder_hidden_states_query_proj,
                joint_tensor_key=encoder_hidden_states_key_proj,
                joint_tensor_value=encoder_hidden_states_value_proj,
                joint_strategy="rear",
            )
            hidden_states = hidden_states.reshape(batch_size, -1, attn.heads * head_dim)

        else:
            if encoder_hidden_states is not None:
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
                hidden_states = hidden_states.reshape(
                    batch_size, -1, attn.heads * head_dim
                )

            else:
                query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
                key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
                value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

                # the output of sdp = (batch, num_heads, seq_len, head_dim)
                # TODO: add support for attn.module.scale when we move to Torch 2.1
                hidden_states = F.scaled_dot_product_attention(
                    query,
                    key,
                    value,
                    attn_mask=attention_mask,
                    dropout_p=0.0,
                    is_causal=False,
                )

                hidden_states = hidden_states.transpose(1, 2).reshape(
                    batch_size, -1, attn.heads * head_dim
                )

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
        if encoder_hidden_states is not None:
            hidden_states, encoder_hidden_states = (
                hidden_states[:, : residual.shape[1]],
                hidden_states[:, residual.shape[1] :],
            )
            if not attn.context_pre_only:
                encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                batch_size, channel, height, width
            )

        if encoder_hidden_states is not None:
            if context_input_ndim == 4:
                encoder_hidden_states = encoder_hidden_states.transpose(-1, -2).reshape(
                    batch_size, channel, height, width
                )
            return hidden_states, encoder_hidden_states
        else:
            return hidden_states


@xFuserAttentionProcessorRegister.register(WanAttnProcessor)
class xFuserWanAttnProcessor(WanAttnProcessor):

    def __init__(self):
        super().__init__()
        use_long_ctx_attn_kvcache = True
        self.use_long_ctx_attn_kvcache = (
            HAS_LONG_CTX_ATTN
            and use_long_ctx_attn_kvcache
            and get_sequence_parallel_world_size() > 1
        )
        set_hybrid_seq_parallel_attn(self, self.use_long_ctx_attn_kvcache)

    def _get_qkv_projections(self, attn: "WanAttention", hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor):
        # encoder_hidden_states is only passed for cross-attention
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        if attn.fused_projections:
            if attn.cross_attention_dim_head is None:
                # In self-attention layers, we can fuse the entire QKV projection into a single linear
                query, key, value = attn.to_qkv(hidden_states).chunk(3, dim=-1)
            else:
                # In cross-attention layers, we can only fuse the KV projections into a single linear
                query = attn.to_q(hidden_states)
                key, value = attn.to_kv(encoder_hidden_states).chunk(2, dim=-1)
        else:
            query = attn.to_q(hidden_states)
            key = attn.to_k(encoder_hidden_states)
            value = attn.to_v(encoder_hidden_states)
        return query, key, value

    def _get_added_kv_projections(self, attn: "WanAttention", encoder_hidden_states_img: torch.Tensor):
        if attn.fused_projections:
            key_img, value_img = attn.to_added_kv(encoder_hidden_states_img).chunk(2, dim=-1)
        else:
            key_img = attn.add_k_proj(encoder_hidden_states_img)
            value_img = attn.add_v_proj(encoder_hidden_states_img)
        return key_img, value_img


    def __call__(
        self,
        attn: "WanAttention",
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_emb: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        encoder_hidden_states_img = None
        if attn.add_k_proj is not None:
            # 512 is the context length of the text encoder, hardcoded for now
            image_context_length = encoder_hidden_states.shape[1] - 512
            encoder_hidden_states_img = encoder_hidden_states[:, :image_context_length]
            encoder_hidden_states = encoder_hidden_states[:, image_context_length:]

        query, key, value = self._get_qkv_projections(attn, hidden_states, encoder_hidden_states)

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        query = query.unflatten(2, (attn.heads, -1))
        key = key.unflatten(2, (attn.heads, -1))
        value = value.unflatten(2, (attn.heads, -1))

        if rotary_emb is not None:

            def apply_rotary_emb(
                hidden_states: torch.Tensor,
                freqs_cos: torch.Tensor,
                freqs_sin: torch.Tensor,
            ):
                x1, x2 = hidden_states.unflatten(-1, (-1, 2)).unbind(-1)
                cos = freqs_cos[..., 0::2]
                sin = freqs_sin[..., 1::2]
                out = torch.empty_like(hidden_states)
                out[..., 0::2] = x1 * cos - x2 * sin
                out[..., 1::2] = x1 * sin + x2 * cos
                return out.type_as(hidden_states)

            query = apply_rotary_emb(query, *rotary_emb)
            key = apply_rotary_emb(key, *rotary_emb)

        # I2V task
        hidden_states_img = None
        if encoder_hidden_states_img is not None:
            key_img, value_img = self._get_added_kv_projections(attn, encoder_hidden_states_img)
            key_img = attn.norm_added_k(key_img)

            key_img = key_img.unflatten(2, (attn.heads, -1))
            value_img = value_img.unflatten(2, (attn.heads, -1))


            hidden_states_img = self.hybrid_seq_parallel_attn(
                None, query, key_img, value_img
            )
            hidden_states_img = hidden_states_img.flatten(2, 3)
            hidden_states_img = hidden_states_img.type_as(query)

        hidden_states = self.hybrid_seq_parallel_attn(
           None, query, key, value
        )
        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.type_as(query)

        if hidden_states_img is not None:
            hidden_states = hidden_states + hidden_states_img

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states


@xFuserAttentionProcessorRegister.register(FluxAttnProcessor2_0)
class xFuserFluxAttnProcessor2_0(FluxAttnProcessor2_0):
    """Attention processor used typically in processing the SD3-like self-attention projections."""

    def __init__(self):
        super().__init__()
        use_long_ctx_attn_kvcache = True
        self.use_long_ctx_attn_kvcache = (
            HAS_LONG_CTX_ATTN
            and use_long_ctx_attn_kvcache
            and get_sequence_parallel_world_size() > 1
        )
        set_hybrid_seq_parallel_attn(self, self.use_long_ctx_attn_kvcache)

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> torch.FloatTensor:
        batch_size, _, _ = (
            hidden_states.shape
            if encoder_hidden_states is None
            else encoder_hidden_states.shape
        )

        # `sample` projections.
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # the attention in FluxSingleTransformerBlock does not use `encoder_hidden_states`
        if encoder_hidden_states is not None:
            # `context` projections.
            encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
            encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
            encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

            encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)

            if attn.norm_added_q is not None:
                encoder_hidden_states_query_proj = attn.norm_added_q(
                    encoder_hidden_states_query_proj
                )
            if attn.norm_added_k is not None:
                encoder_hidden_states_key_proj = attn.norm_added_k(
                    encoder_hidden_states_key_proj
                )

            num_encoder_hidden_states_tokens = encoder_hidden_states_query_proj.shape[2]
            num_query_tokens = query.shape[2]

            # attention
            query = torch.cat([encoder_hidden_states_query_proj, query], dim=2)
            key = torch.cat([encoder_hidden_states_key_proj, key], dim=2)
            value = torch.cat([encoder_hidden_states_value_proj, value], dim=2)
        else:
            num_encoder_hidden_states_tokens = (
                get_runtime_state().max_condition_sequence_length
            )
            num_query_tokens = query.shape[2] - num_encoder_hidden_states_tokens

        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb)
            key = apply_rotary_emb(key, image_rotary_emb)

        #! ---------------------------------------- KV CACHE ----------------------------------------
        if (
            get_runtime_state().num_pipeline_patch > 1
            and not self.use_long_ctx_attn_kvcache
        ):
            encoder_hidden_states_key_proj, key = key.split(
                [num_encoder_hidden_states_tokens, num_query_tokens], dim=2
            )
            encoder_hidden_states_value_proj, value = value.split(
                [num_encoder_hidden_states_tokens, num_query_tokens], dim=2
            )
            key, value = get_cache_manager().update_and_get_kv_cache(
                new_kv=[key, value],
                layer=attn,
                slice_dim=2,
                layer_type="attn",
            )
            key = torch.cat([encoder_hidden_states_key_proj, key], dim=2)
            value = torch.cat([encoder_hidden_states_value_proj, value], dim=2)
        #! ---------------------------------------- KV CACHE ----------------------------------------

        #! ---------------------------------------- ATTENTION ----------------------------------------
        if (
            get_pipeline_parallel_world_size() == 1
            and get_runtime_state().split_text_embed_in_sp
        ):
            hidden_states = USP(query, key, value, dropout_p=0.0, is_causal=False)
            hidden_states = hidden_states.transpose(1, 2).reshape(
                batch_size, -1, attn.heads * head_dim
            )
        elif HAS_LONG_CTX_ATTN and get_sequence_parallel_world_size() > 1:
            query = query.transpose(1, 2)
            key = key.transpose(1, 2)
            value = value.transpose(1, 2)
            if get_runtime_state().split_text_embed_in_sp:
                encoder_hidden_states_query_proj = None
                encoder_hidden_states_key_proj = None
                encoder_hidden_states_value_proj = None
            else:
                encoder_hidden_states_query_proj, query = query.split(
                    [num_encoder_hidden_states_tokens, num_query_tokens], dim=1
                )
                encoder_hidden_states_key_proj, key = key.split(
                    [num_encoder_hidden_states_tokens, num_query_tokens], dim=1
                )
                encoder_hidden_states_value_proj, value = value.split(
                    [num_encoder_hidden_states_tokens, num_query_tokens], dim=1
                )
            hidden_states = self.hybrid_seq_parallel_attn(
                attn if get_runtime_state().num_pipeline_patch > 1 else None,
                query,
                key,
                value,
                dropout_p=0.0,
                causal=False,
                joint_tensor_query=encoder_hidden_states_query_proj,
                joint_tensor_key=encoder_hidden_states_key_proj,
                joint_tensor_value=encoder_hidden_states_value_proj,
                joint_strategy="front",
            )
            hidden_states = hidden_states.reshape(batch_size, -1, attn.heads * head_dim)

        else:
            if HAS_FLASH_ATTN:
                from flash_attn import flash_attn_func

                query = query.transpose(1, 2)
                key = key.transpose(1, 2)
                value = value.transpose(1, 2)
                hidden_states = flash_attn_func(
                    query, key, value, dropout_p=0.0, causal=False
                )
                hidden_states = hidden_states.reshape(
                    batch_size, -1, attn.heads * head_dim
                )
            else:
                hidden_states = F.scaled_dot_product_attention(
                    query, key, value, dropout_p=0.0, is_causal=False
                )
                hidden_states = hidden_states.transpose(1, 2).reshape(
                    batch_size, -1, attn.heads * head_dim
                )
        #! ---------------------------------------- ATTENTION ----------------------------------------

        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            encoder_hidden_states, hidden_states = (
                hidden_states[:, : encoder_hidden_states.shape[1]],
                hidden_states[:, encoder_hidden_states.shape[1] :],
            )

            # linear proj
            hidden_states = attn.to_out[0](hidden_states)
            # dropout
            hidden_states = attn.to_out[1](hidden_states)
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

            return hidden_states, encoder_hidden_states
        else:
            return hidden_states


@xFuserAttentionProcessorRegister.register(HunyuanAttnProcessor2_0)
class xFuserHunyuanAttnProcessor2_0(HunyuanAttnProcessor2_0):
    def __init__(self):
        super().__init__()
        use_long_ctx_attn_kvcache = True
        self.use_long_ctx_attn_kvcache = (
            HAS_LONG_CTX_ATTN
            and use_long_ctx_attn_kvcache
            and get_sequence_parallel_world_size() > 1
        )
        set_hybrid_seq_parallel_attn(self, self.use_long_ctx_attn_kvcache)

    # NOTE() torch.compile dose not works for V100
    @torch_compile_disable_if_v100
    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        latte_temporal_attention: Optional[bool] = False,
    ) -> torch.Tensor:
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(
                batch_size, channel, height * width
            ).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape
            if encoder_hidden_states is None
            else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(
                attention_mask, sequence_length, batch_size
            )
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(
                batch_size, attn.heads, -1, attention_mask.shape[-1]
            )

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(
                1, 2
            )

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(
                encoder_hidden_states
            )

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # Apply RoPE if needed
        # print(f"Q {query.shape}, {key.shape}, {image_rotary_emb[0].shape}")
        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb)
            if not attn.is_cross_attention:
                key = apply_rotary_emb(key, image_rotary_emb)

        #! ---------------------------------------- KV CACHE ----------------------------------------
        if not self.use_long_ctx_attn_kvcache:
            key, value = get_cache_manager().update_and_get_kv_cache(
                new_kv=[key, value],
                layer=attn,
                slice_dim=2,
                layer_type="attn",
            )
        #! ---------------------------------------- KV CACHE ----------------------------------------

        #! ---------------------------------------- ATTENTION ----------------------------------------
        if (
            HAS_LONG_CTX_ATTN
            and get_sequence_parallel_world_size() > 1
            and not attn.is_cross_attention
            and not latte_temporal_attention
        ):
            query = query.transpose(1, 2)
            key = key.transpose(1, 2)
            value = value.transpose(1, 2)

            hidden_states = self.hybrid_seq_parallel_attn(
                attn,
                query,
                key,
                value,
                dropout_p=0.0,
                causal=False,
                joint_strategy="none",
            )
            hidden_states = hidden_states.reshape(batch_size, -1, attn.heads * head_dim)

        else:
            if HAS_FLASH_ATTN:
                from flash_attn import flash_attn_func

                query = query.transpose(1, 2)
                key = key.transpose(1, 2)
                value = value.transpose(1, 2)
                hidden_states = flash_attn_func(
                    query, key, value, dropout_p=0.0, causal=False
                )
                hidden_states = hidden_states.reshape(
                    batch_size, -1, attn.heads * head_dim
                )

            else:
                # the output of sdp = (batch, num_heads, seq_len, head_dim)
                # TODO: add support for attn.module.scale when we move to Torch 2.1
                hidden_states = F.scaled_dot_product_attention(
                    query,
                    key,
                    value,
                    attn_mask=attention_mask,
                    dropout_p=0.0,
                    is_causal=False,
                )

                hidden_states = hidden_states.transpose(1, 2).reshape(
                    batch_size, -1, attn.heads * head_dim
                )

        #! ORIGIN
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

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                batch_size, channel, height, width
            )

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


@xFuserAttentionProcessorRegister.register(CogVideoXAttnProcessor2_0)
class xFuserCogVideoXAttnProcessor2_0(CogVideoXAttnProcessor2_0):
    r"""
    Processor for implementing scaled dot-product attention for the CogVideoX model. It applies a rotary embedding on
    query and key vectors, but does not include spatial normalization.
    """

    def __init__(self):
        super().__init__()
        use_long_ctx_attn_kvcache = True
        self.use_long_ctx_attn_kvcache = (
            HAS_LONG_CTX_ATTN
            and use_long_ctx_attn_kvcache
            and get_sequence_parallel_world_size() > 1
        )
        set_hybrid_seq_parallel_attn(self, self.use_long_ctx_attn_kvcache)

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        text_seq_length = encoder_hidden_states.size(1)
        latent_seq_length = hidden_states.size(1)

        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        batch_size, sequence_length, _ = (
            hidden_states.shape
            if encoder_hidden_states is None
            else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(
                attention_mask, sequence_length, batch_size
            )
            attention_mask = attention_mask.view(
                batch_size, attn.heads, -1, attention_mask.shape[-1]
            )

        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # Apply RoPE if needed
        if image_rotary_emb is not None:
            query[:, :, text_seq_length:] = apply_rotary_emb(
                query[:, :, text_seq_length:], image_rotary_emb
            )
            if not attn.is_cross_attention:
                key[:, :, text_seq_length:] = apply_rotary_emb(
                    key[:, :, text_seq_length:], image_rotary_emb
                )

        #! ---------------------------------------- ATTENTION ----------------------------------------
        if (
            get_pipeline_parallel_world_size() == 1
            and get_runtime_state().split_text_embed_in_sp
        ):
            hidden_states = USP(query, key, value, dropout_p=0.0, is_causal=False)
            hidden_states = hidden_states.transpose(1, 2).reshape(
                batch_size, -1, attn.heads * head_dim
            )
        elif HAS_LONG_CTX_ATTN and get_sequence_parallel_world_size() > 1:
            if get_runtime_state().split_text_embed_in_sp:
                encoder_query = None
                encoder_key = None
                encoder_value = None
            else:
                encoder_query = query[:, :, :text_seq_length, :]
                query = query[:, :, text_seq_length:, :]
                encoder_key = key[:, :, :text_seq_length, :]
                key = key[:, :, text_seq_length:, :]
                encoder_value = value[:, :, :text_seq_length, :]
                value = value[:, :, text_seq_length:, :]

                encoder_query = encoder_query.transpose(1, 2)
                encoder_key = encoder_key.transpose(1, 2)
                encoder_value = encoder_value.transpose(1, 2)

            query = query.transpose(1, 2)
            key = key.transpose(1, 2)
            value = value.transpose(1, 2)

            hidden_states = self.hybrid_seq_parallel_attn(
                None,
                query,
                key,
                value,
                dropout_p=0.0,
                causal=False,
                joint_tensor_query=encoder_query,
                joint_tensor_key=encoder_key,
                joint_tensor_value=encoder_value,
                joint_strategy="front",
            )

            hidden_states = hidden_states.reshape(batch_size, -1, attn.heads * head_dim)
        else:
            if HAS_FLASH_ATTN:
                from flash_attn import flash_attn_func

                query = query.transpose(1, 2)
                key = key.transpose(1, 2)
                value = value.transpose(1, 2)
                hidden_states = flash_attn_func(
                    query, key, value, dropout_p=0.0, causal=False
                )
                hidden_states = hidden_states.reshape(
                    batch_size, -1, attn.heads * head_dim
                )

            else:
                # the output of sdp = (batch, num_heads, seq_len, head_dim)
                # TODO: add support for attn.scale when we move to Torch 2.1
                hidden_states = F.scaled_dot_product_attention(
                    query, key, value, dropout_p=0.0, is_causal=False
                )
                hidden_states = hidden_states.transpose(1, 2).reshape(
                    batch_size, -1, attn.heads * head_dim
                )

        #! ORIGIN
        # hidden_states = F.scaled_dot_product_attention(
        #     query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        # )
        # hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        #! ---------------------------------------- ATTENTION ----------------------------------------

        assert text_seq_length + latent_seq_length == hidden_states.shape[1]
        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        encoder_hidden_states, hidden_states = hidden_states.split(
            [text_seq_length, latent_seq_length], dim=1
        )
        return hidden_states, encoder_hidden_states


@xFuserAttentionProcessorRegister.register(CogVideoXAttnProcessor2_0)
class xFuserConsisIDAttnProcessor2_0(CogVideoXAttnProcessor2_0):
    r"""
    Processor for implementing scaled dot-product attention for the ConsisID model. It applies a rotary embedding on
    query and key vectors, but does not include spatial normalization.
    """

    def __init__(self):
        super().__init__()
        use_long_ctx_attn_kvcache = True
        self.use_long_ctx_attn_kvcache = (
            HAS_LONG_CTX_ATTN
            and use_long_ctx_attn_kvcache
            and get_sequence_parallel_world_size() > 1
        )
        set_hybrid_seq_parallel_attn(self, self.use_long_ctx_attn_kvcache)

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        text_seq_length = encoder_hidden_states.size(1)
        latent_seq_length = hidden_states.size(1)

        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        batch_size, sequence_length, _ = (
            hidden_states.shape
            if encoder_hidden_states is None
            else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(
                attention_mask, sequence_length, batch_size
            )
            attention_mask = attention_mask.view(
                batch_size, attn.heads, -1, attention_mask.shape[-1]
            )

        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # Apply RoPE if needed
        if image_rotary_emb is not None:
            query[:, :, text_seq_length:] = apply_rotary_emb(
                query[:, :, text_seq_length:], image_rotary_emb
            )
            if not attn.is_cross_attention:
                key[:, :, text_seq_length:] = apply_rotary_emb(
                    key[:, :, text_seq_length:], image_rotary_emb
                )

        #! ---------------------------------------- ATTENTION ----------------------------------------
        if (
            get_pipeline_parallel_world_size() == 1
            and get_runtime_state().split_text_embed_in_sp
        ):
            hidden_states = USP(query, key, value, dropout_p=0.0, is_causal=False)
            hidden_states = hidden_states.transpose(1, 2).reshape(
                batch_size, -1, attn.heads * head_dim
            )
        elif HAS_LONG_CTX_ATTN and get_sequence_parallel_world_size() > 1:
            if get_runtime_state().split_text_embed_in_sp:
                encoder_query = None
                encoder_key = None
                encoder_value = None
            else:
                encoder_query = query[:, :, :text_seq_length, :]
                query = query[:, :, text_seq_length:, :]
                encoder_key = key[:, :, :text_seq_length, :]
                key = key[:, :, text_seq_length:, :]
                encoder_value = value[:, :, :text_seq_length, :]
                value = value[:, :, text_seq_length:, :]

                encoder_query = encoder_query.transpose(1, 2)
                encoder_key = encoder_key.transpose(1, 2)
                encoder_value = encoder_value.transpose(1, 2)

            query = query.transpose(1, 2)
            key = key.transpose(1, 2)
            value = value.transpose(1, 2)

            hidden_states = self.hybrid_seq_parallel_attn(
                None,
                query,
                key,
                value,
                dropout_p=0.0,
                causal=False,
                joint_tensor_query=encoder_query,
                joint_tensor_key=encoder_key,
                joint_tensor_value=encoder_value,
                joint_strategy="front",
            )

            hidden_states = hidden_states.reshape(batch_size, -1, attn.heads * head_dim)
        else:
            if HAS_FLASH_ATTN:
                from flash_attn import flash_attn_func

                query = query.transpose(1, 2)
                key = key.transpose(1, 2)
                value = value.transpose(1, 2)
                hidden_states = flash_attn_func(
                    query, key, value, dropout_p=0.0, causal=False
                )
                hidden_states = hidden_states.reshape(
                    batch_size, -1, attn.heads * head_dim
                )

            else:
                # the output of sdp = (batch, num_heads, seq_len, head_dim)
                # TODO: add support for attn.scale when we move to Torch 2.1
                hidden_states = F.scaled_dot_product_attention(
                    query, key, value, dropout_p=0.0, is_causal=False
                )
                hidden_states = hidden_states.transpose(1, 2).reshape(
                    batch_size, -1, attn.heads * head_dim
                )

        #! ORIGIN
        # hidden_states = F.scaled_dot_product_attention(
        #     query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        # )
        # hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        #! ---------------------------------------- ATTENTION ----------------------------------------

        assert text_seq_length + latent_seq_length == hidden_states.shape[1]
        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        encoder_hidden_states, hidden_states = hidden_states.split(
            [text_seq_length, latent_seq_length], dim=1
        )
        return hidden_states, encoder_hidden_states


if HunyuanVideoAttnProcessor2_0 is not None:

    @xFuserAttentionProcessorRegister.register(HunyuanVideoAttnProcessor2_0)
    class xFuserHunyuanVideoAttnProcessor2_0(HunyuanVideoAttnProcessor2_0):
        def __init__(self):
            super().__init__()
            use_long_ctx_attn_kvcache = True
            self.use_long_ctx_attn_kvcache = (
                HAS_LONG_CTX_ATTN
                and use_long_ctx_attn_kvcache
                and get_sequence_parallel_world_size() > 1
            )
            set_hybrid_seq_parallel_attn(self, self.use_long_ctx_attn_kvcache)

        def __call__(
            self,
            attn: Attention,
            hidden_states: torch.Tensor,
            encoder_hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            image_rotary_emb: Optional[torch.Tensor] = None,
            *args,
            **kwargs,
        ) -> torch.Tensor:
            batch_size, _, _ = (
                hidden_states.shape
                if encoder_hidden_states is None
                else encoder_hidden_states.shape
            )

            if attn.add_q_proj is None and encoder_hidden_states is not None:
                hidden_states = torch.cat([hidden_states, encoder_hidden_states], dim=1)

            # 1. QKV projections
            query = attn.to_q(hidden_states)
            key = attn.to_k(hidden_states)
            value = attn.to_v(hidden_states)

            query = query.unflatten(2, (attn.heads, -1)).transpose(1, 2)
            key = key.unflatten(2, (attn.heads, -1)).transpose(1, 2)
            value = value.unflatten(2, (attn.heads, -1)).transpose(1, 2)

            # 2. QK normalization
            if attn.norm_q is not None:
                query = attn.norm_q(query)
            if attn.norm_k is not None:
                key = attn.norm_k(key)

            # 3. Rotational positional embeddings applied to latent stream
            if image_rotary_emb is not None:
                if attn.add_q_proj is None and encoder_hidden_states is not None:
                    query = torch.cat(
                        [
                            apply_rotary_emb(
                                query[:, :, : -encoder_hidden_states.shape[1]],
                                image_rotary_emb,
                            ),
                            query[:, :, -encoder_hidden_states.shape[1] :],
                        ],
                        dim=2,
                    )
                    key = torch.cat(
                        [
                            apply_rotary_emb(
                                key[:, :, : -encoder_hidden_states.shape[1]],
                                image_rotary_emb,
                            ),
                            key[:, :, -encoder_hidden_states.shape[1] :],
                        ],
                        dim=2,
                    )
                else:
                    query = apply_rotary_emb(query, image_rotary_emb)
                    key = apply_rotary_emb(key, image_rotary_emb)

            # 4. Encoder condition QKV projection and normalization
            if attn.add_q_proj is not None and encoder_hidden_states is not None:
                encoder_query = attn.add_q_proj(encoder_hidden_states)
                encoder_key = attn.add_k_proj(encoder_hidden_states)
                encoder_value = attn.add_v_proj(encoder_hidden_states)

                encoder_query = encoder_query.unflatten(2, (attn.heads, -1)).transpose(
                    1, 2
                )
                encoder_key = encoder_key.unflatten(2, (attn.heads, -1)).transpose(1, 2)
                encoder_value = encoder_value.unflatten(2, (attn.heads, -1)).transpose(
                    1, 2
                )

                if attn.norm_added_q is not None:
                    encoder_query = attn.norm_added_q(encoder_query)
                if attn.norm_added_k is not None:
                    encoder_key = attn.norm_added_k(encoder_key)

                query = torch.cat([query, encoder_query], dim=2)
                key = torch.cat([key, encoder_key], dim=2)
                value = torch.cat([value, encoder_value], dim=2)

            if encoder_hidden_states is not None:
                num_encoder_hidden_states_tokens = encoder_hidden_states.shape[1]
                num_query_tokens = query.shape[2] - num_encoder_hidden_states_tokens
            else:
                num_encoder_hidden_states_tokens = (
                    get_runtime_state().max_condition_sequence_length
                )
                num_query_tokens = query.shape[2] - num_encoder_hidden_states_tokens

            #! ---------------------------------------- ATTENTION ----------------------------------------
            if (
                get_pipeline_parallel_world_size() == 1
                and get_runtime_state().split_text_embed_in_sp
            ):
                hidden_states = USP(query, key, value, dropout_p=0.0, is_causal=False)
                hidden_states = hidden_states.transpose(1, 2).flatten(2, 3)
            elif HAS_LONG_CTX_ATTN and get_sequence_parallel_world_size() > 1:
                if get_runtime_state().split_text_embed_in_sp:
                    encoder_query = None
                    encoder_key = None
                    encoder_value = None
                else:
                    query, encoder_query = query.split(
                        [num_query_tokens, num_encoder_hidden_states_tokens], dim=2
                    )
                    key, encoder_key = key.split(
                        [num_query_tokens, num_encoder_hidden_states_tokens], dim=2
                    )
                    value, encoder_value = value.split(
                        [num_query_tokens, num_encoder_hidden_states_tokens], dim=2
                    )

                    encoder_query = encoder_query.transpose(1, 2)
                    encoder_key = encoder_key.transpose(1, 2)
                    encoder_value = encoder_value.transpose(1, 2)

                query = query.transpose(1, 2)
                key = key.transpose(1, 2)
                value = value.transpose(1, 2)

                hidden_states = self.hybrid_seq_parallel_attn(
                    None,
                    query,
                    key,
                    value,
                    dropout_p=0.0,
                    causal=False,
                    joint_tensor_query=encoder_query,
                    joint_tensor_key=encoder_key,
                    joint_tensor_value=encoder_value,
                    joint_strategy="rear",
                )

                hidden_states = hidden_states.flatten(2, 3)
            else:
                if HAS_FLASH_ATTN:
                    from flash_attn import flash_attn_func

                    query = query.transpose(1, 2)
                    key = key.transpose(1, 2)
                    value = value.transpose(1, 2)
                    hidden_states = flash_attn_func(
                        query, key, value, dropout_p=0.0, causal=False
                    )
                    hidden_states = hidden_states.flatten(2, 3)

                else:
                    # the output of sdp = (batch, num_heads, seq_len, head_dim)
                    # TODO: add support for attn.scale when we move to Torch 2.1
                    hidden_states = F.scaled_dot_product_attention(
                        query, key, value, dropout_p=0.0, is_causal=False
                    )
                    hidden_states = hidden_states.transpose(1, 2).flatten(2, 3)

            hidden_states = hidden_states.to(query.dtype)

            # 6. Output projection
            if encoder_hidden_states is not None:
                hidden_states, encoder_hidden_states = (
                    hidden_states[:, : -encoder_hidden_states.shape[1]],
                    hidden_states[:, -encoder_hidden_states.shape[1] :],
                )

                if getattr(attn, "to_out", None) is not None:
                    hidden_states = attn.to_out[0](hidden_states)
                    hidden_states = attn.to_out[1](hidden_states)

                if getattr(attn, "to_add_out", None) is not None:
                    encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

            return hidden_states, encoder_hidden_states

else:
    xFuserHunyuanVideoAttnProcessor2_0 = None


@xFuserAttentionProcessorRegister.register(SanaAttnProcessor2_0)
class xFuserSanaAttnProcessor2_0(SanaAttnProcessor2_0):
    def __init__(self):
        super().__init__()

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            sequence_length = sequence_length * get_sequence_parallel_world_size() \
                if get_runtime_state().split_text_embed_in_sp else sequence_length

            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads


        query = query.view(batch_size, -1, attn.heads, head_dim)
        key = key.view(batch_size, -1, attn.heads, head_dim)
        value = value.view(batch_size, -1, attn.heads, head_dim)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1 (diffuser todo)
        # hidden_states = F.scaled_dot_product_attention(
        #     query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        # )
        if get_runtime_state().split_text_embed_in_sp:
            raise NotImplementedError(
                "Currently SANA not support split_text_embed_in_sp!"
            )
        else:
            query = query.transpose(1, 2)
            key = key.transpose(1, 2)
            value = value.transpose(1, 2)
            hidden_states = F.scaled_dot_product_attention(
                query, key, value, attn_mask=attention_mask, dropout_p=0., is_causal=False
            )
            hidden_states = hidden_states.transpose(1, 2)

        hidden_states = hidden_states.reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states

@xFuserAttentionProcessorRegister.register(SanaLinearAttnProcessor2_0)
class xFuserSanaLinearAttnProcessor2_0(SanaLinearAttnProcessor2_0):
    def __init__(self):
        super().__init__()
        use_long_ctx_attn_kvcache = True
        self.use_long_ctx_attn_kvcache = (
            HAS_LONG_CTX_ATTN
            and use_long_ctx_attn_kvcache
            and get_sequence_parallel_world_size() > 1
        )
        if HAS_LONG_CTX_ATTN and get_sequence_parallel_world_size() > 1:
            from xfuser.core.long_ctx_attention import (
                xFuserSanaLinearLongContextAttention
            )

            if HAS_FLASH_ATTN:
                self.hybrid_seq_parallel_attn = xFuserSanaLinearLongContextAttention(
                    use_kv_cache=self.use_long_ctx_attn_kvcache,
                    attn_type=AttnType.FA,
                )
            else:
                self.hybrid_seq_parallel_attn = xFuserSanaLinearLongContextAttention(
                    use_kv_cache=self.use_long_ctx_attn_kvcache,
                    attn_type=AttnType.TORCH,
                )

        if get_fast_attn_enable():
            self.fast_attn = xFuserFastAttention()

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        original_dtype = hidden_states.dtype

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        batch_size = encoder_hidden_states.size(0)

        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        #! ---------------------------------------- KV CACHE ----------------------------------------
        if not self.use_long_ctx_attn_kvcache:
            key, value = get_cache_manager().update_and_get_kv_cache(
                new_kv=[key, value],
                layer=attn,
                slice_dim=1,
                layer_type="attn",
            )
        #! ---------------------------------------- KV CACHE ----------------------------------------

        query = query.view(batch_size, -1, attn.heads, head_dim)
        key = key.view(batch_size, -1, attn.heads, head_dim)
        value = value.view(batch_size, -1, attn.heads, head_dim)

        if HAS_LONG_CTX_ATTN and get_sequence_parallel_world_size() > 1:
            hidden_states = self.hybrid_seq_parallel_attn(
                    attn,
                    query,
                    key,
                    value,
                    attn_mask=attention_mask,
                    dropout_p=0.0,
                )

        else:
            query = query.transpose(1, 2)
            key = key.transpose(1, 2)
            value = value.transpose(1, 2)
            query = F.relu(query)
            key = F.relu(key)

            query, key, value = query.float(), key.float(), value.float()

            value = F.pad(value, (0, 1, 0, 0), mode="constant", value=1.0)
            scores = key.transpose(-2, -1) @ value
            hidden_states = query @ scores

            hidden_states = hidden_states[..., :-1] / (hidden_states[..., -1:] + 1e-15)
            hidden_states = hidden_states.transpose(1, 2)

        hidden_states = hidden_states.reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(original_dtype)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if original_dtype is not None and original_dtype == torch.float16:
            hidden_states = hidden_states.clip(-65504, 65504)


        return hidden_states
