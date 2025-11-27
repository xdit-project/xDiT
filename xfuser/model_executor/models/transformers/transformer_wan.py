import torch
from typing import Optional

from diffusers.models.transformers.transformer_wan import WanAttnProcessor
from diffusers.models.transformers.transformer_wan import WanAttention

from xfuser.model_executor.layers.usp import USP
from xfuser.core.distributed import get_sequence_parallel_world_size
from xfuser.model_executor.layers.attention_processor import (
    set_hybrid_seq_parallel_attn,
    xFuserAttentionProcessorRegister
)
from xfuser.envs import PACKAGES_CHECKER

env_info = PACKAGES_CHECKER.get_packages_info()
HAS_LONG_CTX_ATTN = env_info["has_long_ctx_attn"]

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
        self.use_fp8_attn = False

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

    def _set_fp8_attn(self, use_fp8_attn: bool):
        self.use_fp8_attn = use_fp8_attn

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


            hidden_states_img = USP(query.transpose(1, 2), key_img.transpose(1, 2), value_img.transpose(1, 2), use_fp8_attn=self.use_fp8_attn).transpose(1, 2)
            hidden_states_img = hidden_states_img.flatten(2, 3)
            hidden_states_img = hidden_states_img.type_as(query)

        hidden_states = USP(query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2), use_fp8_attn=self.use_fp8_attn).transpose(1, 2)
        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.type_as(query)

        if hidden_states_img is not None:
            hidden_states = hidden_states + hidden_states_img

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states
