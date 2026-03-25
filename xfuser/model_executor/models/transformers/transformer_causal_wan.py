import torch
import torch.nn.functional as F
from typing import Optional, Union, Dict, Any, Tuple

from diffusers.models.transformers.transformer_wan import (
    WanAttnProcessor,
    WanAttention,
    WanTransformerBlock,
    WanTransformer3DModel,
)
from diffusers.models.modeling_outputs import Transformer2DModelOutput

from xfuser.model_executor.layers.usp import attention
from xfuser.core.distributed import get_runtime_state


class xFuserCausalWanAttnProcessor(WanAttnProcessor):
    """Attention processor with KV cache support for causal (autoregressive) Wan inference.

    NOT registered via @xFuserAttentionProcessorRegister.register() to avoid
    conflict with the existing xFuserWanAttnProcessor that is registered
    against WanAttnProcessor. Instead, manually assigned in the transformer constructor.
    """

    def __init__(self, is_cross_attention: bool = False) -> None:
        super().__init__()
        self.is_cross_attention = is_cross_attention

    def _get_qkv_projections(self, attn: "WanAttention", hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor):
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        if attn.fused_projections:
            if attn.cross_attention_dim_head is None:
                query, key, value = attn.to_qkv(hidden_states).chunk(3, dim=-1)
            else:
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

    @staticmethod
    def _apply_rotary_emb(
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

    def __call__(
        self,
        attn: "WanAttention",
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_emb: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        kv_cache: Optional[dict] = None,
        crossattn_cache: Optional[dict] = None,
        current_start: int = 0,
        local_attn_size: int = -1,
        sink_size: int = 0,
        max_attention_size: int = 32760,
        **kwargs,
    ) -> torch.Tensor:

        if not self.is_cross_attention:
            return self._self_attention(
                attn, hidden_states, encoder_hidden_states, attention_mask,
                rotary_emb, kv_cache, current_start, local_attn_size,
                sink_size, max_attention_size,
            )
        else:
            return self._cross_attention(
                attn, hidden_states, encoder_hidden_states, attention_mask,
                rotary_emb, crossattn_cache,
            )

    def _self_attention(
        self,
        attn: "WanAttention",
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        rotary_emb: Optional[tuple[torch.Tensor, torch.Tensor]],
        kv_cache: Optional[dict],
        current_start: int,
        local_attn_size: int,
        sink_size: int,
        max_attention_size: int,
    ) -> torch.Tensor:
        query, key, value = self._get_qkv_projections(attn, hidden_states, None)

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        # [B, seq_len, heads, head_dim]
        query = query.unflatten(2, (attn.heads, -1))
        key = key.unflatten(2, (attn.heads, -1))
        value = value.unflatten(2, (attn.heads, -1))

        if rotary_emb is not None:
            query = self._apply_rotary_emb(query, *rotary_emb)
            key = self._apply_rotary_emb(key, *rotary_emb)

        hidden_states = self._cached_self_attention(
            query, key, value, kv_cache, current_start,
            local_attn_size, sink_size, max_attention_size,
        )

        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.type_as(query)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states

    def _cached_self_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: dict,
        current_start: int,
        local_attn_size: int,
        sink_size: int,
        max_attention_size: int,
    ) -> torch.Tensor:
        """Self-attention with KV cache, following FastVideo's CausalWanSelfAttention logic."""
        frame_seqlen = query.shape[1]
        num_new_tokens = query.shape[1]
        current_end = current_start + num_new_tokens
        sink_tokens = sink_size * frame_seqlen

        kv_cache_size = kv_cache["k"].shape[1]

        global_end_index = (
            int(kv_cache["global_end_index"].item())
            if isinstance(kv_cache["global_end_index"], torch.Tensor)
            else int(kv_cache["global_end_index"])
        )
        local_end_index_prev = (
            int(kv_cache["local_end_index"].item())
            if isinstance(kv_cache["local_end_index"], torch.Tensor)
            else int(kv_cache["local_end_index"])
        )

        if local_attn_size != -1 and (current_end > global_end_index) and (
                num_new_tokens + local_end_index_prev > kv_cache_size):
            # Sliding window eviction: shift left, preserve sink tokens
            num_evicted_tokens = num_new_tokens + local_end_index_prev - kv_cache_size
            num_rolled_tokens = local_end_index_prev - num_evicted_tokens - sink_tokens
            kv_cache["k"][:, sink_tokens:sink_tokens + num_rolled_tokens] = \
                kv_cache["k"][:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
            kv_cache["v"][:, sink_tokens:sink_tokens + num_rolled_tokens] = \
                kv_cache["v"][:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
            local_end_index = local_end_index_prev + current_end - global_end_index - num_evicted_tokens
            local_start_index = local_end_index - num_new_tokens
            kv_cache["k"][:, local_start_index:local_end_index] = key
            kv_cache["v"][:, local_start_index:local_end_index] = value
        else:
            local_end_index = local_end_index_prev + current_end - global_end_index
            local_start_index = local_end_index - num_new_tokens
            kv_cache["k"] = kv_cache["k"].detach()
            kv_cache["v"] = kv_cache["v"].detach()
            kv_cache["k"][:, local_start_index:local_end_index] = key
            kv_cache["v"][:, local_start_index:local_end_index] = value

        # Attention over cached keys/values
        # query: [B, seq, heads, head_dim] -> [B, heads, seq, head_dim]
        attn_k = kv_cache["k"][:, max(0, local_end_index - max_attention_size):local_end_index]
        attn_v = kv_cache["v"][:, max(0, local_end_index - max_attention_size):local_end_index]

        out = attention(query.transpose(1, 2), attn_k.transpose(1, 2), attn_v.transpose(1, 2)).transpose(1, 2)

        # Update cache indices
        if isinstance(kv_cache["global_end_index"], torch.Tensor):
            kv_cache["global_end_index"].fill_(current_end)
        else:
            kv_cache["global_end_index"] = current_end
        if isinstance(kv_cache["local_end_index"], torch.Tensor):
            kv_cache["local_end_index"].fill_(local_end_index)
        else:
            kv_cache["local_end_index"] = local_end_index

        return out

    def _cross_attention(
        self,
        attn: "WanAttention",
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        rotary_emb: Optional[tuple[torch.Tensor, torch.Tensor]],
        crossattn_cache: Optional[dict],
    ) -> torch.Tensor:
        backend = get_runtime_state().get_cross_attention_backend()

        encoder_hidden_states_img = None
        if attn.add_k_proj is not None:
            image_context_length = encoder_hidden_states.shape[1] - 512
            encoder_hidden_states_img = encoder_hidden_states[:, :image_context_length]
            encoder_hidden_states = encoder_hidden_states[:, image_context_length:]

        # Cached cross-attention path
        query = attn.to_q(hidden_states)
        query = attn.norm_q(query)
        query = query.unflatten(2, (attn.heads, -1))

        if not crossattn_cache["is_init"]:
            # First block: compute and cache K, V
            if attn.fused_projections and attn.cross_attention_dim_head is not None:
                key, value = attn.to_kv(encoder_hidden_states).chunk(2, dim=-1)
            else:
                key = attn.to_k(encoder_hidden_states)
                value = attn.to_v(encoder_hidden_states)
            key = attn.norm_k(key)
            key = key.unflatten(2, (attn.heads, -1))
            value = value.unflatten(2, (attn.heads, -1))

            crossattn_cache["k"][:, :key.shape[1]] = key
            crossattn_cache["v"][:, :value.shape[1]] = value
            crossattn_cache["is_init"] = True
            crossattn_cache["seq_len"] = key.shape[1]
        else:
            seq_len = crossattn_cache["seq_len"]
            key = crossattn_cache["k"][:, :seq_len]
            value = crossattn_cache["v"][:, :seq_len]

        # I2V image embeddings
        hidden_states_img = None
        if encoder_hidden_states_img is not None:
            key_img, value_img = self._get_added_kv_projections(attn, encoder_hidden_states_img)
            key_img = attn.norm_added_k(key_img)
            key_img = key_img.unflatten(2, (attn.heads, -1))
            value_img = value_img.unflatten(2, (attn.heads, -1))

            hidden_states_img = attention(query.transpose(1, 2), key_img.transpose(1, 2), value_img.transpose(1, 2), backend=backend).transpose(1, 2)
            hidden_states_img = hidden_states_img.flatten(2, 3).type_as(query)

        hidden_states = attention(query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2), backend=backend).transpose(1, 2)
        hidden_states = hidden_states.flatten(2, 3).type_as(query)

        if hidden_states_img is not None:
            hidden_states = hidden_states + hidden_states_img

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states


class xFuserCausalWanTransformerBlock(WanTransformerBlock):
    """WanTransformerBlock subclass that accepts and passes through
    causal-specific kwargs (kv_cache, crossattn_cache, etc.) to the
    attention layers."""

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        rotary_emb: torch.Tensor,
        kv_cache: Optional[dict] = None,
        crossattn_cache: Optional[dict] = None,
        current_start: int = 0,
        local_attn_size: int = -1,
        sink_size: int = 0,
        max_attention_size: int = 32760,
    ) -> torch.Tensor:
        if temb.ndim == 4:
            shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
                self.scale_shift_table.unsqueeze(0) + temb.float()
            ).chunk(6, dim=2)
            shift_msa = shift_msa.squeeze(2)
            scale_msa = scale_msa.squeeze(2)
            gate_msa = gate_msa.squeeze(2)
            c_shift_msa = c_shift_msa.squeeze(2)
            c_scale_msa = c_scale_msa.squeeze(2)
            c_gate_msa = c_gate_msa.squeeze(2)
        else:
            shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
                self.scale_shift_table + temb.float()
            ).chunk(6, dim=1)

        # 1. Self-attention (with causal kwargs)
        norm_hidden_states = (self.norm1(hidden_states.float()) * (1 + scale_msa) + shift_msa).type_as(hidden_states)
        attn_output = self.attn1(
            norm_hidden_states, None, None, rotary_emb,
            kv_cache=kv_cache,
            current_start=current_start,
            local_attn_size=local_attn_size,
            sink_size=sink_size,
            max_attention_size=max_attention_size,
        )
        hidden_states = (hidden_states.float() + attn_output * gate_msa).type_as(hidden_states)

        # 2. Cross-attention (with cache kwargs)
        norm_hidden_states = self.norm2(hidden_states.float()).type_as(hidden_states)
        attn_output = self.attn2(
            norm_hidden_states, encoder_hidden_states, None, None,
            crossattn_cache=crossattn_cache,
        )
        hidden_states = hidden_states + attn_output

        # 3. Feed-forward (unchanged)
        norm_hidden_states = (self.norm3(hidden_states.float()) * (1 + c_scale_msa) + c_shift_msa).type_as(
            hidden_states
        )
        ff_output = self.ffn(norm_hidden_states)
        hidden_states = (hidden_states.float() + ff_output.float() * c_gate_msa).type_as(hidden_states)

        return hidden_states


class xFuserCausalWanTransformer3DWrapper(WanTransformer3DModel):
    """Wan transformer wrapper for causal (autoregressive) inference.

    Uses the same weights as standard Wan models but replaces blocks with
    xFuserCausalWanTransformerBlock and assigns causal attention processors.
    """

    def __init__(
        self,
        patch_size: Tuple[int, ...] = (1, 2, 2),
        num_attention_heads: int = 40,
        attention_head_dim: int = 128,
        in_channels: int = 16,
        out_channels: int = 16,
        text_dim: int = 4096,
        freq_dim: int = 256,
        ffn_dim: int = 13824,
        num_layers: int = 40,
        cross_attn_norm: bool = True,
        qk_norm: Optional[str] = "rms_norm_across_heads",
        eps: float = 1e-6,
        image_dim: Optional[int] = None,
        added_kv_proj_dim: Optional[int] = None,
        rope_max_seq_len: int = 1024,
        pos_embed_seq_len: Optional[int] = None,
    ) -> None:
        super().__init__(
            patch_size,
            num_attention_heads,
            attention_head_dim,
            in_channels,
            out_channels,
            text_dim,
            freq_dim,
            ffn_dim,
            num_layers,
            cross_attn_norm,
            qk_norm,
            eps,
            image_dim,
            added_kv_proj_dim,
            rope_max_seq_len,
            pos_embed_seq_len,
        )

        # Replace each standard WanTransformerBlock with our causal version
        # that passes extra kwargs through to attention
        for i, block in enumerate(self.blocks):
            causal_block = xFuserCausalWanTransformerBlock(
                dim=num_attention_heads * attention_head_dim,
                ffn_dim=ffn_dim,
                num_heads=num_attention_heads,
                qk_norm=qk_norm,
                cross_attn_norm=cross_attn_norm,
                eps=eps,
                added_kv_proj_dim=added_kv_proj_dim,
            )
            # Copy weight references from the original block
            causal_block.load_state_dict(block.state_dict(), strict=True)
            # Assign causal attention processors
            causal_block.attn1.processor = xFuserCausalWanAttnProcessor(is_cross_attention=False)
            causal_block.attn2.processor = xFuserCausalWanAttnProcessor(is_cross_attention=True)
            self.blocks[i] = causal_block


    def _compute_rope_with_offset(
        self,
        hidden_states: torch.Tensor,
        start_frame: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute RoPE embeddings with a temporal frame offset for causal generation."""
        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.config.patch_size
        ppf = num_frames // p_t
        pph = height // p_h
        ppw = width // p_w

        rope = self.rope
        split_sizes = [rope.t_dim, rope.h_dim, rope.w_dim]

        freqs_cos = rope.freqs_cos.split(split_sizes, dim=1)
        freqs_sin = rope.freqs_sin.split(split_sizes, dim=1)

        # Apply start_frame offset to the temporal component
        freqs_cos_f = freqs_cos[0][start_frame:start_frame + ppf].view(ppf, 1, 1, -1).expand(ppf, pph, ppw, -1)
        freqs_cos_h = freqs_cos[1][:pph].view(1, pph, 1, -1).expand(ppf, pph, ppw, -1)
        freqs_cos_w = freqs_cos[2][:ppw].view(1, 1, ppw, -1).expand(ppf, pph, ppw, -1)

        freqs_sin_f = freqs_sin[0][start_frame:start_frame + ppf].view(ppf, 1, 1, -1).expand(ppf, pph, ppw, -1)
        freqs_sin_h = freqs_sin[1][:pph].view(1, pph, 1, -1).expand(ppf, pph, ppw, -1)
        freqs_sin_w = freqs_sin[2][:ppw].view(1, 1, ppw, -1).expand(ppf, pph, ppw, -1)

        freqs_cos_out = torch.cat([freqs_cos_f, freqs_cos_h, freqs_cos_w], dim=-1).reshape(1, ppf * pph * ppw, 1, -1)
        freqs_sin_out = torch.cat([freqs_sin_f, freqs_sin_h, freqs_sin_w], dim=-1).reshape(1, ppf * pph * ppw, 1, -1)

        return freqs_cos_out, freqs_sin_out

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_image: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:

        # Extract causal params from attention_kwargs
        attention_kwargs = attention_kwargs.copy()
        kv_cache = attention_kwargs["kv_cache"]
        crossattn_cache = attention_kwargs["crossattn_cache"]
        current_start = attention_kwargs["current_start"]
        start_frame = attention_kwargs["start_frame"]
        local_attn_size = attention_kwargs["local_attn_size"]
        sink_size = attention_kwargs["sink_size"]
        max_attention_size = attention_kwargs["max_attention_size"]

        get_runtime_state().increment_step_counter()

        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.config.patch_size
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p_h
        post_patch_width = width // p_w

        # 1. RoPE with frame offset for causal generation
        rotary_emb = self._compute_rope_with_offset(hidden_states, start_frame)

        hidden_states = self.patch_embedding(hidden_states)
        hidden_states = hidden_states.flatten(2).transpose(1, 2)

        # timestep shape: batch_size, or batch_size, seq_len (wan 2.2 ti2v)
        if timestep.ndim == 2:
            ts_seq_len = timestep.shape[1]
            timestep = timestep.flatten()
        else:
            ts_seq_len = None

        temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = self.condition_embedder(
            timestep, encoder_hidden_states, encoder_hidden_states_image, timestep_seq_len=ts_seq_len
        )

        if ts_seq_len is not None:
            timestep_proj = timestep_proj.unflatten(2, (6, -1))
        else:
            timestep_proj = timestep_proj.unflatten(1, (6, -1))

        if encoder_hidden_states_image is not None:
            encoder_hidden_states = torch.concat([encoder_hidden_states_image, encoder_hidden_states], dim=1)

        # 4. Transformer blocks
        for i, block in enumerate(self.blocks):
            block_kv_cache = kv_cache[i]
            block_crossattn_cache = crossattn_cache[i]

            hidden_states = block(
                hidden_states,
                encoder_hidden_states,
                timestep_proj,
                rotary_emb,
                kv_cache=block_kv_cache,
                crossattn_cache=block_crossattn_cache,
                current_start=current_start,
                local_attn_size=local_attn_size,
                sink_size=sink_size,
                max_attention_size=max_attention_size,
            )

        # 5. Output norm, projection & unpatchify
        if temb.ndim == 3:
            shift, scale = (self.scale_shift_table.unsqueeze(0).to(temb.device) + temb.unsqueeze(2)).chunk(2, dim=2)
            shift = shift.squeeze(2)
            scale = scale.squeeze(2)
        else:
            shift, scale = (self.scale_shift_table.to(temb.device) + temb.unsqueeze(1)).chunk(2, dim=1)

        shift = shift.to(hidden_states.device)
        scale = scale.to(hidden_states.device)

        hidden_states = (self.norm_out(hidden_states.float()) * (1 + scale) + shift).type_as(hidden_states)
        hidden_states = self.proj_out(hidden_states)

        hidden_states = hidden_states.reshape(
            batch_size, post_patch_num_frames, post_patch_height, post_patch_width, p_t, p_h, p_w, -1
        )
        hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
        output = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)
