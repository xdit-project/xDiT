import torch
import math
from typing import Optional, Union, Dict, Any, Tuple

from diffusers.models.transformers.transformer_wan import WanAttnProcessor
from diffusers.models.transformers.transformer_wan import WanAttention
from diffusers.models.transformers.transformer_wan import WanTransformer3DModel
from diffusers.models.modeling_outputs import Transformer2DModelOutput

from xfuser.model_executor.layers.usp import USP
from xfuser.core.distributed import (
    get_sequence_parallel_world_size,
    get_sequence_parallel_rank,
    get_sp_group,
    get_runtime_state,
)
from xfuser.model_executor.layers.attention_processor import (
    xFuserAttentionProcessorRegister
)
from xfuser.envs import PACKAGES_CHECKER

env_info = PACKAGES_CHECKER.get_packages_info()
HAS_LONG_CTX_ATTN = env_info["has_long_ctx_attn"]

@xFuserAttentionProcessorRegister.register(WanAttnProcessor)
class xFuserWanAttnProcessor(WanAttnProcessor):

    def __init__(self):
        super().__init__()

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


            hidden_states_img = USP(query.transpose(1, 2), key_img.transpose(1, 2), value_img.transpose(1, 2)).transpose(1, 2)
            hidden_states_img = hidden_states_img.flatten(2, 3)
            hidden_states_img = hidden_states_img.type_as(query)

        hidden_states = USP(query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2)).transpose(1, 2)
        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.type_as(query)

        if hidden_states_img is not None:
            hidden_states = hidden_states + hidden_states_img

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states


class xFuserWanTransformer3DWrapper(WanTransformer3DModel):


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
        for block in self.blocks:
            block.attn1.processor = xFuserWanAttnProcessor()
            block.attn2.processor = xFuserWanAttnProcessor()


    def _chunk_and_pad_sequence(self, x: torch.Tensor, sp_world_rank: int, sp_world_size: int, pad_amount: int, dim: int) -> torch.Tensor:
        if pad_amount > 0:
            if dim < 0:
                dim = x.ndim + dim
            pad_shape = list(x.shape)
            pad_shape[dim] = pad_amount
            x = torch.cat([x,
                        torch.zeros(
                            pad_shape,
                            dtype=x.dtype,
                            device=x.device,
                        )], dim=dim)
        x = torch.chunk(x,
                        sp_world_size,
                        dim=dim)[sp_world_rank]
        return x

    def _gather_and_unpad(self, x: torch.Tensor, pad_amount: int, dim: int) -> torch.Tensor:
        x = get_sp_group().all_gather(x, dim=dim)
        size = x.size(dim)
        return x.narrow(dim=dim, start=0, length=size - pad_amount)


    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_image: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:

        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        get_runtime_state().increment_step_counter()

        sp_world_rank = get_sequence_parallel_rank()
        sp_world_size = get_sequence_parallel_world_size()

        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.config.patch_size
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p_h
        post_patch_width = width // p_w

        # 1. RoPE
        rotary_emb = self.rope(hidden_states)

        hidden_states = self.patch_embedding(hidden_states)
        hidden_states = hidden_states.flatten(2).transpose(1, 2)

        # timestep shape: batch_size, or batch_size, seq_len (wan 2.2 ti2v)
        if timestep.ndim == 2:
            ts_seq_len = timestep.shape[1]
            timestep = timestep.flatten()  # batch_size * seq_len
        else:
            ts_seq_len = None

        temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = self.condition_embedder(
            timestep, encoder_hidden_states, encoder_hidden_states_image, timestep_seq_len=ts_seq_len
        )
        if ts_seq_len is not None:
            # batch_size, seq_len, 6, inner_dim
            timestep_proj = timestep_proj.unflatten(2, (6, -1))
        else:
            # batch_size, 6, inner_dim
            timestep_proj = timestep_proj.unflatten(1, (6, -1))

        if encoder_hidden_states_image is not None:
            # We only reach this for Wan2.1, when doing cross attention with image embeddings
            encoder_hidden_states = torch.concat([encoder_hidden_states_image, encoder_hidden_states], dim=1)
        else:
            # Wan2.1 fails if we chunk encoder_hidden_states when cross attention is used. Should cross attention really be sharded?
            encoder_hidden_states = torch.chunk(encoder_hidden_states, get_sequence_parallel_world_size(), dim=-2)[get_sequence_parallel_rank()]

        # Part of sequence parallel: given the resolution, we may need to pad the sequence length to match this prior to chunking
        pad_amount = (sp_world_size - (hidden_states.shape[1] % sp_world_size)) % sp_world_size
        # hidden_states = torch.cat([
        #     hidden_states,
        #     torch.zeros(batch_size, sequence_pad_amount, hidden_states.shape[2], device=hidden_states.device, dtype=hidden_states.dtype)
        # ], dim=1)
        # hidden_states = torch.chunk(hidden_states, get_sequence_parallel_world_size(), dim=-2)[get_sequence_parallel_rank()]
        hidden_states = self._chunk_and_pad_sequence(hidden_states, sp_world_rank, sp_world_size, pad_amount, dim=1)

        if ts_seq_len is not None: # (wan2.2 ti2v)
            # temb = torch.cat([
            #     temb,
            #     torch.zeros(batch_size, sequence_pad_amount, temb.shape[2], device=temb.device, dtype=temb.dtype)
            # ], dim=1)
            # timestep_proj = torch.cat([
            #     timestep_proj,
            #     torch.zeros(batch_size, sequence_pad_amount, timestep_proj.shape[2], timestep_proj.shape[3], device=timestep_proj.device, dtype=timestep_proj.dtype)
            # ], dim=1)
            # temb = torch.chunk(temb, get_sequence_parallel_world_size(), dim=1)[get_sequence_parallel_rank()]
            # timestep_proj = torch.chunk(timestep_proj, get_sequence_parallel_world_size(), dim=1)[get_sequence_parallel_rank()]
            temb = self._chunk_and_pad_sequence(temb, sp_world_rank, sp_world_size, pad_amount, dim=1)
            timestep_proj = self._chunk_and_pad_sequence(timestep_proj, sp_world_rank, sp_world_size, pad_amount, dim=1)

        freqs_cos, freqs_sin = rotary_emb

        def get_rotary_emb_chunk(freqs, pad_amount):
            # freqs = torch.cat([
            #     freqs,
            #     torch.zeros(1, sequence_pad_amount, freqs.shape[2], freqs.shape[3], device=freqs.device, dtype=freqs.dtype)
            # ], dim=1)
            # freqs = torch.chunk(freqs, get_sequence_parallel_world_size(), dim=1)[get_sequence_parallel_rank()]
            freqs = self._chunk_and_pad_sequence(freqs, sp_world_rank, sp_world_size, pad_amount, dim=1)
            return freqs

        freqs_cos = get_rotary_emb_chunk(freqs_cos, pad_amount)
        freqs_sin = get_rotary_emb_chunk(freqs_sin, pad_amount)
        rotary_emb = (freqs_cos, freqs_sin)


        # 4. Transformer blocks
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            for block in self.blocks:
                hidden_states = self._gradient_checkpointing_func(
                    block, hidden_states, encoder_hidden_states, timestep_proj, rotary_emb
                )
        else:
            for block in self.blocks:
                hidden_states = block(hidden_states, encoder_hidden_states, timestep_proj, rotary_emb)

        # 5. Output norm, projection & unpatchify
        if temb.ndim == 3:
            # batch_size, seq_len, inner_dim (wan 2.2 ti2v)
            shift, scale = (self.scale_shift_table.unsqueeze(0).to(temb.device) + temb.unsqueeze(2)).chunk(2, dim=2)
            shift = shift.squeeze(2)
            scale = scale.squeeze(2)
        else:
            # batch_size, inner_dim
            shift, scale = (self.scale_shift_table.to(temb.device) + temb.unsqueeze(1)).chunk(2, dim=1)

        # Move the shift and scale tensors to the same device as hidden_states.
        # When using multi-GPU inference via accelerate these will be on the
        # first device rather than the last device, which hidden_states ends up
        # on.
        shift = shift.to(hidden_states.device)
        scale = scale.to(hidden_states.device)

        hidden_states = (self.norm_out(hidden_states.float()) * (1 + scale) + shift).type_as(hidden_states)
        hidden_states = self.proj_out(hidden_states)

        #hidden_states = get_sp_group().all_gather(hidden_states, dim=-2)
        hidden_states = self._gather_and_unpad(hidden_states, pad_amount, dim=-2)

        # # Removing excess padding to get back to original sequence length
        # hidden_states = hidden_states[:, :math.prod([post_patch_num_frames, post_patch_height, post_patch_width]), :]

        hidden_states = hidden_states.reshape(
            batch_size, post_patch_num_frames, post_patch_height, post_patch_width, p_t, p_h, p_w, -1
        )
        hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
        output = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)