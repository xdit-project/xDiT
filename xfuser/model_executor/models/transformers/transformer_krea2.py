import torch
import torch.nn.functional as F
from typing import Any, Dict, Optional, Tuple, Union

from diffusers.models.transformers.transformer_krea2 import (
    Krea2Attention,
    Krea2Transformer2DModel,
)
from diffusers.models.transformers.transformer_2d import Transformer2DModelOutput
from diffusers.models.embeddings import apply_rotary_emb

from xfuser.core.distributed import (
    get_sequence_parallel_rank,
    get_sequence_parallel_world_size,
)
from xfuser.model_executor.layers.attention_mask import (
    AttentionMaskWithMeta,
    make_attn_mask_with_meta,
)
from xfuser.model_executor.layers.usp import USP
from xfuser.model_executor.models.transformers.transformers_utils import (
    chunk_and_pad_sequence,
    gather_and_unpad,
)


class xFuserKrea2AttnProcessor:
    """SP-aware attention processor for Krea2Attention.

    Krea-2 uses GQA (48 query heads, 12 KV heads). KV heads are expanded to
    match Q heads before the USP call, so Ulysses degrees must divide
    `attn.num_heads` (typically 48).
    """

    def __call__(
        self,
        attn: Krea2Attention,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> torch.Tensor:
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)
        gate = attn.to_gate(hidden_states)

        query = query.unflatten(-1, (attn.num_heads, -1))
        key = key.unflatten(-1, (attn.num_kv_heads, -1))
        value = value.unflatten(-1, (attn.num_kv_heads, -1))

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
            key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)

        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        # Expand KV heads to match Q heads for GQA before passing to USP.
        if attn.num_heads != attn.num_kv_heads:
            repeat_factor = attn.num_heads // attn.num_kv_heads
            key = key.repeat_interleave(repeat_factor, dim=1)
            value = value.repeat_interleave(repeat_factor, dim=1)

        attn_kw = (
            {
                "attn_mask": attention_mask.attn_mask,
                "indices_k": attention_mask.indices_k,
                "cu_seqlens_k": attention_mask.cu_seqlens_k,
                "max_seqlen_k": attention_mask.max_seqlen_k,
            }
            if isinstance(attention_mask, AttentionMaskWithMeta)
            else None
        )
        out = USP(query, key, value, attention_kwargs=attn_kw)

        out = out.transpose(1, 2).flatten(2, 3).to(query.dtype)
        out = out * torch.sigmoid(gate)
        out = attn.to_out[0](out)
        return out


class xFuserKrea2Transformer2DWrapper(Krea2Transformer2DModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for block in self.transformer_blocks:
            block.attn.processor = xFuserKrea2AttnProcessor()
        self._attn_mask_cache: tuple | None = None  # (data_ptr, shape, AttentionMaskWithMeta)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        position_ids: torch.Tensor,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
    ) -> Union[torch.FloatTensor, Transformer2DModelOutput]:
        sp_world_size = get_sequence_parallel_world_size()
        sp_rank = get_sequence_parallel_rank()

        temb_embed = self.time_embed(timestep, hidden_states.dtype)
        temb_mod = self.time_mod_proj(F.gelu(temb_embed, approximate="tanh"))

        text_attn_mask = None
        if encoder_attention_mask is not None:
            text_attn_mask = encoder_attention_mask[:, None, None, :]

        text_projected = self.txt_in(
            self.text_fusion(encoder_hidden_states, text_attn_mask)
        )
        image_projected = self.img_in(hidden_states)

        full_seq = torch.cat([text_projected, image_projected], dim=1)
        total_seq = full_seq.shape[1]
        pad_len = (sp_world_size - total_seq % sp_world_size) % sp_world_size

        # Build a [B, S] key-padding mask covering text, image, and SP pad tokens
        # (1 = valid, 0 = excluded). nonzero/item graph breaks occur only on the
        # first step; subsequent steps hit the cache because encoder_attention_mask
        # is the same tensor object for the entire denoising loop.
        block_attn_mask = None
        if encoder_attention_mask is not None:
            ptr, shape = encoder_attention_mask.data_ptr(), encoder_attention_mask.shape
            if self._attn_mask_cache is not None and self._attn_mask_cache[:2] == (ptr, shape):
                block_attn_mask = self._attn_mask_cache[2]
            else:
                B = hidden_states.shape[0]
                combined = torch.cat(
                    [
                        encoder_attention_mask[:, : text_projected.shape[1]],
                        encoder_attention_mask.new_ones(B, image_projected.shape[1]),
                    ],
                    dim=1,
                )
                if pad_len > 0:
                    combined = torch.cat([combined, combined.new_zeros(B, pad_len)], dim=1)
                block_attn_mask = make_attn_mask_with_meta(combined)
                self._attn_mask_cache = (ptr, shape, block_attn_mask)

        local_seq = chunk_and_pad_sequence(
            full_seq, sp_rank, sp_world_size, pad_len, dim=1
        )
        pos_ids_local = chunk_and_pad_sequence(
            position_ids, sp_rank, sp_world_size, pad_len, dim=0
        )

        image_rotary_emb = self.rotary_emb(pos_ids_local)

        for block in self.transformer_blocks:
            local_seq = block(
                hidden_states=local_seq,
                temb=temb_mod,
                image_rotary_emb=image_rotary_emb,
                attention_mask=block_attn_mask,
            )

        full_out = gather_and_unpad(local_seq, pad_len, dim=1)

        txt_seq = text_projected.shape[1]
        image_out = self.final_layer(full_out[:, txt_seq:, :], temb_embed)

        if return_dict:
            return Transformer2DModelOutput(sample=image_out)
        return (image_out,)
