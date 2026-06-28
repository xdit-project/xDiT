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
    get_cfg_group,
    get_classifier_free_guidance_rank,
    get_classifier_free_guidance_world_size,
    get_ring_parallel_world_size,
    get_sequence_parallel_rank,
    get_sequence_parallel_world_size,
    get_sp_group,
)
from xfuser.logger import init_logger
from xfuser.model_executor.layers.usp import USP

logger = init_logger(__name__)


class xFuserKrea2AttnProcessor:
    """SP-aware attention processor for Krea2Attention.

    Krea-2 uses GQA (48 query heads, 12 KV heads). KV heads are expanded to
    match Q heads before the USP call. Valid Ulysses degrees are divisors of
    12: 1, 2, 3, 4, 6, 12.
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

        # combined QKV all-to-all is faster with ring attention
        # because ring uses Python-mutable rotater that adds graph
        # breaks and prevents q,k,v A2A overlap with computation
        combine_qkv_a2a = get_ring_parallel_world_size() > 1
        out = USP(query, key, value, combine_qkv_a2a=combine_qkv_a2a)

        out = out.transpose(1, 2).flatten(2, 3).to(query.dtype)
        out = out * torch.sigmoid(gate)
        out = attn.to_out[0](out)
        return out


class xFuserKrea2Transformer2DWrapper(Krea2Transformer2DModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for block in self.transformer_blocks:
            block.attn.processor = xFuserKrea2AttnProcessor()

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
        cfg_world_size = get_classifier_free_guidance_world_size()
        cfg_rank = get_classifier_free_guidance_rank()

        if cfg_world_size > 1:
            hidden_states = hidden_states.chunk(cfg_world_size, dim=0)[cfg_rank]
            encoder_hidden_states = encoder_hidden_states.chunk(cfg_world_size, dim=0)[
                cfg_rank
            ]
            if encoder_attention_mask is not None:
                encoder_attention_mask = encoder_attention_mask.chunk(
                    cfg_world_size, dim=0
                )[cfg_rank]
            timestep = timestep.chunk(cfg_world_size, dim=0)[cfg_rank]

        temb_embed = self.time_embed(timestep, hidden_states.dtype)
        temb_mod = self.time_mod_proj(F.gelu(temb_embed, approximate="tanh"))

        text_projected = self.txt_in(
            self.text_fusion(encoder_hidden_states, encoder_attention_mask)
        )
        image_projected = self.img_in(hidden_states)

        full_seq = torch.cat([text_projected, image_projected], dim=1)
        total_seq = full_seq.shape[1]

        pad_len = (sp_world_size - total_seq % sp_world_size) % sp_world_size
        if pad_len > 0:
            pad = torch.zeros(
                full_seq.shape[0],
                pad_len,
                full_seq.shape[2],
                dtype=full_seq.dtype,
                device=full_seq.device,
            )
            full_seq = torch.cat([full_seq, pad], dim=1)

        local_seq = full_seq.chunk(sp_world_size, dim=1)[sp_rank]

        if pad_len > 0:
            pad_pos = torch.zeros(
                pad_len, 3, dtype=position_ids.dtype, device=position_ids.device
            )
            position_ids_padded = torch.cat([position_ids, pad_pos], dim=0)
        else:
            position_ids_padded = position_ids
        pos_ids_local = position_ids_padded.chunk(sp_world_size, dim=0)[sp_rank]

        image_rotary_emb = self.rotary_emb(pos_ids_local)

        for block in self.transformer_blocks:
            local_seq = block(
                hidden_states=local_seq,
                temb=temb_mod,
                image_rotary_emb=image_rotary_emb,
                attention_mask=None,
            )

        full_out = get_sp_group().all_gather(local_seq, dim=1)
        if pad_len > 0:
            full_out = full_out[:, :total_seq, :]

        txt_seq = text_projected.shape[1]
        image_out = self.final_layer(full_out[:, txt_seq:, :], temb_embed)

        image_out = get_cfg_group().all_gather(image_out, dim=0)

        if return_dict:
            return Transformer2DModelOutput(sample=image_out)
        return (image_out,)
