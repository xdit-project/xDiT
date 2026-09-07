from __future__ import annotations

from typing import Any

import torch

from diffusers.models.transformers.transformer_minimax_h3 import (
    MINIMAX_H3_MODALITY_NUM,
    MiniMaxH3AttnProcessor,
    MiniMaxH3Transformer3DModel,
    MiniMaxH3TransformerOutput,
    _apply_rotary_emb,
)
from diffusers.utils import apply_lora_scale

from xfuser.core.distributed import (
    get_runtime_state,
    get_sp_group,
    get_ulysses_parallel_rank,
    get_ulysses_parallel_world_size,
)
from xfuser.model_executor.layers.usp import USP, attention


MINIMAX_H3_PACKED_SEQUENCE_ALIGNMENT = 64


class xFuserMiniMaxH3AttnProcessor(MiniMaxH3AttnProcessor):
    def __init__(
        self,
        use_ulysses_parallel_attention: bool,
        attention_kwargs: dict[str, Any] | None = None,
        backend=None,
    ) -> None:
        super().__init__()
        self.use_ulysses_parallel_attention = use_ulysses_parallel_attention
        self.attention_kwargs = attention_kwargs
        self.backend = backend

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if attention_mask is not None:
            raise ValueError(
                "MiniMax-H3 xDiT attention expects padding to be represented by "
                "the varlen metadata prepared by the transformer wrapper."
            )

        if attn.fused_projections:
            query, key, value = attn.to_qkv(hidden_states).chunk(3, dim=-1)
        else:
            query = attn.to_q(hidden_states)
            key = attn.to_k(hidden_states)
            value = attn.to_v(hidden_states)

        query = query.unflatten(-1, (attn.heads, -1))
        key = key.unflatten(-1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        if rotary_emb is not None:
            query = _apply_rotary_emb(query, *rotary_emb)
            key = _apply_rotary_emb(key, *rotary_emb)

        use_ulysses = (
            self.use_ulysses_parallel_attention
            and get_ulysses_parallel_world_size() > 1
        )
        attention_function = USP if use_ulysses else attention
        attention_args = {
            "dropout_p": 0.0,
            "is_causal": False,
            "attention_kwargs": self.attention_kwargs,
            "head_balance_layer": attn,
            "backend": self.backend,
        }
        if use_ulysses:
            attention_args["combine_qkv_a2a"] = True
        hidden_states = attention_function(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
            **attention_args,
        ).transpose(1, 2)

        hidden_states = hidden_states.flatten(2, 3).type_as(query)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states


class xFuserMiniMaxH3Transformer3DWrapper(MiniMaxH3Transformer3DModel):
    def __init__(self, *args, attention_backend=None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._usp_attention_kwargs: dict[str, Any] = {}

        for block in self.token_refiner.refiner_blocks:
            block.attn.set_processor(
                xFuserMiniMaxH3AttnProcessor(
                    use_ulysses_parallel_attention=False,
                    backend=attention_backend,
                )
            )

        for block in self.transformer_blocks:
            block.attn.set_processor(
                xFuserMiniMaxH3AttnProcessor(
                    use_ulysses_parallel_attention=True,
                    attention_kwargs=self._usp_attention_kwargs,
                    backend=attention_backend,
                )
            )

        self.register_forward_pre_hook(
            lambda module, args: get_runtime_state().increment_step_counter()
        )

    @staticmethod
    def _pad_rows(
        hidden_states: torch.Tensor,
        timestep_indices: torch.Tensor,
        token_tags: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
        sequence_length = position_ids.shape[0]
        padded_length = (
            (sequence_length + MINIMAX_H3_PACKED_SEQUENCE_ALIGNMENT - 1)
            // MINIMAX_H3_PACKED_SEQUENCE_ALIGNMENT
            * MINIMAX_H3_PACKED_SEQUENCE_ALIGNMENT
        )
        pad_amount = padded_length - sequence_length
        if pad_amount == 0:
            return hidden_states, timestep_indices, token_tags, position_ids, 0

        hidden_states = torch.cat(
            (
                hidden_states,
                hidden_states.new_zeros(
                    hidden_states.shape[0],
                    pad_amount,
                    hidden_states.shape[-1],
                ),
            ),
            dim=1,
        )
        timestep_indices = torch.cat(
            (
                timestep_indices,
                timestep_indices.new_zeros(pad_amount),
            )
        )
        token_tags = torch.cat(
            (
                token_tags,
                token_tags.new_full((pad_amount,), -1),
            )
        )
        position_ids = torch.cat(
            (
                position_ids,
                position_ids.new_zeros(pad_amount, position_ids.shape[-1]),
            ),
            dim=0,
        )
        return hidden_states, timestep_indices, token_tags, position_ids, pad_amount

    @apply_lora_scale("attention_kwargs")
    def forward(
        self,
        hidden_states: torch.Tensor,
        audio_hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        timestep_indices: torch.Tensor,
        token_tags: torch.Tensor,
        position_ids: torch.Tensor,
        video_indices: torch.Tensor,
        audio_indices: torch.Tensor,
        text_indices: torch.Tensor,
        attention_kwargs: dict[str, Any] | None = None,
        return_dict: bool = True,
    ) -> MiniMaxH3TransformerOutput | tuple[torch.Tensor, torch.Tensor]:
        if position_ids.ndim != 2 or position_ids.shape[-1] != 3:
            raise ValueError(
                f"`position_ids` must be a `(seq_len, 3)` tensor, got {list(position_ids.shape)}."
            )
        sequence_length = position_ids.shape[0]
        if token_tags.shape != (sequence_length,) or timestep_indices.shape != (
            sequence_length,
        ):
            raise ValueError(
                "`token_tags` and `timestep_indices` must both be `(seq_len,)` "
                f"tensors matching `position_ids`, got {list(token_tags.shape)} "
                f"and {list(timestep_indices.shape)} for seq_len={sequence_length}."
            )

        video_embeds = self.proj_in(hidden_states.to(self.proj_in.weight.dtype))
        audio_embeds = self.audio_proj_in(
            audio_hidden_states.to(self.audio_proj_in.weight.dtype)
        )
        text_embeds = self.context_embedder(
            encoder_hidden_states.to(self.context_embedder.weight.dtype)
        )
        text_embeds = self.token_refiner(text_embeds)

        packed_hidden_states = text_embeds.new_zeros(
            (text_embeds.shape[0], sequence_length, text_embeds.shape[-1])
        )
        packed_hidden_states = packed_hidden_states.index_copy(
            1, text_indices, text_embeds
        )
        packed_hidden_states = packed_hidden_states.index_copy(
            1, video_indices, video_embeds.to(text_embeds.dtype)
        )
        packed_hidden_states = packed_hidden_states.index_copy(
            1, audio_indices, audio_embeds.to(text_embeds.dtype)
        )

        temb = self.time_proj(timestep)
        temb = self.time_embedder(
            temb.to(self.time_embedder.linear_1.weight.dtype)
        )

        (
            packed_hidden_states,
            padded_timestep_indices,
            padded_token_tags,
            padded_position_ids,
            pad_amount,
        ) = self._pad_rows(
            packed_hidden_states,
            timestep_indices,
            token_tags,
            position_ids,
        )

        padded_length = padded_position_ids.shape[0]
        ulysses_world_size = get_ulysses_parallel_world_size()
        ulysses_rank = get_ulysses_parallel_rank()
        if padded_length % ulysses_world_size:
            raise ValueError(
                f"MiniMax-H3 padded sequence length {padded_length} must be "
                f"divisible by Ulysses degree {ulysses_world_size}."
            )

        local_sequence_length = padded_length // ulysses_world_size
        local_start = ulysses_rank * local_sequence_length
        local_stop = local_start + local_sequence_length

        rotary_emb = self.rope(padded_position_ids)
        rotary_emb = (
            rotary_emb[0][local_start:local_stop],
            rotary_emb[1][local_start:local_stop],
        )

        packed_hidden_states = packed_hidden_states[:, local_start:local_stop]
        local_timestep_indices = padded_timestep_indices[local_start:local_stop]
        local_token_tags = padded_token_tags[local_start:local_stop]
        adaln_indices = (
            local_timestep_indices * MINIMAX_H3_MODALITY_NUM
            + local_token_tags.clamp(min=0)
        )

        if pad_amount:
            indices_k = torch.arange(
                sequence_length,
                dtype=torch.long,
                device=packed_hidden_states.device,
            )
            self._usp_attention_kwargs.update(
                {
                    "indices_k": indices_k,
                    "cu_seqlens_k": torch.tensor(
                        [0, sequence_length],
                        dtype=torch.int32,
                        device=packed_hidden_states.device,
                    ),
                    "max_seqlen_k": sequence_length,
                }
            )
        else:
            self._usp_attention_kwargs.pop("indices_k", None)
            self._usp_attention_kwargs.pop("cu_seqlens_k", None)
            self._usp_attention_kwargs.pop("max_seqlen_k", None)

        for block in self.transformer_blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                packed_hidden_states = self._gradient_checkpointing_func(
                    block,
                    packed_hidden_states,
                    temb,
                    adaln_indices,
                    rotary_emb,
                    None,
                )
            else:
                packed_hidden_states = block(
                    packed_hidden_states,
                    temb,
                    adaln_indices,
                    rotary_emb,
                    None,
                )

        packed_hidden_states = self.norm_out(
            packed_hidden_states,
            temb,
            local_timestep_indices,
        ).to(self.proj_out.weight.dtype)
        local_video_output = self.proj_out(packed_hidden_states)
        local_audio_output = self.audio_proj_out(packed_hidden_states)

        if ulysses_world_size > 1:
            video_width = local_video_output.shape[-1]
            packed_output = get_sp_group().all_gather(
                torch.cat((local_video_output, local_audio_output), dim=-1),
                dim=1,
            )
            local_video_output, local_audio_output = packed_output.split(
                (video_width, packed_output.shape[-1] - video_width),
                dim=-1,
            )

        local_video_output = local_video_output[:, :sequence_length]
        local_audio_output = local_audio_output[:, :sequence_length]
        video_output = local_video_output.index_select(1, video_indices)
        audio_output = local_audio_output.index_select(1, audio_indices)

        if not return_dict:
            return (video_output, audio_output)
        return MiniMaxH3TransformerOutput(
            sample=video_output,
            audio_sample=audio_output,
        )
