from __future__ import annotations

from typing import Any

import torch

from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.transformers.transformer_minimax_h3_image import (
    MINIMAX_H3_MODALITY_NUM,
    MiniMaxH3ImageTransformer2DModel,
)

from xfuser.core.distributed import (
    get_runtime_state,
    get_sp_group,
    get_ulysses_parallel_rank,
    get_ulysses_parallel_world_size,
)
from xfuser.model_executor.layers.fused_qk_rope_flydsl import (
    _HAS_FLYDSL,
    prepare_neox_rope_tables,
)
from xfuser.model_executor.models.transformers.transformer_minimax_h3 import (
    pad_packed_rows,
    xFuserMiniMaxH3AttnProcessor,
)


class xFuserMiniMaxH3ImageTransformer2DWrapper(MiniMaxH3ImageTransformer2DModel):
    """MiniMax-H3-Image transformer with Ulysses sequence parallelism.

    The model is MiniMax-H3's packed single-stream transformer without the audio
    stream, so it shards the packed sequence the way the MiniMax-H3 wrapper does:
    the rows are padded to a multiple of 64 at the end, each Ulysses rank runs the
    blocks on its slice, and the attention processor exchanges heads. The text
    refiner runs replicated.

    ``forward`` splits into ``_forward_body``, which ends at the shared output
    norm and is what the runner compiles, and the per-step output head, which
    stays outside the graph: ``pdd_step_index`` is a Python int and would
    otherwise specialize one graph per sampling step.
    """

    def __init__(
        self,
        num_attention_heads: int = 56,
        attention_head_dim: int = 128,
        hidden_size: int = 5376,
        num_layers: int = 50,
        num_refiner_layers: int = 2,
        ffn_dim: int = 14336,
        in_channels: int = 24,
        patch_size: tuple[int, int, int] = (1, 2, 2),
        text_dim: int = 5120,
        freq_dim: int = 256,
        time_embed_hidden_dim: int = 5376,
        time_embed_dim: int = 2688,
        rope_freq_dim: int = 16,
        rope_theta: float = 10000.0,
        norm_eps: float = 1e-5,
        qk_norm_eps: float = 1e-5,
        final_norm_eps: float = 1e-5,
        num_pdd_heads: int = 6,
        attention_backend=None,
    ) -> None:
        super().__init__(
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_refiner_layers=num_refiner_layers,
            ffn_dim=ffn_dim,
            in_channels=in_channels,
            patch_size=patch_size,
            text_dim=text_dim,
            freq_dim=freq_dim,
            time_embed_hidden_dim=time_embed_hidden_dim,
            time_embed_dim=time_embed_dim,
            rope_freq_dim=rope_freq_dim,
            rope_theta=rope_theta,
            norm_eps=norm_eps,
            qk_norm_eps=qk_norm_eps,
            final_norm_eps=final_norm_eps,
            num_pdd_heads=num_pdd_heads,
        )
        # Keys are always present, carrying None when unused. torch.compile
        # guards on this dict's key set, so adding or removing entries between
        # forwards forces a recompile.
        self._usp_attention_kwargs: dict[str, Any] = {
            "indices_k": None,
            "cu_seqlens_k": None,
            "max_seqlen_k": None,
            # The pad rows are a trailing block, so backends without a
            # key-padding mask can slice K/V here instead of packing.
            "valid_kv_len": None,
        }
        if attention_backend is None:
            try:
                attention_backend = get_runtime_state().attention_backend
            except AssertionError:
                attention_backend = None
        self.attention_backend = attention_backend

        for block in self.token_refiner.refiner_blocks:
            block.attn.set_processor(
                xFuserMiniMaxH3AttnProcessor(
                    use_ulysses_parallel_attention=False,
                    backend=self.attention_backend,
                )
            )
        for block in self.transformer_blocks:
            block.attn.set_processor(
                xFuserMiniMaxH3AttnProcessor(
                    use_ulysses_parallel_attention=True,
                    attention_kwargs=self._usp_attention_kwargs,
                    backend=self.attention_backend,
                )
            )

        self.register_forward_pre_hook(lambda module, args: get_runtime_state().increment_step_counter())

    def _forward_body(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        timestep_indices: torch.Tensor,
        token_tags: torch.Tensor,
        position_ids: torch.Tensor,
        image_indices: torch.Tensor,
        text_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Run this rank's slice of the packed sequence up to the output norm.

        Returns the normalized rows of the slice, in the output heads' dtype.
        """
        sequence_length = position_ids.shape[0]

        image_embeds = self.proj_in(hidden_states.to(self.proj_in.weight.dtype))
        text_embeds = self.context_embedder(encoder_hidden_states.to(self.context_embedder.weight.dtype))
        text_embeds = self.token_refiner(text_embeds)

        packed_hidden_states = text_embeds.new_zeros((text_embeds.shape[0], sequence_length, text_embeds.shape[-1]))
        packed_hidden_states = packed_hidden_states.index_copy(1, text_indices, text_embeds)
        packed_hidden_states = packed_hidden_states.index_copy(1, image_indices, image_embeds.to(text_embeds.dtype))

        temb = self.time_proj(timestep)
        temb = self.time_embedder(temb.to(self.time_embedder.linear_1.weight.dtype))

        (
            packed_hidden_states,
            padded_timestep_indices,
            padded_token_tags,
            padded_position_ids,
            pad_amount,
        ) = pad_packed_rows(packed_hidden_states, timestep_indices, token_tags, position_ids)

        padded_length = padded_position_ids.shape[0]
        ulysses_world_size = get_ulysses_parallel_world_size()
        ulysses_rank = get_ulysses_parallel_rank()
        if padded_length % ulysses_world_size:
            raise ValueError(
                f"MiniMax-H3-Image padded sequence length {padded_length} must be "
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
        if _HAS_FLYDSL:
            # The fused QK-norm + RoPE kernel of the shared processor reads the
            # folded tables; built once per forward rather than once per block.
            rotary_emb = rotary_emb + prepare_neox_rope_tables(
                rotary_emb[0], rotary_emb[1], self.config.attention_head_dim
            )

        packed_hidden_states = packed_hidden_states[:, local_start:local_stop]
        local_timestep_indices = padded_timestep_indices[local_start:local_stop]
        local_token_tags = padded_token_tags[local_start:local_stop]
        adaln_indices = local_timestep_indices * MINIMAX_H3_MODALITY_NUM + local_token_tags.clamp(min=0)

        if pad_amount:
            indices_k = torch.arange(sequence_length, dtype=torch.long, device=packed_hidden_states.device)
            cu_seqlens_k = torch.tensor([0, sequence_length], dtype=torch.int32, device=packed_hidden_states.device)
            max_seqlen_k = sequence_length
        else:
            indices_k = None
            cu_seqlens_k = None
            max_seqlen_k = None
        self._usp_attention_kwargs.update(
            {
                "indices_k": indices_k,
                "cu_seqlens_k": cu_seqlens_k,
                "max_seqlen_k": max_seqlen_k,
                # pad_packed_rows appends its rows, so the valid keys are the
                # leading sequence_length rows.
                "valid_kv_len": max_seqlen_k,
            }
        )

        for block in self.transformer_blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                packed_hidden_states = self._gradient_checkpointing_func(
                    block, packed_hidden_states, temb, adaln_indices, rotary_emb
                )
            else:
                packed_hidden_states = block(packed_hidden_states, temb, adaln_indices, rotary_emb)

        return self.norm_out(packed_hidden_states, temb, local_timestep_indices).to(self.pdd_proj_out[0].weight.dtype)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        timestep_indices: torch.Tensor,
        token_tags: torch.Tensor,
        position_ids: torch.Tensor,
        image_indices: torch.Tensor,
        text_indices: torch.Tensor,
        pdd_step_index: int,
        return_dict: bool = True,
    ) -> Transformer2DModelOutput | tuple[torch.Tensor]:
        if position_ids.ndim != 2 or position_ids.shape[-1] != 3:
            raise ValueError(f"`position_ids` must be a `(seq_len, 3)` tensor, got {list(position_ids.shape)}.")
        sequence_length = position_ids.shape[0]
        if token_tags.shape != (sequence_length,) or timestep_indices.shape != (sequence_length,):
            raise ValueError(
                "`token_tags` and `timestep_indices` must both be `(seq_len,)` "
                f"tensors matching `position_ids`, got {list(token_tags.shape)} "
                f"and {list(timestep_indices.shape)} for seq_len={sequence_length}."
            )

        local_hidden_states = self._forward_body(
            hidden_states,
            encoder_hidden_states,
            timestep,
            timestep_indices,
            token_tags,
            position_ids,
            image_indices,
            text_indices,
        )
        # The head projects to the patch width, far narrower than the hidden
        # size, so the sequence is gathered after it rather than before.
        output = self.pdd_proj_out[pdd_step_index](local_hidden_states)
        if get_ulysses_parallel_world_size() > 1:
            output = get_sp_group().all_gather(output, dim=1)
        output = output.index_select(1, image_indices)

        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)
