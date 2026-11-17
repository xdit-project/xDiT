from __future__ import annotations

from typing import Any

import torch
from torch.distributed.fsdp import FSDPModule, FullyShardedDataParallel

from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import get_parameter_dtype
from diffusers.models.transformers.transformer_minimax_h3_image import (
    MINIMAX_H3_MODALITY_NUM,
    MiniMaxH3ImageAdaLayerNormModulation,
    MiniMaxH3ImageAdaLayerNormOut,
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

# Distinct timestep sets whose modulation the wrapper keeps. The sampling grid is
# fixed, so a T2I request uses 6 and an edit another 6; the bound only stops a
# caller sweeping arbitrary timesteps from growing the cache without limit.
_MODULATION_CACHE_SIZE = 32


class _CachedAdaLNModulation(MiniMaxH3ImageAdaLayerNormModulation):
    """A block's AdaLN modulation, read from a table projected ahead of the block.

    ``project`` is the stock projection, its six chunks concatenated back into one
    ``(num_timesteps * MINIMAX_H3_MODALITY_NUM, 6 * hidden_size)`` table; ``forward``
    takes that table in place of ``temb`` and chunks it as the stock module does,
    so the block's own forward runs unchanged.
    """

    def project(self, temb: torch.Tensor) -> torch.Tensor:
        return torch.cat(super().forward(temb), dim=-1)

    def forward(self, modulation: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return modulation.chunk(6, dim=-1)


class _CachedAdaLayerNormOut(MiniMaxH3ImageAdaLayerNormOut):
    """The output norm, modulated by a ``(num_timesteps, 2 * hidden_size)`` table
    from ``project`` in place of ``temb``."""

    def project(self, temb: torch.Tensor) -> torch.Tensor:
        return self.linear(torch.nn.functional.silu(temb).to(get_parameter_dtype(self.linear)))

    def forward(
        self, hidden_states: torch.Tensor, modulation: torch.Tensor, timestep_indices: torch.Tensor
    ) -> torch.Tensor:
        shift, scale = modulation.chunk(2, dim=-1)
        hidden_states = self.norm(hidden_states)
        return hidden_states * (1.0 + scale.index_select(0, timestep_indices)) + shift.index_select(0, timestep_indices)


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

    The AdaLN modulation of every block and of the output norm depends on the
    timestep alone, and the sampler visits the same fixed grid in every request.
    ``forward`` therefore projects each distinct timestep set once, at inference,
    and hands the blocks the projected tables. The block projections are about
    40% of the checkpoint, so this skips reading them on every step.
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
            # The state dict keys stay those of the stock module.
            block.adaln_proj.__class__ = _CachedAdaLNModulation
        self.norm_out.__class__ = _CachedAdaLayerNormOut
        self._modulation_cache: dict[tuple[float, ...], tuple[torch.Tensor, torch.Tensor]] = {}

        self.register_forward_pre_hook(lambda module, args: get_runtime_state().increment_step_counter())

    @staticmethod
    def _project_block_modulation(block: torch.nn.Module, temb: torch.Tensor) -> torch.Tensor:
        # FSDP gathers a block's weights only around its own forward. The runner
        # compiles a sharded model block by block, so unwrap the compiled module first.
        block = getattr(block, "_orig_mod", block)
        if isinstance(block, FullyShardedDataParallel):
            # This runs inside the root unit's forward, which torch documents as
            # unsupported for summon_full_params; it checks only this block's state.
            with FullyShardedDataParallel.summon_full_params(block, recurse=False, writeback=False):
                return block.adaln_proj.project(temb)
        if isinstance(block, FSDPModule):
            block.unshard()
            try:
                return block.adaln_proj.project(temb)
            finally:
                block.reshard()
        return block.adaln_proj.project(temb)

    def _modulation(self, timestep: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """The AdaLN tables of ``timestep``: ``(num_layers, num_timesteps *
        MINIMAX_H3_MODALITY_NUM, 6 * hidden_size)`` for the blocks and
        ``(num_timesteps, 2 * hidden_size)`` for the output norm.

        Cached by timestep value at inference; with autograd on they are projected
        every call, so gradients still reach the projections.
        """
        key = None
        if not torch.is_grad_enabled():
            key = tuple(timestep.tolist())
            cached = self._modulation_cache.get(key)
            if cached is not None:
                return cached

        temb = self.time_proj(timestep)
        temb = self.time_embedder(temb.to(self.time_embedder.linear_1.weight.dtype))
        modulation = (
            torch.stack([self._project_block_modulation(block, temb) for block in self.transformer_blocks]),
            self.norm_out.project(temb),
        )
        if key is not None:
            if len(self._modulation_cache) >= _MODULATION_CACHE_SIZE:
                self._modulation_cache.clear()
            self._modulation_cache[key] = modulation
        return modulation

    def _forward_body(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        block_modulation: torch.Tensor,
        output_modulation: torch.Tensor,
        timestep_indices: torch.Tensor,
        token_tags: torch.Tensor,
        position_ids: torch.Tensor,
        image_indices: torch.Tensor,
        text_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Run this rank's slice of the packed sequence up to the output norm.

        ``block_modulation`` and ``output_modulation`` are the tables of
        ``_modulation``. Returns the normalized rows of the slice, in the output
        heads' dtype.
        """
        sequence_length = position_ids.shape[0]

        image_embeds = self.proj_in(hidden_states.to(self.proj_in.weight.dtype))
        text_embeds = self.context_embedder(encoder_hidden_states.to(self.context_embedder.weight.dtype))
        text_embeds = self.token_refiner(text_embeds)

        packed_hidden_states = text_embeds.new_zeros((text_embeds.shape[0], sequence_length, text_embeds.shape[-1]))
        packed_hidden_states = packed_hidden_states.index_copy(1, text_indices, text_embeds)
        packed_hidden_states = packed_hidden_states.index_copy(1, image_indices, image_embeds.to(text_embeds.dtype))

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

        for block, modulation in zip(self.transformer_blocks, block_modulation):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                packed_hidden_states = self._gradient_checkpointing_func(
                    block, packed_hidden_states, modulation, adaln_indices, rotary_emb
                )
            else:
                packed_hidden_states = block(packed_hidden_states, modulation, adaln_indices, rotary_emb)

        return self.norm_out(packed_hidden_states, output_modulation, local_timestep_indices).to(
            self.pdd_proj_out[0].weight.dtype
        )

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

        block_modulation, output_modulation = self._modulation(timestep)
        local_hidden_states = self._forward_body(
            hidden_states,
            encoder_hidden_states,
            block_modulation,
            output_modulation,
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
