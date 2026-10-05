import math
from typing import Any

import torch
from diffusers import Lumina2Transformer2DModel
from diffusers.models.attention_processor import Attention
from diffusers.models.embeddings import apply_rotary_emb
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.transformers.transformer_lumina2 import Lumina2AttnProcessor2_0
from diffusers.utils import (
    USE_PEFT_BACKEND,
    logging,
    scale_lora_layers,
    unscale_lora_layers,
)

from xfuser.core.distributed import (
    get_ring_parallel_world_size,
    get_sequence_parallel_rank,
    get_sequence_parallel_world_size,
    get_sp_group,
    get_ulysses_parallel_world_size,
)
from xfuser.core.attention.spec import AttentionBackendType
from xfuser.model_executor.layers.attention_processor import (
    xFuserAttentionProcessorRegister,
)
from xfuser.model_executor.layers.usp import USP

from .base_transformer import xFuserTransformerBaseWrapper
from .register import xFuserTransformerWrappersRegister

logger = logging.get_logger(__name__)


@xFuserAttentionProcessorRegister.register(Lumina2AttnProcessor2_0)
class xFuserLumina2AttnProcessor2_0(Lumina2AttnProcessor2_0):
    """Lumina2 GQA attention implemented with xDiT's USP primitive."""

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        image_rotary_emb: torch.Tensor | None = None,
        base_sequence_length: int | None = None,
    ) -> torch.Tensor:
        batch_size, local_sequence_length, _ = hidden_states.shape

        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query_dim = query.shape[-1]
        inner_dim = key.shape[-1]
        head_dim = query_dim // attn.heads
        activation_dtype = query.dtype
        kv_heads = inner_dim // head_dim

        if attn.heads % kv_heads != 0:
            raise ValueError(
                "Lumina2 GQA requires the number of query heads to be divisible "
                f"by the number of KV heads, got {attn.heads} and {kv_heads}."
            )

        query = query.view(batch_size, -1, attn.heads, head_dim)
        key = key.view(batch_size, -1, kv_heads, head_dim)
        value = value.view(batch_size, -1, kv_heads, head_dim)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb, use_real=False)
            key = apply_rotary_emb(key, image_rotary_emb, use_real=False)

        query = query.to(activation_dtype)
        key = key.to(activation_dtype)

        repeats = attn.heads // kv_heads
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        # Lumina2 currently does not pass base_sequence_length from its blocks,
        # but retain the processor API and proportional-attention behavior.
        if base_sequence_length is not None:
            global_sequence_length = (
                attention_mask.shape[-1]
                if attention_mask is not None
                else local_sequence_length * get_sequence_parallel_world_size()
            )
            softmax_scale = (
                math.sqrt(math.log(global_sequence_length, base_sequence_length))
                * attn.scale
            )
            query = query * (softmax_scale / attn.scale)

        attention_kwargs = None
        backend = None
        if attention_mask is not None:
            attention_kwargs = {
                "attn_mask": attention_mask.bool().view(batch_size, 1, 1, -1)
            }
            # Several optimized backends ignore arbitrary masks. PyTorch SDPA
            # supports this broadcastable key mask and still selects an
            # appropriate CUDA kernel internally.
            backend = AttentionBackendType.SDPA

        # Keep the eight KV heads compact through all-to-all; USP repeats each
        # rank's local KV shard only after communication.
        hidden_states = USP(
            query,
            key,
            value,
            dropout_p=0.0,
            is_causal=False,
            combine_qkv_a2a=True,
            kv_head_repeat=repeats,
            backend=backend,
            attention_kwargs=attention_kwargs,
        )
        hidden_states = hidden_states.transpose(1, 2).reshape(
            batch_size, -1, attn.heads * head_dim
        )
        hidden_states = hidden_states.to(activation_dtype)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states


@xFuserTransformerWrappersRegister.register(Lumina2Transformer2DModel)
class xFuserLumina2Transformer2DWrapper(xFuserTransformerBaseWrapper):
    """Lumina2 transformer wrapper with GQA-aware Ulysses sequence parallelism."""

    def __init__(self, transformer: Lumina2Transformer2DModel):
        # Lumina2 does not support PipeFusion yet, so keep every transformer
        # block on every rank and shard only the joint token sequence.
        super().__init__(transformer=transformer, transformer_blocks_name=[])
        self.wrapped_layers = []

        sp_world_size = get_sequence_parallel_world_size()
        if sp_world_size > 1:
            if get_ring_parallel_world_size() > 1:
                raise NotImplementedError(
                    "Lumina2 sequence parallelism currently supports Ulysses only; "
                    "set --ring_degree 1. Ring attention needs per-step masks for "
                    "Lumina2's padded text-image sequences."
                )
            ulysses_world_size = get_ulysses_parallel_world_size()
            if self.module.config.num_attention_heads % ulysses_world_size != 0:
                raise ValueError(
                    "Lumina2's query heads must be divisible by the Ulysses "
                    f"degree, got {self.module.config.num_attention_heads} heads "
                    f"and degree {ulysses_world_size}."
                )
            if self.module.config.num_kv_heads % ulysses_world_size != 0:
                raise ValueError(
                    "Lumina2's KV heads must be divisible by the Ulysses degree, "
                    f"got {self.module.config.num_kv_heads} heads and degree "
                    f"{ulysses_world_size}."
                )
            for layer in self.module.layers:
                layer.attn.set_processor(xFuserLumina2AttnProcessor2_0())

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        attention_kwargs: dict[str, Any] | None = None,
        return_dict: bool = True,
    ) -> torch.Tensor | Transformer2DModelOutput:
        if get_sequence_parallel_world_size() == 1:
            return self.module(
                hidden_states=hidden_states,
                timestep=timestep,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                attention_kwargs=attention_kwargs,
                return_dict=return_dict,
            )

        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            scale_lora_layers(self.module, lora_scale)
        elif attention_kwargs is not None and attention_kwargs.get("scale") is not None:
            logger.warning(
                "Passing `scale` via `attention_kwargs` without the PEFT backend "
                "has no effect."
            )

        batch_size, _, height, width = hidden_states.shape
        temb, encoder_hidden_states = self.module.time_caption_embed(
            hidden_states, timestep, encoder_hidden_states
        )
        (
            hidden_states,
            context_rotary_emb,
            noise_rotary_emb,
            rotary_emb,
            encoder_seq_lengths,
            seq_lengths,
        ) = self.module.rope_embedder(hidden_states, encoder_attention_mask)
        hidden_states = self.module.x_embedder(hidden_states)

        # These short refinement sequences are replicated. Only the long joint
        # text-image sequence is worth sharding, and their native processors
        # avoid unnecessary collectives here.
        for layer in self.module.context_refiner:
            encoder_hidden_states = layer(
                encoder_hidden_states, encoder_attention_mask, context_rotary_emb
            )
        for layer in self.module.noise_refiner:
            hidden_states = layer(hidden_states, None, noise_rotary_emb, temb)

        max_seq_len = max(seq_lengths)
        attention_mask = hidden_states.new_zeros(
            batch_size, max_seq_len, dtype=torch.bool
        )
        joint_hidden_states = hidden_states.new_zeros(
            batch_size, max_seq_len, self.module.config.hidden_size
        )
        for i, (encoder_seq_len, seq_len) in enumerate(
            zip(encoder_seq_lengths, seq_lengths)
        ):
            attention_mask[i, :seq_len] = True
            joint_hidden_states[i, :encoder_seq_len] = encoder_hidden_states[
                i, :encoder_seq_len
            ]
            joint_hidden_states[i, encoder_seq_len:seq_len] = hidden_states[i]

        sp_world_size = get_sequence_parallel_world_size()
        padded_seq_len = math.ceil(max_seq_len / sp_world_size) * sp_world_size
        if padded_seq_len != max_seq_len:
            padded_hidden_states = joint_hidden_states.new_zeros(
                batch_size, padded_seq_len, self.module.config.hidden_size
            )
            padded_hidden_states[:, :max_seq_len] = joint_hidden_states
            joint_hidden_states = padded_hidden_states

            padded_attention_mask = attention_mask.new_zeros(batch_size, padded_seq_len)
            padded_attention_mask[:, :max_seq_len] = attention_mask
            attention_mask = padded_attention_mask

            # Multiplication by 1+0j leaves padded Q/K unchanged before the key
            # mask removes them. This also preserves rotary_emb's complex dtype.
            padded_rotary_emb = torch.ones(
                batch_size,
                padded_seq_len,
                rotary_emb.shape[-1],
                dtype=rotary_emb.dtype,
                device=rotary_emb.device,
            )
            padded_rotary_emb[:, :max_seq_len] = rotary_emb
            rotary_emb = padded_rotary_emb

        local_seq_len = padded_seq_len // sp_world_size
        local_start = get_sequence_parallel_rank() * local_seq_len
        local_end = local_start + local_seq_len
        hidden_states = joint_hidden_states[:, local_start:local_end].contiguous()
        local_rotary_emb = rotary_emb[:, local_start:local_end].contiguous()

        # A mask is necessary for SP padding and for unequal caption lengths.
        # Skip it only when every token is real so optimized dense backends stay
        # available.
        use_mask = padded_seq_len != max_seq_len or len(set(seq_lengths)) > 1
        layer_attention_mask = attention_mask if use_mask else None

        for layer in self.module.layers:
            if torch.is_grad_enabled() and self.module.gradient_checkpointing:
                hidden_states = self.module._gradient_checkpointing_func(
                    layer,
                    hidden_states,
                    layer_attention_mask,
                    local_rotary_emb,
                    temb,
                )
            else:
                hidden_states = layer(
                    hidden_states,
                    layer_attention_mask,
                    local_rotary_emb,
                    temb,
                )

        hidden_states = get_sp_group().all_gather(hidden_states, dim=1)
        hidden_states = hidden_states[:, :max_seq_len]
        hidden_states = self.module.norm_out(hidden_states, temb)

        p = self.module.config.patch_size
        output = []
        for i, (encoder_seq_len, seq_len) in enumerate(
            zip(encoder_seq_lengths, seq_lengths)
        ):
            output.append(
                hidden_states[i][encoder_seq_len:seq_len]
                .view(height // p, width // p, p, p, self.module.out_channels)
                .permute(4, 0, 2, 1, 3)
                .flatten(3, 4)
                .flatten(1, 2)
            )
        output = torch.stack(output, dim=0)

        if USE_PEFT_BACKEND:
            unscale_lora_layers(self.module, lora_scale)

        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)
