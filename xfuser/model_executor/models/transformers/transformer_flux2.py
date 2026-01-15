import torch
from typing import Optional, Tuple
from diffusers.models.transformers.transformer_flux2 import (
    Flux2Attention,
    Flux2AttnProcessor,
    Flux2Transformer2DModel,
    Flux2ParallelSelfAttention,
    Flux2ParallelSelfAttnProcessor,
    _get_qkv_projections,
)
from diffusers.models.embeddings import apply_rotary_emb

from xfuser.model_executor.layers.attention_processor import (
    xFuserAttentionBaseWrapper,
    xFuserAttentionProcessorRegister
)

from xfuser.core.distributed import (
    get_sequence_parallel_world_size,
    get_sequence_parallel_rank,
    get_classifier_free_guidance_world_size,
    get_classifier_free_guidance_rank,
    get_sp_group,
    get_cfg_group,
    get_runtime_state,
)

from xfuser.model_executor.layers.usp import USP
from xfuser.model_executor.layers import xFuserLayerWrappersRegister
from xfuser.model_executor.models.transformers.transformer_flux import xFuserFluxAttentionWrapper


@xFuserAttentionProcessorRegister.register(Flux2AttnProcessor)
class xFuserFlux2AttnProcessor(Flux2AttnProcessor):

    def __init__(self):
        super().__init__()

    def __call__(
        self,
        attn: "Flux2Attention",
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        query, key, value, encoder_query, encoder_key, encoder_value = _get_qkv_projections(
            attn, hidden_states, encoder_hidden_states
        )

        query = query.unflatten(-1, (attn.heads, -1))
        key = key.unflatten(-1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        if attn.added_kv_proj_dim is not None:
            encoder_query = encoder_query.unflatten(-1, (attn.heads, -1))
            encoder_key = encoder_key.unflatten(-1, (attn.heads, -1))
            encoder_value = encoder_value.unflatten(-1, (attn.heads, -1))

            encoder_query = attn.norm_added_q(encoder_query)
            encoder_key = attn.norm_added_k(encoder_key)

            query = torch.cat([encoder_query, query], dim=1)
            key = torch.cat([encoder_key, key], dim=1)
            value = torch.cat([encoder_value, value], dim=1)

        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
            key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)


        # Transpose for attention computation
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        hidden_states = USP(query, key, value)

        # Transpose back to original shape
        hidden_states = hidden_states.transpose(1, 2)

        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            encoder_hidden_states, hidden_states = hidden_states.split_with_sizes(
                [encoder_hidden_states.shape[1], hidden_states.shape[1] - encoder_hidden_states.shape[1]], dim=1
            )
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if encoder_hidden_states is not None:
            return hidden_states, encoder_hidden_states
        else:
            return hidden_states

@xFuserAttentionProcessorRegister.register(Flux2ParallelSelfAttnProcessor)
class xFuserFlux2ParallelSelfAttnProcessor(Flux2ParallelSelfAttnProcessor):

    def __init__(self):
        super().__init__()

    def __call__(
        self,
        attn: "Flux2ParallelSelfAttention",
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Parallel in (QKV + MLP in) projection
        hidden_states = attn.to_qkv_mlp_proj(hidden_states)
        qkv, mlp_hidden_states = torch.split(
            hidden_states, [3 * attn.inner_dim, attn.mlp_hidden_dim * attn.mlp_mult_factor], dim=-1
        )

        # Handle the attention logic
        query, key, value = qkv.chunk(3, dim=-1)

        query = query.unflatten(-1, (attn.heads, -1))
        key = key.unflatten(-1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
            key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)

        # Transpose for attention computation
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        hidden_states = USP(query, key, value, combine_qkv_a2a=True)

        # Transpose back to original shape
        hidden_states = hidden_states.transpose(1, 2)

        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.to(query.dtype)

        # Handle the feedforward (FF) logic
        mlp_hidden_states = attn.mlp_act_fn(mlp_hidden_states)

        # Concatenate and parallel output projection
        hidden_states = torch.cat([hidden_states, mlp_hidden_states], dim=-1)
        hidden_states = attn.to_out(hidden_states)

        return hidden_states


@xFuserLayerWrappersRegister.register(Flux2ParallelSelfAttention)
class xFuserFlux2ParallelSelfAttention(xFuserAttentionBaseWrapper):

    def __init__(self, attention: Flux2ParallelSelfAttention):
        super().__init__(attention=attention)
        self.processor = xFuserAttentionProcessorRegister.get_processor(
            attention.processor
        )()


    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:

        return super().forward(
            hidden_states,
            attention_mask,
            image_rotary_emb,
            **kwargs,
        )


class xFuserFlux2Transformer2DWrapper(Flux2Transformer2DModel):

    def __init__(
        self,
        patch_size: int = 1,
        in_channels: int = 128,
        out_channels: Optional[int] = None,
        num_layers: int = 8,
        num_single_layers: int = 48,
        attention_head_dim: int = 128,
        num_attention_heads: int = 48,
        joint_attention_dim: int = 15360,
        timestep_guidance_channels: int = 256,
        mlp_ratio: float = 3.0,
        axes_dims_rope: Tuple[int, ...] = (32, 32, 32, 32),
        rope_theta: int = 2000,
        eps: float = 1e-6,
    )
        super().__init__(
            patch_size=patch_size,
            in_channels=in_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            num_single_layers=num_single_layers,
            attention_head_dim=attention_head_dim,
            num_attention_heads=num_attention_heads,
            joint_attention_dim=joint_attention_dim,
            timestep_guidance_channels=timestep_guidance_channels,
            mlp_ratio=mlp_ratio,
            axes_dims_rope=axes_dims_rope,
            rope_theta=rope_theta,
            eps=eps,
        )

        for block in self.transformer_blocks:
            block.attn.processor = xFuserFlux2AttnProcessor()
        for block in self.single_transformer_blocks:
            block.attn.processor = xFuserFlux2ParallelSelfAttnProcessor()


    def pad_to_sp_divisible(self, tensor: torch.Tensor, padding_length: int, dim: int) -> torch.Tensor:
        padding =  torch.zeros(
            *tensor.shape[:dim], padding_length, *tensor.shape[dim + 1 :], dtype=tensor.dtype, device=tensor.device
        )
        tensor = torch.cat([tensor, padding], dim=dim)
        return tensor

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        *args,
        timestep: torch.LongTensor = None,
        img_ids: torch.Tensor = None,
        txt_ids: torch.Tensor = None,
        **kwargs,
    ):

        sp_world_size = get_sequence_parallel_world_size()
        sequence_length = hidden_states.shape[1]
        padding_length = (sp_world_size - (sequence_length % sp_world_size)) % sp_world_size
        if padding_length > 0:
            hidden_states = self._pad_to_sp_divisible(hidden_states, padding_length, dim=1)
            img_ids = self._pad_to_sp_divisible(img_ids, padding_length, dim=0)
        assert (
            hidden_states.shape[0] % get_classifier_free_guidance_world_size() == 0
        ), f"Cannot split dim 0 of hidden_states ({hidden_states.shape[0]}) into {get_classifier_free_guidance_world_size()} parts."
        if encoder_hidden_states.shape[-2] % get_sequence_parallel_world_size() != 0:
            get_runtime_state().split_text_embed_in_sp = False
        else:
            get_runtime_state().split_text_embed_in_sp = True

        if (
            isinstance(timestep, torch.Tensor)
            and timestep.ndim != 0
            and timestep.shape[0] == hidden_states.shape[0]
        ):
            timestep = torch.chunk(
                timestep, get_classifier_free_guidance_world_size(), dim=0
            )[get_classifier_free_guidance_rank()]
        hidden_states = torch.chunk(
            hidden_states, get_classifier_free_guidance_world_size(), dim=0
        )[get_classifier_free_guidance_rank()]
        hidden_states = torch.chunk(
            hidden_states, get_sequence_parallel_world_size(), dim=-2
        )[get_sequence_parallel_rank()]
        encoder_hidden_states = torch.chunk(
            encoder_hidden_states, get_classifier_free_guidance_world_size(), dim=0
        )[get_classifier_free_guidance_rank()]
        if get_runtime_state().split_text_embed_in_sp:
            encoder_hidden_states = torch.chunk(
                encoder_hidden_states, get_sequence_parallel_world_size(), dim=-2
            )[get_sequence_parallel_rank()]
        img_ids = torch.chunk(img_ids, get_sequence_parallel_world_size(), dim=-2)[
            get_sequence_parallel_rank()
        ]
        if get_runtime_state().split_text_embed_in_sp:
            txt_ids = torch.chunk(txt_ids, get_sequence_parallel_world_size(), dim=-2)[
                get_sequence_parallel_rank()
            ]

        output = super().forward(
            hidden_states,
            encoder_hidden_states,
            *args,
            timestep=timestep,
            img_ids=img_ids,
            txt_ids=txt_ids,
            **kwargs,
        )

        return_dict = not isinstance(output, tuple)
        sample = output[0]
        sample = get_sp_group().all_gather(sample, dim=-2)
        sample = get_cfg_group().all_gather(sample, dim=0)
        if padding_length > 0:
            sample = sample[:, :-padding_length, :]
        if return_dict:
            return output.__class__(sample, *output[1:])
        return (sample, *output[1:])