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
    xFuserAttentionProcessorRegister,
)

from xfuser.core.distributed.parallel_state import (
    is_pipeline_first_stage,
    is_pipeline_last_stage,
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
from xfuser.core.cache_manager.cache_manager import get_cache_manager
from xfuser.core.distributed.parallel_state import _SP
from xfuser.envs import PACKAGES_CHECKER

from xfuser.model_executor.layers.usp import USP
from xfuser.model_executor.layers import (
    xFuserLayerWrappersRegister,
    xFuserLayerBaseWrapper,
)
from xfuser.model_executor.models.transformers.transformer_flux import (
    xFuserFluxAttentionWrapper,
)
from xfuser.model_executor.models.transformers.register import (
    xFuserTransformerWrappersRegister,
)
from xfuser.model_executor.models.transformers.base_transformer import (
    xFuserTransformerBaseWrapper,
)

env_info = PACKAGES_CHECKER.get_packages_info()
HAS_LONG_CTX_ATTN = env_info["has_long_ctx_attn"]


@xFuserAttentionProcessorRegister.register(Flux2AttnProcessor)
class xFuserFlux2AttnProcessor(Flux2AttnProcessor):

    def __init__(self):
        super().__init__()
        self.use_long_ctx_attn_kvcache = (
            HAS_LONG_CTX_ATTN and _SP and get_sequence_parallel_world_size() > 1
        )

    def __call__(
        self,
        attn: "Flux2Attention",
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        query, key, value, encoder_query, encoder_key, encoder_value = (
            _get_qkv_projections(attn, hidden_states, encoder_hidden_states)
        )

        query = query.unflatten(-1, (attn.heads, -1))
        key = key.unflatten(-1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        num_encoder_tokens = 0
        if attn.added_kv_proj_dim is not None:
            encoder_query = encoder_query.unflatten(-1, (attn.heads, -1))
            encoder_key = encoder_key.unflatten(-1, (attn.heads, -1))
            encoder_value = encoder_value.unflatten(-1, (attn.heads, -1))

            encoder_query = attn.norm_added_q(encoder_query)
            encoder_key = attn.norm_added_k(encoder_key)

            num_encoder_tokens = encoder_query.shape[1]

            query = torch.cat([encoder_query, query], dim=1)
            key = torch.cat([encoder_key, key], dim=1)
            value = torch.cat([encoder_value, value], dim=1)

        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
            key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)

        # PipeFusion stale-KV: cache image KV per patch, keep encoder (text) KV fresh.
        # Mirrors xFuserFluxAttnProcessor.
        distri_cache_updated = False
        if (
            get_runtime_state().num_pipeline_patch > 1
            and not self.use_long_ctx_attn_kvcache
            and num_encoder_tokens > 0
        ):
            encoder_key, key = key.split(
                [num_encoder_tokens, key.shape[1] - num_encoder_tokens], dim=1
            )
            encoder_value, value = value.split(
                [num_encoder_tokens, value.shape[1] - num_encoder_tokens], dim=1
            )
            key, value = get_cache_manager().update_and_get_kv_cache(
                new_kv=[key, value],
                layer=attn,
                slice_dim=1,
                layer_type="attn",
            )
            key = torch.cat([encoder_key, key], dim=1)
            value = torch.cat([encoder_value, value], dim=1)
            distri_cache_updated = True

        # Transpose for attention computation
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        if get_runtime_state().num_pipeline_patch > 1 and not distri_cache_updated:
            # SP+PP hybrid: split text/image QKV, pass attn_layer for KV buffer
            # (yunchang long-context-attention manages stale-KV across SP ranks)
            if get_runtime_state().split_text_embed_in_sp:
                joint_q = joint_k = joint_v = None
            else:
                joint_q, query = query.split(
                    [num_encoder_tokens, query.shape[2] - num_encoder_tokens], dim=2
                )
                joint_k, key = key.split(
                    [num_encoder_tokens, key.shape[2] - num_encoder_tokens], dim=2
                )
                joint_v, value = value.split(
                    [num_encoder_tokens, value.shape[2] - num_encoder_tokens], dim=2
                )
            hidden_states = USP(
                query,
                key,
                value,
                dropout_p=0.0,
                is_causal=False,
                joint_query=joint_q,
                joint_key=joint_k,
                joint_value=joint_v,
                joint_strategy="front",
                attn_layer=attn,
            )
        else:
            hidden_states = USP(query, key, value)

        # Transpose back to original shape
        hidden_states = hidden_states.transpose(1, 2)

        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            encoder_hidden_states, hidden_states = hidden_states.split_with_sizes(
                [
                    encoder_hidden_states.shape[1],
                    hidden_states.shape[1] - encoder_hidden_states.shape[1],
                ],
                dim=1,
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
        self.use_long_ctx_attn_kvcache = (
            HAS_LONG_CTX_ATTN and _SP and get_sequence_parallel_world_size() > 1
        )

    def __call__(
        self,
        attn: "Flux2ParallelSelfAttention",
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        # Parallel in (QKV + MLP in) projection
        hidden_states = attn.to_qkv_mlp_proj(hidden_states)
        qkv, mlp_hidden_states = torch.split(
            hidden_states,
            [3 * attn.inner_dim, attn.mlp_hidden_dim * attn.mlp_mult_factor],
            dim=-1,
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

        # PipeFusion stale-KV: single block runs on concatenated [text, image].
        # Cache only the image portion (text KV stays fresh). MLP part is NOT cached.
        # num_txt_tokens is passed from the transformer wrapper via joint_attention_kwargs.
        num_txt_tokens = kwargs.get("num_txt_tokens", 0)
        distri_cache_updated = False
        if (
            get_runtime_state().num_pipeline_patch > 1
            and not self.use_long_ctx_attn_kvcache
        ):
            if num_txt_tokens > 0:
                text_key, key = key.split(
                    [num_txt_tokens, key.shape[1] - num_txt_tokens], dim=1
                )
                text_value, value = value.split(
                    [num_txt_tokens, value.shape[1] - num_txt_tokens], dim=1
                )
                key, value = get_cache_manager().update_and_get_kv_cache(
                    new_kv=[key, value],
                    layer=attn,
                    slice_dim=1,
                    layer_type="attn",
                )
                key = torch.cat([text_key, key], dim=1)
                value = torch.cat([text_value, value], dim=1)
            else:
                key, value = get_cache_manager().update_and_get_kv_cache(
                    new_kv=[key, value],
                    layer=attn,
                    slice_dim=1,
                    layer_type="attn",
                )
            distri_cache_updated = True

        # Transpose for attention computation
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        if (
            get_runtime_state().num_pipeline_patch > 1
            and not distri_cache_updated
            and num_txt_tokens > 0
        ):
            # SP+PP hybrid: split text/image QKV, pass attn_layer for KV buffer
            joint_q, query = query.split(
                [num_txt_tokens, query.shape[2] - num_txt_tokens], dim=2
            )
            joint_k, key = key.split(
                [num_txt_tokens, key.shape[2] - num_txt_tokens], dim=2
            )
            joint_v, value = value.split(
                [num_txt_tokens, value.shape[2] - num_txt_tokens], dim=2
            )
            hidden_states = USP(
                query,
                key,
                value,
                dropout_p=0.0,
                is_causal=False,
                joint_query=joint_q,
                joint_key=joint_k,
                joint_value=joint_v,
                joint_strategy="front",
                attn_layer=attn,
                combine_qkv_a2a=True,
            )
        else:
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


@xFuserLayerWrappersRegister.register(Flux2Attention)
class xFuserFlux2AttentionWrapper(xFuserAttentionBaseWrapper):
    """Layer wrapper for Flux2Attention (joint double-stream block).

    Inherits xFuserAttentionBaseWrapper so _register_cache recognises it, but
    bypasses the to_k/to_v asserts in the base __init__ (Flux2Attention may use
    a fused to_qkv) by calling xFuserLayerBaseWrapper.__init__ directly.
    """

    def __init__(self, attention: Flux2Attention):
        xFuserLayerBaseWrapper.__init__(self, module=attention)
        self.processor = xFuserAttentionProcessorRegister.get_processor(
            attention.processor
        )()

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        return self.processor(
            self,
            hidden_states,
            encoder_hidden_states,
            attention_mask,
            image_rotary_emb,
            **kwargs,
        )


@xFuserLayerWrappersRegister.register(Flux2ParallelSelfAttention)
class xFuserFlux2ParallelSelfAttention(xFuserAttentionBaseWrapper):
    """Layer wrapper for Flux2ParallelSelfAttention (single-stream block).

    Inherits xFuserAttentionBaseWrapper so _register_cache recognises it, but
    bypasses the to_k/to_v asserts (Flux2ParallelSelfAttention uses a fused
    to_qkv_mlp_proj and has no to_k/to_v) by calling xFuserLayerBaseWrapper.__init__.
    """

    def __init__(self, attention: Flux2ParallelSelfAttention):
        xFuserLayerBaseWrapper.__init__(self, module=attention)
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
        return self.processor(
            self, hidden_states, attention_mask, image_rotary_emb, **kwargs
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
        guidance_embeds: bool = True,
    ):
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
            guidance_embeds=guidance_embeds,
        )

        for block in self.transformer_blocks:
            block.attn.processor = xFuserFlux2AttnProcessor()
        for block in self.single_transformer_blocks:
            block.attn.processor = xFuserFlux2ParallelSelfAttnProcessor()

    def _pad_to_sp_divisible(
        self, tensor: torch.Tensor, padding_length: int, dim: int
    ) -> torch.Tensor:
        padding = torch.zeros(
            *tensor.shape[:dim],
            padding_length,
            *tensor.shape[dim + 1 :],
            dtype=tensor.dtype,
            device=tensor.device,
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
        padding_length = (
            sp_world_size - (sequence_length % sp_world_size)
        ) % sp_world_size
        if padding_length > 0:
            hidden_states = self._pad_to_sp_divisible(
                hidden_states, padding_length, dim=1
            )
            img_ids = self._pad_to_sp_divisible(img_ids, padding_length, dim=1)

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
        encoder_hidden_states = torch.chunk(
            encoder_hidden_states, get_sequence_parallel_world_size(), dim=-2
        )[get_sequence_parallel_rank()]
        img_ids = torch.chunk(img_ids, get_sequence_parallel_world_size(), dim=-2)[
            get_sequence_parallel_rank()
        ]
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


@xFuserTransformerWrappersRegister.register(Flux2Transformer2DModel)
class xFuserFlux2Transformer2DModelWrapper(xFuserTransformerBaseWrapper):
    """PipeFusion-enabled wrapper for Flux2Transformer2DModel.

    Registered so that xFuserPipelineBaseWrapper picks it up when the pipefusion
    pipeline is built. Splits transformer blocks across pipeline stages and gates
    the embedders / output layers to the first / last stage. Attention processors
    carry the stale-KV logic for patch-level pipeline parallelism.
    """

    def __init__(self, transformer: Flux2Transformer2DModel):
        super().__init__(
            transformer=transformer,
            submodule_name_to_wrap=["attn"],
            transformer_blocks_name=["transformer_blocks", "single_transformer_blocks"],
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        *args,
        timestep: torch.LongTensor = None,
        img_ids: torch.Tensor = None,
        txt_ids: torch.Tensor = None,
        guidance: torch.Tensor = None,
        joint_attention_kwargs: Optional[dict] = None,
        return_dict: bool = True,
        **kwargs,
    ):
        # 1. timestep embedding + modulation (cheap; depends only on timestep)
        timestep = timestep.to(hidden_states.dtype) * 1000
        if guidance is not None:
            guidance = guidance.to(hidden_states.dtype) * 1000
        temb = self.time_guidance_embed(timestep, guidance)
        double_stream_mod_img = self.double_stream_modulation_img(temb)
        double_stream_mod_txt = self.double_stream_modulation_txt(temb)
        single_stream_mod = self.single_stream_modulation(temb)

        # 2. input projection (first stage only); other stages receive already
        #    embedded streams from the previous stage via P2P.
        if is_pipeline_first_stage():
            hidden_states = self.x_embedder(hidden_states)
            encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        # 3. RoPE (computed on every stage; ids are passed to all stages)
        if img_ids.ndim == 3:
            img_ids = img_ids[0]
        if txt_ids.ndim == 3:
            txt_ids = txt_ids[0]
        num_txt_tokens = txt_ids.shape[0]
        image_rotary_emb = self.pos_embed(img_ids)
        text_rotary_emb = self.pos_embed(txt_ids)
        concat_rotary_emb = (
            torch.cat([text_rotary_emb[0], image_rotary_emb[0]], dim=0),
            torch.cat([text_rotary_emb[1], image_rotary_emb[1]], dim=0),
        )

        if joint_attention_kwargs is None:
            joint_attention_kwargs = {}
        # single blocks run on [text, image] and need num_txt to split stale-KV
        single_kwargs = {**joint_attention_kwargs, "num_txt_tokens": num_txt_tokens}

        # 4. double stream blocks (this stage's subset; may be empty)
        for block in self.transformer_blocks:
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb_mod_img=double_stream_mod_img,
                temb_mod_txt=double_stream_mod_txt,
                image_rotary_emb=concat_rotary_emb,
                joint_attention_kwargs=joint_attention_kwargs,
            )

        # 5. single stream blocks operate on concatenated [text, image].
        #    Flux2 single blocks run on the concatenated stream (unlike Flux1
        #    which keeps streams separate). Cat before the single loop.
        if len(self.single_transformer_blocks) > 0:
            if encoder_hidden_states is not None:
                hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
            for block in self.single_transformer_blocks:
                hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=None,
                    temb_mod=single_stream_mod,
                    image_rotary_emb=concat_rotary_emb,
                    joint_attention_kwargs=single_kwargs,
                )

        # 6. output: last stage unpatchifies, others pass streams to next stage.
        #    The "sample" is a (noise_pred, encoder_hidden_states) tuple, mirroring
        #    xFuserFluxTransformer2DWrapper so _backbone_forward can unpack it.
        #    Use is_pipeline_last_stage() instead of after_flags, because when
        #    pp_degree > num_blocks, the last block may land on a non-last stage
        #    (after_flags=True) while the true last stage is empty (after_flags=False).
        is_last = is_pipeline_last_stage()
        if is_last:
            # Last stage: ensure [text, image] concatenated, then slice text, unpatchify.
            # Empty stage (no blocks ran) may have separate encoder/hidden that need cat.
            if (
                len(self.single_transformer_blocks) == 0
                and encoder_hidden_states is not None
            ):
                hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
            hidden_states = hidden_states[:, num_txt_tokens:, ...]
            hidden_states = self.norm_out(hidden_states, temb)
            output = (self.proj_out(hidden_states), None)
        else:
            # Non-last stage: split the concatenated stream back into (text,
            # image) so P2P always carries separate streams (matches the Flux1
            # _backbone_forward / pipeline contract). Stages that only ran double
            # blocks already have separate encoder/hidden and skip the split.
            if len(self.single_transformer_blocks) > 0:
                encoder_hidden_states = hidden_states[:, :num_txt_tokens, :]
                hidden_states = hidden_states[:, num_txt_tokens:, :]
            output = (hidden_states, encoder_hidden_states)

        if not return_dict:
            return (output,)
        from diffusers.models.transformers.transformer_flux2 import (
            Flux2Transformer2DModelOutput,
        )

        return Flux2Transformer2DModelOutput(sample=output)
