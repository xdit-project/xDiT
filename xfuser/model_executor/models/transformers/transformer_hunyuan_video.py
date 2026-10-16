import torch
from typing import Optional, Dict, Any, Union
from diffusers.models.transformers.transformer_hunyuan_video import HunyuanVideoTransformer3DModel
from diffusers.models.embeddings import apply_rotary_emb
from diffusers.models.attention_processor import Attention
from diffusers.models.modeling_outputs import Transformer2DModelOutput

from xfuser.core.attention import registry as attention_registry
from xfuser.core.attention.spec import Sparsity
from xfuser.model_executor.layers.usp import USP
from xfuser.model_executor.models.transformers.transformers_utils import chunk_and_pad_sequence, gather_and_unpad
from xfuser.core.distributed import (
    get_classifier_free_guidance_world_size,
    get_classifier_free_guidance_rank,
    get_pipeline_parallel_world_size,
    get_sequence_parallel_world_size,
    get_sequence_parallel_rank,
    get_runtime_state,
    get_cfg_group,
)


def _refuse_padding_under_vsa(pad_amount: int, backend) -> None:
    spec = attention_registry.find(backend)
    if pad_amount and spec is not None and spec.sparsity == Sparsity.VSA:
        # USP trims the padded video keys, joins the text after the trim and leaves valid_kv_len in
        # the kwargs, so VSA's dense fallback crops K to it a second time and drops the text keys.
        # Delete this once USP clears valid_kv_len after the trim when joint tensors are present.
        raise NotImplementedError(
            f"HunyuanVideo pads the sequence by {pad_amount} token(s) to split it across the sequence-parallel "
            f"ranks, which --attention_backend {backend.name} does not support. Use another attention backend "
            "or a sequence-parallel degree that divides the video token count."
        )


class xFuserHunyuanVideoAttnProcessor:
    def __init__(self, attention_kwargs: Optional[Dict[str, Any]] = None):
        # Shared with the transformer, which sets ``valid_kv_len`` to the real
        # video token count when it pads the sequence to split evenly.
        self.attention_kwargs = attention_kwargs

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        batch_size, _, _ = hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape

        if attn.add_q_proj is None and encoder_hidden_states is not None:
            hidden_states = torch.cat([hidden_states, encoder_hidden_states], dim=1)

        # 1. QKV projections
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        query = query.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        key = key.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        value = value.unflatten(2, (attn.heads, -1)).transpose(1, 2)

        # 2. QK normalization
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # 3. Rotational positional embeddings applied to latent stream
        if image_rotary_emb is not None:
            if attn.add_q_proj is None and encoder_hidden_states is not None:
                query = torch.cat(
                    [
                        apply_rotary_emb(
                            query[:, :, : -encoder_hidden_states.shape[1]],
                            image_rotary_emb,
                        ),
                        query[:, :, -encoder_hidden_states.shape[1] :],
                    ],
                    dim=2,
                )
                key = torch.cat(
                    [
                        apply_rotary_emb(
                            key[:, :, : -encoder_hidden_states.shape[1]],
                            image_rotary_emb,
                        ),
                        key[:, :, -encoder_hidden_states.shape[1] :],
                    ],
                    dim=2,
                )
            else:
                query = apply_rotary_emb(query, image_rotary_emb)
                key = apply_rotary_emb(key, image_rotary_emb)

        # 4. Encoder condition QKV projection and normalization
        if attn.add_q_proj is not None and encoder_hidden_states is not None:
            encoder_query = attn.add_q_proj(encoder_hidden_states)
            encoder_key = attn.add_k_proj(encoder_hidden_states)
            encoder_value = attn.add_v_proj(encoder_hidden_states)

            encoder_query = encoder_query.unflatten(2, (attn.heads, -1)).transpose(1, 2)
            encoder_key = encoder_key.unflatten(2, (attn.heads, -1)).transpose(1, 2)
            encoder_value = encoder_value.unflatten(2, (attn.heads, -1)).transpose(1, 2)

            if attn.norm_added_q is not None:
                encoder_query = attn.norm_added_q(encoder_query)
            if attn.norm_added_k is not None:
                encoder_key = attn.norm_added_k(encoder_key)

            query = torch.cat([query, encoder_query], dim=2)
            key = torch.cat([key, encoder_key], dim=2)
            value = torch.cat([value, encoder_value], dim=2)

        if encoder_hidden_states is not None:
            num_encoder_hidden_states_tokens = encoder_hidden_states.shape[1]
            num_query_tokens = query.shape[2] - num_encoder_hidden_states_tokens
        else:
            num_encoder_hidden_states_tokens = get_runtime_state().max_condition_sequence_length
            num_query_tokens = query.shape[2] - num_encoder_hidden_states_tokens

        #! ---------------------------------------- ATTENTION ----------------------------------------
        if get_pipeline_parallel_world_size() == 1 and get_runtime_state().split_text_embed_in_sp:
            # The text is only split when the video tokens are not padded (see
            # forward), so valid_kv_len is None here; were it set, a trailing trim
            # would also cut into the per-rank text chunks inside the sequence.
            hidden_states = USP(query, key, value, dropout_p=0.0, is_causal=False)
            hidden_states = hidden_states.transpose(1, 2).flatten(2, 3)
        elif get_sequence_parallel_world_size() > 1:
            if get_runtime_state().split_text_embed_in_sp:
                encoder_query = None
                encoder_key = None
                encoder_value = None
            else:
                query, encoder_query = query.split([num_query_tokens, num_encoder_hidden_states_tokens], dim=2)
                key, encoder_key = key.split([num_query_tokens, num_encoder_hidden_states_tokens], dim=2)
                value, encoder_value = value.split([num_query_tokens, num_encoder_hidden_states_tokens], dim=2)

            hidden_states = USP(
                query,
                key,
                value,
                dropout_p=0.0,
                is_causal=False,
                joint_query=encoder_query,
                joint_key=encoder_key,
                joint_value=encoder_value,
                joint_strategy="rear",
                # Zero-padded video keys trail the gathered video K/V (the text
                # is appended after USP's trim), so keep only the real ones.
                attention_kwargs=self.attention_kwargs,
            )

            hidden_states = hidden_states.transpose(1, 2)
            hidden_states = hidden_states.flatten(2, 3)
        else:
            hidden_states = USP(query, key, value, dropout_p=0.0, is_causal=False)
            hidden_states = hidden_states.transpose(1, 2).flatten(2, 3)

        hidden_states = hidden_states.to(query.dtype)

        # 6. Output projection
        if encoder_hidden_states is not None:
            hidden_states, encoder_hidden_states = (
                hidden_states[:, : -encoder_hidden_states.shape[1]],
                hidden_states[:, -encoder_hidden_states.shape[1] :],
            )

            if getattr(attn, "to_out", None) is not None:
                hidden_states = attn.to_out[0](hidden_states)
                hidden_states = attn.to_out[1](hidden_states)

            if getattr(attn, "to_add_out", None) is not None:
                encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

        return hidden_states, encoder_hidden_states


class xFuserHunyuanVideoTransformer3DWrapper(HunyuanVideoTransformer3DModel):
    def install_xdit_attention_processors(self):
        # The key is always present because torch.compile guards on the key set.
        self._usp_attention_kwargs = {"valid_kv_len": None}
        for block in self.transformer_blocks + self.single_transformer_blocks:
            block.attn.processor = xFuserHunyuanVideoAttnProcessor(attention_kwargs=self._usp_attention_kwargs)
        return self

    @classmethod
    def from_config(cls, config, return_unused_kwargs=False, **kwargs):
        """Keep the parent config signature, then install xDiT processors."""
        result = super().from_config(
            config,
            return_unused_kwargs=return_unused_kwargs,
            **kwargs,
        )
        model = result[0] if return_unused_kwargs else result
        model.install_xdit_attention_processors()
        return result

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        pooled_projections: torch.Tensor,
        guidance: torch.Tensor = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            attention_kwargs.pop("scale", None)

        get_runtime_state().increment_step_counter()

        batch_size, num_channels, num_frames, height, width = hidden_states.shape

        assert batch_size % get_classifier_free_guidance_world_size() == 0, (
            f"Cannot split dim 0 of hidden_states ({batch_size}) into {get_classifier_free_guidance_world_size()} parts."
        )

        p, p_t = self.config.patch_size, self.config.patch_size_t
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p
        post_patch_width = width // p

        # 1. RoPE
        image_rotary_emb = self.rope(hidden_states)

        # 2. Conditional embeddings
        temb, _ = self.time_text_embed(timestep=timestep, pooled_projection=pooled_projections, guidance=guidance)
        hidden_states = self.x_embedder(hidden_states)
        encoder_hidden_states = self.context_embedder(encoder_hidden_states, timestep, encoder_attention_mask)

        hidden_states = hidden_states.reshape(
            batch_size, post_patch_num_frames, post_patch_height, post_patch_width, -1
        )
        hidden_states = hidden_states.flatten(1, 3)

        hidden_states = torch.chunk(hidden_states, get_classifier_free_guidance_world_size(), dim=0)[
            get_classifier_free_guidance_rank()
        ]
        sp_world_size = get_sequence_parallel_world_size()
        sp_world_rank = get_sequence_parallel_rank()
        video_seq_len = hidden_states.shape[-2]
        # Pad the video tokens so every rank gets an equal shard; the padded
        # keys are excluded from attention and the rows dropped after gather.
        pad_amount = (sp_world_size - video_seq_len % sp_world_size) % sp_world_size
        if pad_amount:
            _refuse_padding_under_vsa(pad_amount, get_runtime_state().attention_backend)
        hidden_states = chunk_and_pad_sequence(hidden_states, sp_world_rank, sp_world_size, pad_amount, dim=-2)
        self._usp_attention_kwargs["valid_kv_len"] = video_seq_len if pad_amount else None

        encoder_attention_mask = encoder_attention_mask.to(torch.bool).any(dim=0)
        encoder_hidden_states = encoder_hidden_states[:, encoder_attention_mask, :]
        # Padded video keys have to form one trailing suffix for USP to trim, so a
        # padded sequence sends the text through the joint "rear" path unsplit
        # instead of splicing it into the sequence.
        get_runtime_state().split_text_embed_in_sp = (
            not pad_amount and encoder_hidden_states.shape[-2] % sp_world_size == 0
        )

        encoder_hidden_states = torch.chunk(encoder_hidden_states, get_classifier_free_guidance_world_size(), dim=0)[
            get_classifier_free_guidance_rank()
        ]
        if get_runtime_state().split_text_embed_in_sp:
            encoder_hidden_states = torch.chunk(encoder_hidden_states, sp_world_size, dim=-2)[sp_world_rank]

        freqs_cos, freqs_sin = image_rotary_emb

        freqs_cos = chunk_and_pad_sequence(freqs_cos, sp_world_rank, sp_world_size, pad_amount, dim=0)
        freqs_sin = chunk_and_pad_sequence(freqs_sin, sp_world_rank, sp_world_size, pad_amount, dim=0)
        image_rotary_emb = (freqs_cos, freqs_sin)

        # 4. Transformer blocks
        if torch.is_grad_enabled() and self.gradient_checkpointing:

            def create_custom_forward(module, return_dict=None):
                def custom_forward(*inputs):
                    if return_dict is not None:
                        return module(*inputs, return_dict=return_dict)
                    else:
                        return module(*inputs)

                return custom_forward

            ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False}

            for block in self.transformer_blocks:
                hidden_states, encoder_hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    encoder_hidden_states,
                    temb,
                    None,
                    image_rotary_emb,
                    **ckpt_kwargs,
                )

            for block in self.single_transformer_blocks:
                hidden_states, encoder_hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    encoder_hidden_states,
                    temb,
                    None,
                    image_rotary_emb,
                    **ckpt_kwargs,
                )

        else:
            for block in self.transformer_blocks:
                hidden_states, encoder_hidden_states = block(
                    hidden_states, encoder_hidden_states, temb, None, image_rotary_emb
                )

            for block in self.single_transformer_blocks:
                hidden_states, encoder_hidden_states = block(
                    hidden_states, encoder_hidden_states, temb, None, image_rotary_emb
                )

        # 5. Output projection
        hidden_states = self.norm_out(hidden_states, temb)
        hidden_states = self.proj_out(hidden_states)

        hidden_states = gather_and_unpad(hidden_states, pad_amount, dim=-2)
        hidden_states = get_cfg_group().all_gather(hidden_states, dim=0)

        hidden_states = hidden_states.reshape(
            batch_size, post_patch_num_frames, post_patch_height, post_patch_width, -1, p_t, p, p
        )

        hidden_states = hidden_states.permute(0, 4, 1, 5, 2, 6, 3, 7)
        hidden_states = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

        if not return_dict:
            return (hidden_states,)

        return Transformer2DModelOutput(sample=hidden_states)
