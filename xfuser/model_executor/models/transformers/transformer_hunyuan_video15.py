import torch
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple, Union
from diffusers.models.transformers.transformer_hunyuan_video15 import HunyuanVideo15Transformer3DModel
from diffusers.models.embeddings import apply_rotary_emb
from diffusers.models.attention_processor import Attention
from diffusers.models.modeling_outputs import Transformer2DModelOutput

from xfuser.model_executor.layers.attention_mask import MaskMetaCache
from xfuser.model_executor.layers.usp import USP, key_padding_attention_kwargs, key_padding_mask
from xfuser.model_executor.models.transformers.transformers_utils import (
    chunk_and_pad_sequence,
    gather_and_unpad,
    local_sp_pad_rows,
    prefix_key_mask,
    sp_pad_spans_ranks,
    text_image_key_mask,
    zero_sp_pad_rows_hook,
)
from xfuser.core.attention import registry as attention_registry
from xfuser.core.attention.spec import Sparsity
from xfuser.core.distributed import (
    get_runtime_state,
    get_sequence_parallel_world_size,
    get_sequence_parallel_rank,
)

_SSTA_BACKENDS = attention_registry.types_where(sparsity=Sparsity.SSTA)


def _uses_sparse_layout(attention_kwargs) -> bool:
    """Whether this step's attention takes SSTA's [image, text] layout.

    A sparse checkpoint's dense steps (hybrid schedule) select a dense backend and do not.
    """
    return (
        attention_kwargs.get("attn_sparse_type") in ("ssta", "sta", "moba")
        and get_runtime_state().attention_backend in _SSTA_BACKENDS
    )


def _prompt_image_key_mask(
    _key, prompt_lengths, prompt_tokens, image_tokens, sp_pad, sp_world_size, joint_strategy, device
) -> torch.Tensor:
    """``text_image_key_mask`` for prompts whose valid tokens come first, ``prompt_lengths`` of them per row."""
    prompt_mask = prefix_key_mask(None, prompt_lengths, prompt_tokens, device)
    return text_image_key_mask(prompt_mask, image_tokens, sp_pad, sp_world_size, joint_strategy)


class xFuserHunyuanVideo15AttnProcessor:
    def __init__(self, attention_kwargs: Optional[Dict[str, Any]] = None):
        self.attention_kwargs = attention_kwargs if attention_kwargs is not None else {}

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # 1. QKV projections
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        query = query.unflatten(2, (attn.heads, -1))
        key = key.unflatten(2, (attn.heads, -1))
        value = value.unflatten(2, (attn.heads, -1))

        # 2. QK normalization
        query = attn.norm_q(query)
        key = attn.norm_k(key)

        # 3. Rotational positional embeddings applied to latent stream
        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
            key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)

        # 4. Encoder condition QKV projection and normalization
        if encoder_hidden_states is not None:
            encoder_query = attn.add_q_proj(encoder_hidden_states)
            encoder_key = attn.add_k_proj(encoder_hidden_states)
            encoder_value = attn.add_v_proj(encoder_hidden_states)

            encoder_query = encoder_query.unflatten(2, (attn.heads, -1))
            encoder_key = encoder_key.unflatten(2, (attn.heads, -1))
            encoder_value = encoder_value.unflatten(2, (attn.heads, -1))

            if attn.norm_added_q is not None:
                encoder_query = attn.norm_added_q(encoder_query)
            if attn.norm_added_k is not None:
                encoder_key = attn.norm_added_k(encoder_key)

        if encoder_hidden_states is not None:
            num_encoder_hidden_states_tokens = encoder_hidden_states.shape[1]
        else:
            num_encoder_hidden_states_tokens = get_runtime_state().max_condition_sequence_length

        split_text = get_runtime_state().split_text_embed_in_sp
        sparse = _uses_sparse_layout(self.attention_kwargs)
        # Dense attention has no spatial layout constraint. Putting the text
        # first keeps the image's SP padding at the end of the gathered keys.
        # SSTA needs the symmetric [image, text] layout used by its tiling code.
        text_first = split_text and not sparse
        if encoder_hidden_states is not None and split_text:
            if text_first:
                query = torch.cat([encoder_query, query], dim=1)
                key = torch.cat([encoder_key, key], dim=1)
                value = torch.cat([encoder_value, value], dim=1)
            else:
                query = torch.cat([query, encoder_query], dim=1)
                key = torch.cat([key, encoder_key], dim=1)
                value = torch.cat([value, encoder_value], dim=1)

        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        #! ---------------------------------------- ATTENTION ----------------------------------------
        self.attention_kwargs["encoder_sequence_length"] = num_encoder_hidden_states_tokens
        # attention_mask is the wrapper's key-padding mask, None when no key is masked.
        attention_kwargs = key_padding_attention_kwargs(self.attention_kwargs, attention_mask)
        if encoder_hidden_states is not None and not split_text:
            hidden_states = USP(
                query,
                key,
                value,
                dropout_p=0.0,
                is_causal=False,
                joint_query=encoder_query.transpose(1, 2),
                joint_key=encoder_key.transpose(1, 2),
                joint_value=encoder_value.transpose(1, 2),
                joint_strategy="rear",
                attention_kwargs=attention_kwargs,
            )
        else:
            hidden_states = USP(
                query,
                key,
                value,
                dropout_p=0.0,
                is_causal=False,
                attention_kwargs=attention_kwargs,
            )

        # Transpose back to original shape
        hidden_states = hidden_states.transpose(1, 2)

        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.to(query.dtype)

        # 6. Output projection
        if encoder_hidden_states is not None:
            if text_first:
                encoder_hidden_states, hidden_states = hidden_states.split(
                    [num_encoder_hidden_states_tokens, hidden_states.shape[1] - num_encoder_hidden_states_tokens], dim=1
                )
            else:
                hidden_states, encoder_hidden_states = hidden_states.split(
                    [hidden_states.shape[1] - num_encoder_hidden_states_tokens, num_encoder_hidden_states_tokens], dim=1
                )

            if getattr(attn, "to_out", None) is not None:
                hidden_states = attn.to_out[0](hidden_states)
                hidden_states = attn.to_out[1](hidden_states)

            if getattr(attn, "to_add_out", None) is not None:
                encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

        return hidden_states, encoder_hidden_states


class xFuserHunyuanVideo15Transformer3DWrapper(HunyuanVideo15Transformer3DModel):
    def __init__(
        self,
        in_channels: int = 65,
        out_channels: int = 32,
        num_attention_heads: int = 16,
        attention_head_dim: int = 128,
        num_layers: int = 54,
        num_refiner_layers: int = 2,
        mlp_ratio: float = 4.0,
        patch_size: int | list = 1,
        patch_size_t: int = 1,
        qk_norm: str = "rms_norm",
        text_embed_dim: int = 3584,
        text_embed_2_dim: int = 1472,
        image_embed_dim: int = 1152,
        rope_theta: float = 256.0,
        rope_axes_dim: Tuple[int, ...] = (16, 56, 56),
        # YiYi Notes: config based on target_size_config https://github.com/yiyixuxu/hy15/blob/main/hyvideo/pipelines/hunyuan_video_pipeline.py#L205
        target_size: int = 640,  # did not name sample_size since it is in pixel spaces
        task_type: str = "i2v",
        use_meanflow: bool = False,
        attention_kwargs: Optional[Dict[str, Any]] = None,
    ):
        if isinstance(patch_size, list):
            patch_size = 1
        if qk_norm == "rms":
            qk_norm = "rms_norm"
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            num_layers=num_layers,
            num_refiner_layers=num_refiner_layers,
            mlp_ratio=mlp_ratio,
            patch_size=patch_size,
            patch_size_t=patch_size_t,
            qk_norm=qk_norm,
            text_embed_dim=text_embed_dim,
            text_embed_2_dim=text_embed_2_dim,
            image_embed_dim=image_embed_dim,
            rope_theta=rope_theta,
            rope_axes_dim=rope_axes_dim,
            target_size=target_size,
            task_type=task_type,
            use_meanflow=use_meanflow,
        )
        self._self_attn_kwargs = attention_kwargs if attention_kwargs is not None else {}
        self._self_attn_kwargs.setdefault("valid_kv_len", None)
        self._sp_pad_image = {"local_rows": 0}
        self._sp_pad_text = {"local_rows": 0}
        for block in self.transformer_blocks:
            block.attn.processor = xFuserHunyuanVideo15AttnProcessor(attention_kwargs=self._self_attn_kwargs)
            # Diffusers' HunyuanVideo15TransformerBlock returns (image, text).
            block.register_forward_hook(zero_sp_pad_rows_hook(self._sp_pad_image, index=0))
            block.register_forward_hook(zero_sp_pad_rows_hook(self._sp_pad_text, index=1))
        self._key_padding_cache = MaskMetaCache()

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        timestep_r: Optional[torch.LongTensor] = None,
        encoder_hidden_states_2: Optional[torch.Tensor] = None,
        encoder_attention_mask_2: Optional[torch.Tensor] = None,
        image_embeds: Optional[torch.Tensor] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
    ) -> Union[Tuple[torch.Tensor], Transformer2DModelOutput]:
        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.config.patch_size_t, self.config.patch_size, self.config.patch_size
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p_h
        post_patch_width = width // p_w
        # The step's scheduled backend is selected before this forward. Its
        # dense steps need the same trailing-pad layout as a dense checkpoint.
        sparse_layout = _uses_sparse_layout(self._self_attn_kwargs)
        if sparse_layout:
            self._self_attn_kwargs["thw"] = (
                post_patch_num_frames,
                post_patch_height,
                post_patch_width,
            )  # Should modify reference in xFuserHunyuanVideo15AttnProcessor.

        sp_world_rank = get_sequence_parallel_rank()
        sp_world_size = get_sequence_parallel_world_size()

        # 1. RoPE
        image_rotary_emb = self.rope(hidden_states)

        # 2. Conditional embeddings
        temb = self.time_embed(timestep, timestep_r=timestep_r)

        hidden_states = self.x_embedder(hidden_states)

        # qwen text embedding
        encoder_hidden_states = self.context_embedder(encoder_hidden_states, timestep, encoder_attention_mask)

        encoder_hidden_states_cond_emb = self.cond_type_embed(
            torch.zeros_like(encoder_hidden_states[:, :, 0], dtype=torch.long)
        )
        encoder_hidden_states = encoder_hidden_states + encoder_hidden_states_cond_emb

        # byt5 text embedding
        encoder_hidden_states_2 = self.context_embedder_2(encoder_hidden_states_2)

        encoder_hidden_states_2_cond_emb = self.cond_type_embed(
            torch.ones_like(encoder_hidden_states_2[:, :, 0], dtype=torch.long)
        )
        encoder_hidden_states_2 = encoder_hidden_states_2 + encoder_hidden_states_2_cond_emb

        # image embed
        encoder_hidden_states_3 = self.image_embedder(image_embeds)
        is_t2v = torch.all(image_embeds == 0)
        if is_t2v:
            encoder_hidden_states_3 = encoder_hidden_states_3 * 0.0
            encoder_attention_mask_3 = torch.zeros(
                (batch_size, encoder_hidden_states_3.shape[1]),
                dtype=encoder_attention_mask.dtype,
                device=encoder_attention_mask.device,
            )
        else:
            encoder_attention_mask_3 = torch.ones(
                (batch_size, encoder_hidden_states_3.shape[1]),
                dtype=encoder_attention_mask.dtype,
                device=encoder_attention_mask.device,
            )
        encoder_hidden_states_3_cond_emb = self.cond_type_embed(
            2
            * torch.ones_like(
                encoder_hidden_states_3[:, :, 0],
                dtype=torch.long,
            )
        )
        encoder_hidden_states_3 = encoder_hidden_states_3 + encoder_hidden_states_3_cond_emb

        # reorder and combine text tokens: combine valid tokens first, then padding
        encoder_attention_mask = encoder_attention_mask.bool()
        encoder_attention_mask_2 = encoder_attention_mask_2.bool()
        encoder_attention_mask_3 = encoder_attention_mask_3.bool()
        new_encoder_hidden_states = []
        new_encoder_attention_mask = []

        for text, text_mask, text_2, text_mask_2, image, image_mask in zip(
            encoder_hidden_states,
            encoder_attention_mask,
            encoder_hidden_states_2,
            encoder_attention_mask_2,
            encoder_hidden_states_3,
            encoder_attention_mask_3,
        ):
            # Concatenate: [valid_image, valid_byt5, valid_mllm, invalid_image, invalid_byt5, invalid_mllm]
            new_encoder_hidden_states.append(
                torch.cat(
                    [
                        image[image_mask],  # valid image
                        text_2[text_mask_2],  # valid byt5
                        text[text_mask],  # valid mllm
                        image[~image_mask],  # invalid image
                        torch.zeros_like(text_2[~text_mask_2]),  # invalid byt5 (zeroed)
                        torch.zeros_like(text[~text_mask]),  # invalid mllm (zeroed)
                    ],
                    dim=0,
                )
            )

            # Apply same reordering to attention masks
            new_encoder_attention_mask.append(
                torch.cat(
                    [
                        image_mask[image_mask],
                        text_mask_2[text_mask_2],
                        text_mask[text_mask],
                        image_mask[~image_mask],
                        text_mask_2[~text_mask_2],
                        text_mask[~text_mask],
                    ],
                    dim=0,
                )
            )

        encoder_hidden_states = torch.stack(new_encoder_hidden_states)
        encoder_attention_mask = torch.stack(new_encoder_attention_mask)

        # sequence parallel
        hidden_states_pad_amount = (sp_world_size - (hidden_states.shape[1] % sp_world_size)) % sp_world_size
        image_sequence_length = hidden_states.shape[1]
        hidden_states = chunk_and_pad_sequence(
            hidden_states, sp_world_rank, sp_world_size, hidden_states_pad_amount, dim=1
        )
        cos, sin = image_rotary_emb
        cos = chunk_and_pad_sequence(cos, sp_world_rank, sp_world_size, hidden_states_pad_amount, dim=0)
        sin = chunk_and_pad_sequence(sin, sp_world_rank, sp_world_size, hidden_states_pad_amount, dim=0)
        image_rotary_emb = (cos, sin)
        self._sp_pad_image["local_rows"] = local_sp_pad_rows(
            image_sequence_length, hidden_states.shape[1], sp_world_rank
        )

        any_valid = encoder_attention_mask.to(torch.bool).any(dim=0)
        encoder_hidden_states = encoder_hidden_states[:, any_valid, :]
        encoder_attention_mask = encoder_attention_mask.to(torch.bool)[:, any_valid]

        encoder_sequence_length = encoder_hidden_states.shape[1]
        enc_pad = 0
        # If a tiny image has pad rows on multiple ranks, interleaving each
        # rank's [text, image] would put some pads between real text keys.
        # Keep that prompt joint so USP's global trailing-length trim applies.
        image_padding_spans_ranks = sp_pad_spans_ranks(image_sequence_length, sp_world_size)
        if encoder_sequence_length % sp_world_size != 0 or (not sparse_layout and image_padding_spans_ranks):
            if sparse_layout:
                # Sparse models requires symmetric [image, text] layout in Q/K/V.
                # Pad text to be divisible by sp_world_size so it can be chunked,
                # ensuring split_text_embed_in_sp=True and avoiding the asymmetric
                # joint_strategy path which is not compatible.
                enc_pad = (-encoder_sequence_length) % sp_world_size
                encoder_hidden_states = F.pad(encoder_hidden_states, (0, 0, 0, enc_pad))
                encoder_attention_mask = F.pad(encoder_attention_mask, (0, enc_pad), value=False)
                get_runtime_state().split_text_embed_in_sp = True
                encoder_hidden_states = torch.chunk(encoder_hidden_states, sp_world_size, dim=1)[sp_world_rank]
            else:
                get_runtime_state().split_text_embed_in_sp = False
        else:
            get_runtime_state().split_text_embed_in_sp = True
            encoder_hidden_states = torch.chunk(encoder_hidden_states, sp_world_size, dim=1)[sp_world_rank]
        # SSTA gathers the text after the image, so this padding ends the
        # gathered text; SSTA drops it, as it does the image padding.
        self._self_attn_kwargs["encoder_sp_padding"] = enc_pad
        self._sp_pad_text["local_rows"] = (
            local_sp_pad_rows(encoder_sequence_length, encoder_hidden_states.shape[1], sp_world_rank)
            if get_runtime_state().split_text_embed_in_sp
            else 0
        )
        # USP expects a global length, including sharded text in the dense
        # [text, image] path, but excluding the separately joined prompt.
        valid_kv_len = image_sequence_length
        if get_runtime_state().split_text_embed_in_sp:
            valid_kv_len += encoder_sequence_length
        self._self_attn_kwargs["valid_kv_len"] = valid_kv_len if hidden_states_pad_amount or enc_pad else None

        # SSTA applies the per-sample text mask itself. A dense layout gets a key-padding mask over the
        # keys in the order the attention sees them, and only when prompts of different lengths leave
        # padded text keys in the batch (a single prompt's padding was dropped above). The reordering
        # above put each prompt's valid tokens first, so its length describes its mask: read on the host,
        # it keys the mask cache, as the mask itself is rebuilt every step.
        self._self_attn_kwargs["text_mask"] = encoder_attention_mask if sparse_layout else None
        attention_mask = None
        if not sparse_layout and batch_size > 1:
            prompt_lengths = tuple(encoder_attention_mask.sum(dim=1).tolist())
            if min(prompt_lengths) < encoder_sequence_length:
                attention_mask = key_padding_mask(
                    self._key_padding_cache,
                    None,
                    _prompt_image_key_mask,
                    prompt_lengths,
                    encoder_sequence_length,
                    image_sequence_length,
                    hidden_states_pad_amount,
                    sp_world_size,
                    None if get_runtime_state().split_text_embed_in_sp else "rear",
                    encoder_attention_mask.device,
                )

        # 4. Transformer blocks
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            for block in self.transformer_blocks:
                hidden_states, encoder_hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    temb,
                    attention_mask,
                    image_rotary_emb,
                )

        else:
            for block in self.transformer_blocks:
                hidden_states, encoder_hidden_states = block(
                    hidden_states,
                    encoder_hidden_states,
                    temb,
                    attention_mask,
                    image_rotary_emb,
                )

        # 5. Output projection
        hidden_states = self.norm_out(hidden_states, temb)
        hidden_states = self.proj_out(hidden_states)

        hidden_states = gather_and_unpad(hidden_states, hidden_states_pad_amount, dim=1)

        hidden_states = hidden_states.reshape(
            batch_size, post_patch_num_frames, post_patch_height, post_patch_width, -1, p_t, p_h, p_w
        )
        hidden_states = hidden_states.permute(0, 4, 1, 5, 2, 6, 3, 7)
        hidden_states = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

        if not return_dict:
            return (hidden_states,)

        return Transformer2DModelOutput(sample=hidden_states)
