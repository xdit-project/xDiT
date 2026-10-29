import torch
from math import prod
import numpy as np
from typing import Optional, Dict, Any, Union, Tuple, List
from diffusers.models.transformers.transformer_2d import Transformer2DModelOutput
from diffusers.models.transformers.transformer_qwenimage import (
    QwenImageTransformer2DModel,
    apply_rotary_emb_qwen,
    compute_text_seq_len_from_mask,
)
from diffusers.utils import (
    USE_PEFT_BACKEND,
    scale_lora_layers,
    unscale_lora_layers,
)
from xfuser.model_executor.layers.attention_mask import MaskMetaCache
from xfuser.model_executor.layers.usp import USP, key_padding_attention_kwargs, key_padding_mask
from xfuser.core.distributed.fp8_comms import register_fp8_comms_eligible_modules
from xfuser.core.distributed import (
    get_runtime_state,
    get_sequence_parallel_rank,
    get_sequence_parallel_world_size,
)
from xfuser.model_executor.models.transformers.transformers_utils import (
    chunk_and_pad_sequence,
    gather_and_unpad,
    local_sp_pad_rows,
    sp_pad_spans_ranks,
    text_image_key_mask,
    zero_sp_pad_rows_hook,
)

from xfuser.model_executor.layers.fused_qk_rope_flydsl import (
    flydsl_fused_qk_norm_rope,
    _HAS_FLYDSL,
)


def _qwen_cos_sin(freqs: torch.Tensor):
    """Real ``(cos, sin)`` tables equivalent to the complex Qwen ``freqs_cis``.

    ``apply_rotary_emb_qwen(x, freqs, use_real=False)`` treats each interleaved
    channel pair ``(2k, 2k+1)`` of ``x`` as a complex number and multiplies by
    ``freqs`` -- the interleaved (GPT-J) rotation the fused kernel implements
    with ``cos = repeat_interleave(Re(freqs), 2)`` and
    ``sin = repeat_interleave(Im(freqs), 2)``.
    """
    fr = torch.view_as_real(freqs.contiguous())  # [S, D/2, 2] fp32
    cos = fr[..., 0].repeat_interleave(2, dim=-1)
    sin = fr[..., 1].repeat_interleave(2, dim=-1)
    return cos, sin


class xFuserQwenDoubleStreamAttnProcessor:
    def __init__(self, attention_kwargs=None):
        self.attention_kwargs = attention_kwargs if attention_kwargs is not None else {}

    def __call__(
        self,
        attn,
        hidden_states: torch.FloatTensor,  # Image stream
        encoder_hidden_states: torch.FloatTensor = None,  # Text stream
        encoder_hidden_states_mask: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.FloatTensor:
        seq_txt = encoder_hidden_states.shape[1]

        # Compute QKV for image stream (sample projections)
        img_query = attn.to_q(hidden_states)
        img_key = attn.to_k(hidden_states)
        img_value = attn.to_v(hidden_states)

        # Compute QKV for text stream (context projections)
        txt_query = attn.add_q_proj(encoder_hidden_states)
        txt_key = attn.add_k_proj(encoder_hidden_states)
        txt_value = attn.add_v_proj(encoder_hidden_states)

        # Reshape for multi-head attention
        img_query = img_query.unflatten(-1, (attn.heads, -1))
        img_key = img_key.unflatten(-1, (attn.heads, -1))
        img_value = img_value.unflatten(-1, (attn.heads, -1))

        txt_query = txt_query.unflatten(-1, (attn.heads, -1))
        txt_key = txt_key.unflatten(-1, (attn.heads, -1))
        txt_value = txt_value.unflatten(-1, (attn.heads, -1))

        # Fuse QK-RMSNorm and RoPE into one kernel per stream when AITER/FlyDSL
        # is present. RoPE is per-token, so fusing it per stream before the
        # joint concat is identical to the unfused norm-then-rope path below.
        # Qwen's rotary emb is a complex freqs pair, so convert it to the real
        # (cos, sin) tables the kernel expects.
        if _HAS_FLYDSL and image_rotary_emb is not None:
            img_freqs, txt_freqs = image_rotary_emb
            img_query, img_key = flydsl_fused_qk_norm_rope(
                img_query, img_key, attn.norm_q, attn.norm_k, _qwen_cos_sin(img_freqs)
            )
            txt_query, txt_key = flydsl_fused_qk_norm_rope(
                txt_query, txt_key, attn.norm_added_q, attn.norm_added_k, _qwen_cos_sin(txt_freqs)
            )
        else:
            # Apply QK normalization
            if attn.norm_q is not None:
                img_query = attn.norm_q(img_query)
            if attn.norm_k is not None:
                img_key = attn.norm_k(img_key)
            if attn.norm_added_q is not None:
                txt_query = attn.norm_added_q(txt_query)
            if attn.norm_added_k is not None:
                txt_key = attn.norm_added_k(txt_key)

            # Apply RoPE
            if image_rotary_emb is not None:
                img_freqs, txt_freqs = image_rotary_emb
                img_query = apply_rotary_emb_qwen(img_query, img_freqs, use_real=False)
                img_key = apply_rotary_emb_qwen(img_key, img_freqs, use_real=False)
                txt_query = apply_rotary_emb_qwen(txt_query, txt_freqs, use_real=False)
                txt_key = apply_rotary_emb_qwen(txt_key, txt_freqs, use_real=False)

        # The wrapper's key-padding mask, when the prompts are padded, is in the
        # order the attention sees the keys.
        attention_kwargs = key_padding_attention_kwargs(self.attention_kwargs, attention_mask)

        if self.attention_kwargs.get("replicated_prompt_tokens"):
            # Every rank holds the whole prompt rather than a shard of it. Attend to it
            # once, as joint tokens: left in the sharded sequence its keys would be
            # gathered once per rank. The gathered image keys then end with the padding,
            # which USP drops, or which the key-padding mask covers.
            joint_hidden_states = USP(
                img_query.transpose(1, 2),
                img_key.transpose(1, 2),
                img_value.transpose(1, 2),
                dropout_p=0.0,
                is_causal=False,
                joint_query=txt_query.transpose(1, 2),
                joint_key=txt_key.transpose(1, 2),
                joint_value=txt_value.transpose(1, 2),
                joint_strategy="front",
                attn_layer=attn,
                attention_kwargs=attention_kwargs,
            ).transpose(1, 2)
            joint_dtype = img_query.dtype
        else:
            # Concatenate for joint attention
            # Order: [text, image]. Each rank holds [text shard, image shard], and
            # only the last rank's image shard ends with padding, so the gathered
            # keys end with it too.
            joint_query = torch.cat([txt_query, img_query], dim=1)
            joint_key = torch.cat([txt_key, img_key], dim=1)
            joint_value = torch.cat([txt_value, img_value], dim=1)

            joint_hidden_states = USP(
                joint_query.transpose(1, 2),
                joint_key.transpose(1, 2),
                joint_value.transpose(1, 2),
                dropout_p=0.0,
                is_causal=False,
                attn_layer=attn,
                attention_kwargs=attention_kwargs,
            ).transpose(1, 2)
            joint_dtype = joint_query.dtype

        # Reshape back
        joint_hidden_states = joint_hidden_states.flatten(2, 3)
        joint_hidden_states = joint_hidden_states.to(joint_dtype)

        # Split attention outputs back
        txt_attn_output = joint_hidden_states[:, :seq_txt, :]  # Text part
        img_attn_output = joint_hidden_states[:, seq_txt:, :]  # Image part

        # Apply output projections
        img_attn_output = attn.to_out[0](img_attn_output.contiguous())
        if len(attn.to_out) > 1:
            img_attn_output = attn.to_out[1](img_attn_output)  # dropout

        txt_attn_output = attn.to_add_out(txt_attn_output.contiguous())

        return img_attn_output, txt_attn_output


class xFuserQwenImageTransformerWrapper(QwenImageTransformer2DModel):
    def __init__(
        self,
        patch_size: int = 2,
        in_channels: int = 64,
        out_channels: Optional[int] = 16,
        num_layers: int = 60,
        attention_head_dim: int = 128,
        num_attention_heads: int = 24,
        joint_attention_dim: int = 3584,
        guidance_embeds: bool = False,  # TODO: this should probably be removed
        axes_dims_rope: Tuple[int, int, int] = (16, 56, 56),
        zero_cond_t: bool = False,
        use_additional_t_cond: bool = False,
        use_layer3d_rope: bool = False,
    ):
        super().__init__(
            patch_size=patch_size,
            in_channels=in_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            attention_head_dim=attention_head_dim,
            num_attention_heads=num_attention_heads,
            joint_attention_dim=joint_attention_dim,
            guidance_embeds=guidance_embeds,
            axes_dims_rope=axes_dims_rope,
            zero_cond_t=zero_cond_t,
            use_additional_t_cond=use_additional_t_cond,
            use_layer3d_rope=use_layer3d_rope,
        )

        self._self_attn_kwargs = {"valid_kv_len": None, "replicated_prompt_tokens": 0}
        self._sp_pad = {"local_rows": 0}
        for block in self.transformer_blocks:
            block.attn.processor = xFuserQwenDoubleStreamAttnProcessor(self._self_attn_kwargs)
            # Qwen returns (encoder, image) and its text stream is never SP padded.
            block.register_forward_hook(zero_sp_pad_rows_hook(self._sp_pad, index=1))
        register_fp8_comms_eligible_modules(self, [block.attn for block in self.transformer_blocks])
        self._key_padding_cache = MaskMetaCache()

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        encoder_hidden_states_mask: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        img_shapes: Optional[List[Tuple[int, int, int]]] = None,
        txt_seq_lens: Optional[List[int]] = None,
        guidance: torch.Tensor = None,  # TODO: this should probably be removed
        attention_kwargs: Optional[Dict[str, Any]] = None,
        controlnet_block_samples=None,
        additional_t_cond=None,
        return_dict: bool = True,
    ) -> Union[torch.Tensor, Transformer2DModelOutput]:
        sp_world_rank = get_sequence_parallel_rank()
        sp_world_size = get_sequence_parallel_world_size()

        image_seq_len = hidden_states.shape[1]
        full_text_seq_len = encoder_hidden_states.shape[1]
        pad_amount = (sp_world_size - (image_seq_len % sp_world_size)) % sp_world_size
        local_image_len = (image_seq_len + pad_amount) // sp_world_size
        # Replicate a prompt that cannot split evenly without padded text keys.
        # Also replicate it when image padding spans multiple shards: placing
        # text shards between those image shards would make the pad nontrailing.
        replicated_text = full_text_seq_len % sp_world_size != 0 or sp_pad_spans_ranks(image_seq_len, sp_world_size)
        if replicated_text and get_runtime_state().fp8_comms is not None:
            # The replicated prompt joins attention as joint tokens, which FP8
            # communication does not handle: it would silently fall back to bf16.
            if full_text_seq_len % sp_world_size != 0:
                reason = (
                    f"the prompt length ({full_text_seq_len} tokens) to be divisible by "
                    f"the sequence parallel degree ({sp_world_size}). Change the prompt, "
                    "--max_sequence_length or the parallel degrees"
                )
            else:
                reason = (
                    f"an image whose padding fits in one shard (got {image_seq_len} tokens at "
                    f"sequence parallel degree {sp_world_size}). Use a larger image or smaller parallel degrees"
                )
            raise ValueError(f"--use_fp8_comms needs {reason}, or drop --use_fp8_comms.")

        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)

        # The pipeline's own mask tensor, the same object at every step, keys the key-padding cache.
        prompt_mask = encoder_hidden_states_mask
        hidden_states = self.img_in(hidden_states)

        timestep = timestep.to(hidden_states.dtype)

        if self.zero_cond_t:
            timestep = torch.cat([timestep, timestep * 0], dim=0)
            modulate_index = torch.tensor(
                [[0] * prod(sample[0]) + [1] * sum([prod(s) for s in sample[1:]]) for sample in img_shapes],
                device=timestep.device,
                dtype=torch.int,
            )
        else:
            modulate_index = None

        encoder_hidden_states = self.txt_norm(encoder_hidden_states)
        encoder_hidden_states = self.txt_in(encoder_hidden_states)

        # Use the encoder_hidden_states sequence length for RoPE computation and normalize mask
        text_seq_len, _, encoder_hidden_states_mask = compute_text_seq_len_from_mask(
            encoder_hidden_states, encoder_hidden_states_mask
        )

        if guidance is not None:
            guidance = guidance.to(hidden_states.dtype) * 1000

        temb = (
            self.time_text_embed(timestep, hidden_states, additional_t_cond)
            if guidance is None
            else self.time_text_embed(timestep, guidance, hidden_states, additional_t_cond)
        )

        image_rotary_emb = self.pos_embed(img_shapes, max_txt_seq_len=text_seq_len, device=hidden_states.device)

        hidden_states = chunk_and_pad_sequence(hidden_states, sp_world_rank, sp_world_size, pad_amount, dim=1)
        valid_kv_len = image_seq_len if replicated_text else image_seq_len + full_text_seq_len
        self._self_attn_kwargs["valid_kv_len"] = valid_kv_len if pad_amount else None
        self._self_attn_kwargs["replicated_prompt_tokens"] = full_text_seq_len if replicated_text else 0
        self._sp_pad["local_rows"] = local_sp_pad_rows(image_seq_len, local_image_len, sp_world_rank)
        txt_freqs = image_rotary_emb[1]
        if not replicated_text:
            encoder_hidden_states = chunk_and_pad_sequence(
                encoder_hidden_states, sp_world_rank, sp_world_size, 0, dim=1
            )
            txt_freqs = chunk_and_pad_sequence(txt_freqs, sp_world_rank, sp_world_size, 0, dim=0)

        image_rotary_emb = [
            chunk_and_pad_sequence(image_rotary_emb[0], sp_world_rank, sp_world_size, pad_amount, dim=0),
            txt_freqs,
        ]

        # Construct joint attention mask once to avoid reconstructing in every block
        # This eliminates 60 GPU syncs during training while maintaining torch.compile compatibility
        block_attention_kwargs = attention_kwargs.copy() if attention_kwargs is not None else {}
        if prompt_mask is not None:
            # diffusers' Qwen-Image pipelines pass None when no prompt token is padding, so a mask here
            # means prompts of different lengths.
            block_attention_kwargs["attention_mask"] = key_padding_mask(
                self._key_padding_cache,
                prompt_mask,
                text_image_key_mask,
                image_seq_len,
                pad_amount,
                sp_world_size,
                "front" if replicated_text else None,
            )

        for index_block, block in enumerate(self.transformer_blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                encoder_hidden_states, hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    None,  # Don't pass encoder_hidden_states_mask (using attention_mask instead)
                    temb,
                    image_rotary_emb,
                    block_attention_kwargs,
                    modulate_index,
                )

            else:
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_hidden_states_mask=None,  # Don't pass (using attention_mask instead)
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    joint_attention_kwargs=block_attention_kwargs,
                    modulate_index=modulate_index,
                )

            # controlnet residual
            if controlnet_block_samples is not None:
                interval_control = len(self.transformer_blocks) / len(controlnet_block_samples)
                interval_control = int(np.ceil(interval_control))
                hidden_states = hidden_states + controlnet_block_samples[index_block // interval_control]
                # The pad-row hook ran before this residual was added, and the controlnet
                # output is not zero on the pad rows: clear them again.
                local_pad_rows = self._sp_pad["local_rows"]
                if local_pad_rows:
                    hidden_states[:, -local_pad_rows:] = 0

        if self.zero_cond_t:
            temb = temb.chunk(2, dim=0)[0]
        # Use only the image part (hidden_states) from the dual-stream blocks
        hidden_states = self.norm_out(hidden_states, temb)
        output = self.proj_out(hidden_states)

        output = gather_and_unpad(output, pad_amount, dim=1)

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)
