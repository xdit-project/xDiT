import torch
from math import prod
import numpy as np
from typing import Optional, Dict, Any, Union, Tuple, List
from diffusers.models.transformers.transformer_2d import Transformer2DModelOutput
from diffusers.models.transformers.transformer_qwenimage import (
    QwenDoubleStreamAttnProcessor2_0,
    QwenImageTransformer2DModel,
    apply_rotary_emb_qwen,
    compute_text_seq_len_from_mask,
)
from diffusers.utils import (
    USE_PEFT_BACKEND,
    scale_lora_layers,
    unscale_lora_layers,
)
from xfuser.model_executor.layers.usp import USP
from xfuser.core.distributed.fp8_comms import register_fp8_comms_eligible_modules
from xfuser.core.distributed import (
    is_pipeline_first_stage,
    is_pipeline_last_stage,
    get_sequence_parallel_rank,
    get_sequence_parallel_world_size,
)
from xfuser.core.distributed.parallel_state import _SP
from xfuser.core.cache_manager.cache_manager import get_cache_manager
from xfuser.core.distributed.runtime_state import get_runtime_state
from xfuser.envs import PACKAGES_CHECKER
from xfuser.model_executor.layers.attention_processor import (
    xFuserAttentionProcessorRegister,
)
from xfuser.model_executor.models.transformers.base_transformer import (
    xFuserTransformerBaseWrapper,
)
from xfuser.model_executor.models.transformers.register import (
    xFuserTransformerWrappersRegister,
)
from xfuser.model_executor.models.transformers.transformers_utils import chunk_and_pad_sequence, gather_and_unpad

from xfuser.model_executor.layers.fused_qk_rope_flydsl import (
    flydsl_fused_qk_norm_rope,
    _HAS_FLYDSL,
)

HAS_LONG_CTX_ATTN = PACKAGES_CHECKER.get_packages_info()["has_long_ctx_attn"]


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


@xFuserAttentionProcessorRegister.register(QwenDoubleStreamAttnProcessor2_0)
class xFuserQwenDoubleStreamAttnProcessor(QwenDoubleStreamAttnProcessor2_0):
    def __init__(self):
        super().__init__()
        self.use_long_ctx_attn_kvcache = HAS_LONG_CTX_ATTN and _SP and get_sequence_parallel_world_size() > 1

    def __call__(
        self,
        attn,
        hidden_states: torch.FloatTensor,  # Image stream
        encoder_hidden_states: torch.FloatTensor = None,  # Text stream
        encoder_hidden_states_mask: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        latte_temporal_attention: bool = False,
    ) -> torch.FloatTensor:
        del latte_temporal_attention
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

        uses_pipefusion = get_runtime_state().num_pipeline_patch > 1
        if uses_pipefusion:
            if self.use_long_ctx_attn_kvcache:
                raise RuntimeError("Qwen Image PipeFusion does not support sequence parallel attention.")
            img_key, img_value = get_cache_manager().update_and_get_kv_cache(
                new_kv=[img_key, img_value],
                layer=attn,
                slice_dim=1,
                layer_type="attn",
            )

        if uses_pipefusion and attention_mask is None and encoder_hidden_states_mask is not None:
            image_mask = torch.ones(
                (hidden_states.shape[0], img_key.shape[1]),
                dtype=torch.bool,
                device=hidden_states.device,
            )
            attention_mask = torch.cat([encoder_hidden_states_mask, image_mask], dim=1)

        if attention_mask is not None and attention_mask.ndim == 2:
            attention_mask = attention_mask[:, None, None, :]
        # The pre-existing Ulysses wrapper constructs a rank-local joint mask
        # whose ordering does not describe the sequence after all-to-all.
        # Keep that path unchanged; PipeFusion's image cache uses this mask.
        usp_attention_mask = attention_mask if uses_pipefusion else None

        # Order: [text, image]. Text QKV is always recomputed; only image
        # K/V participates in PipeFusion's patch cache.
        joint_query = torch.cat([txt_query, img_query], dim=1)
        joint_key = torch.cat([txt_key, img_key], dim=1)
        joint_value = torch.cat([txt_value, img_value], dim=1)
        joint_hidden_states = USP(
            joint_query.transpose(1, 2),
            joint_key.transpose(1, 2),
            joint_value.transpose(1, 2),
            dropout_p=0.0,
            is_causal=False,
            attention_kwargs={"attn_mask": usp_attention_mask} if usp_attention_mask is not None else None,
            attn_layer=None if uses_pipefusion else attn,
            head_balance_layer=attn if uses_pipefusion else None,
        ).transpose(1, 2)

        # Reshape back
        joint_hidden_states = joint_hidden_states.flatten(2, 3)
        joint_hidden_states = joint_hidden_states.to(img_query.dtype)

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

        for block in self.transformer_blocks:
            block.attn.processor = xFuserQwenDoubleStreamAttnProcessor()
        register_fp8_comms_eligible_modules(self, qwen_attn_modules(self))

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

        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)

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

        pad_amount = (sp_world_size - (hidden_states.shape[1] % sp_world_size)) % sp_world_size
        encoder_pad_amount = (sp_world_size - (encoder_hidden_states.shape[1] % sp_world_size)) % sp_world_size
        hidden_states = chunk_and_pad_sequence(hidden_states, sp_world_rank, sp_world_size, pad_amount, dim=1)
        encoder_hidden_states = chunk_and_pad_sequence(
            encoder_hidden_states, sp_world_rank, sp_world_size, encoder_pad_amount, dim=1
        )

        image_rotary_emb = [
            chunk_and_pad_sequence(image_rotary_emb[0], sp_world_rank, sp_world_size, pad_amount, dim=0),
            chunk_and_pad_sequence(image_rotary_emb[1], sp_world_rank, sp_world_size, encoder_pad_amount, dim=0),
        ]

        # Construct joint attention mask once to avoid reconstructing in every block
        # This eliminates 60 GPU syncs during training while maintaining torch.compile compatibility
        block_attention_kwargs = attention_kwargs.copy() if attention_kwargs is not None else {}
        if encoder_hidden_states_mask is not None:
            # Build joint mask: [text_mask, all_ones_for_image]
            batch_size, image_seq_len = hidden_states.shape[:2]
            image_mask = torch.ones((batch_size, image_seq_len), dtype=torch.bool, device=hidden_states.device)
            joint_attention_mask = torch.cat([encoder_hidden_states_mask, image_mask], dim=1)
            block_attention_kwargs["attention_mask"] = joint_attention_mask

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


def qwen_attn_modules(transformer) -> list[torch.nn.Module]:
    return [block.attn for block in transformer.transformer_blocks]


@xFuserTransformerWrappersRegister.register(QwenImageTransformer2DModel)
class xFuserQwenImageTransformer2DWrapper(xFuserTransformerBaseWrapper):
    """Stage-local Qwen Image backbone for generic PipeFusion schedules."""

    def __init__(self, transformer: QwenImageTransformer2DModel):
        super().__init__(
            transformer=transformer,
            submodule_name_to_wrap=["attn"],
            transformer_blocks_name=["transformer_blocks"],
        )
        register_fp8_comms_eligible_modules(self, qwen_attn_modules(self))

    def pipefusion_compile_capture_forward(
        self,
        hidden_states,
        encoder_hidden_states=None,
        *args,
        return_dict=True,
        **kwargs,
    ):
        if is_pipeline_first_stage():
            hidden_states = hidden_states.new_zeros((*hidden_states.shape[:-1], self.img_in.out_features))
            encoder_hidden_states = encoder_hidden_states.new_zeros(
                (*encoder_hidden_states.shape[:-1], self.txt_in.out_features)
            )
        if is_pipeline_last_stage():
            hidden_states = hidden_states.new_zeros((*hidden_states.shape[:-1], self.proj_out.out_features))
            output = (hidden_states, None)
        else:
            output = (hidden_states, encoder_hidden_states)
        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        encoder_hidden_states_mask: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        img_shapes: Optional[List[Tuple[int, int, int]]] = None,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        txt_seq_lens: Optional[List[int]] = None,
        guidance: torch.Tensor = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        controlnet_block_samples=None,
        additional_t_cond=None,
        return_dict: bool = True,
    ) -> Union[torch.Tensor, Transformer2DModelOutput]:
        del txt_seq_lens
        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            scale_lora_layers(self, lora_scale)

        if is_pipeline_first_stage():
            hidden_states = self.img_in(hidden_states)
            encoder_hidden_states = self.txt_in(self.txt_norm(encoder_hidden_states))

        timestep = timestep.to(hidden_states.dtype)
        if self.zero_cond_t:
            timestep = torch.cat([timestep, timestep * 0], dim=0)
            modulate_index = torch.tensor(
                [[0] * prod(sample[0]) + [1] * sum(prod(shape) for shape in sample[1:]) for sample in img_shapes],
                device=timestep.device,
                dtype=torch.int,
            )
        else:
            modulate_index = None

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
        if image_rotary_emb is None:
            image_rotary_emb = self.pos_embed(
                img_shapes,
                max_txt_seq_len=text_seq_len,
                device=hidden_states.device,
            )

        block_attention_kwargs = attention_kwargs.copy() if attention_kwargs is not None else {}
        if encoder_hidden_states_mask is not None:
            sample_shapes = img_shapes if isinstance(img_shapes[0], tuple) else img_shapes[0]
            image_seq_len = sum(prod(shape) for shape in sample_shapes)
            image_mask = torch.ones(
                (hidden_states.shape[0], image_seq_len),
                dtype=torch.bool,
                device=hidden_states.device,
            )
            block_attention_kwargs["attention_mask"] = torch.cat(
                (encoder_hidden_states_mask, image_mask),
                dim=1,
            )

        for index_block, block in enumerate(self.transformer_blocks):
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                encoder_hidden_states_mask=None,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
                joint_attention_kwargs=block_attention_kwargs,
                modulate_index=modulate_index,
            )
            if controlnet_block_samples is not None:
                total_blocks = self.config.num_layers
                interval_control = int(np.ceil(total_blocks / len(controlnet_block_samples)))
                global_index = getattr(block, "_xfuser_checkpoint_block_index", index_block)
                hidden_states = hidden_states + controlnet_block_samples[global_index // interval_control]

        if self.zero_cond_t:
            temb = temb.chunk(2, dim=0)[0]

        if is_pipeline_last_stage():
            hidden_states = self.norm_out(hidden_states, temb)
            output = (self.proj_out(hidden_states), None)
        else:
            output = (hidden_states, encoder_hidden_states)

        if USE_PEFT_BACKEND:
            unscale_lora_layers(self, lora_scale)
        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)
