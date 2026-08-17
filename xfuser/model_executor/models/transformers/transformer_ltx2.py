from typing import Any

import torch
from diffusers.models.transformers.transformer_ltx2 import (
    AudioVisualModelOutput,
    LTX2VideoTransformer3DModel,
    apply_interleaved_rotary_emb,
    apply_split_rotary_emb,
)
from diffusers.utils import USE_PEFT_BACKEND, scale_lora_layers, unscale_lora_layers

from xfuser.core.distributed import (
    get_sequence_parallel_rank,
    get_sequence_parallel_world_size,
    get_sp_group,
)
from xfuser.model_executor.layers.attention_mask import (
    AttentionMaskWithMeta,
    make_attn_mask_with_meta,
)
from xfuser.model_executor.layers.usp import USP, attention


def _get_mask_meta(cache: dict, mask: torch.Tensor | None) -> object | None:
    """Convert a 2-D boolean attention mask to AttentionMaskWithMeta, cached per tensor."""
    if mask is None or mask.ndim != 2:
        return mask
    key = (mask.data_ptr(), tuple(mask.shape))
    if key not in cache:
        cache[key] = make_attn_mask_with_meta(mask)
    return cache[key]


class xFuserLTX2PerturbedAttnProcessor:
    def __init__(self, use_parallel_attention: bool = True, gather_kv=False):
        if use_parallel_attention:
            self.attention_method = USP
        else:
            self.attention_method = attention
        self.gather_kv = gather_kv

    def __call__(
        self,
        attn: Any,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        query_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        key_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        perturbation_mask: torch.Tensor | None = None,
        all_perturbed: bool | None = None,
    ) -> torch.Tensor:
        batch_size, sequence_length, _ = (
            hidden_states.shape
            if encoder_hidden_states is None
            else encoder_hidden_states.shape
        )
        if self.gather_kv:
            encoder_hidden_states = get_sp_group().all_gather(
                encoder_hidden_states, dim=1
            )
            key_rotary_emb = [x.contiguous() for x in key_rotary_emb]
            key_rotary_emb = [
                get_sp_group().all_gather(x, dim=2) for x in key_rotary_emb
            ]

        if isinstance(attention_mask, AttentionMaskWithMeta):
            # Only forward attn_mask. The varlen fields (indices_k etc.) are for
            # self-attention where Q and K share the same sequence length; here the
            # mask covers text encoder tokens (K) while Q is video tokens, so the
            # varlen pack path in _varlen_pack_keys would mis-reshape the key tensor.
            attn_kw = {"attn_mask": attention_mask.attn_mask}
        elif attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(
                attention_mask, sequence_length, batch_size
            )
            attention_mask = attention_mask.view(
                batch_size, attn.heads, -1, attention_mask.shape[-1]
            )
            attn_kw = {"attn_mask": attention_mask}
        else:
            attn_kw = None

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        if attn.to_gate_logits is not None:
            gate_logits = attn.to_gate_logits(hidden_states)
        value = attn.to_v(encoder_hidden_states)
        if all_perturbed is None:
            all_perturbed = (
                torch.all(perturbation_mask == 0)
                if perturbation_mask is not None
                else False
            )

        if all_perturbed:
            hidden_states = value
        else:
            query = attn.to_q(hidden_states)
            key = attn.to_k(encoder_hidden_states)

            query = attn.norm_q(query)
            key = attn.norm_k(key)

            if query_rotary_emb is not None:
                if attn.rope_type == "interleaved":
                    query = apply_interleaved_rotary_emb(query, query_rotary_emb)
                    key = apply_interleaved_rotary_emb(
                        key,
                        key_rotary_emb
                        if key_rotary_emb is not None
                        else query_rotary_emb,
                    )
                elif attn.rope_type == "split":
                    query = apply_split_rotary_emb(query, query_rotary_emb)
                    key = apply_split_rotary_emb(
                        key,
                        key_rotary_emb
                        if key_rotary_emb is not None
                        else query_rotary_emb,
                    )

            query = query.unflatten(2, (attn.heads, -1)).transpose(1, 2)
            key = key.unflatten(2, (attn.heads, -1)).transpose(1, 2)
            value = value.unflatten(2, (attn.heads, -1)).transpose(1, 2)

            hidden_states = self.attention_method(
                query,
                key,
                value,
                dropout_p=0.0,
                is_causal=False,
                attention_kwargs=attn_kw,
            )
            hidden_states = hidden_states.transpose(1, 2).flatten(2, 3)
            hidden_states = hidden_states.to(query.dtype)

            if perturbation_mask is not None:
                value = value.transpose(1, 2).flatten(2, 3)
                hidden_states = torch.lerp(value, hidden_states, perturbation_mask)

        if attn.to_gate_logits is not None:
            hidden_states = hidden_states.unflatten(2, (attn.heads, -1))
            gates = 2.0 * torch.sigmoid(gate_logits)
            hidden_states = hidden_states * gates.unsqueeze(-1)
            hidden_states = hidden_states.flatten(2, 3)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states


class xFuserLTX2AudioVideoAttnProcessor:
    def __init__(self, use_parallel_attention: bool = True, gather_kv=False):
        if use_parallel_attention:
            self.attention_method = USP
        else:
            self.attention_method = attention
        self.gather_kv = gather_kv

    def __call__(
        self,
        attn: Any,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        query_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        key_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        batch_size, sequence_length, _ = (
            hidden_states.shape
            if encoder_hidden_states is None
            else encoder_hidden_states.shape
        )

        if self.gather_kv:
            encoder_hidden_states = get_sp_group().all_gather(
                encoder_hidden_states, dim=1
            )
            key_rotary_emb = [x.contiguous() for x in key_rotary_emb]
            key_rotary_emb = [
                get_sp_group().all_gather(x, dim=2) for x in key_rotary_emb
            ]

        if isinstance(attention_mask, AttentionMaskWithMeta):
            # Only forward attn_mask. The varlen fields (indices_k etc.) are for
            # self-attention where Q and K share the same sequence length; here the
            # mask covers text encoder tokens (K) while Q is video tokens, so the
            # varlen pack path in _varlen_pack_keys would mis-reshape the key tensor.
            attn_kw = {"attn_mask": attention_mask.attn_mask}
        elif attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(
                attention_mask, sequence_length, batch_size
            )
            attention_mask = attention_mask.view(
                batch_size, attn.heads, -1, attention_mask.shape[-1]
            )
            attn_kw = {"attn_mask": attention_mask}
        else:
            attn_kw = None

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        if attn.to_gate_logits is not None:
            gate_logits = attn.to_gate_logits(hidden_states)

        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        if query_rotary_emb is not None:
            if attn.rope_type == "interleaved":
                query = apply_interleaved_rotary_emb(query, query_rotary_emb)
                key = apply_interleaved_rotary_emb(
                    key,
                    key_rotary_emb if key_rotary_emb is not None else query_rotary_emb,
                )
            elif attn.rope_type == "split":
                query = apply_split_rotary_emb(query, query_rotary_emb)
                key = apply_split_rotary_emb(
                    key,
                    key_rotary_emb if key_rotary_emb is not None else query_rotary_emb,
                )

        query = query.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        key = key.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        value = value.unflatten(2, (attn.heads, -1)).transpose(1, 2)

        hidden_states = self.attention_method(
            query, key, value, dropout_p=0.0, is_causal=False, attention_kwargs=attn_kw
        )
        hidden_states = hidden_states.transpose(1, 2).flatten(2, 3)
        hidden_states = hidden_states.to(query.dtype)

        if attn.to_gate_logits is not None:
            hidden_states = hidden_states.unflatten(2, (attn.heads, -1))
            gates = 2.0 * torch.sigmoid(gate_logits)
            hidden_states = hidden_states * gates.unsqueeze(-1)
            hidden_states = hidden_states.flatten(2, 3)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states


class xFuserLTX2VideoTransformer3DWrapper(LTX2VideoTransformer3DModel):
    def __init__(
        self,
        in_channels: int = 128,  # Video Arguments
        out_channels: int | None = 128,
        patch_size: int = 1,
        patch_size_t: int = 1,
        num_attention_heads: int = 32,
        attention_head_dim: int = 128,
        cross_attention_dim: int = 4096,
        vae_scale_factors: tuple[int, int, int] = (8, 32, 32),
        pos_embed_max_pos: int = 20,
        base_height: int = 2048,
        base_width: int = 2048,
        gated_attn: bool = False,
        cross_attn_mod: bool = False,
        audio_in_channels: int = 128,  # Audio Arguments
        audio_out_channels: int | None = 128,
        audio_patch_size: int = 1,
        audio_patch_size_t: int = 1,
        audio_num_attention_heads: int = 32,
        audio_attention_head_dim: int = 64,
        audio_cross_attention_dim: int = 2048,
        audio_scale_factor: int = 4,
        audio_pos_embed_max_pos: int = 20,
        audio_sampling_rate: int = 16000,
        audio_hop_length: int = 160,
        audio_gated_attn: bool = False,
        audio_cross_attn_mod: bool = False,
        num_layers: int = 48,  # Shared arguments
        activation_fn: str = "gelu-approximate",
        qk_norm: str = "rms_norm_across_heads",
        norm_elementwise_affine: bool = False,
        norm_eps: float = 1e-6,
        caption_channels: int = 3840,
        attention_bias: bool = True,
        attention_out_bias: bool = True,
        rope_theta: float = 10000.0,
        rope_double_precision: bool = True,
        causal_offset: int = 1,
        timestep_scale_multiplier: int = 1000,
        cross_attn_timestep_scale_multiplier: int = 1000,
        rope_type: str = "interleaved",
        use_prompt_embeddings: bool = True,
        perturbed_attn: bool = False,
        ff_bias: bool = False,
        audio_ff_bias: bool = False,
        use_prompt_adaln_single: bool = False,
        use_keyframes_abs_pos_embedding: bool = False,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            patch_size=patch_size,
            patch_size_t=patch_size_t,
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            cross_attention_dim=cross_attention_dim,
            vae_scale_factors=vae_scale_factors,
            pos_embed_max_pos=pos_embed_max_pos,
            base_height=base_height,
            base_width=base_width,
            gated_attn=gated_attn,
            cross_attn_mod=cross_attn_mod,
            audio_in_channels=audio_in_channels,  # Audio Arguments
            audio_out_channels=audio_out_channels,
            audio_patch_size=audio_patch_size,
            audio_patch_size_t=audio_patch_size_t,
            audio_num_attention_heads=audio_num_attention_heads,
            audio_attention_head_dim=audio_attention_head_dim,
            audio_cross_attention_dim=audio_cross_attention_dim,
            audio_scale_factor=audio_scale_factor,
            audio_pos_embed_max_pos=audio_pos_embed_max_pos,
            audio_sampling_rate=audio_sampling_rate,
            audio_hop_length=audio_hop_length,
            audio_gated_attn=audio_gated_attn,
            audio_cross_attn_mod=audio_cross_attn_mod,
            num_layers=num_layers,  # Shared arguments
            activation_fn=activation_fn,
            qk_norm=qk_norm,
            norm_elementwise_affine=norm_elementwise_affine,
            norm_eps=norm_eps,
            caption_channels=caption_channels,
            attention_bias=attention_bias,
            attention_out_bias=attention_out_bias,
            rope_theta=rope_theta,
            rope_double_precision=rope_double_precision,
            causal_offset=causal_offset,
            timestep_scale_multiplier=timestep_scale_multiplier,
            cross_attn_timestep_scale_multiplier=cross_attn_timestep_scale_multiplier,
            rope_type=rope_type,
            use_prompt_embeddings=use_prompt_embeddings,
            perturbed_attn=perturbed_attn,
            ff_bias=ff_bias,
            audio_ff_bias=audio_ff_bias,
            use_prompt_adaln_single=use_prompt_adaln_single,
            use_keyframes_abs_pos_embedding=use_keyframes_abs_pos_embedding,
        )

        self._enc_mask_cache: dict = {}
        self._audio_enc_mask_cache: dict = {}

        if perturbed_attn:
            attn_processor_cls = xFuserLTX2PerturbedAttnProcessor
        else:
            attn_processor_cls = xFuserLTX2AudioVideoAttnProcessor

        for block in self.transformer_blocks:
            block.attn1.processor = attn_processor_cls()
            block.attn2.processor = attn_processor_cls(use_parallel_attention=False)
            block.audio_attn1.processor = attn_processor_cls(
                use_parallel_attention=False
            )
            block.audio_attn2.processor = attn_processor_cls(
                use_parallel_attention=False
            )
            block.audio_to_video_attn.processor = attn_processor_cls(
                use_parallel_attention=False
            )
            block.video_to_audio_attn.processor = attn_processor_cls(
                use_parallel_attention=False, gather_kv=True
            )

    def _chunk_and_pad_sequence(
        self,
        x: torch.Tensor,
        sp_world_rank: int,
        sp_world_size: int,
        pad_amount: int,
        dim: int,
    ) -> torch.Tensor:
        if pad_amount > 0:
            if dim < 0:
                dim = x.ndim + dim
            pad_shape = list(x.shape)
            pad_shape[dim] = pad_amount
            x = torch.cat(
                [
                    x,
                    torch.zeros(
                        pad_shape,
                        dtype=x.dtype,
                        device=x.device,
                    ),
                ],
                dim=dim,
            )
        x = torch.chunk(x, sp_world_size, dim=dim)[sp_world_rank]
        return x

    def _gather_and_unpad(
        self, x: torch.Tensor, pad_amount: int, dim: int
    ) -> torch.Tensor:
        x = get_sp_group().all_gather(x, dim=dim)
        size = x.size(dim)
        return x.narrow(dim=dim, start=0, length=size - pad_amount)

    def forward(
        self,
        hidden_states: torch.Tensor,
        audio_hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        audio_encoder_hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        audio_timestep: torch.LongTensor | None = None,
        sigma: torch.Tensor | None = None,
        audio_sigma: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        audio_encoder_attention_mask: torch.Tensor | None = None,
        num_frames: int | None = None,
        height: int | None = None,
        width: int | None = None,
        fps: float = 24.0,
        audio_num_frames: int | None = None,
        video_coords: torch.Tensor | None = None,
        audio_coords: torch.Tensor | None = None,
        isolate_modalities: bool = False,
        spatio_temporal_guidance_blocks: list[int] | None = None,
        perturbation_mask: torch.Tensor | None = None,
        use_cross_timestep: bool = False,
        attention_kwargs: dict[str, Any] | None = None,
        return_dict: bool = True,
    ) -> torch.Tensor:
        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            scale_lora_layers(self, lora_scale)

        sp_world_rank = get_sequence_parallel_rank()
        sp_world_size = get_sequence_parallel_world_size()

        full_seq_len = hidden_states.shape[1]
        pad_amount = (sp_world_size - (full_seq_len % sp_world_size)) % sp_world_size
        hidden_states = self._chunk_and_pad_sequence(
            hidden_states, sp_world_rank, sp_world_size, pad_amount, dim=1
        )

        # Determine timestep for audio before chunking the video-side timestep.
        # The pipeline passes audio_timestep as a separate per-sample tensor, so
        # it must not inherit the (to-be-chunked) per-token video timestep.
        audio_timestep = audio_timestep if audio_timestep is not None else timestep
        audio_sigma = audio_sigma if audio_sigma is not None else sigma

        # If the video timestep is per-token [batch, num_video_tokens] rather than
        # per-sample, chunk it to match the local hidden_states shard.  This avoids
        # the broadcast mismatch between chunked hidden_states [batch, local_seq, dim]
        # and embedded_timestep [batch, full_seq, dim] in the output scale/shift layer.
        if timestep.ndim == 2 and timestep.shape[1] == full_seq_len:
            timestep = self._chunk_and_pad_sequence(
                timestep, sp_world_rank, sp_world_size, pad_amount, dim=1
            )

        encoder_attention_mask = _get_mask_meta(
            self._enc_mask_cache, encoder_attention_mask
        )
        audio_encoder_attention_mask = _get_mask_meta(
            self._audio_enc_mask_cache, audio_encoder_attention_mask
        )

        batch_size = hidden_states.size(0)

        # 1. Prepare RoPE positional embeddings
        if video_coords is None:
            video_coords = self.rope.prepare_video_coords(
                batch_size, num_frames, height, width, hidden_states.device, fps=fps
            )
        if audio_coords is None:
            audio_coords = self.audio_rope.prepare_audio_coords(
                batch_size, audio_num_frames, audio_hidden_states.device
            )

        video_coords = self._chunk_and_pad_sequence(
            video_coords, sp_world_rank, sp_world_size, pad_amount, dim=2
        )

        video_rotary_emb = self.rope(video_coords, device=hidden_states.device)

        audio_rotary_emb = self.audio_rope(
            audio_coords, device=audio_hidden_states.device
        )

        video_cross_attn_rotary_emb = self.cross_attn_rope(
            video_coords[:, 0:1, :], device=hidden_states.device
        )
        audio_cross_attn_rotary_emb = self.cross_attn_audio_rope(
            audio_coords[:, 0:1, :], device=audio_hidden_states.device
        )

        # 2. Patchify input projections
        hidden_states = self.proj_in(hidden_states)
        audio_hidden_states = self.audio_proj_in(audio_hidden_states)

        # 3. Prepare timestep embeddings and modulation parameters
        timestep_cross_attn_gate_scale_factor = (
            self.config.cross_attn_timestep_scale_multiplier
            / self.config.timestep_scale_multiplier
        )

        # 3.1. Prepare global modality (video and audio) timestep embedding and modulation parameters
        # temb is used in the transformer blocks (as expected), while embedded_timestep is used for the output layer
        # modulation with scale_shift_table (and similarly for audio)
        temb, embedded_timestep = self.time_embed(
            timestep.flatten(),
            batch_size=batch_size,
            hidden_dtype=hidden_states.dtype,
        )

        temb = temb.view(batch_size, -1, temb.size(-1))
        embedded_timestep = embedded_timestep.view(
            batch_size, -1, embedded_timestep.size(-1)
        )

        temb_audio, audio_embedded_timestep = self.audio_time_embed(
            audio_timestep.flatten(),
            batch_size=batch_size,
            hidden_dtype=audio_hidden_states.dtype,
        )

        temb_audio = temb_audio.view(batch_size, -1, temb_audio.size(-1))

        audio_embedded_timestep = audio_embedded_timestep.view(
            batch_size, -1, audio_embedded_timestep.size(-1)
        )

        if self.prompt_modulation and self.config.use_prompt_adaln_single:
            temb_prompt, _ = self.prompt_adaln(
                sigma.flatten(), batch_size=batch_size, hidden_dtype=hidden_states.dtype
            )
            temb_prompt_audio, _ = self.audio_prompt_adaln(
                audio_sigma.flatten(),
                batch_size=batch_size,
                hidden_dtype=audio_hidden_states.dtype,
            )
            temb_prompt = temb_prompt.view(batch_size, -1, temb_prompt.size(-1))
            temb_prompt_audio = temb_prompt_audio.view(
                batch_size, -1, temb_prompt_audio.size(-1)
            )
        else:
            temb_prompt = temb_prompt_audio = None

        # 3.2. Prepare global modality cross attention modulation parameters
        # LTX-2.3: use the cross-modality sigma (audio sigma for video CA, video sigma for audio CA)
        video_ca_timestep = (
            audio_sigma.flatten() if use_cross_timestep else timestep.flatten()
        )
        audio_ca_timestep = (
            sigma.flatten() if use_cross_timestep else audio_timestep.flatten()
        )

        video_cross_attn_scale_shift, _ = self.av_cross_attn_video_scale_shift(
            video_ca_timestep,
            batch_size=batch_size,
            hidden_dtype=hidden_states.dtype,
        )

        video_cross_attn_a2v_gate, _ = self.av_cross_attn_video_a2v_gate(
            video_ca_timestep * timestep_cross_attn_gate_scale_factor,
            batch_size=batch_size,
            hidden_dtype=hidden_states.dtype,
        )
        video_cross_attn_scale_shift = video_cross_attn_scale_shift.view(
            batch_size, -1, video_cross_attn_scale_shift.shape[-1]
        )
        video_cross_attn_a2v_gate = video_cross_attn_a2v_gate.view(
            batch_size, -1, video_cross_attn_a2v_gate.shape[-1]
        )

        audio_cross_attn_scale_shift, _ = self.av_cross_attn_audio_scale_shift(
            audio_ca_timestep,
            batch_size=batch_size,
            hidden_dtype=audio_hidden_states.dtype,
        )
        audio_cross_attn_v2a_gate, _ = self.av_cross_attn_audio_v2a_gate(
            audio_ca_timestep * timestep_cross_attn_gate_scale_factor,
            batch_size=batch_size,
            hidden_dtype=audio_hidden_states.dtype,
        )
        audio_cross_attn_scale_shift = audio_cross_attn_scale_shift.view(
            batch_size, -1, audio_cross_attn_scale_shift.shape[-1]
        )
        audio_cross_attn_v2a_gate = audio_cross_attn_v2a_gate.view(
            batch_size, -1, audio_cross_attn_v2a_gate.shape[-1]
        )

        # 4. Prepare prompt embeddings (LTX-2.0)
        if self.config.use_prompt_embeddings:
            encoder_hidden_states = self.caption_projection(encoder_hidden_states)
            encoder_hidden_states = encoder_hidden_states.view(
                batch_size, -1, hidden_states.size(-1)
            )

            audio_encoder_hidden_states = self.audio_caption_projection(
                audio_encoder_hidden_states
            )
            audio_encoder_hidden_states = audio_encoder_hidden_states.view(
                batch_size, -1, audio_hidden_states.size(-1)
            )

        # 5. Run transformer blocks
        stg_blocks = set(spatio_temporal_guidance_blocks or [])
        if stg_blocks and perturbation_mask is None:
            default_all_perturbed = True
        else:
            default_all_perturbed = False
            if perturbation_mask is not None and perturbation_mask.ndim == 1:
                perturbation_mask = perturbation_mask[:, None, None]

        for block_i, block in enumerate(self.transformer_blocks):
            is_stg_block = block_i in stg_blocks
            block_all_perturbed = default_all_perturbed if is_stg_block else False
            block_perturbation_mask = (
                perturbation_mask
                if (is_stg_block and not default_all_perturbed)
                else None
            )

            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states, audio_hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    audio_hidden_states,
                    encoder_hidden_states,
                    audio_encoder_hidden_states,
                    temb,
                    temb_audio,
                    video_cross_attn_scale_shift,
                    audio_cross_attn_scale_shift,
                    video_cross_attn_a2v_gate,
                    audio_cross_attn_v2a_gate,
                    temb_prompt,
                    temb_prompt_audio,
                    video_rotary_emb,
                    audio_rotary_emb,
                    video_cross_attn_rotary_emb,
                    audio_cross_attn_rotary_emb,
                    encoder_attention_mask,
                    audio_encoder_attention_mask,
                    None,  # self_attention_mask
                    None,  # audio_self_attention_mask
                    None,  # a2v_cross_attention_mask
                    None,  # v2a_cross_attention_mask
                    not isolate_modalities,  # use_a2v_cross_attention
                    not isolate_modalities,  # use_v2a_cross_attention
                    block_perturbation_mask,
                    block_all_perturbed,
                )
            else:
                hidden_states, audio_hidden_states = block(
                    hidden_states=hidden_states,
                    audio_hidden_states=audio_hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    audio_encoder_hidden_states=audio_encoder_hidden_states,
                    temb=temb,
                    temb_audio=temb_audio,
                    temb_ca_scale_shift=video_cross_attn_scale_shift,
                    temb_ca_audio_scale_shift=audio_cross_attn_scale_shift,
                    temb_ca_gate=video_cross_attn_a2v_gate,
                    temb_ca_audio_gate=audio_cross_attn_v2a_gate,
                    temb_prompt=temb_prompt,
                    temb_prompt_audio=temb_prompt_audio,
                    video_rotary_emb=video_rotary_emb,
                    audio_rotary_emb=audio_rotary_emb,
                    ca_video_rotary_emb=video_cross_attn_rotary_emb,
                    ca_audio_rotary_emb=audio_cross_attn_rotary_emb,
                    encoder_attention_mask=encoder_attention_mask,
                    audio_encoder_attention_mask=audio_encoder_attention_mask,
                    use_a2v_cross_attention=not isolate_modalities,
                    use_v2a_cross_attention=not isolate_modalities,
                    perturbation_mask=block_perturbation_mask,
                    all_perturbed=block_all_perturbed,
                )
                # Workaround for pytorch/pytorch#152887: when compile_repeated_blocks
                # is used with mode="reduce-overhead", every block call hits the same
                # compiled callable. CUDA Graph Trees cannot handle the resulting
                # ``x = f(x)`` chain (the previous call's static-memory output is the
                # next call's input). Cloning into fresh user memory breaks the alias.
                hidden_states = hidden_states.clone()
                audio_hidden_states = audio_hidden_states.clone()

        # 6. Output layers (including unpatchification)
        scale_shift_values = (
            self.scale_shift_table[None, None] + embedded_timestep[:, :, None]
        )
        shift, scale = scale_shift_values[:, :, 0], scale_shift_values[:, :, 1]

        hidden_states = self.norm_out(hidden_states)
        hidden_states = hidden_states * (1 + scale) + shift
        output = self.proj_out(hidden_states)

        output = self._gather_and_unpad(output, pad_amount, dim=1)

        audio_scale_shift_values = (
            self.audio_scale_shift_table[None, None]
            + audio_embedded_timestep[:, :, None]
        )
        audio_shift, audio_scale = (
            audio_scale_shift_values[:, :, 0],
            audio_scale_shift_values[:, :, 1],
        )

        audio_hidden_states = self.audio_norm_out(audio_hidden_states)
        audio_hidden_states = audio_hidden_states * (1 + audio_scale) + audio_shift
        audio_output = self.audio_proj_out(audio_hidden_states)

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output, audio_output)
        return AudioVisualModelOutput(sample=output, audio_sample=audio_output)
