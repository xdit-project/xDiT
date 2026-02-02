import torch
from typing import Optional, Dict, Any, Tuple
from diffusers.models.transformers.transformer_ltx2 import (
    LTX2VideoTransformer3DModel,
    apply_interleaved_rotary_emb,
    apply_split_rotary_emb,
    AudioVisualModelOutput,
)



from xfuser.model_executor.layers.usp import (
    USP,
    attention,
)

from xfuser.core.distributed import (
    get_sequence_parallel_world_size,
    get_sequence_parallel_rank,
    get_sp_group,
)

class xFuserLTX2AudioVideoAttnProcessor:

    def __init__(self, use_parallel_attention: bool = True, gather=False):
        if use_parallel_attention:
            self.attention_method = USP
        else:
            self.attention_method = attention
        self.gather = gather

    def __call__(
        self,
        attn: "LTX2Attention",
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        query_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        key_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if self.gather:
            encoder_hidden_states = get_sp_group().all_gather(encoder_hidden_states, dim=1)
            key_rotary_emb = [x.contiguous() for x in key_rotary_emb]
            key_rotary_emb = [get_sp_group().all_gather(x, dim=2) for x in key_rotary_emb]

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        if query_rotary_emb is not None:
            if attn.rope_type == "interleaved":
                query = apply_interleaved_rotary_emb(query, query_rotary_emb)
                key = apply_interleaved_rotary_emb(
                    key, key_rotary_emb if key_rotary_emb is not None else query_rotary_emb
                )
            elif attn.rope_type == "split":
                query = apply_split_rotary_emb(query, query_rotary_emb)
                key = apply_split_rotary_emb(key, key_rotary_emb if key_rotary_emb is not None else query_rotary_emb)

        query = query.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        key = key.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        value = value.unflatten(2, (attn.heads, -1)).transpose(1, 2)

        hidden_states = self.attention_method(
            query,
            key,
            value,
            dropout_p=0.0,
            is_causal=False,
        )
        hidden_states = hidden_states.transpose(1, 2).flatten(2, 3)
        hidden_states = hidden_states.to(query.dtype)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states

class xFuserLTX2VideoTransformer3DWrapper(LTX2VideoTransformer3DModel):

    def __init__(
        self,
        in_channels: int = 128,  # Video Arguments
        out_channels: Optional[int] = 128,
        patch_size: int = 1,
        patch_size_t: int = 1,
        num_attention_heads: int = 32,
        attention_head_dim: int = 128,
        cross_attention_dim: int = 4096,
        vae_scale_factors: Tuple[int, int, int] = (8, 32, 32),
        pos_embed_max_pos: int = 20,
        base_height: int = 2048,
        base_width: int = 2048,
        audio_in_channels: int = 128,  # Audio Arguments
        audio_out_channels: Optional[int] = 128,
        audio_patch_size: int = 1,
        audio_patch_size_t: int = 1,
        audio_num_attention_heads: int = 32,
        audio_attention_head_dim: int = 64,
        audio_cross_attention_dim: int = 2048,
        audio_scale_factor: int = 4,
        audio_pos_embed_max_pos: int = 20,
        audio_sampling_rate: int = 16000,
        audio_hop_length: int = 160,
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
        )
        for block in self.transformer_blocks:

            block.attn1.processor = xFuserLTX2AudioVideoAttnProcessor()
            block.attn2.processor = xFuserLTX2AudioVideoAttnProcessor(use_parallel_attention=False)
            block.audio_attn1.processor = xFuserLTX2AudioVideoAttnProcessor(use_parallel_attention=False)
            block.audio_attn2.processor = xFuserLTX2AudioVideoAttnProcessor(use_parallel_attention=False)
            block.audio_to_video_attn.processor = xFuserLTX2AudioVideoAttnProcessor(use_parallel_attention=False)
            block.video_to_audio_attn.processor = xFuserLTX2AudioVideoAttnProcessor(use_parallel_attention=False, gather=True)


    def _chunk_and_pad_sequence(self, x: torch.Tensor, sp_world_rank: int, sp_world_size: int, pad_amount: int, dim: int) -> torch.Tensor:
        if pad_amount > 0:
            if dim < 0:
                dim = x.ndim + dim
            pad_shape = list(x.shape)
            pad_shape[dim] = pad_amount
            x = torch.cat([x,
                        torch.zeros(
                            pad_shape,
                            dtype=x.dtype,
                            device=x.device,
                        )], dim=dim)
        x = torch.chunk(x,
                        sp_world_size,
                        dim=dim)[sp_world_rank]
        return x

    def _gather_and_unpad(self, x: torch.Tensor, pad_amount: int, dim: int) -> torch.Tensor:
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
        audio_timestep: Optional[torch.LongTensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        audio_encoder_attention_mask: Optional[torch.Tensor] = None,
        num_frames: Optional[int] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        fps: float = 24.0,
        audio_num_frames: Optional[int] = None,
        video_coords: Optional[torch.Tensor] = None,
        audio_coords: Optional[torch.Tensor] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
    ) -> torch.Tensor:
        """
        Forward pass for LTX-2.0 audiovisual video transformer.

        Args:
            hidden_states (`torch.Tensor`):
                Input patchified video latents of shape `(batch_size, num_video_tokens, in_channels)`.
            audio_hidden_states (`torch.Tensor`):
                Input patchified audio latents of shape `(batch_size, num_audio_tokens, audio_in_channels)`.
            encoder_hidden_states (`torch.Tensor`):
                Input video text embeddings of shape `(batch_size, text_seq_len, self.config.caption_channels)`.
            audio_encoder_hidden_states (`torch.Tensor`):
                Input audio text embeddings of shape `(batch_size, text_seq_len, self.config.caption_channels)`.
            timestep (`torch.Tensor`):
                Input timestep of shape `(batch_size, num_video_tokens)`. These should already be scaled by
                `self.config.timestep_scale_multiplier`.
            audio_timestep (`torch.Tensor`, *optional*):
                Input timestep of shape `(batch_size,)` or `(batch_size, num_audio_tokens)` for audio modulation
                params. This is only used by certain pipelines such as the I2V pipeline.
            encoder_attention_mask (`torch.Tensor`, *optional*):
                Optional multiplicative text attention mask of shape `(batch_size, text_seq_len)`.
            audio_encoder_attention_mask (`torch.Tensor`, *optional*):
                Optional multiplicative text attention mask of shape `(batch_size, text_seq_len)` for audio modeling.
            num_frames (`int`, *optional*):
                The number of latent video frames. Used if calculating the video coordinates for RoPE.
            height (`int`, *optional*):
                The latent video height. Used if calculating the video coordinates for RoPE.
            width (`int`, *optional*):
                The latent video width. Used if calculating the video coordinates for RoPE.
            fps: (`float`, *optional*, defaults to `24.0`):
                The desired frames per second of the generated video. Used if calculating the video coordinates for
                RoPE.
            audio_num_frames: (`int`, *optional*):
                The number of latent audio frames. Used if calculating the audio coordinates for RoPE.
            video_coords (`torch.Tensor`, *optional*):
                The video coordinates to be used when calculating the rotary positional embeddings (RoPE) of shape
                `(batch_size, 3, num_video_tokens, 2)`. If not supplied, this will be calculated inside `forward`.
            audio_coords (`torch.Tensor`, *optional*):
                The audio coordinates to be used when calculating the rotary positional embeddings (RoPE) of shape
                `(batch_size, 1, num_audio_tokens, 2)`. If not supplied, this will be calculated inside `forward`.
            attention_kwargs (`Dict[str, Any]`, *optional*):
                Optional dict of keyword args to be passed to the attention processor.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether to return a dict-like structured output of type `AudioVisualModelOutput` or a tuple.

        Returns:
            `AudioVisualModelOutput` or `tuple`:
                If `return_dict` is `True`, returns a structured output of type `AudioVisualModelOutput`, otherwise a
                `tuple` is returned where the first element is the denoised video latent patch sequence and the second
                element is the denoised audio latent patch sequence.
        """

        """
            _cp_plan = {
        "": {
            "hidden_states": ContextParallelInput(split_dim=1, expected_dims=3, split_output=False),
            "encoder_hidden_states": ContextParallelInput(split_dim=1, expected_dims=3, split_output=False),
            "encoder_attention_mask": ContextParallelInput(split_dim=1, expected_dims=2, split_output=False),
        },
        "rope": {
            0: ContextParallelInput(split_dim=1, expected_dims=3, split_output=True),
            1: ContextParallelInput(split_dim=1, expected_dims=3, split_output=True),
        },
        "proj_out": ContextParallelOutput(gather_dim=1, expected_dims=3),
    }

        """

        sp_world_rank = get_sequence_parallel_rank()
        sp_world_size = get_sequence_parallel_world_size()

        pad_amount = (sp_world_size - (hidden_states.shape[1] % sp_world_size)) % sp_world_size
        #audio_pad_amount  = (sp_world_size - (audio_hidden_states.shape[1] % sp_world_size)) % sp_world_size
        #encoder_pad_amount = (sp_world_size - (encoder_hidden_states.shape[1] % sp_world_size)) % sp_world_size
        #audio_encoder_pad_amount = (sp_world_size - (audio_encoder_hidden_states.shape[1] % sp_world_size)) % sp_world_size
        hidden_states = self._chunk_and_pad_sequence(hidden_states, sp_world_rank, sp_world_size, pad_amount, dim=1)
        #audio_hidden_states = self._chunk_and_pad_sequence(audio_hidden_states, sp_world_rank, sp_world_size, audio_pad_amount, dim=1)
        #encoder_hidden_states = self._chunk_and_pad_sequence(encoder_hidden_states, sp_world_rank, sp_world_size, encoder_pad_amount, dim=1)
        #audio_encoder_hidden_states = self._chunk_and_pad_sequence(audio_encoder_hidden_states, sp_world_rank, sp_world_size, audio_encoder_pad_amount, dim=1)
        #encoder_attention_mask = self._chunk_and_pad_sequence(encoder_attention_mask, sp_world_rank, sp_world_size, encoder_pad_amount, dim=1) if encoder_attention_mask is not None else None
        #audio_encoder_attention_mask = self._chunk_and_pad_sequence(audio_encoder_attention_mask, sp_world_rank, sp_world_size, audio_encoder_pad_amount, dim=1) if audio_encoder_attention_mask is not None else None

        # Determine timestep for audio.
        audio_timestep = audio_timestep if audio_timestep is not None else timestep

        # if len(timestep.shape) == 2:
        #     timestep = self._chunk_and_pad_sequence(timestep, sp_world_rank, sp_world_size, pad_amount, dim=1)

        # if len(audio_timestep.shape) == 2:
        #     print(audio_timestep.shape)
        #     audio_timestep = self._chunk_and_pad_sequence(audio_timestep, sp_world_rank, sp_world_size, audio_pad_amount, dim=1)

        # convert encoder_attention_mask to a bias the same way we do for attention_mask
        if encoder_attention_mask is not None and encoder_attention_mask.ndim == 2:
            encoder_attention_mask = (1 - encoder_attention_mask.to(hidden_states.dtype)) * -10000.0
            encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

        if audio_encoder_attention_mask is not None and audio_encoder_attention_mask.ndim == 2:
            audio_encoder_attention_mask = (1 - audio_encoder_attention_mask.to(audio_hidden_states.dtype)) * -10000.0
            audio_encoder_attention_mask = audio_encoder_attention_mask.unsqueeze(1)

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


        video_coords = self._chunk_and_pad_sequence(video_coords, sp_world_rank, sp_world_size, pad_amount, dim=2)
        #audio_coords = self._chunk_and_pad_sequence(audio_coords, sp_world_rank, sp_world_size, audio_pad_amount, dim=2)

        video_rotary_emb = self.rope(video_coords, device=hidden_states.device)
        #video_rotary_emb = [self._chunk_and_pad_sequence(x, sp_world_rank, sp_world_size, pad_amount, dim=2) for x in video_rotary_emb]

        audio_rotary_emb = self.audio_rope(audio_coords, device=audio_hidden_states.device)

        video_cross_attn_rotary_emb = self.cross_attn_rope(video_coords[:, 0:1, :], device=hidden_states.device)
        audio_cross_attn_rotary_emb = self.cross_attn_audio_rope(
            audio_coords[:, 0:1, :], device=audio_hidden_states.device
        )

        # 2. Patchify input projections
        hidden_states = self.proj_in(hidden_states)
        audio_hidden_states = self.audio_proj_in(audio_hidden_states)

        # 3. Prepare timestep embeddings and modulation parameters
        timestep_cross_attn_gate_scale_factor = (
            self.config.cross_attn_timestep_scale_multiplier / self.config.timestep_scale_multiplier
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
        embedded_timestep = embedded_timestep.view(batch_size, -1, embedded_timestep.size(-1))

        temb_audio, audio_embedded_timestep = self.audio_time_embed(
            audio_timestep.flatten(),
            batch_size=batch_size,
            hidden_dtype=audio_hidden_states.dtype,
        )

        temb_audio = temb_audio.view(batch_size, -1, temb_audio.size(-1))

        audio_embedded_timestep = audio_embedded_timestep.view(batch_size, -1, audio_embedded_timestep.size(-1))

        # 3.2. Prepare global modality cross attention modulation parameters
        video_cross_attn_scale_shift, _ = self.av_cross_attn_video_scale_shift(
            timestep.flatten(),
            batch_size=batch_size,
            hidden_dtype=hidden_states.dtype,
        )

        video_cross_attn_a2v_gate, _ = self.av_cross_attn_video_a2v_gate(
            timestep.flatten() * timestep_cross_attn_gate_scale_factor,
            batch_size=batch_size,
            hidden_dtype=hidden_states.dtype,
        )
        video_cross_attn_scale_shift = video_cross_attn_scale_shift.view(
            batch_size, -1, video_cross_attn_scale_shift.shape[-1]
        )
        video_cross_attn_a2v_gate = video_cross_attn_a2v_gate.view(batch_size, -1, video_cross_attn_a2v_gate.shape[-1])

        audio_cross_attn_scale_shift, _ = self.av_cross_attn_audio_scale_shift(
            audio_timestep.flatten(),
            batch_size=batch_size,
            hidden_dtype=audio_hidden_states.dtype,
        )
        audio_cross_attn_v2a_gate, _ = self.av_cross_attn_audio_v2a_gate(
            audio_timestep.flatten() * timestep_cross_attn_gate_scale_factor,
            batch_size=batch_size,
            hidden_dtype=audio_hidden_states.dtype,
        )
        audio_cross_attn_scale_shift = audio_cross_attn_scale_shift.view(
            batch_size, -1, audio_cross_attn_scale_shift.shape[-1]
        )
        audio_cross_attn_v2a_gate = audio_cross_attn_v2a_gate.view(batch_size, -1, audio_cross_attn_v2a_gate.shape[-1])

        # 4. Prepare prompt embeddings
        encoder_hidden_states = self.caption_projection(encoder_hidden_states)
        encoder_hidden_states = encoder_hidden_states.view(batch_size, -1, hidden_states.size(-1))

        audio_encoder_hidden_states = self.audio_caption_projection(audio_encoder_hidden_states)
        audio_encoder_hidden_states = audio_encoder_hidden_states.view(batch_size, -1, audio_hidden_states.size(-1))

        # 5. Run transformer blocks

        for block_i, block in enumerate(self.transformer_blocks):
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
                    video_rotary_emb,
                    audio_rotary_emb,
                    video_cross_attn_rotary_emb,
                    audio_cross_attn_rotary_emb,
                    encoder_attention_mask,
                    audio_encoder_attention_mask,
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
                    video_rotary_emb=video_rotary_emb,
                    audio_rotary_emb=audio_rotary_emb,
                    ca_video_rotary_emb=video_cross_attn_rotary_emb,
                    ca_audio_rotary_emb=audio_cross_attn_rotary_emb,
                    encoder_attention_mask=encoder_attention_mask,
                    audio_encoder_attention_mask=audio_encoder_attention_mask,
                )

        # 6. Output layers (including unpatchification)
        scale_shift_values = self.scale_shift_table[None, None] + embedded_timestep[:, :, None]
        shift, scale = scale_shift_values[:, :, 0], scale_shift_values[:, :, 1]

        hidden_states = self.norm_out(hidden_states)
        hidden_states = hidden_states * (1 + scale) + shift
        output = self.proj_out(hidden_states)

        output = self._gather_and_unpad(output, pad_amount, dim=1)

        audio_scale_shift_values = self.audio_scale_shift_table[None, None] + audio_embedded_timestep[:, :, None]
        audio_shift, audio_scale = audio_scale_shift_values[:, :, 0], audio_scale_shift_values[:, :, 1]

        audio_hidden_states = self.audio_norm_out(audio_hidden_states)
        audio_hidden_states = audio_hidden_states * (1 + audio_scale) + audio_shift
        audio_output = self.audio_proj_out(audio_hidden_states)

        #audio_output = self._gather_and_unpad(audio_output, audio_pad_amount, dim=1)

        if not return_dict:
            return (output, audio_output)
        return AudioVisualModelOutput(sample=output, audio_sample=audio_output)