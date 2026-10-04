"""Sequence-parallel SkyReels-V2 transformer.

SkyReels-V2 is built on Wan 2.1: its attention module has the same projections, QK
norm, RoPE and 512-token I2V context split, so the Wan USP attention processor serves
it unchanged. What differs is the transformer forward (an optional fps embedding, a
5-argument block call carrying a causal mask, and diffusion-forcing timesteps), which
is ported here with the Wan sequence-parallel split around the blocks.

Only the non-causal path is supported: one timestep per sample and full attention
(``num_frame_per_block == 1``). Diffusion forcing needs per-frame timesteps and a
block-causal attention mask, and USP has no attention-mask path, so it is refused
instead of silently running without the mask.
"""

from typing import Any, Dict, Optional, Tuple, Union

import torch
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.transformers.transformer_skyreels_v2 import (
    SkyReelsV2AttnProcessor,
    SkyReelsV2Transformer3DModel,
)

from xfuser.core.distributed import (
    get_ring_parallel_world_size,
    get_runtime_state,
    get_sequence_parallel_rank,
    get_sequence_parallel_world_size,
)
from xfuser.model_executor.layers.attention_processor import (
    xFuserAttentionProcessorRegister,
)
from xfuser.model_executor.models.transformers.transformer_wan import (
    xFuserWanAttnProcessor,
    xFuserWanTransformer3DWrapper,
)


@xFuserAttentionProcessorRegister.register(SkyReelsV2AttnProcessor)
class xFuserSkyReelsV2AttnProcessor(xFuserWanAttnProcessor, SkyReelsV2AttnProcessor):
    """The Wan USP processor, refusing the attention mask it cannot apply."""

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        if attention_mask is not None:
            raise NotImplementedError(
                "xDiT's SkyReels-V2 attention does not support attention masks: "
                "sequence-parallel attention has no mask path."
            )
        return super().__call__(attn, hidden_states, encoder_hidden_states, None, rotary_emb)


class xFuserSkyReelsV2Transformer3DWrapper(SkyReelsV2Transformer3DModel):
    # Same sequence-parallel split as Wan.
    _chunk_and_pad_sequence = xFuserWanTransformer3DWrapper._chunk_and_pad_sequence
    _gather_and_unpad = xFuserWanTransformer3DWrapper._gather_and_unpad

    def __init__(
        self,
        patch_size: Tuple[int, ...] = (1, 2, 2),
        num_attention_heads: int = 16,
        attention_head_dim: int = 128,
        in_channels: int = 16,
        out_channels: int = 16,
        text_dim: int = 4096,
        freq_dim: int = 256,
        ffn_dim: int = 8192,
        num_layers: int = 32,
        cross_attn_norm: bool = True,
        qk_norm: Optional[str] = "rms_norm_across_heads",
        eps: float = 1e-6,
        image_dim: Optional[int] = None,
        added_kv_proj_dim: Optional[int] = None,
        rope_max_seq_len: int = 1024,
        pos_embed_seq_len: Optional[int] = None,
        inject_sample_info: bool = False,
        num_frame_per_block: int = 1,
    ) -> None:
        super().__init__(
            patch_size,
            num_attention_heads,
            attention_head_dim,
            in_channels,
            out_channels,
            text_dim,
            freq_dim,
            ffn_dim,
            num_layers,
            cross_attn_norm,
            qk_norm,
            eps,
            image_dim,
            added_kv_proj_dim,
            rope_max_seq_len,
            pos_embed_seq_len,
            inject_sample_info,
            num_frame_per_block,
        )
        # Shared with every self-attention processor. When the token count does not
        # divide the sequence-parallel degree the sequence is zero-padded; publishing
        # the real length lets USP drop the padded keys after the Ulysses exchange.
        # The key is always present because torch.compile guards on the key set.
        self._usp_attention_kwargs: Dict[str, Any] = {"valid_kv_len": None}
        for block in self.blocks:
            block.attn1.processor = xFuserSkyReelsV2AttnProcessor(attention_kwargs=self._usp_attention_kwargs)
            block.attn2.processor = xFuserSkyReelsV2AttnProcessor(
                use_ulysses_parallel_attention=False, is_cross_attention=True
            )

    def _check_supported(self, timestep: torch.Tensor, enable_diffusion_forcing: bool) -> None:
        if enable_diffusion_forcing or timestep.ndim > 1:
            raise NotImplementedError(
                "xDiT supports SkyReels-V2 text-to-video and image-to-video only; diffusion "
                "forcing (per-frame timesteps) is not supported."
            )
        if self.config.num_frame_per_block > 1:
            raise NotImplementedError(
                "xDiT does not support SkyReels-V2 block-causal attention "
                f"(num_frame_per_block={self.config.num_frame_per_block}): "
                "sequence-parallel attention has no mask path."
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_image: Optional[torch.Tensor] = None,
        enable_diffusion_forcing: bool = False,
        fps: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Union[torch.Tensor, Transformer2DModelOutput]:
        self._check_supported(timestep, enable_diffusion_forcing)
        get_runtime_state().increment_step_counter()

        sp_rank = get_sequence_parallel_rank()
        sp_world_size = get_sequence_parallel_world_size()

        batch_size, _, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.config.patch_size
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p_h
        post_patch_width = width // p_w

        # RoPE is built for the full sequence, then split like the tokens.
        rotary_emb = self.rope(hidden_states)

        hidden_states = self.patch_embedding(hidden_states)
        hidden_states = hidden_states.flatten(2).transpose(1, 2)

        temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = self.condition_embedder(
            timestep, encoder_hidden_states, encoder_hidden_states_image
        )
        timestep_proj = timestep_proj.unflatten(-1, (6, -1))

        if encoder_hidden_states_image is not None:
            encoder_hidden_states = torch.concat([encoder_hidden_states_image, encoder_hidden_states], dim=1)

        if self.config.inject_sample_info:
            fps = torch.tensor(fps, dtype=torch.long, device=hidden_states.device)
            fps_emb = self.fps_embedding(fps)
            timestep_proj = timestep_proj + self.fps_projection(fps_emb).unflatten(1, (6, -1))

        seq_len = hidden_states.shape[1]
        pad_amount = (sp_world_size - seq_len % sp_world_size) % sp_world_size
        if pad_amount and get_ring_parallel_world_size() > 1:
            # Ring attention sees one chunk of keys per step, so the padded keys
            # cannot be dropped and would leak into every real token's attention.
            raise ValueError(
                f"SkyReels-V2 with ring attention needs the latent token count ({seq_len}) "
                f"to be divisible by the sequence-parallel degree ({sp_world_size}); "
                "change the resolution or frame count, or use Ulysses parallelism only."
            )
        self._usp_attention_kwargs["valid_kv_len"] = seq_len if pad_amount else None

        hidden_states = self._chunk_and_pad_sequence(hidden_states, sp_rank, sp_world_size, pad_amount, dim=1)
        rotary_emb = tuple(
            self._chunk_and_pad_sequence(freqs, sp_rank, sp_world_size, pad_amount, dim=1) for freqs in rotary_emb
        )

        if torch.is_grad_enabled() and self.gradient_checkpointing:
            for block in self.blocks:
                hidden_states = self._gradient_checkpointing_func(
                    block, hidden_states, encoder_hidden_states, timestep_proj, rotary_emb, None
                )
        else:
            for block in self.blocks:
                hidden_states = block(hidden_states, encoder_hidden_states, timestep_proj, rotary_emb, None)

        shift, scale = (self.scale_shift_table + temb.unsqueeze(1)).chunk(2, dim=1)
        shift = shift.to(hidden_states.device)
        scale = scale.to(hidden_states.device)

        hidden_states = (self.norm_out(hidden_states.float()) * (1 + scale) + shift).type_as(hidden_states)
        hidden_states = self.proj_out(hidden_states)

        if sp_world_size > 1:
            hidden_states = self._gather_and_unpad(hidden_states, pad_amount, dim=-2)

        hidden_states = hidden_states.reshape(
            batch_size, post_patch_num_frames, post_patch_height, post_patch_width, p_t, p_h, p_w, -1
        )
        hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
        output = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)
