import torch
import math
from typing import Optional, Union, Dict, Any, Tuple, List

from diffusers.models.transformers.transformer_wan_vace import WanVACETransformer3DModel
from diffusers.models.modeling_outputs import Transformer2DModelOutput

from xfuser.core.distributed import (
    get_sequence_parallel_world_size,
    get_sequence_parallel_rank,
    get_sp_group,
    get_runtime_state,
)
from xfuser.model_executor.layers.attention_processor import (
    xFuserAttentionProcessorRegister
)
from xfuser.envs import PACKAGES_CHECKER

env_info = PACKAGES_CHECKER.get_packages_info()
HAS_LONG_CTX_ATTN = env_info["has_long_ctx_attn"]


from xfuser.model_executor.models.transformers.transformer_wan import xFuserWanAttnProcessor
from xfuser.model_executor.models.transformers.transformers_utils import (
    chunk_and_pad_sequence,
    gather_and_unpad,
)

class xFuserWanVACETransformer3DWrapper(WanVACETransformer3DModel):

    def __init__(
        self,
        patch_size: Tuple[int, ...] = (1, 2, 2),
        num_attention_heads: int = 40,
        attention_head_dim: int = 128,
        in_channels: int = 16,
        out_channels: int = 16,
        text_dim: int = 4096,
        freq_dim: int = 256,
        ffn_dim: int = 13824,
        num_layers: int = 40,
        cross_attn_norm: bool = True,
        qk_norm: Optional[str] = "rms_norm_across_heads",
        eps: float = 1e-6,
        image_dim: Optional[int] = None,
        added_kv_proj_dim: Optional[int] = None,
        rope_max_seq_len: int = 1024,
        pos_embed_seq_len: Optional[int] = None,
        vace_layers: List[int] = [0, 5, 10, 15, 20, 25, 30, 35],
        vace_in_channels: int = 96,
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
            vace_layers,
            vace_in_channels,
        )

        for block in self.blocks:
            block.attn1.processor = xFuserWanAttnProcessor()
            block.attn2.processor = xFuserWanAttnProcessor(use_parallel_attention=False)

        for block in self.vace_blocks:
            block.attn1.processor = xFuserWanAttnProcessor()
            block.attn2.processor = xFuserWanAttnProcessor(use_parallel_attention=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_image: Optional[torch.Tensor] = None,
        control_hidden_states: torch.Tensor = None,
        control_hidden_states_scale: torch.Tensor = None,
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:

        # if attention_kwargs is not None:
        #     attention_kwargs = attention_kwargs.copy()
        #     lora_scale = attention_kwargs.pop("scale", 1.0)
        # else:
        #     lora_scale = 1.0

        # if USE_PEFT_BACKEND:
        #     # weight the lora layers by setting `lora_scale` for each PEFT layer
        #     scale_lora_layers(self, lora_scale)
        # else:
        #     if attention_kwargs is not None and attention_kwargs.get("scale", None) is not None:
        #         logger.warning(
        #             "Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective."
        #         )

        get_runtime_state().increment_step_counter()

        sp_world_rank = get_sequence_parallel_rank()
        sp_world_size = get_sequence_parallel_world_size()

        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.config.patch_size
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p_h
        post_patch_width = width // p_w

        if control_hidden_states_scale is None:
            control_hidden_states_scale = control_hidden_states.new_ones(len(self.config.vace_layers))
        control_hidden_states_scale = torch.unbind(control_hidden_states_scale)
        if len(control_hidden_states_scale) != len(self.config.vace_layers):
            raise ValueError(
                f"Length of `control_hidden_states_scale` {len(control_hidden_states_scale)} should be "
                f"equal to {len(self.config.vace_layers)}."
            )


        # 1. Rotary position embedding
        rotary_emb = self.rope(hidden_states)

        # 2. Patch embedding
        hidden_states = self.patch_embedding(hidden_states)
        hidden_states = hidden_states.flatten(2).transpose(1, 2)

        control_hidden_states = self.vace_patch_embedding(control_hidden_states)
        control_hidden_states = control_hidden_states.flatten(2).transpose(1, 2)
        control_hidden_states_padding = control_hidden_states.new_zeros(
            batch_size, hidden_states.size(1) - control_hidden_states.size(1), control_hidden_states.size(2)
        )
        control_hidden_states = torch.cat([control_hidden_states, control_hidden_states_padding], dim=1)

        pad_amount = (sp_world_size - (hidden_states.shape[1] % sp_world_size)) % sp_world_size
        hidden_states = chunk_and_pad_sequence(hidden_states, sp_world_rank, sp_world_size, pad_amount, dim=1)
        control_hidden_states = chunk_and_pad_sequence(control_hidden_states, sp_world_rank, sp_world_size, pad_amount, dim=1)

        rotary_emb = [chunk_and_pad_sequence(freqs, sp_world_rank, sp_world_size, pad_amount, dim=1) for freqs in rotary_emb]

        # 3. Time embedding
        temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = self.condition_embedder(
            timestep, encoder_hidden_states, encoder_hidden_states_image
        )
        timestep_proj = timestep_proj.unflatten(1, (6, -1))

        # 4. Image embedding
        if encoder_hidden_states_image is not None:
            encoder_hidden_states = torch.concat([encoder_hidden_states_image, encoder_hidden_states], dim=1)

        # 5. Transformer blocks
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            # Prepare VACE hints
            control_hidden_states_list = []
            for i, block in enumerate(self.vace_blocks):
                conditioning_states, control_hidden_states = self._gradient_checkpointing_func(
                    block, hidden_states, encoder_hidden_states, control_hidden_states, timestep_proj, rotary_emb
                )
                control_hidden_states_list.append((conditioning_states, control_hidden_states_scale[i]))
            control_hidden_states_list = control_hidden_states_list[::-1]

            for i, block in enumerate(self.blocks):
                hidden_states = self._gradient_checkpointing_func(
                    block, hidden_states, encoder_hidden_states, timestep_proj, rotary_emb
                )
                if i in self.config.vace_layers:
                    control_hint, scale = control_hidden_states_list.pop()
                    hidden_states = hidden_states + control_hint * scale
        else:
            # Prepare VACE hints
            control_hidden_states_list = []
            for i, block in enumerate(self.vace_blocks):
                conditioning_states, control_hidden_states = block(
                    hidden_states, encoder_hidden_states, control_hidden_states, timestep_proj, rotary_emb
                )
                control_hidden_states_list.append((conditioning_states, control_hidden_states_scale[i]))
            control_hidden_states_list = control_hidden_states_list[::-1]

            for i, block in enumerate(self.blocks):
                hidden_states = block(hidden_states, encoder_hidden_states, timestep_proj, rotary_emb)
                if i in self.config.vace_layers:
                    control_hint, scale = control_hidden_states_list.pop()
                    hidden_states = hidden_states + control_hint * scale

        # 6. Output norm, projection & unpatchify
        shift, scale = (self.scale_shift_table.to(temb.device) + temb.unsqueeze(1)).chunk(2, dim=1)

        # Move the shift and scale tensors to the same device as hidden_states.
        # When using multi-GPU inference via accelerate these will be on the
        # first device rather than the last device, which hidden_states ends up
        # on.
        shift = shift.to(hidden_states.device)
        scale = scale.to(hidden_states.device)

        hidden_states = (self.norm_out(hidden_states.float()) * (1 + scale) + shift).type_as(hidden_states)
        hidden_states = self.proj_out(hidden_states)

        hidden_states = gather_and_unpad(hidden_states, pad_amount, dim=1)

        hidden_states = hidden_states.reshape(
            batch_size, post_patch_num_frames, post_patch_height, post_patch_width, p_t, p_h, p_w, -1
        )
        hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
        output = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

        # if USE_PEFT_BACKEND:
        #     # remove `lora_scale` from each PEFT layer
        #     unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)