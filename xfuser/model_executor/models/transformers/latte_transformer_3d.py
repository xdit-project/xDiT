from typing import Optional, Dict, Any, Union, List, Optional, Tuple, Type
import torch
import torch.distributed
import torch.nn as nn

from diffusers.models.embeddings import PatchEmbed

from diffusers.models import LatteTransformer3DModel
from diffusers.models.transformers.transformer_2d import Transformer2DModelOutput
from diffusers.utils import is_torch_version, scale_lora_layers, USE_PEFT_BACKEND, unscale_lora_layers

from xfuser.model_executor.models import xFuserModelBaseWrapper
from xfuser.distributed.runtime_state import get_runtime_state
from xfuser.logger import init_logger
from xfuser.model_executor.base_wrapper import xFuserBaseWrapper
from xfuser.distributed import (
    get_data_parallel_world_size,
    get_sequence_parallel_world_size,
    get_pipeline_parallel_world_size,
    get_classifier_free_guidance_world_size,
    get_classifier_free_guidance_rank,
    get_pipeline_parallel_rank,
    get_pp_group,
    get_world_group,
    get_cfg_group,
    get_sp_group,
    get_runtime_state, 
    initialize_runtime_state
)

from xfuser.model_executor.models.transformers.register import xFuserTransformerWrappersRegister
from xfuser.model_executor.models.transformers.base_transformer import xFuserTransformerBaseWrapper

logger = init_logger(__name__)


@xFuserTransformerWrappersRegister.register(LatteTransformer3DModel)
class xFuserLatteTransformer3DWrapper(xFuserTransformerBaseWrapper):
    def __init__(
        self,
        transformer: LatteTransformer3DModel,
    ):
        super().__init__(
            transformer=transformer,
            submodule_classes_to_wrap=[nn.Conv2d, PatchEmbed],
            submodule_name_to_wrap=["attn1"]
        )
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: Optional[torch.LongTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        enable_temporal_attentions: bool = True,
        return_dict: bool = True,
    ):
        """
        The [`LatteTransformer3DModel`] forward method.

        Args:
            hidden_states shape `(batch size, channel, num_frame, height, width)`:
                Input `hidden_states`.
            timestep ( `torch.LongTensor`, *optional*):
                Used to indicate denoising step. Optional timestep to be applied as an embedding in `AdaLayerNorm`.
            encoder_hidden_states ( `torch.FloatTensor` of shape `(batch size, sequence len, embed dims)`, *optional*):
                Conditional embeddings for cross attention layer. If not given, cross-attention defaults to
                self-attention.
            encoder_attention_mask ( `torch.Tensor`, *optional*):
                Cross-attention mask applied to `encoder_hidden_states`. Two formats supported:

                    * Mask `(batcheight, sequence_length)` True = keep, False = discard.
                    * Bias `(batcheight, 1, sequence_length)` 0 = keep, -10000 = discard.

                If `ndim == 2`: will be interpreted as a mask, then converted into a bias consistent with the format
                above. This bias will be added to the cross-attention scores.
            enable_temporal_attentions:
                (`bool`, *optional*, defaults to `True`): Whether to enable temporal attentions.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.unet_2d_condition.UNet2DConditionOutput`] instead of a plain
                tuple.

        Returns:
            If `return_dict` is True, an [`~models.transformer_2d.Transformer2DModelOutput`] is returned, otherwise a
            `tuple` where the first element is the sample tensor.
        """

        # Reshape hidden states
        batch_size, channels, num_frame, height, width = hidden_states.shape
        # batch_size channels num_frame height width -> (batch_size * num_frame) channels height width
        hidden_states = hidden_states.permute(0, 2, 1, 3, 4).reshape(-1, channels, height, width)

        # Input
        height, width = (
            hidden_states.shape[-2] // self.config.patch_size,
            hidden_states.shape[-1] // self.config.patch_size,
        )
        num_patches = height * width

        hidden_states = self.pos_embed(hidden_states)  # alrady add positional embeddings

        added_cond_kwargs = {"resolution": None, "aspect_ratio": None}
        timestep, embedded_timestep = self.adaln_single(
            timestep, added_cond_kwargs=added_cond_kwargs, batch_size=batch_size, hidden_dtype=hidden_states.dtype
        )

        # Prepare text embeddings for spatial block
        # batch_size num_tokens hidden_size -> (batch_size * num_frame) num_tokens hidden_size
        encoder_hidden_states = self.caption_projection(encoder_hidden_states)  # 3 120 1152
        encoder_hidden_states_spatial = encoder_hidden_states.repeat_interleave(num_frame, dim=0).view(
            -1, encoder_hidden_states.shape[-2], encoder_hidden_states.shape[-1]
        )

        # Prepare timesteps for spatial and temporal block
        timestep_spatial = timestep.repeat_interleave(num_frame, dim=0).view(-1, timestep.shape[-1])
        timestep_temp = timestep.repeat_interleave(num_patches, dim=0).view(-1, timestep.shape[-1])
        
        # Spatial and temporal transformer blocks
        for i, (spatial_block, temp_block) in enumerate(
            zip(self.transformer_blocks, self.temporal_transformer_blocks)
        ):
            if self.training and self.gradient_checkpointing:
                hidden_states = torch.utils.checkpoint.checkpoint(
                    spatial_block,
                    hidden_states,
                    None,  # attention_mask
                    encoder_hidden_states_spatial,
                    encoder_attention_mask,
                    timestep_spatial,
                    None,  # cross_attention_kwargs
                    None,  # class_labels
                    use_reentrant=False,
                )
            else:
                hidden_states = spatial_block(
                    hidden_states,
                    None,  # attention_mask
                    encoder_hidden_states_spatial,
                    encoder_attention_mask,
                    timestep_spatial,
                    None,  # cross_attention_kwargs
                    None,  # class_labels
                )

            if enable_temporal_attentions:
                # (batch_size * num_frame) num_tokens hidden_size -> (batch_size * num_tokens) num_frame hidden_size
                hidden_states = hidden_states.reshape(
                    batch_size, -1, hidden_states.shape[-2], hidden_states.shape[-1]
                ).permute(0, 2, 1, 3)
                hidden_states = hidden_states.reshape(-1, hidden_states.shape[-2], hidden_states.shape[-1])

                if i == 0 and num_frame > 1:
                    hidden_states = hidden_states + self.temp_pos_embed

                if self.training and self.gradient_checkpointing:
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        temp_block,
                        hidden_states,
                        None,  # attention_mask
                        None,  # encoder_hidden_states
                        None,  # encoder_attention_mask
                        timestep_temp,
                        None,  # cross_attention_kwargs
                        None,  # class_labels
                        use_reentrant=False,
                    )
                else:
                    hidden_states = temp_block(
                        hidden_states,
                        None,  # attention_mask
                        None,  # encoder_hidden_states
                        None,  # encoder_attention_mask
                        timestep_temp,
                        None,  # cross_attention_kwargs
                        None,  # class_labels
                    )

                # (batch_size * num_tokens) num_frame hidden_size -> (batch_size * num_frame) num_tokens hidden_size
                hidden_states = hidden_states.reshape(
                    batch_size, -1, hidden_states.shape[-2], hidden_states.shape[-1]
                ).permute(0, 2, 1, 3)
                hidden_states = hidden_states.reshape(-1, hidden_states.shape[-2], hidden_states.shape[-1])
                
        embedded_timestep = embedded_timestep.repeat_interleave(num_frame, dim=0).view(-1, embedded_timestep.shape[-1])
        shift, scale = (self.scale_shift_table[None] + embedded_timestep[:, None]).chunk(2, dim=1)
        hidden_states = self.norm_out(hidden_states)
        # Modulation
        hidden_states = hidden_states * (1 + scale) + shift
        hidden_states = self.proj_out(hidden_states)

        # unpatchify
        if self.adaln_single is None:
            height = width = int(hidden_states.shape[1] ** 0.5)
        hidden_states = hidden_states.reshape(
            shape=(-1, height, width, self.config.patch_size, self.config.patch_size, self.out_channels)
        )
        hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states)
        output = hidden_states.reshape(
            shape=(-1, self.out_channels, height * self.config.patch_size, width * self.config.patch_size)
        )
        output = output.reshape(batch_size, -1, output.shape[-3], output.shape[-2], output.shape[-1]).permute(
            0, 2, 1, 3, 4
        )

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)