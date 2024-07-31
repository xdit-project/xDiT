from typing import Optional, Dict, Any, Union
import torch
import torch.distributed
import torch.nn as nn

from diffusers.models.embeddings import PatchEmbed
from diffusers.models.transformers.transformer_sd3 import SD3Transformer2DModel
from diffusers.models.transformers.transformer_2d import Transformer2DModelOutput
from diffusers.utils import is_torch_version, scale_lora_layers, USE_PEFT_BACKEND, unscale_lora_layers

from xfuser.logger import init_logger
from xfuser.model_executor.base_wrapper import xFuserBaseWrapper
from xfuser.distributed import (
    get_pipeline_parallel_rank,
    get_pipeline_parallel_world_size,
)
from .register import xFuserTransformerWrappersRegister
from .base_transformer import xFuserTransformerBaseWrapper

logger = init_logger(__name__)


@xFuserTransformerWrappersRegister.register(SD3Transformer2DModel)
class xFuserSD3Transformer2DWrapper(xFuserTransformerBaseWrapper):
    def __init__(
        self,
        transformer: SD3Transformer2DModel,
    ):
        super().__init__(
            transformer=transformer,
            submodule_classes_to_wrap=[nn.Conv2d, PatchEmbed],
            submodule_name_to_wrap=["attn"],
        )

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        pooled_projections: torch.FloatTensor = None,
        timestep: torch.LongTensor = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
    ) -> Union[torch.FloatTensor, Transformer2DModelOutput]:
        """
        The [`SD3Transformer2DModel`] forward method.

        Args:
            hidden_states (`torch.FloatTensor` of shape `(batch size, channel, height, width)`):
                Input `hidden_states`.
            encoder_hidden_states (`torch.FloatTensor` of shape `(batch size, sequence_len, embed_dims)`):
                Conditional embeddings (embeddings computed from the input conditions such as prompts) to use.
            pooled_projections (`torch.FloatTensor` of shape `(batch_size, projection_dim)`): Embeddings projected
                from the embeddings of input conditions.
            timestep ( `torch.LongTensor`):
                Used to indicate denoising step.
            joint_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.transformer_2d.Transformer2DModelOutput`] instead of a plain
                tuple.

        Returns:
            If `return_dict` is True, an [`~models.transformer_2d.Transformer2DModelOutput`] is returned, otherwise a
            `tuple` where the first element is the sample tensor.
        """
        if joint_attention_kwargs is not None:
            joint_attention_kwargs = joint_attention_kwargs.copy()
            lora_scale = joint_attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            logger.warning(
                "Passing `scale` via `joint_attention_kwargs` when not using the PEFT backend is ineffective."
            )

#! ---------------------------------------- MODIFIED BELOW ----------------------------------------
        #* get height & width from runtime state
        height, width = self._get_patch_height_width()

        #* only pp rank 0 needs pos_embed (patchify)
        if get_pipeline_parallel_rank() == 0:
            hidden_states = self.pos_embed(hidden_states)  # takes care of adding positional embeddings too.

        #! ORIGIN:
        # height, width = hidden_states.shape[-2:]
        # hidden_states = self.pos_embed(hidden_states)  # takes care of adding positional embeddings too.
#! ---------------------------------------- MODIFIED ABOVE ----------------------------------------
        temb = self.time_text_embed(timestep, pooled_projections)
        print(95, encoder_hidden_states.shape)
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)
        print(97, encoder_hidden_states.shape)

        for block in self.transformer_blocks:
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    encoder_hidden_states,
                    temb,
                    **ckpt_kwargs,
                )

            else:
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states, encoder_hidden_states=encoder_hidden_states, temb=temb
                )


        #* only the last pp rank needs unpatchify
#! ---------------------------------------- ADD BELOW ----------------------------------------
        if get_pipeline_parallel_rank() == get_pipeline_parallel_world_size() - 1:
#! ---------------------------------------- ADD ABOVE ----------------------------------------
            hidden_states = self.norm_out(hidden_states, temb)
            hidden_states = self.proj_out(hidden_states)

            # unpatchify
            patch_size = self.config.patch_size
            # height = height // patch_size
            # width = width // patch_size

            hidden_states = hidden_states.reshape(
                shape=(hidden_states.shape[0], height, width, patch_size, patch_size, self.out_channels)
            )
            hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states)
            output = hidden_states.reshape(
                shape=(hidden_states.shape[0], self.out_channels, height * patch_size, width * patch_size)
            ), None

            if USE_PEFT_BACKEND:
                # remove `lora_scale` from each PEFT layer
                unscale_lora_layers(self, lora_scale)
#! ---------------------------------------- ADD BELOW ----------------------------------------
        else:
            output = hidden_states, encoder_hidden_states
#! ---------------------------------------- ADD ABOVE ----------------------------------------
        

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)