# Adapted from
# https://github.com/huggingface/diffusers/blob/3e1097cb63c724f5f063704df5ede8a18f472f29/src/diffusers/models/transformers/transformer_2d.py

from pipefuser.logger import init_logger

logger = init_logger(__name__)

# from diffusers import Transformer2DModel
from diffusers.models.transformers.transformer_sd3 import SD3Transformer2DModel

from diffusers.models.transformers.transformer_2d import Transformer2DModelOutput
from typing import Optional, Dict, Any, Union, Tuple
import torch
import torch.nn.functional as F
from torch import distributed as dist
from pipefuser.models.base_model import BaseModule, BaseModel
from pipefuser.utils import DistriConfig


class DistriSD3Transformer2DModel(BaseModule):
    def __init__(self, module: SD3Transformer2DModel, distri_config: DistriConfig):
        super().__init__(module, distri_config)
        current_rank = (
            distri_config.rank - 1 + distri_config.world_size
        ) % distri_config.world_size

        # logger.info(f"attn_num {distri_config.attn_num}")
        # logger.info(f"{len{self.module.transformer_blocks}}")

        if distri_config.attn_num is not None:
            assert sum(distri_config.attn_num) == len(self.module.transformer_blocks)
            assert len(distri_config.attn_num) == distri_config.world_size

            if current_rank == 0:
                self.module.transformer_blocks = self.module.transformer_blocks[
                    : distri_config.attn_num[0]
                ]
            else:
                self.module.transformer_blocks = self.module.transformer_blocks[
                    sum(distri_config.attn_num[: current_rank - 1]) : sum(
                        distri_config.attn_num[:current_rank]
                    )
                ]
        else:

            block_len = (
                len(self.module.transformer_blocks) + distri_config.world_size - 1
            ) // distri_config.world_size
            start_idx = block_len * current_rank
            end_idx = min(
                block_len * (current_rank + 1), len(self.module.transformer_blocks)
            )
            self.module.transformer_blocks = self.module.transformer_blocks[
                start_idx:end_idx
            ]

        if distri_config.rank != 1:
            self.module.pos_embed = None

        self.config = module.config
        self.batch_idx = 0

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
        module = self.module
        distri_config = self.distri_config
        is_warmup = distri_config.mode == "full_sync" or self.counter <= distri_config.warmup_steps
        if joint_attention_kwargs is not None:
            joint_attention_kwargs = joint_attention_kwargs.copy()
            lora_scale = joint_attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        # if USE_PEFT_BACKEND:
        #     # weight the lora layers by setting `lora_scale` for each PEFT layer
        #     scale_lora_layers(module, lora_scale)
        # else:
        #     logger.warning(
        #         "Passing `scale` via `joint_attention_kwargs` when not using the PEFT backend is ineffective."
        #     )

        if distri_config.rank == 1:
            hidden_states = module.pos_embed(hidden_states)  # takes care of adding positional embeddings too.
            encoder_hidden_states = module.context_embedder(encoder_hidden_states)
        
        temb = module.time_text_embed(timestep, pooled_projections)

        for block in module.transformer_blocks:
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states, encoder_hidden_states=encoder_hidden_states, temb=temb
            )
        
        if distri_config.rank == 0:

            hidden_states = module.norm_out(hidden_states, temb)
            hidden_states = module.proj_out(hidden_states)

            patch_size = module.config.patch_size
            height, width = hidden_states.shape[-2] // module.out_channels // patch_size, hidden_states.shape[-1] * patch_size
            # if not is_warmup:
                # height //= distri_config.pp_num_patch

            # unpatchify
            height = height // patch_size
            width = width // patch_size

            hidden_states = hidden_states.reshape(
                shape=(hidden_states.shape[0], height, width, patch_size, patch_size, module.out_channels)
            )
            hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states)
            output = hidden_states.reshape(
                shape=(hidden_states.shape[0], module.out_channels, height * patch_size, width * patch_size)
            ), None
    
            # if USE_PEFT_BACKEND:
            #     # remove `lora_scale` from each PEFT layer
            #     unscale_lora_layers(module, lora_scale)
        
        else:
            output = hidden_states, encoder_hidden_states

        if is_warmup:
            self.counter += 1
        else:
            self.batch_idx += 1
            if self.batch_idx == distri_config.pp_num_patch:
                self.counter += 1
                self.batch_idx = 0

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)
