from typing import Optional, Union, Dict, Any, Tuple

import torch
from diffusers import SanaTransformer2DModel
from diffusers.models.modeling_outputs import Transformer2DModelOutput

from xfuser.logger import init_logger
from diffusers.models.embeddings import PatchEmbed
from diffusers.utils import (
    scale_lora_layers,
    USE_PEFT_BACKEND,
    unscale_lora_layers,
)
from xfuser.core.distributed.runtime_state import get_runtime_state
from xfuser.model_executor.base_wrapper import xFuserBaseWrapper
from xfuser.core.distributed import (
    get_pipeline_parallel_rank,
    get_pipeline_parallel_world_size,
    get_ulysses_parallel_rank,
    is_pipeline_first_stage,
    is_pipeline_last_stage,
)
from .register import xFuserTransformerWrappersRegister
from .base_transformer import xFuserTransformerBaseWrapper

logger = init_logger(__name__)

@xFuserTransformerWrappersRegister.register(SanaTransformer2DModel)
class xFuserSanaTransformer2DWrapper(xFuserTransformerBaseWrapper):
    def __init__(self, transformer: SanaTransformer2DModel):
        super().__init__(
            transformer=transformer,
            submodule_classes_to_wrap=[PatchEmbed],
            submodule_name_to_wrap=['attn1', 'attn2', 'ff']
            )
        
        # Used for pipefusion. Encoder_hidden_state only need to be calculated once at patch 0
        self.encoder_hidden_state_cache = [
            None for _ in range(len(self.transformer_blocks))
        ]

        self.attention_mask = None
        self.encoder_attention_mask = None

    def forward(self,
                hidden_states: torch.Tensor,
                encoder_hidden_states: torch.Tensor,
                timestep: torch.Tensor,
                guidance: Optional[torch.Tensor] = None,
                encoder_attention_mask: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                attention_kwargs: Optional[Dict[str, Any]] = None,
                return_dict: bool = True,
            ) -> Union[Tuple[torch.Tensor, ...], Transformer2DModelOutput]:

        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if attention_kwargs is not None and attention_kwargs.get("scale", None) is not None:
                logger.warning(
                    "Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective."
                )

        # ensure attention_mask is a bias, and give it a singleton query_tokens dimension.
        #   we may have done this conversion already, e.g. if we came here via UNet2DConditionModel#forward.
        #   we can tell by counting dims; if ndim == 2: it's a mask rather than a bias.
        # expects mask of shape:
        #   [batch, key_tokens]
        # adds singleton query_tokens dimension:
        #   [batch,                    1, key_tokens]
        # this helps to broadcast it as a bias over attention scores, which will be in one of the following shapes:
        #   [batch,  heads, query_tokens, key_tokens] (e.g. torch sdp attn)
        #   [batch * heads, query_tokens, key_tokens] (e.g. xformers or classic attn)
        
        if attention_mask is not None and attention_mask.ndim == 2:
            # assume that mask is expressed as:
            #   (1 = keep,      0 = discard)
            # convert mask into a bias that can be added to attention scores:
            #       (keep = +0,     discard = -10000.0)
            attention_mask = (1 - attention_mask.to(hidden_states.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        # convert encoder_attention_mask to a bias the same way we do for attention_mask
        if encoder_attention_mask is not None and encoder_attention_mask.ndim == 2:
            encoder_attention_mask = (1 - encoder_attention_mask.to(hidden_states.dtype)) * -10000.0
            encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

        if attention_mask is not None:
            self.attention_mask = attention_mask
        if encoder_attention_mask is not None:
            self.encoder_attention_mask = encoder_attention_mask
            
        # 1. Input
        # batch_size, num_channels, _, _ = hidden_states.shape
        batch_size = hidden_states.size(0)
        height, width = self._get_patch_height_width()
        p = self.config.patch_size
        post_patch_height, post_patch_width = height // p, width // p

        if is_pipeline_first_stage():
            hidden_states = self.patch_embed(hidden_states)

        if guidance is not None:
            timestep, embedded_timestep = self.time_embed(
                timestep, guidance=guidance, hidden_dtype=hidden_states.dtype
            )
        else:
            timestep, embedded_timestep = self.time_embed(
                timestep, batch_size=batch_size, hidden_dtype=hidden_states.dtype
            )

        encoder_hidden_states = self.caption_projection(encoder_hidden_states)
        encoder_hidden_states = encoder_hidden_states.view(batch_size, -1, hidden_states.shape[-1])

        encoder_hidden_states = self.caption_norm(encoder_hidden_states)

        # 2. Transformer blocks
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            assert "Currently, we don't support training with gradient checkpointing."
            for block in self.transformer_blocks:
                hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    attention_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    timestep,
                    post_patch_height,
                    post_patch_width,
                )

        else:
            for block in self.transformer_blocks:
                hidden_states = block(
                    hidden_states,
                    self.attention_mask,
                    encoder_hidden_states,
                    self.encoder_attention_mask,
                    timestep,
                    post_patch_height,
                    post_patch_width,
                )

        if is_pipeline_last_stage():
            # 3. Normalization
            hidden_states = self.norm_out(hidden_states, embedded_timestep, self.scale_shift_table)

            hidden_states = self.proj_out(hidden_states)

            # 5. Unpatchify
            hidden_states = hidden_states.reshape(
                batch_size, post_patch_height, post_patch_width, self.config.patch_size, self.config.patch_size, -1
            )
            hidden_states = hidden_states.permute(0, 5, 1, 3, 2, 4)
            output = hidden_states.reshape(batch_size, -1, post_patch_height * p, post_patch_width * p)

            if USE_PEFT_BACKEND:
                # remove `lora_scale` from each PEFT layer
                unscale_lora_layers(self, lora_scale)
        else:
            output = hidden_states

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)
