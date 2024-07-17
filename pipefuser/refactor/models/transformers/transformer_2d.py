# Adapted from
# https://github.com/huggingface/diffusers/blob/3e1097cb63c724f5f063704df5ede8a18f472f29/src/diffusers/models/transformers/transformer_2d.py
from typing import Optional, Dict, Any
import torch
import torch.nn as nn

from diffusers import Transformer2DModel
from diffusers.models.embeddings import PatchEmbed
from diffusers.models.transformers.transformer_2d import (
    Transformer2DModelOutput)
from diffusers.utils import is_torch_version

from pipefuser.refactor.models.transformers import (
    PipeFuserTransformerBaseWrapper
)
from .register import PipeFuserTransformerWrappersRegister
from pipefuser.refactor.config.config import InputConfig, ParallelConfig, RuntimeConfig
from pipefuser.refactor.distributed.parallel_state import (
    get_pipeline_parallel_rank,
    get_pipeline_parallel_world_size,
)
from pipefuser.logger import init_logger

logger = init_logger(__name__)

@PipeFuserTransformerWrappersRegister.register(Transformer2DModel)
class PipeFuserTransformer2DWrapper(PipeFuserTransformerBaseWrapper):
    def __init__(
        self, 
        transformer: Transformer2DModel,
        parallel_config: ParallelConfig,
        runtime_config: RuntimeConfig,
    ):
        super().__init__(
            self,
            transformer=transformer,
            parallel_config=parallel_config,
            runtime_config=runtime_config,
            submodule_classes_to_wrap=[nn.Conv2d, PatchEmbed],
            submodule_name_to_wrap=["attn1"],
        )

    @PipeFuserTransformerBaseWrapper.forward_check_condition
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        timestep: Optional[torch.LongTensor] = None,
        added_cond_kwargs: Dict[str, torch.Tensor] = None,
        class_labels: Optional[torch.LongTensor] = None,
        cross_attention_kwargs: Dict[str, Any] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ):
        """
        The [`Transformer2DModel`] forward method.

        Args:
            hidden_states (`torch.LongTensor` of shape `(batch size, num latent pixels)` if discrete, `torch.Tensor` of shape `(batch size, channel, height, width)` if continuous):
                Input `hidden_states`.
            encoder_hidden_states ( `torch.Tensor` of shape `(batch size, sequence len, embed dims)`, *optional*):
                Conditional embeddings for cross attention layer. If not given, cross-attention defaults to
                self-attention.
            timestep ( `torch.LongTensor`, *optional*):
                Used to indicate denoising step. Optional timestep to be applied as an embedding in `AdaLayerNorm`.
            class_labels ( `torch.LongTensor` of shape `(batch size, num classes)`, *optional*):
                Used to indicate class labels conditioning. Optional class labels to be applied as an embedding in
                `AdaLayerZeroNorm`.
            cross_attention_kwargs ( `Dict[str, Any]`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            attention_mask ( `torch.Tensor`, *optional*):
                An attention mask of shape `(batch, key_tokens)` is applied to `encoder_hidden_states`. If `1` the mask
                is kept, otherwise if `0` it is discarded. Mask will be converted into a bias, which adds large
                negative values to the attention scores corresponding to "discard" tokens.
            encoder_attention_mask ( `torch.Tensor`, *optional*):
                Cross-attention mask applied to `encoder_hidden_states`. Two formats supported:

                    * Mask `(batch, sequence_length)` True = keep, False = discard.
                    * Bias `(batch, 1, sequence_length)` 0 = keep, -10000 = discard.

                If `ndim == 2`: will be interpreted as a mask, then converted into a bias consistent with the format
                above. This bias will be added to the cross-attention scores.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.unets.unet_2d_condition.UNet2DConditionOutput`] instead of a plain
                tuple.

        Returns:
            If `return_dict` is True, an [`~models.transformers.transformer_2d.Transformer2DModelOutput`] is returned,
            otherwise a `tuple` where the first element is the sample tensor.
        """
        if get_pipeline_parallel_world_size() == 1:
            return self.module(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                timestep=timestep,
                added_cond_kwargs=added_cond_kwargs,
                class_labels=class_labels,
                cross_attention_kwargs=cross_attention_kwargs,
                attention_mask=attention_mask,
                encoder_attention_mask=encoder_attention_mask,
                return_dict=return_dict,
            )
        assert not self.is_input_continuous, (
            "Continuous inputs are not supported in pipeline parallelism yet")
        if cross_attention_kwargs is not None:
            if cross_attention_kwargs.get("scale", None) is not None:
                logger.warning("Passing `scale` to `cross_attention_kwargs` is deprecated. `scale` will be ignored.")
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
            attention_mask = (1 - attention_mask.to(hidden_states.dtype)
                              ) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        # convert encoder_attention_mask to a bias the same way we do for attention_mask
        if (encoder_attention_mask is not None and 
            encoder_attention_mask.ndim == 2):
            encoder_attention_mask = (
                1 - encoder_attention_mask.to(hidden_states.dtype)) * -10000.0
            encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

        # 1. Input
        if self.is_input_continuous:
            #TODO residual processing (rank 1 to rank 0 p2p...)
            residual = hidden_states
            if get_pipeline_parallel_rank() == 0:
                hidden_states, inner_dim = \
                    self._operate_on_continuous_inputs(hidden_states)
            else:
                _, inner_dim = self._operate_on_continuous_inputs(hidden_states)
        elif self.is_input_vectorized:
            if get_pipeline_parallel_rank() == 0:
                hidden_states = self.latent_image_embedding(hidden_states)
        elif self.is_input_patches:
            (hidden_states, encoder_hidden_states, timestep, embedded_timestep
             ) = self._operate_on_patched_inputs(
                 hidden_states, encoder_hidden_states, timestep, 
                 added_cond_kwargs
                 )

        # 2. Blocks
        for block in self.transformer_blocks:
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = (
                    {"use_reentrant": False} 
                    if is_torch_version(">=", "1.11.0") else {}
                )
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    attention_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    timestep,
                    cross_attention_kwargs,
                    class_labels,
                    **ckpt_kwargs,
                )
            else:
                hidden_states = block(
                    hidden_states,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    timestep=timestep,
                    cross_attention_kwargs=cross_attention_kwargs,
                    class_labels=class_labels,
                )

        # 3. Output
        if get_pipeline_parallel_rank() == get_pipeline_parallel_world_size() - 1:
            if self.is_input_continuous:
                batch_size = hidden_states.shape[0]
                height, width = (
                    self.input_config.height // 8,
                    self.input_config.width // 8
                )
                output = self._get_output_for_continuous_inputs(
                    hidden_states=hidden_states,
                    residual=residual,
                    batch_size=batch_size,
                    height=height,
                    width=width,
                    inner_dim=inner_dim,
                )
            elif self.is_input_vectorized:
                output = self._get_output_for_vectorized_inputs(hidden_states)
            elif self.is_input_patches:
                #* height & width must obtain from input_config
                height, width = (
                    self.input_config.height // self.patch_size // 8, 
                    self.input_config.width // self.patch_size // 8
                )
                if self.patched_mode:
                    height //= self.num_pipeline_patch
                 
                output = self._get_output_for_patched_inputs(
                    hidden_states=hidden_states,
                    timestep=timestep,
                    class_labels=class_labels,
                    embedded_timestep=embedded_timestep,
                    height=height,
                    width=width,
                )
        else:
            output = hidden_states

        # if self.in_warmup_stage():
        #     self.round_step()
        # else:
        #     self.patch_step()

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)

    def _operate_on_continuous_inputs(self, hidden_states):
        batch, _, height, width = hidden_states.shape
        hidden_states = self.norm(hidden_states)

        if not self.use_linear_projection:
            hidden_states = self.proj_in(hidden_states)
            inner_dim = hidden_states.shape[1]
            hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(
                batch, height * width, inner_dim)
        else:
            inner_dim = hidden_states.shape[1]
            hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(
                batch, height * width, inner_dim)
            hidden_states = self.proj_in(hidden_states)

        return hidden_states, inner_dim

    def _operate_on_patched_inputs(self, hidden_states, encoder_hidden_states, 
                                   timestep, added_cond_kwargs):
        batch_size = hidden_states.shape[0]
        embedded_timestep = None

        if get_pipeline_parallel_rank() == 0:
            hidden_states = self.pos_embed(hidden_states)

        if self.adaln_single is not None:
            if self.use_additional_conditions and added_cond_kwargs is None:
                raise ValueError(
                    "`added_cond_kwargs` cannot be None when using additional "
                    "conditions for `adaln_single`."
                )
            timestep, embedded_timestep = self.adaln_single(
                timestep, added_cond_kwargs, batch_size=batch_size, 
                hidden_dtype=hidden_states.dtype
            )

        if self.caption_projection is not None:
            encoder_hidden_states = self.caption_projection(
                encoder_hidden_states)
            encoder_hidden_states = encoder_hidden_states.view(
                batch_size, -1, hidden_states.shape[-1])

        return hidden_states, encoder_hidden_states, timestep, embedded_timestep