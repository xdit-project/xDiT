# Adapted from
# https://github.com/huggingface/diffusers/blob/3e1097cb63c724f5f063704df5ede8a18f472f29/src/diffusers/models/transformers/transformer_2d.py

from distrifuser.logger import init_logger

logger = init_logger(__name__)

# from diffusers import Transformer2DModel
from distrifuser.models.diffusers import Transformer2DModel

from diffusers.models.transformers.transformer_2d import Transformer2DModelOutput
from typing import Optional, Dict, Any
import torch
import torch.nn.functional as F
from torch import distributed as dist
from distrifuser.modules.base_module import BaseModule
from distrifuser.utils import DistriConfig


class DistriTransformer2DModel(BaseModule):
    def __init__(self, module: Transformer2DModel, distri_config: DistriConfig):
        super().__init__(module, distri_config)
        self.config = module.config

    def _forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        timestep: Optional[torch.LongTensor] = None,
        class_labels: Optional[torch.LongTensor] = None,
        cross_attention_kwargs: Dict[str, Any] = None,
    ):
        module = self.module

        for block in module.transformer_blocks:
            hidden_states = block(
                hidden_states,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                timestep=timestep,
                cross_attention_kwargs=cross_attention_kwargs,
                class_labels=class_labels,
            )
        return hidden_states

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
            hidden_states (`torch.LongTensor` of shape `(batch size, num latent pixels)` if discrete, `torch.FloatTensor` of shape `(batch size, channel, height, width)` if continuous):
                Input `hidden_states`.
            encoder_hidden_states ( `torch.FloatTensor` of shape `(batch size, sequence len, embed dims)`, *optional*):
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
            If `return_dict` is True, an [`~models.transformer_2d.Transformer2DModelOutput`] is returned, otherwise a
            `tuple` where the first element is the sample tensor.
        """
        module = self.module

        if cross_attention_kwargs is not None:
            if cross_attention_kwargs.get("scale", None) is not None:
                logger.warning(
                    "Passing `scale` to `cross_attention_kwargs` is deprecated. `scale` will be ignored."
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
            encoder_attention_mask = (
                1 - encoder_attention_mask.to(hidden_states.dtype)
            ) * -10000.0
            encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

        # 1. Input
        if module.is_input_continuous:
            batch, _, height, width = hidden_states.shape
            residual = hidden_states

            hidden_states = module.norm(hidden_states)
            if not module.use_linear_projection:
                hidden_states = module.proj_in(hidden_states)
                inner_dim = hidden_states.shape[1]
                hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(
                    batch, height * width, inner_dim
                )
            else:
                inner_dim = hidden_states.shape[1]
                hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(
                    batch, height * width, inner_dim
                )
                hidden_states = module.proj_in(hidden_states)

        elif module.is_input_vectorized:
            hidden_states = module.latent_image_embedding(hidden_states)
        elif module.is_input_patches:
            height, width = (
                hidden_states.shape[-2] // module.patch_size,
                hidden_states.shape[-1] // module.patch_size,
            )
            hidden_states = module.pos_embed(hidden_states)

            if module.adaln_single is not None:
                if module.use_additional_conditions and added_cond_kwargs is None:
                    raise ValueError(
                        "`added_cond_kwargs` cannot be None when using additional conditions for `adaln_single`."
                    )
                batch_size = hidden_states.shape[0]
                timestep, embedded_timestep = module.adaln_single(
                    timestep,
                    added_cond_kwargs,
                    batch_size=batch_size,
                    hidden_dtype=hidden_states.dtype,
                )

        # 2. Blocks
        if module.is_input_patches and module.caption_projection is not None:
            batch_size = hidden_states.shape[0]
            encoder_hidden_states = module.caption_projection(encoder_hidden_states)
            encoder_hidden_states = encoder_hidden_states.view(
                batch_size, -1, hidden_states.shape[-1]
            )

        distri_config = self.distri_config
        if (
            distri_config.world_size > 1
            and distri_config.n_device_per_batch > 1
            and distri_config.parallelism == "patch"
        ):
            b, c, h = hidden_states.shape
            sliced_hidden_states = hidden_states.view(
                b, distri_config.n_device_per_batch, -1, h
            )[:, distri_config.split_idx()]
            output = self._forward(
                hidden_states=sliced_hidden_states,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                timestep=timestep,
                class_labels=class_labels,
                cross_attention_kwargs=cross_attention_kwargs,
            )

            if self.buffer_list is None:
                self.buffer_list = [
                    torch.empty_like(output.view(-1))
                    for _ in range(distri_config.world_size)
                ]
            dist.all_gather(
                self.buffer_list, output.contiguous().view(-1), async_op=False
            )
            buffer_list = [buffer.view(output.shape) for buffer in self.buffer_list]
            hidden_states = torch.cat(buffer_list, dim=1)
        else:
            hidden_states = self._forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                timestep=timestep,
                class_labels=class_labels,
                cross_attention_kwargs=cross_attention_kwargs,
            )

        # 3. Output
        if module.is_input_continuous:
            if not module.use_linear_projection:
                hidden_states = (
                    hidden_states.reshape(batch, height, width, inner_dim)
                    .permute(0, 3, 1, 2)
                    .contiguous()
                )
                hidden_states = module.proj_out(hidden_states)
            else:
                hidden_states = module.proj_out(hidden_states)
                hidden_states = (
                    hidden_states.reshape(batch, height, width, inner_dim)
                    .permute(0, 3, 1, 2)
                    .contiguous()
                )

            output = hidden_states + residual
        elif module.is_input_vectorized:
            hidden_states = module.norm_out(hidden_states)
            logits = module.out(hidden_states)
            # (batch, module.num_vector_embeds - 1, self.num_latent_pixels)
            logits = logits.permute(0, 2, 1)

            # log(p(x_0))
            output = F.log_softmax(logits.double(), dim=1).float()

        if module.is_input_patches:
            if module.config.norm_type != "ada_norm_single":
                conditioning = module.transformer_blocks[0].norm1.emb(
                    timestep, class_labels, hidden_dtype=hidden_states.dtype
                )
                shift, scale = module.proj_out_1(F.silu(conditioning)).chunk(2, dim=1)
                hidden_states = (
                    module.norm_out(hidden_states) * (1 + scale[:, None])
                    + shift[:, None]
                )
                hidden_states = module.proj_out_2(hidden_states)
            elif module.config.norm_type == "ada_norm_single":
                shift, scale = (
                    module.scale_shift_table[None] + embedded_timestep[:, None]
                ).chunk(2, dim=1)
                hidden_states = module.norm_out(hidden_states)
                # Modulation
                hidden_states = hidden_states * (1 + scale) + shift
                hidden_states = module.proj_out(hidden_states)
                hidden_states = hidden_states.squeeze(1)

            # unpatchify
            if module.adaln_single is None:
                height = width = int(hidden_states.shape[1] ** 0.5)
            hidden_states = hidden_states.reshape(
                shape=(
                    -1,
                    height,
                    width,
                    module.patch_size,
                    module.patch_size,
                    module.out_channels,
                )
            )
            hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states)
            output = hidden_states.reshape(
                shape=(
                    -1,
                    module.out_channels,
                    height * module.patch_size,
                    width * module.patch_size,
                )
            )

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)
