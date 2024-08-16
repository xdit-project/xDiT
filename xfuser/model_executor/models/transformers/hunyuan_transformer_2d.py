from typing import Optional, Dict, Any
import torch
import torch.distributed
import torch.nn as nn

from diffusers import HunyuanDiT2DModel
from diffusers.models.embeddings import PatchEmbed
from diffusers.models.attention_processor import Attention
from diffusers.models.transformers.transformer_2d import Transformer2DModelOutput
from diffusers.utils import is_torch_version

from xfuser.logger import init_logger
from xfuser.distributed.runtime_state import get_runtime_state
from xfuser.model_executor.base_wrapper import xFuserBaseWrapper
from xfuser.distributed import (
    get_pipeline_parallel_rank,
    get_pipeline_parallel_world_size,
    is_pipeline_first_stage,
    is_pipeline_last_stage,
)
from .register import xFuserTransformerWrappersRegister
from .base_transformer import xFuserTransformerBaseWrapper

logger = init_logger(__name__)


@xFuserTransformerWrappersRegister.register(HunyuanDiT2DModel)
class xFuserHunyuanDiT2DWrapper(xFuserTransformerBaseWrapper):
    def __init__(
        self,
        transformer: HunyuanDiT2DModel,
    ):
        super().__init__(
            transformer=transformer,
            submodule_classes_to_wrap=[nn.Conv2d, PatchEmbed],
            submodule_name_to_wrap=["attn1"],
        )

    def _split_transformer_blocks(
        self,
        transformer: nn.Module,
    ):
        if not hasattr(transformer, "blocks"):
            raise AttributeError(
                f"'{transformer.__class__.__name__}' object has no attribute 'blocks'"
            )

        # transformer layer split
        attn_layer_num_for_pp = (
            get_runtime_state().parallel_config.pp_config.attn_layer_num_for_pp
        )
        pp_rank = get_pipeline_parallel_rank()
        pp_world_size = get_pipeline_parallel_world_size()
        if attn_layer_num_for_pp is not None:
            assert sum(attn_layer_num_for_pp) == len(transformer.blocks), (
                "Sum of attn_layer_num_for_pp should be equal to the "
                "number of transformer blocks"
            )
            if is_pipeline_first_stage():
                transformer.blocks = transformer.blocks[: attn_layer_num_for_pp[0]]
            else:
                transformer.blocks = transformer.blocks[
                    sum(attn_layer_num_for_pp[: pp_rank - 1]) : sum(
                        attn_layer_num_for_pp[:pp_rank]
                    )
                ]
        else:
            num_blocks_per_stage = (
                len(transformer.blocks) + pp_world_size - 1
            ) // pp_world_size
            start_idx = pp_rank * num_blocks_per_stage
            end_idx = min(
                (pp_rank + 1) * num_blocks_per_stage,
                len(transformer.blocks),
            )
            transformer.blocks = transformer.blocks[start_idx:end_idx]
        # position embedding
        if not is_pipeline_first_stage():
            transformer.pos_embed = None
        if not is_pipeline_last_stage():
            transformer.norm_out = None
            transformer.proj_out = None
        return transformer

    @xFuserBaseWrapper.forward_check_condition
    def forward(
        self,
        hidden_states,
        timestep,
        encoder_hidden_states=None,
        text_embedding_mask=None,
        encoder_hidden_states_t5=None,
        text_embedding_mask_t5=None,
        image_meta_size=None,
        style=None,
        image_rotary_emb=None,
        skips=None,
        controlnet_block_samples=None,
        return_dict=True,
    ):
        """
        The [`HunyuanDiT2DModel`] forward method.

        Args:
        hidden_states (`torch.Tensor` of shape `(batch size, dim, height, width)`):
            The input tensor.
        timestep ( `torch.LongTensor`, *optional*):
            Used to indicate denoising step.
        encoder_hidden_states ( `torch.Tensor` of shape `(batch size, sequence len, embed dims)`, *optional*):
            Conditional embeddings for cross attention layer. This is the output of `BertModel`.
        text_embedding_mask: torch.Tensor
            An attention mask of shape `(batch, key_tokens)` is applied to `encoder_hidden_states`. This is the output
            of `BertModel`.
        encoder_hidden_states_t5 ( `torch.Tensor` of shape `(batch size, sequence len, embed dims)`, *optional*):
            Conditional embeddings for cross attention layer. This is the output of T5 Text Encoder.
        text_embedding_mask_t5: torch.Tensor
            An attention mask of shape `(batch, key_tokens)` is applied to `encoder_hidden_states`. This is the output
            of T5 Text Encoder.
        image_meta_size (torch.Tensor):
            Conditional embedding indicate the image sizes
        style: torch.Tensor:
            Conditional embedding indicate the style
        image_rotary_emb (`torch.Tensor`):
            The image rotary embeddings to apply on query and key tensors during attention calculation.
        return_dict: bool
            Whether to return a dictionary.
        """
        #! ---------------------------------------- MODIFIED BELOW ----------------------------------------
        assert controlnet_block_samples is None
        # * get height & width from runtime state
        height, width = self._get_patch_height_width()
        # * only pp rank 0 needs pos_embed (patchify)
        if is_pipeline_first_stage():
            hidden_states = self.pos_embed(hidden_states)

        #! ORIGIN
        # height, width = hidden_states.shape[-2:]

        # hidden_states = self.pos_embed(hidden_states)
        #! ---------------------------------------- MODIFIED ABOVE ----------------------------------------

        temb = self.time_extra_emb(
            timestep,
            encoder_hidden_states_t5,
            image_meta_size,
            style,
            hidden_dtype=timestep.dtype,
        )  # [B, D]

        # text projection
        batch_size, sequence_length, _ = encoder_hidden_states_t5.shape
        encoder_hidden_states_t5 = self.text_embedder(
            encoder_hidden_states_t5.view(-1, encoder_hidden_states_t5.shape[-1])
        )
        encoder_hidden_states_t5 = encoder_hidden_states_t5.view(
            batch_size, sequence_length, -1
        )

        encoder_hidden_states = torch.cat(
            [encoder_hidden_states, encoder_hidden_states_t5], dim=1
        )
        text_embedding_mask = torch.cat(
            [text_embedding_mask, text_embedding_mask_t5], dim=-1
        )
        text_embedding_mask = text_embedding_mask.unsqueeze(2).bool()

        encoder_hidden_states = torch.where(
            text_embedding_mask, encoder_hidden_states, self.text_embedding_padding
        )

        #! ---------------------------------------- MODIFIED BELOW ----------------------------------------
        if get_pipeline_parallel_world_size() == 1:
            skips = []
            num_layers = len(self.blocks)
            for layer, block in enumerate(self.blocks):
                if layer > num_layers // 2:
                    skip = skips.pop()
                    hidden_states = block(
                        hidden_states,
                        temb=temb,
                        encoder_hidden_states=encoder_hidden_states,
                        image_rotary_emb=image_rotary_emb,
                        skip=skip,
                    )  # (N, L, D)
                else:
                    hidden_states = block(
                        hidden_states,
                        temb=temb,
                        encoder_hidden_states=encoder_hidden_states,
                        image_rotary_emb=image_rotary_emb,
                    )  # (N, L, D)
                if layer < (num_layers // 2 - 1):
                    skips.append(hidden_states)

        else:
            if get_pipeline_parallel_rank() >= get_pipeline_parallel_world_size() // 2:
                assert skips is not None
                skips = list(skips.unbind(0))
                for layer, block in enumerate(self.blocks):
                    skip = skips.pop()
                    hidden_states = block(
                        hidden_states,
                        temb=temb,
                        encoder_hidden_states=encoder_hidden_states,
                        image_rotary_emb=image_rotary_emb,
                        skip=skip,
                    )  # (N, L, D)
                assert len(skips) == 0
            else:
                skips = []
                for layer, block in enumerate(self.blocks):
                    hidden_states = block(
                        hidden_states,
                        temb=temb,
                        encoder_hidden_states=encoder_hidden_states,
                        image_rotary_emb=image_rotary_emb,
                    )  # (N, L, D)
                    skips.append(hidden_states)
                skips = torch.stack(skips, dim=0)
        #! ---------------------------------------- MODIFIED ABOVE ----------------------------------------

        # * only the last pp rank needs unpatchify
        #! ---------------------------------------- ADD BELOW ----------------------------------------
        if is_pipeline_last_stage():
            #! ---------------------------------------- ADD ABOVE ----------------------------------------
            # final layer
            hidden_states = self.norm_out(hidden_states, temb.to(torch.float32))
            hidden_states = self.proj_out(hidden_states)
            # (N, L, patch_size ** 2 * out_channels)

            # unpatchify: (N, out_channels, H, W)
            patch_size = get_runtime_state().backbone_patch_size

            hidden_states = hidden_states.reshape(
                shape=(
                    hidden_states.shape[0],
                    height,
                    width,
                    patch_size,
                    patch_size,
                    self.out_channels,
                )
            )
            hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states)
            output = hidden_states.reshape(
                shape=(
                    hidden_states.shape[0],
                    self.out_channels,
                    height * patch_size,
                    width * patch_size,
                )
            )
        #! ---------------------------------------- ADD BELOW ----------------------------------------
        elif get_pipeline_parallel_rank() >= get_pipeline_parallel_world_size() // 2:
            output = hidden_states
        else:
            output = hidden_states, skips
        #! ---------------------------------------- ADD ABOVE ----------------------------------------

        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)
