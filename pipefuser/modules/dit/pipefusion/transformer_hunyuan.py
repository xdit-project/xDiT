# Adapted from
# https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/transformers/hunyuan_transformer_2d.py

from pipefuser.logger import init_logger

logger = init_logger(__name__)

from diffusers import HunyuanDiT2DModel

from diffusers.models.transformers.transformer_2d import Transformer2DModelOutput
from typing import Optional, Dict, Any
import torch
import torch.nn.functional as F
from torch import distributed as dist
from pipefuser.modules.base_module import BaseModule
from pipefuser.utils import DistriConfig


class DistriHunyuanTransformer2DModel(BaseModule):
    def __init__(self, module: HunyuanDiT2DModel, distri_config: DistriConfig):
        super().__init__(module, distri_config)
        current_rank = (
            distri_config.rank - 1 + distri_config.n_device_per_batch
        ) % distri_config.n_device_per_batch

        # logger.info(f"attn_num {distri_config.attn_num}")
        # logger.info(f"{len{self.module.transformer_blocks}}")

        if distri_config.attn_num is not None:
            assert sum(distri_config.attn_num) == len(self.module.blocks)
            assert len(distri_config.attn_num) == distri_config.n_device_per_batch

            if current_rank == 0:
                self.module.blocks = self.module.blocks[
                    : distri_config.attn_num[0]
                ]
            else:
                self.module.blocks = self.module.blocks[
                    sum(distri_config.attn_num[: current_rank - 1]) : sum(
                        distri_config.attn_num[:current_rank]
                    )
                ]
        else:

            block_len = (
                len(self.module.blocks) + distri_config.n_device_per_batch - 1
            ) // distri_config.n_device_per_batch
            start_idx = block_len * current_rank
            end_idx = min(
                block_len * (current_rank + 1), len(self.module.blocks)
            )
            self.module.blocks = self.module.blocks[start_idx:end_idx]

        if distri_config.rank != 1:
            self.module.pos_embed = None

        self.config = module.config
        self.batch_idx = 0
        
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
        module = self.module
        distri_config = self.distri_config
        assert controlnet_block_samples is None

        if distri_config.rank == 0:
            height, width = (
                distri_config.height // 8,
                distri_config.width // 8,
            )
            if (
                self.counter <= distri_config.warmup_steps
                or distri_config.mode == "full_sync"
            ):
                pass
            else:
                height //= distri_config.pp_num_patch

        if distri_config.rank == 1:
            hidden_states = module.pos_embed(hidden_states)

        temb = module.time_extra_emb(
            timestep, encoder_hidden_states_t5, image_meta_size, style, hidden_dtype=timestep.dtype
        )  # [B, D]

        # text projection
        batch_size, sequence_length, _ = encoder_hidden_states_t5.shape
        encoder_hidden_states_t5 = module.text_embedder(
            encoder_hidden_states_t5.view(-1, encoder_hidden_states_t5.shape[-1])
        )
        encoder_hidden_states_t5 = encoder_hidden_states_t5.view(batch_size, sequence_length, -1)

        encoder_hidden_states = torch.cat([encoder_hidden_states, encoder_hidden_states_t5], dim=1)
        text_embedding_mask = torch.cat([text_embedding_mask, text_embedding_mask_t5], dim=-1)
        text_embedding_mask = text_embedding_mask.unsqueeze(2).bool()

        encoder_hidden_states = torch.where(text_embedding_mask, encoder_hidden_states, module.text_embedding_padding)

        if distri_config.rank == 0 or distri_config.rank > distri_config.n_device_per_batch//2:
            assert skips is not None
            skips = list(skips.unbind(0)) 
            for layer, block in enumerate(module.blocks):
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
            for layer, block in enumerate(module.blocks):
                hidden_states = block(
                    hidden_states,
                    temb=temb,
                    encoder_hidden_states=encoder_hidden_states,
                    image_rotary_emb=image_rotary_emb,
                )  # (N, L, D)
                skips.append(hidden_states)
            skips = torch.stack(skips, dim=0)

        if distri_config.rank == 0:
            # final layer
            hidden_states = module.norm_out(hidden_states, temb.to(torch.float32))
            hidden_states = module.proj_out(hidden_states)
            # (N, L, patch_size ** 2 * out_channels)

            # unpatchify: (N, out_channels, H, W)
            patch_size = module.config.patch_size
            height = height // patch_size
            width = width // patch_size

            hidden_states = hidden_states.reshape(
                shape=(hidden_states.shape[0], height, width, patch_size, patch_size, module.out_channels)
            )
            hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states)
            output = hidden_states.reshape(
                shape=(hidden_states.shape[0], module.out_channels, height * patch_size, width * patch_size)
            )
        elif distri_config.rank > distri_config.n_device_per_batch//2:
            output = hidden_states
        else:
            output = hidden_states, skips
        
        
        if (
            distri_config.mode == "full_sync"
            or self.counter <= distri_config.warmup_steps
        ):
            self.counter += 1
        else:
            self.batch_idx += 1
            if self.batch_idx == distri_config.pp_num_patch:
                self.counter += 1
                self.batch_idx = 0
            
        if not return_dict:
            return (output,)
        
        return Transformer2DModelOutput(sample=output)