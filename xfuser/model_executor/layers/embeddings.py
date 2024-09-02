# adapted from https://github.com/huggingface/diffusers/blob/v0.29.0/src/diffusers/models/embeddings.py
import torch

from diffusers.models.embeddings import PatchEmbed, get_2d_sincos_pos_embed, CogVideoXPatchEmbed
import torch.distributed
from xfuser.core.distributed.runtime_state import get_runtime_state
from xfuser.model_executor.layers import xFuserLayerBaseWrapper
from xfuser.model_executor.layers import xFuserLayerWrappersRegister
from xfuser.logger import init_logger

logger = init_logger(__name__)


@xFuserLayerWrappersRegister.register(PatchEmbed)
class xFuserPatchEmbedWrapper(xFuserLayerBaseWrapper):
    def __init__(
        self,
        patch_embedding: PatchEmbed,
    ):
        super().__init__(
            module=patch_embedding,
        )
        self.module: PatchEmbed
        self.pos_embed = None

    def forward(self, latent):
        height = (
            get_runtime_state().input_config.height
            // get_runtime_state().vae_scale_factor
        )
        width = latent.shape[-1]
        if not get_runtime_state().patch_mode:
            if getattr(self.module, "pos_embed_max_size", None) is not None:
                pass
            else:
                height, width = (
                    height // self.module.patch_size,
                    width // self.module.patch_size,
                )
        else:
            if getattr(self.module, "pos_embed_max_size", None) is not None:
                pass
            else:
                height, width = (
                    height // self.module.patch_size,
                    width // self.module.patch_size,
                )

        latent = self.module.proj(latent)
        if self.module.flatten:
            latent = latent.flatten(2).transpose(1, 2)  # BCHW -> BNC
        if self.module.layer_norm:
            # TODO: NOT SURE whether compatible with norm
            latent = self.module.norm(latent)

        # [2, 4096 / c, 1152]

        if self.module.pos_embed is None:
            return latent.to(latent.dtype)

        # Interpolate positional embeddings if needed.
        # (For PixArt-Alpha: https://github.com/PixArt-alpha/PixArt-alpha/blob/0f55e922376d8b797edd44d25d0e7464b260dcab/diffusion/model/nets/PixArtMS.py#L162C151-L162C160)

        # TODO: There might be a more faster way to generate a smaller pos_embed
        if getattr(self.module, "pos_embed_max_size", None):
            pos_embed = self.module.cropped_pos_embed(height, width)
        else:
            if self.module.height != height or self.module.width != width:
                pos_embed = get_2d_sincos_pos_embed(
                    embed_dim=self.module.pos_embed.shape[-1],
                    grid_size=(height, width),
                    base_size=self.module.base_size,
                    interpolation_scale=self.module.interpolation_scale,
                )
                pos_embed = torch.from_numpy(pos_embed)
                self.module.pos_embed = pos_embed.float().unsqueeze(0).to(latent.device)
                self.module.height = height
                self.module.width = width
                pos_embed = self.module.pos_embed
            else:
                pos_embed = self.module.pos_embed
        b, c, h = pos_embed.shape

        if get_runtime_state().patch_mode:
            start, end = get_runtime_state().pp_patches_token_start_end_idx_global[
                get_runtime_state().pipeline_patch_idx
            ]
            pos_embed = pos_embed[
                :,
                start:end,
                :,
            ]
        else:
            pos_embed_list = [
                pos_embed[
                    :,
                    get_runtime_state()
                    .pp_patches_token_start_end_idx_global[i][0] : get_runtime_state()
                    .pp_patches_token_start_end_idx_global[i][1],
                    :,
                ]
                for i in range(get_runtime_state().num_pipeline_patch)
            ]
            pos_embed = torch.cat(pos_embed_list, dim=1)

        return (latent + pos_embed).to(latent.dtype)


@xFuserLayerWrappersRegister.register(CogVideoXPatchEmbed)
class xFuserCogVideoXPatchEmbedWrapper(xFuserLayerBaseWrapper):
    def __init__(
        self,
        patch_embedding: CogVideoXPatchEmbed,
    ):
        super().__init__(
            module=patch_embedding,
        )

    def forward(self, text_embeds: torch.Tensor, image_embeds: torch.Tensor):
        r"""
        Args:
            text_embeds (`torch.Tensor`):
                Input text embeddings. Expected shape: (batch_size, seq_length, embedding_dim).
            image_embeds (`torch.Tensor`):
                Input image embeddings. Expected shape: (batch_size, num_frames, channels, height, width).
        """
        text_embeds = self.text_proj(text_embeds)
        batch, num_frames, channels, height, width = image_embeds.shape
        if torch.distributed.get_rank() == 0: print(f"ori image_embeds.shape: {image_embeds.shape}")
        image_embeds = image_embeds.reshape(-1, channels, height, width)
        if torch.distributed.get_rank() == 0: print(f"reshape image_embeds.shape: {image_embeds.shape}")
        image_embeds = self.proj(image_embeds)
        if torch.distributed.get_rank() == 0: print(f"proj image_embeds.shape: {image_embeds.shape}")
        image_embeds = image_embeds.view(batch, num_frames, *image_embeds.shape[1:])
        if torch.distributed.get_rank() == 0: print(f"view image_embeds.shape: {image_embeds.shape}")
        image_embeds = image_embeds.flatten(3)
        if torch.distributed.get_rank() == 0: print(f"flatten(3) image_embeds.shape: {image_embeds.shape}")
        image_embeds = image_embeds.transpose(2, 3)  # [batch, num_frames, height x width, channels]
        if torch.distributed.get_rank() == 0: print(f".transpose(2, 3) image_embeds.shape: {image_embeds.shape}")
        image_embeds = image_embeds.flatten(1, 2)  # [batch, num_frames x height x width, channels]
        if torch.distributed.get_rank() == 0: print(f"flatten(1, 2) image_embeds.shape: {image_embeds.shape}")
        # if get_runtime_state().patch_mode:
        #     start, end = get_runtime_state().pp_patches_token_start_end_idx_global[
        #         get_runtime_state().pipeline_patch_idx
        #     ]
        #     image_embeds = image_embeds[
        #         :,
        #         start:end,
        #         :,
        #     ]
        # else:
        #     image_embeds_list = [
        #         image_embeds[
        #             :,
        #             get_runtime_state()
        #             .pp_patches_token_start_end_idx_global[i][0] : get_runtime_state()
        #             .pp_patches_token_start_end_idx_global[i][1],
        #             :,
        #         ]
        #         for i in range(get_runtime_state().num_pipeline_patch)
        #     ]
        #     image_embeds = torch.cat(image_embeds_list, dim=1)
        
        embeds = torch.cat(
            [text_embeds, image_embeds], dim=1
        ).contiguous()  # [batch, seq_length + num_frames x height x width, channels]
        return embeds
