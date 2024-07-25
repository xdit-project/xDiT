# adapted from https://github.com/huggingface/diffusers/blob/v0.29.0/src/diffusers/models/embeddings.py
import torch

from diffusers.models.embeddings import PatchEmbed, get_2d_sincos_pos_embed
import torch.distributed
from pipefuser.config import ParallelConfig, RuntimeConfig
from pipefuser.model_executor.layers import PipeFuserLayerBaseWrapper
from pipefuser.model_executor.layers import PipeFuserLayerWrappersRegister
from pipefuser.logger import init_logger

logger = init_logger(__name__)


@PipeFuserLayerWrappersRegister.register(PatchEmbed)
class PipeFuserPatchEmbedWrapper(PipeFuserLayerBaseWrapper):
    def __init__(
        self,
        patch_embedding: PatchEmbed,
        parallel_config: ParallelConfig,
        runtime_config: RuntimeConfig,
    ):
        super().__init__(
            module=patch_embedding,
            parallel_config=parallel_config,
            runtime_config=runtime_config,
        )
        self.module: PatchEmbed
        self.pos_embed = None

    def forward(self, latent):
        # is_warmup = self.in_warmup_stage()
        height = self.patches_start_idx[-1][-1]
        width = latent.shape[-1]
        if not self.patched_mode:
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

        if self.patched_mode:
            pos_embed = pos_embed[
                :, 
                self.pp_patches_token_start_end_idx[self.current_patch_idx][0]:
                self.pp_patches_token_start_end_idx[self.current_patch_idx][1],
                :,
            ]
        else:
            pos_embed_list = [
                pos_embed[
                    :,
                    self.pp_patches_token_start_end_idx[i][0]:
                    self.pp_patches_token_start_end_idx[i][1],
                    :,
                ]
                for i in range(self.num_pipeline_patch)
            ]
            pos_embed = torch.cat(pos_embed_list, dim=1)

        if self.patched_mode:
            self.patch_step()

        return (latent + pos_embed).to(latent.dtype)
