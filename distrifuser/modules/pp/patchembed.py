import torch

from diffusers.models.embeddings import PatchEmbed, get_2d_sincos_pos_embed
from distrifuser.modules.base_module import BaseModule
from distrifuser.utils import DistriConfig
from distrifuser.logger import init_logger
logger = init_logger(__name__)

class DistriPatchEmbed(BaseModule):
    def __init__(self, module: PatchEmbed, distri_config: DistriConfig):
        super(DistriPatchEmbed, self).__init__(module, distri_config)

    def forward(self, latent):
        module = self.module
        distri_config = self.distri_config
        height, width = latent.shape[-2] // module.patch_size, latent.shape[-1] // module.patch_size

        latent = module.proj(latent)
        if module.flatten:
            latent = latent.flatten(2).transpose(1, 2)  # BCHW -> BNC
        if module.layer_norm:
            latent = module.norm(latent)

        # Interpolate positional embeddings if needed.
        # (For PixArt-Alpha: https://github.com/PixArt-alpha/PixArt-alpha/blob/0f55e922376d8b797edd44d25d0e7464b260dcab/diffusion/model/nets/PixArtMS.py#L162C151-L162C160)
        if module.height != height or module.width != width:
            pos_embed = get_2d_sincos_pos_embed(
                embed_dim=module.pos_embed.shape[-1],
                grid_size=(height, width),
                base_size=module.base_size,
                interpolation_scale=module.interpolation_scale,
            )
            pos_embed = torch.from_numpy(pos_embed)
            pos_embed = pos_embed.float().unsqueeze(0).to(latent.device)
        else:
            pos_embed = module.pos_embed
        logger.info(f"pos_embed shape: {pos_embed.shape}, latent shape: {latent.shape}")
        b, c, h = pos_embed.shape
        pos_embed = pos_embed.view(
            b, distri_config.n_device_per_batch, -1, h)[
                :, distri_config.split_idx()]
        logger.info(f"pos_embed shape: {pos_embed.shape}, latent shape: {latent.shape}")
        return (latent + pos_embed).to(latent.dtype)