# adapted from https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/embeddings.py
import torch

from diffusers.models.embeddings import PatchEmbed, get_2d_sincos_pos_embed
from pipefuser.models.base_model import BaseModule, BaseModel
from pipefuser.utils import DistriConfig
from pipefuser.logger import init_logger

logger = init_logger(__name__)


class DistriPatchEmbed(BaseModule):
    def __init__(self, module: PatchEmbed, distri_config: DistriConfig):
        super(DistriPatchEmbed, self).__init__(module, distri_config)
        self.batch_idx = 0
        self.pos_embed = None

    def forward(self, latent):
        module = self.module
        distri_config = self.distri_config
        is_warmup = (
            distri_config.mode == "full_sync"
            or self.counter <= distri_config.warmup_steps
        )

        if is_warmup:
            height, width = (
                latent.shape[-2] // module.patch_size,
                latent.shape[-1] // module.patch_size,
            )
        else:
            height, width = (
                latent.shape[-2] // module.patch_size * distri_config.pp_num_patch,
                latent.shape[-1] // module.patch_size,
            )

        latent = module.proj(latent)
        if module.flatten:
            latent = latent.flatten(2).transpose(1, 2)  # BCHW -> BNC
        if module.layer_norm:
            # TODO: NOT SURE whether compatible with norm
            latent = module.norm(latent)

        # [2, 4096 / c, 1152]

        # Interpolate positional embeddings if needed.
        # (For PixArt-Alpha: https://github.com/PixArt-alpha/PixArt-alpha/blob/0f55e922376d8b797edd44d25d0e7464b260dcab/diffusion/model/nets/PixArtMS.py#L162C151-L162C160)

        # TODO: There might be a more faster way to generate a smaller pos_embed
        if module.height != height or module.width != width:
            pos_embed = get_2d_sincos_pos_embed(
                embed_dim=module.pos_embed.shape[-1],
                grid_size=(height, width),
                base_size=module.base_size,
                interpolation_scale=module.interpolation_scale,
            )
            pos_embed = torch.from_numpy(pos_embed)
            module.pos_embed = pos_embed.float().unsqueeze(0).to(latent.device)
            module.height = height
            module.width = width
            pos_embed = module.pos_embed
        else:
            pos_embed = module.pos_embed
        b, c, h = pos_embed.shape

        if not is_warmup:
            pos_embed = pos_embed.view(b, distri_config.pp_num_patch, -1, h)[
                :, self.batch_idx
            ]

        if is_warmup:
            self.counter += 1
        else:
            self.batch_idx += 1
            if self.batch_idx == distri_config.pp_num_patch:
                self.batch_idx = 0
                self.counter += 1

        return (latent + pos_embed).to(latent.dtype)
