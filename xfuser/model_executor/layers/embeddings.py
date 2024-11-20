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
        self.module: CogVideoXPatchEmbed

    def forward(self, text_embeds: torch.Tensor, image_embeds: torch.Tensor):
        r"""
        Args:
            text_embeds (`torch.Tensor`):
                Input text embeddings. Expected shape: (batch_size, seq_length, embedding_dim).
            image_embeds (`torch.Tensor`):
                Input image embeddings. Expected shape: (batch_size, num_frames, channels, height, width).
        """
        # height is the height of a batch on a GPU, sum_height is the total height of the video
        sum_height = (
            get_runtime_state().input_config.height
            // get_runtime_state().vae_scale_factor_spatial
        )
        text_embeds = self.text_proj(text_embeds)

        batch_size, num_frames, channels, height, width = image_embeds.shape
        
        if self.patch_size_t is None:
            image_embeds = image_embeds.reshape(-1, channels, height, width)
            image_embeds = self.proj(image_embeds)
            image_embeds = image_embeds.view(batch_size, num_frames, *image_embeds.shape[1:])
            image_embeds = image_embeds.flatten(3).transpose(2, 3)  # [batch, num_frames, height x width, channels]
            image_embeds = image_embeds.flatten(1, 2)  # [batch, num_frames x height x width, channels]
        else:
            p = self.patch_size
            p_t = self.patch_size_t

            image_embeds = image_embeds.permute(0, 1, 3, 4, 2)
            image_embeds = image_embeds.reshape(
                batch_size, num_frames // p_t, p_t, height // p, p, width // p, p, channels
            )
            image_embeds = image_embeds.permute(0, 1, 3, 5, 7, 2, 4, 6).flatten(4, 7).flatten(1, 3)
            image_embeds = self.proj(image_embeds)
 
        embeds = torch.cat(
            [text_embeds, image_embeds], dim=1
        ).contiguous()  # [batch, seq_length + num_frames x height x width, channels]
           
        if self.use_positional_embeddings or self.use_learned_positional_embeddings:
            if self.use_learned_positional_embeddings and (self.sample_width != width or self.sample_height != sum_height):
                raise ValueError(
                    "It is currently not possible to generate videos at a different resolution that the defaults. This should only be the case with 'THUDM/CogVideoX-5b-I2V'."
                    "If you think this is incorrect, please open an issue at https://github.com/huggingface/diffusers/issues."
                )

            pre_time_compression_frames = (num_frames - 1) * self.temporal_compression_ratio + 1

            if (
                self.sample_height != sum_height
                or self.sample_width != width
                or self.sample_frames != pre_time_compression_frames
            ):
                pos_embedding = self._get_positional_embeddings(sum_height, width, pre_time_compression_frames)
                pos_embedding = pos_embedding.to(embeds.device, dtype=embeds.dtype)
            else:
                pos_embedding = self.pos_embedding

            # extract the image part of the positional embedding
            pos_embedding = pos_embedding[:, self.max_text_seq_length :]

            # slice the positional embedding
            post_patch_height = sum_height // self.patch_size
            post_patch_width = width // self.patch_size
            post_time_compression_frames = (pre_time_compression_frames - 1) // self.temporal_compression_ratio + 1

            pos_embed_list = [
                pos_embedding[
                    :,
                    post_patch_height * post_patch_width * i + get_runtime_state().pp_patches_token_start_end_idx_global[0][0]: 
                    post_patch_height * post_patch_width * i + get_runtime_state().pp_patches_token_start_end_idx_global[0][1],
                    :,
                ]
                for i in range(post_time_compression_frames)
            ]
            pos_embedding = torch.cat(pos_embed_list, dim=1)

            embeds[:, self.max_text_seq_length :] += pos_embedding

        return embeds
