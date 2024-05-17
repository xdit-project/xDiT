import torch
from diffusers import UNet2DConditionModel
from diffusers.models.attention_processor import Attention
from diffusers.models.unets.unet_2d_condition import UNet2DConditionOutput
from torch import distributed as dist, nn

from .base_model import BaseModel
from distrifuser.modules.base_module import BaseModule
# from distrifuser.modules.pp.attn import DistriCrossAttentionPP, DistriSelfAttentionPP
# from distrifuser.modules.pp.conv2d import DistriConv2dPP
# from distrifuser.modules.pp.groupnorm import DistriGroupNorm
from distrifuser.modules.pip import (
    DistriSelfAttentionPiP,
    DistriConv2dPiP,
    DistriPatchEmbed,
    DistriGroupNormPiP
)
from ..utils import DistriConfig

from distrifuser.logger import init_logger
logger = init_logger(__name__)


class DistriSDXLUNetPiP(BaseModel):  # for Pipeline Parallelism
    def __init__(self, model: UNet2DConditionModel, distri_config: DistriConfig):
        assert isinstance(model, UNet2DConditionModel)
        if distri_config.world_size > 1 and distri_config.n_device_per_batch > 1:
            for name, module in model.named_modules():
                if isinstance(module, BaseModule):
                    continue
                for subname, submodule in module.named_children():
                    if isinstance(submodule, nn.Conv2d):
                        kernel_size = submodule.kernel_size
                        if kernel_size == (1, 1) or kernel_size == 1:
                            continue
                        wrapped_submodule = DistriConv2dPiP(
                            submodule, distri_config, is_first_layer=subname == "conv_in"
                        )
                        setattr(module, subname, wrapped_submodule)
                    elif isinstance(submodule, Attention):
                        if subname == "attn1":  # self attention
                            wrapped_submodule = DistriSelfAttentionPiP(submodule, distri_config)
                        setattr(module, subname, wrapped_submodule)
                    elif isinstance(submodule, nn.GroupNorm):
                        wrapped_submodule = DistriGroupNormPiP(submodule, distri_config)
                        setattr(module, subname, wrapped_submodule)

        super(DistriSDXLUNetPiP, self).__init__(model, distri_config)

        self.batch_idx = 0

    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: torch.Tensor or float or int,
        encoder_hidden_states: torch.Tensor,
        class_labels: torch.Tensor or None = None,
        timestep_cond: torch.Tensor or None = None,
        attention_mask: torch.Tensor or None = None,
        cross_attention_kwargs: dict[str, any] or None = None,
        added_cond_kwargs: dict[str, torch.Tensor] or None = None,
        down_block_additional_residuals: tuple[torch.Tensor] or None = None,
        mid_block_additional_residual: torch.Tensor or None = None,
        down_intrablock_additional_residuals: tuple[torch.Tensor] or None = None,
        encoder_attention_mask: torch.Tensor or None = None,
        return_dict: bool = True,
        record: bool = False,
    ):
        distri_config = self.distri_config
        b, c, h, w = sample.shape
        assert (
            class_labels is None
            and timestep_cond is None
            and attention_mask is None
            and cross_attention_kwargs is None
            and down_block_additional_residuals is None
            and mid_block_additional_residual is None
            and down_intrablock_additional_residuals is None
            and encoder_attention_mask is None
        )

        output = self.model(
            sample,
            timestep,
            encoder_hidden_states,
            class_labels=class_labels,
            timestep_cond=timestep_cond,
            attention_mask=attention_mask,
            cross_attention_kwargs=cross_attention_kwargs,
            added_cond_kwargs=added_cond_kwargs,
            down_block_additional_residuals=down_block_additional_residuals,
            mid_block_additional_residual=mid_block_additional_residual,
            down_intrablock_additional_residuals=down_intrablock_additional_residuals,
            encoder_attention_mask=encoder_attention_mask,
            return_dict=False,
        )[0]

        if return_dict:
            output = UNet2DConditionOutput(sample=output)
        else:
            output = (output,)

        self.counter += 1
        return output

    @property
    def add_embedding(self):
        return self.model.add_embedding
