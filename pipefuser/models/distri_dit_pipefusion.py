# from diffusers.models.attention_processor import Attention
from pipefuser.models.diffusers import Attention

from diffusers.models.transformers.transformer_2d import Transformer2DModelOutput
from diffusers.models.embeddings import PatchEmbed
from pipefuser.models.diffusers import Transformer2DModel
from torch import distributed as dist, nn
import torch

from pipefuser.models.base_model import BaseModule, BaseModel
from pipefuser.modules.dit.pipefusion import (
    DistriSelfAttentionPiP,
    DistriTransformer2DModel,
    DistriConv2dPiP,
    DistriPatchEmbed,
)

from .base_model import BaseModel
from ..utils import DistriConfig
from pipefuser.logger import init_logger

logger = init_logger(__name__)

from typing import Optional, Dict, Any


class DistriDiTPipeFusion(BaseModel):  # for Pipeline Parallelism
    def __init__(self, model: Transformer2DModel, distri_config: DistriConfig):
        assert isinstance(model, Transformer2DModel), f"{type(model)} is not Transformer2DModel"
        model = DistriTransformer2DModel(model, distri_config)
        for name, module in model.named_modules():
            if isinstance(module, BaseModule):
                continue
            for subname, submodule in module.named_children():
                if isinstance(submodule, nn.Conv2d):
                    kernel_size = submodule.kernel_size
                    if kernel_size == (1, 1) or kernel_size == 1:
                        continue
                    wrapped_submodule = DistriConv2dPiP(
                        submodule, distri_config, is_first_layer=True
                    )
                    setattr(module, subname, wrapped_submodule)
                elif isinstance(submodule, PatchEmbed):
                    wrapped_submodule = DistriPatchEmbed(submodule, distri_config)
                    setattr(module, subname, wrapped_submodule)
                elif isinstance(submodule, Attention):
                    if subname == "attn1":  # self attention
                        wrapped_submodule = DistriSelfAttentionPiP(
                            submodule, distri_config
                        )
                        setattr(module, subname, wrapped_submodule)
        logger.info(
            f"Using pipeline parallelism, world_size: {distri_config.world_size} and n_device_per_batch: {distri_config.n_device_per_batch}"
        )
        super(DistriDiTPipeFusion, self).__init__(model, distri_config)

        self.batch_idx = 0

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
        record: bool = False,
    ):
        distri_config = self.distri_config

        # hidden_states.shape = [2, 4, 32, 32]
        # b, c, h, w = hidden_states.shape
        # b, c, h, w = sample.shape
        assert (
            hidden_states is not None
            and cross_attention_kwargs is None
            and attention_mask is None
            # and encoder_attention_mask is None
        )
        output = self.model(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timestep,
            added_cond_kwargs=added_cond_kwargs,
            class_labels=class_labels,
            cross_attention_kwargs=cross_attention_kwargs,
            attention_mask=attention_mask,
            encoder_attention_mask=encoder_attention_mask,
            return_dict=False,
        )[0]

        if return_dict:
            output = Transformer2DModelOutput(sample=output)
        else:
            output = (output,)

        self.counter += 1
        return output

    @property
    def add_embedding(self):
        return self.model.add_embedding
