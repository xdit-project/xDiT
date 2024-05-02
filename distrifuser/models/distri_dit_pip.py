import torch

# from diffusers.models.attention_processor import Attention
from distrifuser.models.diffusers import Attention

from diffusers.models.transformers.transformer_2d import Transformer2DModelOutput
from diffusers.models.embeddings import PatchEmbed
from distrifuser.models.diffusers import Transformer2DModel
from torch import distributed as dist, nn
import torch

from distrifuser.modules.base_module import BaseModule
from distrifuser.modules.pip import (
    DistriSelfAttentionPiP,
    DistriTransformer2DModel
)

from .base_model import BaseModel
from ..utils import DistriConfig
from distrifuser.logger import init_logger

logger = init_logger(__name__)

from typing import Optional, Dict, Any


class DistriDiTPiP(BaseModel):  # for Pipeline Parallelism
    def __init__(self, model: Transformer2DModel, distri_config: DistriConfig):
        assert isinstance(model, Transformer2DModel)
        model = DistriTransformer2DModel(model, distri_config)

        if distri_config.world_size > 1 and distri_config.n_device_per_batch > 1:
            for name, module in model.named_modules():
                if isinstance(module, BaseModule):
                    continue
                for subname, submodule in module.named_children():
                    # if isinstance(submodule, nn.Conv2d):
                    #     kernel_size = submodule.kernel_size
                    #     if kernel_size == (1, 1) or kernel_size == 1:
                    #         continue
                    #     wrapped_submodule = DistriConv2dPP(
                    #         submodule, distri_config, is_first_layer=True
                    #     )
                    #     setattr(module, subname, wrapped_submodule)
                    # elif isinstance(submodule, PatchEmbed): 
                        # wrapped_submodule = DistriPatchEmbed(submodule, distri_config)
                        # setattr(module, subname, wrapped_submodule)
                    if isinstance(submodule, Attention):
                        if subname == "attn1":  # self attention
                            wrapped_submodule = DistriSelfAttentionPiP(
                                submodule, distri_config
                            )
                            setattr(module, subname, wrapped_submodule)
            logger.info(
                f"Using parallelism for DiT, world_size: {distri_config.world_size} and n_device_per_batch: {distri_config.n_device_per_batch}"
            )
        else:
            logger.info("Not using parallelism for DiT")
        super(DistriDiTPiP, self).__init__(model, distri_config)

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
        b, c, h, w = hidden_states.shape
        # b, c, h, w = sample.shape
        assert (
            hidden_states is not None
            and cross_attention_kwargs is None
            and attention_mask is None
            # and encoder_attention_mask is None
        )
        if distri_config.use_cuda_graph and not record:
            static_inputs = self.static_inputs
            assert hidden_states.shape == static_inputs['hidden_states'].shape
            static_inputs['hidden_states'].copy_(hidden_states)
            if torch.is_tensor(timestep):
                if timestep.ndim == 0:
                    for b in range(static_inputs["timestep"].shape[0]):
                        static_inputs["timestep"][b] = timestep.item()
                else:
                    assert static_inputs["timestep"].shape == timestep.shape
                    static_inputs["timestep"].copy_(timestep)
            else:
                for b in range(static_inputs["timestep"].shape[0]):
                    static_inputs["timestep"][b] = timestep
            if encoder_hidden_states is not None:
                assert static_inputs['encoder_hidden_states'].shape == encoder_hidden_states.shape
                static_inputs['encoder_hidden_states'].copy_(encoder_hidden_states)
            if class_labels is not None:
                static_inputs['class_labels'].copy_(class_labels)
            if added_cond_kwargs is not None:
                for k in added_cond_kwargs:
                    assert static_inputs["added_cond_kwargs"][k].shape == added_cond_kwargs[k].shape
            if encoder_attention_mask is not None:
                static_inputs['encoder_attention_mask'].copy_(encoder_attention_mask)

            if self.counter <= distri_config.warmup_steps:
                graph_idx = 0
            elif self.counter == distri_config.warmup_steps + 1:
                graph_idx = 1
            else:
                graph_idx = 2

            self.cuda_graphs[graph_idx].replay()
            output = self.static_outputs[graph_idx]
        else:
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
            if record:
                if self.static_inputs is None:
                    self.static_inputs = {
                        "hidden_states": hidden_states,
                        "class_labels": class_labels,
                        "timestep": timestep,
                        "encoder_hidden_states": encoder_hidden_states,
                        "encoder_attention_mask": encoder_attention_mask,
                        "added_cond_kwargs": added_cond_kwargs,
                    }
                self.synchronize()

        if return_dict:
            output = Transformer2DModelOutput(sample=output)
        else:
            output = (output,)

        self.counter += 1
        return output

    @property
    def add_embedding(self):
        return self.model.add_embedding
