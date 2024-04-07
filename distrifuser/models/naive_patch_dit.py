import torch
from distrifuser.modules.pp import DistriTransformer2DModel
from diffusers.models.transformers.transformer_2d import Transformer2DModelOutput
# from diffusers.models.transformers.transformer_2d import Transformer2DModel
from distrifuser.models.diffusers.transformers_2d import Transformer2DModel

from torch import distributed as dist

from .base_model import BaseModel
from ..utils import DistriConfig
from distrifuser.logger import init_logger
logger = init_logger(__name__)

from typing import Optional, Dict, Any

class NaivePatchDiT(BaseModel):  # for Patch Parallelism
    def __init__(self, model: Transformer2DModel, distri_config: DistriConfig):
        model = DistriTransformer2DModel(model, distri_config)
        super(NaivePatchDiT, self).__init__(model, distri_config)

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
        # record: bool = False,
    ):
        distri_config = self.distri_config

        # hidden_states.shape = [2, 4, 32, 32]
        b, c, h, w = hidden_states.shape
        # b, c, h, w = sample.shape
        assert (
            # encoder_hidden_states is not None
            timestep is not None
            # and added_cond_kwargs is not None
            and class_labels is not None
            # and cross_attention_kwargs is not None
            # and attention_mask is not None
            # and encoder_attention_mask is not None
        )

        if distri_config.world_size == 1:
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
            )[0] # [2, 8, 32, 32]
        else:

            if distri_config.split_scheme == "row":
                split_dim = 2
            elif distri_config.split_scheme == "col":
                split_dim = 3
            elif distri_config.split_scheme == "alternate":
                split_dim = 2 if self.counter % 2 == 0 else 3
            else:
                raise NotImplementedError

            if split_dim == 2:
                sliced_hidden_states = hidden_states.view(b, c, distri_config.n_device_per_batch, -1, w)[
                    :, :, distri_config.split_idx()
                ]
            else:
                assert split_dim == 3
                sliced_hidden_states = hidden_states.view(b, c, h, distri_config.n_device_per_batch, -1)[
                    ..., distri_config.split_idx(), :
                ]
            # logger.info(f"sliced_hidden_states.shape {sliced_hidden_states.shape}") # [2, 4, 16, 32]
            output = self.model(
                hidden_states=sliced_hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                timestep=timestep,
                added_cond_kwargs=added_cond_kwargs,
                class_labels=class_labels,
                cross_attention_kwargs=cross_attention_kwargs,
                attention_mask=attention_mask,
                encoder_attention_mask=encoder_attention_mask,
                return_dict=False,
            )[0]

            # logger.info(f"world_size>1: output.shape {output.shape}")

            if self.output_buffer is None:
                self.output_buffer = torch.empty((b, c, h, w), device=output.device, dtype=output.dtype)
            if self.buffer_list is None:
                self.buffer_list = [torch.empty_like(output.view(-1)) for _ in range(distri_config.world_size)]
            dist.all_gather(self.buffer_list, output.contiguous().view(-1), async_op=False)
            buffer_list = [buffer.view(output.shape) for buffer in self.buffer_list]
            torch.cat(buffer_list, dim=split_dim, out=self.output_buffer)
            output = self.output_buffer

        if return_dict:
            output = Transformer2DModelOutput(sample=output)
        else:
            output = (output,)

        self.counter += 1
        return output

    @property
    def add_embedding(self):
        return self.model.add_embedding
