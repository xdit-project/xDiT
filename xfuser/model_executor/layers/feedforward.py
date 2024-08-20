# https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention.py

from diffusers.models.attention import FeedForward, GELU, GEGLU
from torch import nn
from xfuser.core.distributed.parallel_state import (
    get_tensor_model_parallel_world_size,
    get_tensor_model_parallel_rank,
    get_tp_group,
)
import torch
from xfuser.model_executor.layers.base_layer import xFuserLayerBaseWrapper
from xfuser.model_executor.layers.register import xFuserLayerWrappersRegister


@xFuserLayerWrappersRegister.register(FeedForward)
class xFuserFeedForwardWrapper(xFuserLayerBaseWrapper):
    def __init__(self, feedforward: FeedForward):
        super(xFuserFeedForwardWrapper, self).__init__(module=feedforward)

        tp_degree = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()

        if isinstance(self.module.net[0], GELU):
            self.module.net[0].proj.weight.data = self.module.net[
                0
            ].proj.weight.data.chunk(tp_degree, dim=0)[tp_rank]
            if self.module.net[0].proj.bias is not None:
                self.module.net[0].proj.bias.data = self.module.net[
                    0
                ].proj.bias.data.chunk(tp_degree, dim=0)[tp_rank]
        elif isinstance(self.module.net[0], GEGLU):
            weight_buff = self.module.net[0].proj.weight.data.chunk(2, dim=0)
            a = weight_buff[0].chunk(tp_degree, dim=0)[tp_rank]
            b = weight_buff[1].chunk(tp_degree, dim=0)[tp_rank]
            c = torch.cat([a, b], dim=0)

            self.module.net[0].proj.weight.data = c

            bias_buff = self.module.net[0].proj.bias.data.chunk(2, dim=0)
            a = bias_buff[0].chunk(tp_degree, dim=0)[tp_rank]
            b = bias_buff[1].chunk(tp_degree, dim=0)[tp_rank]
            c = torch.cat([a, b], dim=0)
            self.module.net[0].proj.bias.data = c

        else:
            raise TypeError(
                f"activation_fn {type(isinstance(self.module.net[0]))} not supported"
            )

        self.module.net[2].weight.data = self.module.net[2].weight.chunk(
            tp_degree, dim=1
        )[tp_rank]

        self.has_output_bias = False
        if self.module.net[2].bias is not None:
            self.register_parameter(
                "output_bias", nn.Parameter(self.module.net[2].bias.data.clone())
            )
            self.module.net[2].bias = None
            self.has_output_bias = True

        torch.cuda.empty_cache()

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        hidden_states = self.module(hidden_states, *args, **kwargs)
        get_tp_group().all_reduce(hidden_states)
        if self.has_output_bias:
            hidden_states += self.output_bias
        return hidden_states
