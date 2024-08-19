# https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention.py

from diffusers.models.attention import FeedForward
from xfuser.distributed.runtime_state import get_runtime_state
from torch import nn
from xfuser.distributed.parallel_state import (
    get_tensor_model_parallel_world_size,
    get_tensor_model_parallel_rank,
    get_tp_group,
)
import torch
import torch.distributed as dist


# @xFuserLayerWrappersRegister.register(FeedForward)
# class xFuserFeedForwardWrapper(xFuserFeedForwardWrapper):
class FeedForward_TP(nn.Module):
    def __init__(
        self,
        module: FeedForward,
        activation_fn: str = "geglu",
    ):
        super(FeedForward_TP, self).__init__()

        assert activation_fn in [
            "gelu",
            "geglu",
        ], f"activation_fn {activation_fn} not supported"
        self.module = module
        tp_degree = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()

        if "gelu" == activation_fn:
            self.module.net[0].proj.weight.data = self.module.net[
                0
            ].proj.weight.data.chunk(tp_degree, dim=0)[tp_rank]
            if self.module.net[0].proj.bias is not None:
                self.module.net[0].proj.bias.data = self.module.net[
                    0
                ].proj.bias.data.chunk(tp_degree, dim=0)[tp_rank]
        elif "geglu" == activation_fn:
            weight_buff = self.module.net[0].proj.weight.data.chunk(2, dim=0)
            a = weight_buff[0].chunk(tp_degree, dim=0)[tp_rank]
            b = weight_buff[1].chunk(tp_degree, dim=0)[tp_rank]
            c = torch.cat([a, b], dim=0)
            self.module.net[0].proj.weight.data = c

        self.module.net[2].weight.data = self.module.net[2].weight.chunk(
            tp_degree, dim=1
        )[tp_rank]

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        hidden_states = self.module(hidden_states, *args, **kwargs)
        get_tp_group().all_reduce(hidden_states)
        return hidden_states
