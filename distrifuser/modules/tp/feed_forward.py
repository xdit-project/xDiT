import torch.cuda
from diffusers.models.attention import FeedForward, GEGLU
from torch import distributed as dist
from torch import nn
from torch.nn import functional as F

from ..base_module import BaseModule
from ...utils import DistriConfig


class DistriFeedForwardTP(BaseModule):
    def __init__(self, module: FeedForward, distri_config: DistriConfig):
        super(DistriFeedForwardTP, self).__init__(module, distri_config)
        assert isinstance(module.net[0], GEGLU)
        assert module.net[0].proj.out_features % (distri_config.n_device_per_batch * 2) == 0
        assert module.net[2].in_features % distri_config.n_device_per_batch == 0

        mid_features = module.net[2].in_features // distri_config.n_device_per_batch

        sharded_fc1 = nn.Linear(
            module.net[0].proj.in_features,
            mid_features * 2,
            bias=module.net[0].proj.bias is not None,
            device=module.net[0].proj.weight.device,
            dtype=module.net[0].proj.weight.dtype,
        )
        start_idx = distri_config.split_idx() * mid_features
        end_idx = (distri_config.split_idx() + 1) * mid_features
        sharded_fc1.weight.data[:mid_features].copy_(module.net[0].proj.weight.data[start_idx:end_idx])
        if module.net[0].proj.bias is not None:
            sharded_fc1.bias.data[:mid_features].copy_(module.net[0].proj.bias.data[start_idx:end_idx])
        start_idx = (distri_config.n_device_per_batch + distri_config.split_idx()) * mid_features
        end_idx = (distri_config.n_device_per_batch + distri_config.split_idx() + 1) * mid_features
        sharded_fc1.weight.data[mid_features:].copy_(module.net[0].proj.weight.data[start_idx:end_idx])
        if module.net[0].proj.bias is not None:
            sharded_fc1.bias.data[mid_features:].copy_(module.net[0].proj.bias.data[start_idx:end_idx])

        sharded_fc2 = nn.Linear(
            mid_features,
            module.net[2].out_features,
            bias=module.net[2].bias is not None,
            device=module.net[2].weight.device,
            dtype=module.net[2].weight.dtype,
        )
        sharded_fc2.weight.data.copy_(
            module.net[2].weight.data[
                :, distri_config.split_idx() * mid_features : (distri_config.split_idx() + 1) * mid_features
            ]
        )
        if module.net[2].bias is not None:
            sharded_fc2.bias.data.copy_(module.net[2].bias.data)

        old_fc1 = module.net[0].proj
        old_fc2 = module.net[2]

        module.net[0].proj = sharded_fc1
        module.net[2] = sharded_fc2

        del old_fc1
        del old_fc2
        torch.cuda.empty_cache()

    def forward(self, hidden_states: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
        distri_config = self.distri_config
        module = self.module

        assert scale == 1.0
        for i, submodule in enumerate(module.net):
            if i == 0:
                hidden_states, gate = submodule.proj(hidden_states).chunk(2, dim=-1)
                hidden_states = hidden_states * submodule.gelu(gate)
            elif i == 2:
                hidden_states = F.linear(hidden_states, submodule.weight, None)
            else:
                hidden_states = submodule(hidden_states)

        dist.all_reduce(hidden_states, op=dist.ReduceOp.SUM, group=distri_config.batch_group, async_op=False)
        if module.net[2].bias is not None:
            hidden_states = hidden_states + module.net[2].bias.view(1, 1, -1)

        self.counter += 1

        return hidden_states
