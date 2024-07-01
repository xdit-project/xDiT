import torch
from torch import distributed as dist
from torch import nn
from torch.nn import functional as F

from pipefuser.modules.base_module import BaseModule
from pipefuser.utils import DistriConfig


class DistriConv2dTP(BaseModule):
    def __init__(self, module: nn.Conv2d, distri_config: DistriConfig):
        super(DistriConv2dTP, self).__init__(module, distri_config)
        assert (
            module.in_channels % distri_config.n_device_per_batch == 0
        ), f"in_channels: {module.in_channels} vs n_device_per_batch: {distri_config.n_device_per_batch}"

        sharded_module = nn.Conv2d(
            module.in_channels // distri_config.n_device_per_batch,
            module.out_channels,
            module.kernel_size,
            module.stride,
            module.padding,
            module.dilation,
            module.groups,
            module.bias is not None,
            module.padding_mode,
            device=module.weight.device,
            dtype=module.weight.dtype,
        )
        start_idx = distri_config.split_idx() * (
            module.in_channels // distri_config.n_device_per_batch
        )
        end_idx = (distri_config.split_idx() + 1) * (
            module.in_channels // distri_config.n_device_per_batch
        )
        sharded_module.weight.data.copy_(module.weight.data[:, start_idx:end_idx])
        if module.bias is not None:
            sharded_module.bias.data.copy_(module.bias.data)

        self.module = sharded_module
        del module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        distri_config = self.distri_config

        b, c, h, w = x.shape
        start_idx = distri_config.split_idx() * (c // distri_config.n_device_per_batch)
        end_idx = (distri_config.split_idx() + 1) * (
            c // distri_config.n_device_per_batch
        )
        output = F.conv2d(
            x[:, start_idx:end_idx],
            self.module.weight,
            bias=None,
            stride=self.module.stride,
            padding=self.module.padding,
            dilation=self.module.dilation,
            groups=self.module.groups,
        )
        dist.all_reduce(
            output,
            op=dist.ReduceOp.SUM,
            group=distri_config.batch_parallel_group,
            async_op=False,
        )
        if self.module.bias is not None:
            output = output + self.module.bias.view(1, -1, 1, 1)

        self.counter += 1
        return output
