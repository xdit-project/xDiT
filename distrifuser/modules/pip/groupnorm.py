import torch
from torch import distributed as dist
from torch import nn

from distrifuser.modules.base_module import BaseModule
from distrifuser.utils import DistriConfig
from distrifuser.logger import init_logger

logger = init_logger(__name__)

class DistriGroupNormPiP(BaseModule):
    def __init__(self, module: nn.GroupNorm, distri_config: DistriConfig):
        assert isinstance(module, nn.GroupNorm)
        super(DistriGroupNormPiP, self).__init__(module, distri_config)
        self.batch_idx = 0

    def naive_forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        distri_config = self.distri_config

        n, c, h, w = x.shape
        num_groups = self.module.num_groups
        group_size = c // num_groups

        x = x.view([n, num_groups, group_size, h, w])
        x_mean = x.mean(dim=[2, 3, 4], keepdim=True)  # [1, num_groups, 1, 1, 1]
        x2_mean = (x**2).mean(dim=[2, 3, 4], keepdim=True)  # [1, num_groups, 1, 1, 1]
        mean = torch.stack([x_mean, x2_mean], dim=0)
        dist.all_reduce(mean, op=dist.ReduceOp.SUM, group=distri_config.batch_group)
        mean = mean / distri_config.n_device_per_batch
        x_mean = mean[0]
        x2_mean = mean[1]
        var = x2_mean - x_mean**2
        num_elements = group_size * h * w
        var = var * (num_elements / (num_elements - 1))
        std = (var + self.module.eps).sqrt()
        output = (x - x_mean) / std
        output = output.view([n, c, h, w])
        if self.module.affine:
            output = output * self.module.weight.view([1, -1, 1, 1])
            output = output + self.module.bias.view([1, -1, 1, 1])
        return output
    
    def sliced_forward(self, x: torch.Tensor) -> torch.Tensor:
        distri_config = self.distri_config
        n, c, h, w = x.shape
        assert h % distri_config.num_micro_batch == 0

        num_groups = self.module.num_groups
        group_size = c // num_groups
        idx = self.batch_idx
        x = x.view([n, num_groups, group_size, h, w])
        x_mean = x.mean(dim=[2, 3, 4], keepdim=True)  # [1, num_groups, 1, 1, 1]
        x2_mean = (x**2).mean(dim=[2, 3, 4], keepdim=True)  # [1, num_groups, 1, 1, 1]
        slice_mean = torch.stack([x_mean, x2_mean], dim=0)

        correction = slice_mean - self.buffer_list[idx]
        full_mean = sum(self.buffer_list) / distri_config.num_micro_batch + correction

        full_x_mean, full_x2_mean = full_mean[0], full_mean[1]
        var = full_x2_mean - full_x_mean**2

        slice_x_mean, slice_x2_mean = slice_mean[0], slice_mean[1]
        slice_var = slice_x2_mean - slice_x_mean**2
        var = torch.where(var < 0, slice_var, var)  # Correct negative variance

        num_elements = group_size * h * w
        var = var * (num_elements / (num_elements - 1))
        std = (var + self.module.eps).sqrt()
        output = (x - full_x_mean) / std
        output = output.view([n, c, h, w])
        if self.module.affine:
            output = output * self.module.weight.view([1, -1, 1, 1])
            output = output + self.module.bias.view([1, -1, 1, 1])

        return output



    def forward(self, x: torch.Tensor) -> torch.Tensor:
        distri_config = self.distri_config
        if distri_config.num_micro_batch == 1:
            output = self.naive_forward(x)
        else:
            full_x = self.buffer_list
            if distri_config.mode == "full_sync" or self.counter <= distri_config.warmup_steps:
                full_x = x
                output = self.naive_forward(full_x)
            else:
                _, _, cc, _ = full_x.shape
                _, _, c, _ = x.shape
                assert cc // distri_config.num_micro_batch == c
                full_x[:, :, c * self.batch_idx : c * (self.batch_idx + 1), :] = x
                output = self.sliced_forward(full_x)
            self.buffer_list = full_x

        if distri_config.mode == "full_sync" or self.counter <= distri_config.warmup_steps:
            self.counter += 1
        else:
            self.batch_idx += 1
            if self.batch_idx == distri_config.num_micro_batch:
                self.counter += 1
                self.batch_idx = 0
        return output