import torch
from torch import distributed as dist
from torch import nn

from distrifuser.modules.base_module import BaseModule
from distrifuser.utils import DistriConfig


class DistriGroupNorm(BaseModule):
    def __init__(self, module: nn.GroupNorm, distri_config: DistriConfig):
        assert isinstance(module, nn.GroupNorm)
        super(DistriGroupNorm, self).__init__(module, distri_config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        module = self.module
        assert isinstance(module, nn.GroupNorm)
        distri_config = self.distri_config

        if self.comm_manager is not None and self.comm_manager.handles is not None and self.idx is not None:
            if self.comm_manager.handles[self.idx] is not None:
                self.comm_manager.handles[self.idx].wait()
                self.comm_manager.handles[self.idx] = None

        assert x.ndim == 4
        n, c, h, w = x.shape
        num_groups = module.num_groups
        group_size = c // num_groups

        if distri_config.mode in ["stale_gn", "corrected_async_gn"]:
            if self.buffer_list is None:
                if self.comm_manager.buffer_list is None:
                    n, c, h, w = x.shape
                    self.idx = self.comm_manager.register_tensor(
                        shape=[2, n, num_groups, 1, 1, 1], torch_dtype=x.dtype, layer_type="gn"
                    )
                else:
                    self.buffer_list = self.comm_manager.get_buffer_list(self.idx)
            x = x.view([n, num_groups, group_size, h, w])
            x_mean = x.mean(dim=[2, 3, 4], keepdim=True)  # [1, num_groups, 1, 1, 1]
            x2_mean = (x**2).mean(dim=[2, 3, 4], keepdim=True)  # [1, num_groups, 1, 1, 1]
            slice_mean = torch.stack([x_mean, x2_mean], dim=0)

            if self.buffer_list is None:
                full_mean = slice_mean
            elif self.counter <= distri_config.warmup_steps:
                dist.all_gather(self.buffer_list, slice_mean, group=distri_config.batch_group, async_op=False)
                full_mean = sum(self.buffer_list) / distri_config.n_device_per_batch
            else:
                if distri_config.mode == "corrected_async_gn":
                    correction = slice_mean - self.buffer_list[distri_config.split_idx()]
                    full_mean = sum(self.buffer_list) / distri_config.n_device_per_batch + correction
                else:
                    new_buffer_list = [buffer for buffer in self.buffer_list]
                    new_buffer_list[distri_config.split_idx()] = slice_mean
                    full_mean = sum(new_buffer_list) / distri_config.n_device_per_batch
                self.comm_manager.enqueue(self.idx, slice_mean)

            full_x_mean, full_x2_mean = full_mean[0], full_mean[1]
            var = full_x2_mean - full_x_mean**2
            if distri_config.mode == "corrected_async_gn":
                slice_x_mean, slice_x2_mean = slice_mean[0], slice_mean[1]
                slice_var = slice_x2_mean - slice_x_mean**2
                var = torch.where(var < 0, slice_var, var)  # Correct negative variance

            num_elements = group_size * h * w
            var = var * (num_elements / (num_elements - 1))
            std = (var + module.eps).sqrt()
            output = (x - full_x_mean) / std
            output = output.view([n, c, h, w])
            if module.affine:
                output = output * module.weight.view([1, -1, 1, 1])
                output = output + module.bias.view([1, -1, 1, 1])
        else:
            if self.counter <= distri_config.warmup_steps or distri_config.mode in ["sync_gn", "full_sync"]:
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
                std = (var + module.eps).sqrt()
                output = (x - x_mean) / std
                output = output.view([n, c, h, w])
                if module.affine:
                    output = output * module.weight.view([1, -1, 1, 1])
                    output = output + module.bias.view([1, -1, 1, 1])
            elif distri_config.mode in ["separate_gn", "no_sync"]:
                output = module(x)
            else:
                raise NotImplementedError
        self.counter += 1
        return output
