import torch
from torch import distributed as dist
from torch import nn
from torch.nn import functional as F

from distrifuser.modules.base_module import BaseModule
from distrifuser.utils import DistriConfig


class DistriConv2dPP(BaseModule):
    def __init__(self, module: nn.Conv2d, distri_config: DistriConfig, is_first_layer: bool = False):
        super(DistriConv2dPP, self).__init__(module, distri_config)
        self.is_first_layer = is_first_layer

    def naive_forward(self, x: torch.Tensor) -> torch.Tensor:
        #  x: [B, C, H, W]
        output = self.module(x)
        return output

    def sliced_forward(self, x: torch.Tensor) -> torch.Tensor:
        config = self.distri_config
        b, c, h, w = x.shape
        assert h % config.n_device_per_batch == 0

        stride = self.module.stride[0]
        padding = self.module.padding[0]

        output_h = x.shape[2] // stride // config.n_device_per_batch
        idx = config.split_idx()
        h_begin = output_h * idx * stride - padding
        h_end = output_h * (idx + 1) * stride + padding
        final_padding = [padding, padding, 0, 0]
        if h_begin < 0:
            h_begin = 0
            final_padding[2] = padding
        if h_end > h:
            h_end = h
            final_padding[3] = padding
        sliced_input = x[:, :, h_begin:h_end, :]
        padded_input = F.pad(sliced_input, final_padding, mode="constant")
        return F.conv2d(padded_input, self.module.weight, self.module.bias, stride=stride, padding="valid")

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        distri_config = self.distri_config

        if self.comm_manager is not None and self.comm_manager.handles is not None and self.idx is not None:
            if self.comm_manager.handles[self.idx] is not None:
                self.comm_manager.handles[self.idx].wait()
                self.comm_manager.handles[self.idx] = None

        if distri_config.n_device_per_batch == 1:
            output = self.naive_forward(x)
        else:
            if self.is_first_layer:
                full_x = x
                output = self.sliced_forward(full_x)
            else:
                boundary_size = self.module.padding[0]
                if self.buffer_list is None:
                    if self.comm_manager.buffer_list is None:
                        self.idx = self.comm_manager.register_tensor(
                            shape=[2, x.shape[0], x.shape[1], boundary_size, x.shape[3]],
                            torch_dtype=x.dtype,
                            layer_type="conv2d",
                        )
                    else:
                        self.buffer_list = self.comm_manager.get_buffer_list(self.idx)
                if self.buffer_list is None:
                    output = self.naive_forward(x)
                else:

                    def create_padded_x():
                        if distri_config.split_idx() == 0:
                            concat_x = torch.cat([x, self.buffer_list[distri_config.split_idx() + 1][0]], dim=2)
                            padded_x = F.pad(concat_x, [0, 0, boundary_size, 0], mode="constant")
                        elif distri_config.split_idx() == distri_config.n_device_per_batch - 1:
                            concat_x = torch.cat([self.buffer_list[distri_config.split_idx() - 1][1], x], dim=2)
                            padded_x = F.pad(concat_x, [0, 0, 0, boundary_size], mode="constant")
                        else:
                            padded_x = torch.cat(
                                [
                                    self.buffer_list[distri_config.split_idx() - 1][1],
                                    x,
                                    self.buffer_list[distri_config.split_idx() + 1][0],
                                ],
                                dim=2,
                            )
                        return padded_x

                    boundary = torch.stack([x[:, :, :boundary_size, :], x[:, :, -boundary_size:, :]], dim=0)

                    if distri_config.mode == "full_sync" or self.counter <= distri_config.warmup_steps:
                        dist.all_gather(self.buffer_list, boundary, group=distri_config.batch_group, async_op=False)
                        padded_x = create_padded_x()
                        output = F.conv2d(
                            padded_x,
                            self.module.weight,
                            self.module.bias,
                            stride=self.module.stride[0],
                            padding=(0, self.module.padding[1]),
                        )
                    else:
                        padded_x = create_padded_x()
                        output = F.conv2d(
                            padded_x,
                            self.module.weight,
                            self.module.bias,
                            stride=self.module.stride[0],
                            padding=(0, self.module.padding[1]),
                        )
                        if distri_config.mode != "no_sync":
                            self.comm_manager.enqueue(self.idx, boundary)

        self.counter += 1
        return output
