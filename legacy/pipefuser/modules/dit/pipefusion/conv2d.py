import torch
from torch import distributed as dist
from torch import nn
from torch.nn import functional as F

from legacy.pipefuser.modules.base_module import BaseModule
from legacy.pipefuser.utils import DistriConfig
from legacy.pipefuser.logger import init_logger

logger = init_logger(__name__)


class DistriConv2dPiP(BaseModule):
    def __init__(
        self,
        module: nn.Conv2d,
        distri_config: DistriConfig,
        is_first_layer: bool = False,
    ):
        super(DistriConv2dPiP, self).__init__(module, distri_config)
        self.is_first_layer = is_first_layer
        self.batch_idx = 0

    def naive_forward(self, x: torch.Tensor) -> torch.Tensor:
        #  x: [B, C, H, W]
        output = self.module(x)
        return output

    def sliced_forward(self, x: torch.Tensor) -> torch.Tensor:
        distri_config = self.distri_config
        b, c, h, w = x.shape
        assert h % distri_config.pp_num_patch == 0

        stride = self.module.stride[0]
        padding = self.module.padding[0]

        output_h = x.shape[2] // stride // distri_config.pp_num_patch
        idx = self.batch_idx
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
        return F.conv2d(
            padded_input,
            self.module.weight,
            self.module.bias,
            stride=stride,
            padding="valid",
        )

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        # [2, 4, 128, 128]

        distri_config = self.distri_config
        if distri_config.pp_num_patch == 1:
            output = self.naive_forward(x)
        else:
            if self.is_first_layer:
                full_x = self.buffer_list
                if (
                    distri_config.mode == "full_sync"
                    or self.counter <= distri_config.warmup_steps
                ):
                    full_x = x
                    output = self.naive_forward(full_x)
                    # [2, 1152, 64, 64]
                else:
                    _, _, cc, _ = full_x.shape
                    _, _, c, _ = x.shape
                    assert cc // distri_config.pp_num_patch == c
                    full_x[:, :, c * self.batch_idx : c * (self.batch_idx + 1), :] = x
                    output = self.sliced_forward(full_x)
                self.buffer_list = full_x
            else:
                raise NotImplementedError

            # else:
            #     boundary_size = self.module.padding[0]
            #     # if self.buffer_list is None:
            #     #     if self.comm_manager.buffer_list is None:
            #     #         self.idx = self.comm_manager.register_tensor(
            #     #             shape=[2, x.shape[0], x.shape[1], boundary_size, x.shape[3]],
            #     #             torch_dtype=x.dtype,
            #     #             layer_type="conv2d",
            #     #         )
            #     #     else:
            #     #         self.buffer_list = self.comm_manager.get_buffer_list(self.idx)
            #     if self.buffer_list is None:
            #         output = self.naive_forward(x)
            #         self.buffer_list = x
            #     else:

            #         def create_padded_x():
            #             if distri_config.split_idx() == 0:
            #                 concat_x = torch.cat([x, self.buffer_list[distri_config.split_idx() + 1][0]], dim=2)
            #                 padded_x = F.pad(concat_x, [0, 0, boundary_size, 0], mode="constant")
            #             elif distri_config.split_idx() == distri_config.n_device_per_batch - 1:
            #                 concat_x = torch.cat([self.buffer_list[distri_config.split_idx() - 1][1], x], dim=2)
            #                 padded_x = F.pad(concat_x, [0, 0, 0, boundary_size], mode="constant")
            #             else:
            #                 padded_x = torch.cat(
            #                     [
            #                         self.buffer_list[distri_config.split_idx() - 1][1],
            #                         x,
            #                         self.buffer_list[distri_config.split_idx() + 1][0],
            #                     ],
            #                     dim=2,
            #                 )
            #             return padded_x

            #         boundary = torch.stack([x[:, :, :boundary_size, :], x[:, :, -boundary_size:, :]], dim=0)

            #         if distri_config.mode == "full_sync" or self.counter <= distri_config.warmup_steps:
            #             dist.all_gather(self.buffer_list, boundary, group=distri_config.batch_group, async_op=False)
            #             padded_x = create_padded_x()
            #             output = F.conv2d(
            #                 padded_x,
            #                 self.module.weight,
            #                 self.module.bias,
            #                 stride=self.module.stride[0],
            #                 padding=(0, self.module.padding[1]),
            #             )
            #         else:
            #             padded_x = create_padded_x()
            #             output = F.conv2d(
            #                 padded_x,
            #                 self.module.weight,
            #                 self.module.bias,
            #                 stride=self.module.stride[0],
            #                 padding=(0, self.module.padding[1]),
            #             )
            #             if distri_config.mode != "no_sync":
            #                 self.comm_manager.enqueue(self.idx, boundary)

        if (
            distri_config.mode == "full_sync"
            or self.counter <= distri_config.warmup_steps
        ):
            self.counter += 1
        else:
            self.batch_idx += 1
            if self.batch_idx == distri_config.pp_num_patch:
                self.counter += 1
                self.batch_idx = 0
        return output
