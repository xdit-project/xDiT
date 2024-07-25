import torch
from torch import nn
from torch.nn import functional as F

from pipefuser.config import ParallelConfig, RuntimeConfig
from pipefuser.model_executor.layers import PipeFuserLayerBaseWrapper
from pipefuser.logger import init_logger
from pipefuser.model_executor.layers import PipeFuserLayerWrappersRegister
from pipefuser.distributed import (
    get_pipeline_parallel_world_size,
)

logger = init_logger(__name__)


@PipeFuserLayerWrappersRegister.register(nn.Conv2d)
class PipeFuserConv2dWrapper(PipeFuserLayerBaseWrapper):
    def __init__(
        self,
        conv2d: nn.Conv2d,
        parallel_config: ParallelConfig,
        runtime_config: RuntimeConfig,
        *,
        is_first_layer: bool = True,
    ):
        super().__init__(
            module=conv2d,
            parallel_config=parallel_config,
            runtime_config=runtime_config,
        )
        self.is_first_layer = is_first_layer

    def naive_forward(self, x: torch.Tensor) -> torch.Tensor:
        #  x: [B, C, H, W]
        output = self.module(x)
        return output

    # TODO fix implementation problems in sliced_forward
    # only available for patchify process
    def sliced_forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape

        stride = self.module.stride[0]
        padding = self.module.padding[0]

        output_h = h // self.num_pipeline_patch // stride
        idx = self.current_patch_idx
        h_begin = self.pp_patches_start_idx_local[idx] - padding
        h_end = self.pp_patches_start_idx_local[idx + 1] + padding
        final_padding = [padding, padding, 0, 0]
        if h_begin < 0:
            h_begin = 0
            final_padding[2] = padding
        if h_end > h:
            h_end = h
            final_padding[3] = padding
        sliced_input = x[:, :, h_begin:h_end, :]
        padded_input = F.pad(sliced_input, final_padding, mode="constant")
        result = F.conv2d(
            padded_input,
            self.module.weight,
            self.module.bias,
            stride=stride,
            padding="valid",
        )
        return result

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        if (
            get_pipeline_parallel_world_size() == 1
            or self.module.kernel_size == (1, 1)
            or self.module.kernel_size == 1
        ):
            output = self.naive_forward(x)
        else:
            if self.is_first_layer:
                if not self.patched_mode or self.num_pipeline_patch == 1:
                    self.activation_cache = x
                    output = self.naive_forward(self.activation_cache)
                else:
                    if self.activation_cache is None:
                        self.activation_cache = torch.zeros(
                            [
                                x.shape[0],
                                x.shape[1],
                                self.pp_patches_start_idx_local[-1],
                                x.shape[3],
                            ],
                            dtype=x.dtype,
                            device=x.device,
                        )

                    self.activation_cache[
                        :,
                        :,
                        self.pp_patches_start_idx_local[self.current_patch_idx]: 
                        self.pp_patches_start_idx_local[self.current_patch_idx+1],
                        :,
                    ] = x
                    output = self.sliced_forward(self.activation_cache)

            else:
                raise NotImplementedError

            # else:
            #     boundary_size = self.module.padding[0]
            #     # if self.buffer_list is None:
            #     #     if self.comm_manager.buffer_list is None: #     #         self.idx = self.comm_manager.register_tensor( #     #             shape=[2, x.shape[0], x.shape[1], boundary_size, x.shape[3]], #     #             torch_dtype=x.dtype,
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

        if self.patched_mode:
            self.patch_step()
        return output
