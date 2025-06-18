from typing import List
import torch
from torch import nn
from torch.nn import functional as F
import torch.distributed as dist

from diffusers.models.transformers.sana_transformer import GLUMBConv

from xfuser.config import ParallelConfig, RuntimeConfig
from xfuser.core.distributed.parallel_state import get_sequence_parallel_world_size
from xfuser.core.distributed.runtime_state import get_runtime_state
from xfuser.model_executor.layers import xFuserLayerBaseWrapper
from xfuser.logger import init_logger
from xfuser.model_executor.layers import xFuserLayerWrappersRegister
from xfuser.core.distributed import (
    get_pipeline_parallel_world_size,
    get_sp_group,
)

logger = init_logger(__name__)


@xFuserLayerWrappersRegister.register(nn.Conv2d)
class xFuserConv2dWrapper(xFuserLayerBaseWrapper):
    def __init__(
        self,
        conv2d: nn.Conv2d,
        *,
        is_first_layer: bool = True,
    ):
        super().__init__(
            module=conv2d,
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

        idx = get_runtime_state().pipeline_patch_idx
        pp_patches_start_idx_local = get_runtime_state().pp_patches_start_idx_local
        h_begin = pp_patches_start_idx_local[idx] - padding
        h_end = pp_patches_start_idx_local[idx + 1] + padding
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
            (
                get_pipeline_parallel_world_size() == 1
                and get_sequence_parallel_world_size() == 1
            )
            or self.module.kernel_size == (1, 1)
            or self.module.kernel_size == 1
        ):
            output = self.naive_forward(x)
        else:
            if self.is_first_layer:
                if (
                    not get_runtime_state().patch_mode
                    or get_runtime_state().num_pipeline_patch == 1
                ):
                    self.activation_cache = x
                    output = self.naive_forward(self.activation_cache)
                else:
                    if self.activation_cache is None:
                        self.activation_cache = torch.zeros(
                            [
                                x.shape[0],
                                x.shape[1],
                                get_runtime_state().pp_patches_start_idx_local[-1],
                                x.shape[3],
                            ],
                            dtype=x.dtype,
                            device=x.device,
                        )

                    self.activation_cache[
                        :,
                        :,
                        get_runtime_state()
                        .pp_patches_start_idx_local[
                            get_runtime_state().pipeline_patch_idx
                        ] : get_runtime_state()
                        .pp_patches_start_idx_local[
                            get_runtime_state().pipeline_patch_idx + 1
                        ],
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

        return output


@xFuserLayerWrappersRegister.register(GLUMBConv)
class xFuserGLUMBConvWrapper(xFuserLayerBaseWrapper):
    def __init__(self, module):
        super().__init__(module)

        self.conv_depth_wo_bias = nn.Conv2d(
            self.conv_depth.in_channels,
            self.conv_depth.out_channels,
            self.conv_depth.kernel_size,
            self.conv_depth.stride,
            self.conv_depth.padding,
            groups=self.conv_depth.groups,
            dilation=self.conv_depth.dilation,
            bias=False,
        ).to(self.conv_depth.weight)
        #* Reference to the original conv_depth weight
        self.conv_depth_wo_bias.weight = self.conv_depth.weight
        self.patch_numbers = get_runtime_state().num_pipeline_patch

        self.cache = None
        self.head_cache = None
        self.tail_cache = None

    # Overlap communication
    def _pad_conv(self, x: torch.Tensor, pad: List[int], value=0):

        x = self.conv_inverted(x)
        x = self.nonlinearity(x)
        x = F.pad(x, pad, value=value)
        x = self.conv_depth_wo_bias(x)
        return x
    
    def _prepare_hidden_send_recv(self, hidden_states):
        head_rows = [
            hidden_states[:, :, idx] for idx in get_runtime_state().pp_patches_start_idx_local[:-1]
        ]
        tail_rows = [
            hidden_states[:, :, idx-1] for idx in get_runtime_state().pp_patches_start_idx_local[1:]
        ]
        head_rows= torch.stack(head_rows, dim=2).contiguous()
        tail_rows = torch.stack(tail_rows, dim=2).contiguous()
        
        overlap_head_rows = torch.empty_like(tail_rows)
        overlap_tail_rows = torch.empty_like(head_rows)
        
        return head_rows, tail_rows, overlap_head_rows, overlap_tail_rows

    def _prepare_hidden_send_recv_patch(self, hidden_states: torch.Tensor, pp_patch_idx: int) -> List[torch.Tensor]:
        tail_row = hidden_states[:, :, -1:].contiguous()

        if self.is_sp_first_rank:
            head_row = self.head_cache[:, :, pp_patch_idx: pp_patch_idx+1] \
                if not self.is_last_patch else torch.empty_like(tail_row)
            head_row = head_row.contiguous()
        else:
            head_row = hidden_states[:, :, :1].contiguous()
        
        overlap_head_row = torch.empty_like(tail_row)
        overlap_tail_row = torch.empty_like(head_row)
        
        return head_row, tail_row, overlap_head_row, overlap_tail_row

    @staticmethod
    def _wait(reqs):
        for req in reqs:
            req.wait()
    
    def forward(self, hidden_states: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        if self.residual_connection:
            residual = hidden_states

        patch_mode = get_runtime_state().patch_mode

        pp_patch_idx = get_runtime_state().pipeline_patch_idx
        self.is_first_patch = is_first_patch = pp_patch_idx == 0
        self.is_last_patch = is_last_patch = pp_patch_idx == self.patch_numbers - 1
        self.is_sp_first_rank = is_sp_first_rank = get_sp_group().is_first_rank
        self.is_sp_last_rank = is_sp_last_rank = get_sp_group().is_last_rank
        sp_world_size = get_sp_group().world_size

        if not patch_mode:
            head_row, tail_row, overlap_head_rows, overlap_tail_rows = self._prepare_hidden_send_recv(hidden_states)
        else:
            head_row, tail_row, overlap_head_rows, overlap_tail_rows = self._prepare_hidden_send_recv_patch(hidden_states, pp_patch_idx)
        
        send_head = dist.P2POp(dist.isend, head_row, get_sp_group().prev_rank, group=get_sp_group().device_group)
        recv_head = dist.P2POp(dist.irecv, overlap_tail_rows, get_sp_group().next_rank, group=get_sp_group().device_group)
        reqs_head = dist.batch_isend_irecv([send_head, recv_head])
        
        send_tail = dist.P2POp(dist.isend, tail_row, get_sp_group().next_rank, group=get_sp_group().device_group)
        recv_tail = dist.P2POp(dist.irecv, overlap_head_rows, get_sp_group().prev_rank, group=get_sp_group().device_group)
        reqs_tail = dist.batch_isend_irecv([send_tail, recv_tail])
            

        if get_pipeline_parallel_world_size() > 1:
            if patch_mode:
                if not is_first_patch and is_sp_first_rank:
                    self.head_cache[:, :, pp_patch_idx - 1] = hidden_states[:, :, 0]
                
                if sp_world_size == 1 and not is_last_patch:
                    self.tail_cache[:, :, pp_patch_idx] = hidden_states[:, :, -1]

            else:
                head_rows = [
                    hidden_states[:, :, idx] for idx in get_runtime_state().pp_patches_start_idx_local[1:-1]
                ]
                tail_rows = [
                    hidden_states[:, :, idx-1] for idx in get_runtime_state().pp_patches_start_idx_local[1:-1]
                ]
                self.head_cache = torch.stack(head_rows, dim=2) if head_rows else None
                self.tail_cache = torch.stack(tail_rows, dim=2) if tail_rows else None
            
        hidden_states = self.conv_inverted(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)

        if not patch_mode and get_pipeline_parallel_world_size() > 1:
            # Pipeline Warmup
            hidden_list = hidden_states.split(get_runtime_state().pp_patches_height, dim=2)
            hidden_list_out = [None] * len(hidden_list)
            for i, hidden_states in enumerate(hidden_list):
                hidden_list_out[i] = self.conv_depth(hidden_states)
            
            self._wait(reqs_head)
            for i, hidden_states in enumerate(hidden_list_out):
                idx = i if not is_sp_last_rank else i + 1
                if is_sp_last_rank and i == len(hidden_list_out) - 1:
                    continue
                overlap_tail_i = self._pad_conv(overlap_tail_rows[:, :, idx: idx+1], pad=[0, 0, 2, 0]) 
                hidden_list_out[i][:, :, -1] = hidden_list_out[i][:, :, -1] + overlap_tail_i[:, :, 1]
            
            self._wait(reqs_tail)
            for i, hidden_states in enumerate(hidden_list_out):
                idx = i if not is_sp_first_rank else i - 1
                if is_sp_first_rank and i == 0:
                    continue
                overlap_head_i = self._pad_conv(overlap_head_rows[:, :, idx: idx+1], pad=[0, 0, 0, 2])
                hidden_list_out[i][:, :, 0] = hidden_list_out[i][:, :, 0] + overlap_head_i[:, :, 1]
            hidden_states = torch.cat(hidden_list_out, dim=2) 

        else:
            hidden_states = self.conv_depth(hidden_states)

        if get_pipeline_parallel_world_size() > 1:
            if patch_mode:
                self._wait(reqs_tail)
                if is_first_patch and is_sp_first_rank:
                    pass
                elif is_sp_first_rank:
                    overlap_head = self._pad_conv(self.tail_cache[:, :, pp_patch_idx - 1: pp_patch_idx], pad=[0, 0, 0, 2])
                    hidden_states[:, :, 0] = hidden_states[:, :, 0] + overlap_head[:, :, 1]
                else:
                    overlap_head = self._pad_conv(overlap_head_rows, pad=[0, 0, 0, 2])
                    hidden_states[:, :, 0] = hidden_states[:, :, 0] + overlap_head[:, :, 1]
            
                self._wait(reqs_head)
                if is_last_patch and is_sp_last_rank:
                    pass
                elif sp_world_size == 1:
                    overlap_tail = self._pad_conv(self.head_cache[:, :, pp_patch_idx: pp_patch_idx + 1], pad=[0, 0, 2, 0])
                    hidden_states[:, :, -1] = hidden_states[:, :, -1] + overlap_tail[:, :, 1]
                else:
                    overlap_tail = self._pad_conv(overlap_tail_rows, pad=[0, 0, 2, 0])
                    hidden_states[:, :, -1] = hidden_states[:, :, -1] + overlap_tail[:, :, 1]
                    
                    if  not is_last_patch and is_sp_first_rank:
                        self.tail_cache[:, :, pp_patch_idx] = overlap_head_rows[:, :, 0]

        else:
            self._wait(reqs_tail)
            if not is_sp_first_rank:
                overlap_head = self._pad_conv(overlap_head_rows, pad=[0, 0, 0, 2])
                hidden_states[:, :, 0] = hidden_states[:, :, 0] + overlap_head[:, :, 1]
            
            self._wait(reqs_head)
            if not is_sp_last_rank:
                overlap_tail = self._pad_conv(overlap_tail_rows, pad=[0, 0, 2, 0])
                hidden_states[:, :, -1] = hidden_states[:, :, -1] + overlap_tail[:, :, 1]

        hidden_states, gate = torch.chunk(hidden_states, 2, dim=1)
        hidden_states = hidden_states * self.nonlinearity(gate)
        hidden_states = self.conv_point(hidden_states)

        if self.norm_type == "rms_norm":
            # move channel to the last dimension so we apply RMSnorm across channel dimension
            hidden_states = self.norm(hidden_states.movedim(1, -1)).movedim(-1, 1)

        if self.residual_connection:
            hidden_states = hidden_states + residual

        return hidden_states
