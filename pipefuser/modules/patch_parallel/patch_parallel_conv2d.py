# Copyright (c) 2024.6.12, PC Yang. All rights reserved.

from typing import Union, Optional, Any
from enum import Enum
import torch
import torch.distributed as dist
from torch import nn
from torch import Tensor
from torch.nn import functional as F
from torch.nn.common_types import _size_2_t

import logging

logger = logging.getLogger(__name__)


class PATCH_PARA_KIND(Enum):
    PATCH_PARA_KIND_NORMAL = 0
    PATCH_PARA_KIND_FIRST = 1
    PATCH_PARA_KIND_LAST = 2


class PatchParallelismConv2d(nn.Conv2d):
    __doc__ = r"""
    When the order_idx is even:
        rank0 -> rank1 -> rank2 -> ... -> rank-n
    When the order_idx is odd:
        rank-n -> rank-n-1 -> ... -> rank1 -> rank0
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: _size_2_t,
                 stride: _size_2_t = 1,
                 padding: Union[str, _size_2_t] = 0,
                 dilation: _size_2_t = 1,
                 groups: int = 1,
                 bias: bool = True,
                 padding_mode: str = 'zeros',
                 device=None,
                 dtype=None,
                 previous_group=None,
                 next_group=None,
                 order_idx: int = 0) -> None:
        super().__init__(in_channels, out_channels, kernel_size, stride,
                         padding, dilation, groups, bias, padding_mode, device,
                         dtype)
        self.previous_group = previous_group
        self.next_group = next_group
        self.order_idx = order_idx

    # for the even
    def even_calc_send_start(self, end, kernel_size, stride):
        # The raw Equal: ((end - (kernel_size-1) + stride - 1) // stride) * stride + (kernel_size-1) - kernel_size
        return ((end - (kernel_size - 1) + stride - 1) // stride) * stride - 1

    def even_calc_cur_end(self, end, kernel_size, stride):
        return ((end - (kernel_size - 1) + stride - 1) // stride -
                1) * stride + kernel_size - 1

    # for the odd
    def odd_correct_end(self, end, kernel_size, stride):
        return ((end + stride - 1) // stride - 1) * stride + kernel_size

    def odd_correct_start(self, start, stride):
        return ((start + stride - 1) // stride) * stride

    def get_world_size_and_rank(self):
        rank = 0
        world_size = 1
        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
            world_size = dist.get_world_size()
        return world_size, rank

    def get_kernel_size_and_stride(self):
        # formatting kernel_size and stride
        if isinstance(self.kernel_size, int):
            kernel_size_h, kernel_size_w = self.kernel_size, self.kernel_size
        elif isinstance(self.kernel_size, tuple):
            kernel_size_h, kernel_size_w = self.kernel_size
        else:
            raise ValueError(
                f"kernel_size should be int or tuple, type:{type(self.kernel_size)}"
            )

        if isinstance(self.stride, int):
            stride_h, stride_w = self.stride, self.stride
        elif isinstance(self.stride, tuple):
            stride_h, stride_w = self.stride
        else:
            raise ValueError(
                f"stride should be int or tuple, type: {type(self.stride)}")
        return kernel_size_h, kernel_size_w, stride_h, stride_w

    def _conv_forward(self, input: Tensor, weight: Tensor,
                      bias: Optional[Tensor]) -> Tensor:
        inner_input = input
        world_size, rank = self.get_world_size_and_rank()

        def adjust_padding(padding: list, rank: int):
            if rank == 0:
                padding[-1] = 0
            elif rank == world_size - 1:
                padding[-2] = 0
            else:
                padding[-2:] = [0, 0]

        if self.padding_mode != 'zeros':
            copy_padding = self._reversed_padding_repeated_twice[:]
            if world_size > 1:
                adjust_padding(copy_padding, rank)
            inner_input = F.pad(input, copy_padding, mode=self.padding_mode)

        if world_size == 1:
            return F.conv2d(inner_input, weight, bias, self.stride,
                            self.padding, self.dilation, self.groups)

        if self.padding != 0:
            if isinstance(self.padding, int):
                padding_tuple = (self.padding, ) * 4
            elif isinstance(self.padding, tuple):
                padding_tuple = (self.padding[0], ) * 2 + (
                    self.padding[1], ) * 2
            else:
                raise ValueError(
                    f"padding should be int or tuple, type:{type(self.padding)}"
                )
            adjust_padding(padding_tuple, rank)
            inner_input = F.pad(inner_input, padding_tuple, mode="constant")

        b, c, h, w = inner_input.shape
        cpu = torch.device('cpu:')
        segment_list = [h]
        if world_size > 1:
            tensor_list = [
                torch.zeros(1, dtype=torch.int64, device=cpu)
                for _ in range(world_size)
            ]
            dist.all_gather(tensor_list, torch.tensor([h], device=cpu))
            segment_list = [t.item() for t in tensor_list]

        kernel_size_h, _, stride_h, _ = self.get_kernel_size_and_stride()

        cuda = torch.device(f"cuda:{rank}")
        previous_recv_tensor = None
        next_recv_tensor = None
        cur_end = sum(segment_list[:rank + 1])
        if self.previous_group:
            if self.order_idx % 2 == 0:
                pre_end = sum(segment_list[:rank])
                pre_start = self.even_calc_send_start(pre_end, kernel_size_h,
                                                      stride_h)
                previous_recv_tensor = torch.zeros(
                    [b, c, pre_end - pre_start, w], device=cuda)
                dist.recv(previous_recv_tensor, rank - 1, self.previous_group)
            else:
                pre_end = sum(segment_list[:rank])
                pre_max_end = self.odd_correct_end(pre_end, kernel_size_h,
                                                   stride_h)
                dist.send(inner_input[:, :, :pre_max_end - pre_end, :],
                          rank - 1, self.previous_group)

        if self.next_group:
            if self.order_idx % 2 == 0:
                cur_end = sum(segment_list[:rank + 1])
                cur_send_start = self.even_calc_send_start(
                    cur_end, kernel_size_h, stride_h)
                dist.send(inner_input[:, :, cur_send_start:cur_end, :],
                          rank + 1, self.next_group)
            else:
                cur_end = sum(segment_list[:rank + 1])
                cur_max_end = self.odd_correct_end(cur_end, kernel_size_h,
                                                   stride_h)
                next_recv_tensor = torch.zeros(
                    [b, c, cur_max_end - cur_end, w], device=cuda)
                dist.recv(next_recv_tensor, rank + 1, self.next_group)

        if self.order_idx % 2 == 0:
            # even: rank0 -> rank1 -> ... -> rank-n
            if rank + 1 < world_size:
                cur_end = sum(segment_list[:rank + 1])
                cur_fix_end = self.even_calc_cur_end(cur_end, kernel_size_h,
                                                     stride_h)
                inner_input = inner_input[:, :, :-(cur_end - cur_fix_end), :]
            if previous_recv_tensor is not None:
                inner_input = torch.cat([previous_recv_tensor, inner_input],
                                        dim=2)
            return F.conv2d(inner_input, weight, bias, self.stride, 0,
                            self.dilation, self.groups)
        else:
            # odd: rank-n -> rank-n-1 ... -> rank0
            if rank > 0:
                pre_end = sum(segment_list[:rank])
                cur_start = self.odd_correct_start(pre_end, stride_h)
                inner_input = inner_input[:, :, (cur_start - pre_end):, :]
            if next_recv_tensor is not None:
                inner_input = torch.cat([inner_input, next_recv_tensor], dim=2)
            return F.conv2d(inner_input, weight, bias, self.stride, 0,
                            self.dilation, self.groups)


class PatchParallelismConv2dFirst(PatchParallelismConv2d):
    __doc__ = r"""
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: Union[str, _size_2_t] = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',
        device=None,
        dtype=None,
    ) -> None:
        super().__init__(in_channels, out_channels, kernel_size, stride,
                         padding, dilation, groups, bias, padding_mode, device,
                         dtype)

    def _conv_forward(self, input: Tensor, weight: Tensor,
                      bias: Optional[Tensor]) -> Tensor:
        inner_input = input
        if self.padding_mode != 'zeros':
            inner_input = F.pad(input,
                                self._reversed_padding_repeated_twice,
                                mode=self.padding_mode)
        _, _, h, w = inner_input.shape
        if self.chunk_size <= 0 or (h <= self.chunk_size
                                    and w <= self.chunk_size):
            return F.conv2d(inner_input, weight, bias, self.stride,
                            self.padding, self.dilation, self.groups)

        if self.padding != 0:
            if isinstance(self.padding, int):
                padding_tuple = (self.padding, ) * 4
            elif isinstance(self.padding, tuple):
                padding_tuple = (self.padding[0], ) * 2 + (
                    self.padding[1], ) * 2
            else:
                raise ValueError(
                    f"padding should be int or tuple, type:{type(self.padding)}"
                )
            inner_input = F.pad(inner_input, padding_tuple, mode="constant")

        _, _, h, w = inner_input.shape
        world_size, rank = self.get_world_size_and_rank()

        unit_chunk_size_h = (h + world_size - 1) // world_size
        kernel_size_h, _, stride_h, _ = self.get_kernel_size_and_stride()

        start_h = rank * unit_chunk_size_h
        end_h = (rank + 1) * unit_chunk_size_h
        if rank + 1 < world_size:
            end_h = self.odd_correct_end(end_h, kernel_size_h, stride_h)
        else:
            end_h = h

        if rank > 0:
            start_h = self.odd_correct_start(start_h, stride_h)

        return F.conv2d(inner_input[:, :, start_h:end_h, :], weight, bias,
                        self.stride, 0, self.dilation, self.groups)


class PatchParallelismConv2dLast(PatchParallelismConv2d):
    __doc__ = r"""
    When the order_idx is even:
        rank0 -> rank1 -> rank2 -> ... -> rank-n
    When the order_idx is odd:
        rank-n -> rank-n-1 -> ... -> rank1 -> rank0
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: _size_2_t,
                 stride: _size_2_t = 1,
                 padding: Union[str, _size_2_t] = 0,
                 dilation: _size_2_t = 1,
                 groups: int = 1,
                 bias: bool = True,
                 padding_mode: str = 'zeros',
                 device=None,
                 dtype=None,
                 previous_group=None,
                 next_group=None,
                 order_idx: int = 0) -> None:
        super().__init__(in_channels, out_channels, kernel_size, stride,
                         padding, dilation, groups, bias, padding_mode, device,
                         dtype)
        self.previous_group = previous_group
        self.next_group = next_group
        self.order_idx = order_idx

    def _conv_forward(self, input: Tensor, weight: Tensor,
                      bias: Optional[Tensor]) -> Tensor:
        result = super()._conv_forward(input, weight, bias)
        b, c, h, w = result.shape
        world_size, rank = self.get_world_size_and_rank()
        if world_size == 1:
            return result
        tensor_list = [
            torch.zeros(1, dtype=torch.int64) for _ in range(world_size)
        ]
        dist.all_gather(tensor_list,
                        torch.tensor([h], dtype=torch.int64, device="cpu"))
        segment_list = [t.item() for t in tensor_list]
        max_h = max(segment_list)
        padding = [0, 0, 0, max_h - h]
        result = F.pad(result, padding, mode="constant")
        result_list = [
            torch.zeros([b, c, max_h, w],
                        dtype=result.dtype,
                        device=f"cuda:{rank}") for _ in range(world_size)
        ]
        dist.all_gather(result_list, result)
        post_result_list = [
            res[:, :, :segment_list[rank], :] for res in result_list
        ]
        return torch.cat(post_result_list, dim=2)
