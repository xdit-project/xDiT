from typing import Union, Optional
import torch
from torch import nn
from torch import Tensor
from torch.nn import functional as F
from torch.nn.common_types import _size_2_t

from legacy.pipefuser.logger import init_logger

logger = init_logger(__name__)


class ChunkConv2d(nn.Conv2d):
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
        padding_mode: str = "zeros",
        device=None,
        dtype=None,
        chunk_size: int = -1,
    ) -> None:
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
            device,
            dtype,
        )
        self.chunk_size = chunk_size

    def _conv_forward(
        self, input: Tensor, weight: Tensor, bias: Optional[Tensor]
    ) -> Tensor:
        inner_input = input
        if self.padding_mode != "zeros":
            inner_input = F.pad(
                input, self._reversed_padding_repeated_twice, mode=self.padding_mode
            )
        _, _, h, w = inner_input.shape
        if self.chunk_size <= 0 or (h <= self.chunk_size and w <= self.chunk_size):
            return F.conv2d(
                inner_input,
                weight,
                bias,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
            )

        if self.padding != 0 and self.padding_mode == "zeros":
            if isinstance(self.padding, int):
                padding_tuple = (self.padding,) * 4
            elif isinstance(self.padding, tuple):
                padding_tuple = (self.padding[0],) * 2 + (self.padding[1],) * 2
            else:
                raise ValueError(
                    f"padding should be int or tuple, type:{type(self.padding)}"
                )
            inner_input = F.pad(inner_input, padding_tuple, mode="constant")
        _, _, h, w = inner_input.shape
        num_chunks_in_h = (h + self.chunk_size - 1) // self.chunk_size
        num_chunks_in_w = (w + self.chunk_size - 1) // self.chunk_size
        unit_chunk_size_h = h // num_chunks_in_h
        unit_chunk_size_w = w // num_chunks_in_w
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
                f"stride should be int or tuple, type: {type(self.stride)}"
            )

        def correct_end(end, kernel_size, stride):
            return ((end + stride - 1) // stride - 1) * stride + kernel_size

        def correct_start(start, stride):
            return ((start + stride - 1) // stride) * stride

        outputs = []
        for idx_h in range(num_chunks_in_h):
            inner_output = []
            for idx_w in range(num_chunks_in_w):
                start_w = idx_w * unit_chunk_size_w
                start_h = idx_h * unit_chunk_size_h
                end_w = (idx_w + 1) * unit_chunk_size_w
                end_h = (idx_h + 1) * unit_chunk_size_h
                if idx_w + 1 < num_chunks_in_w:
                    end_w = correct_end(end_w, kernel_size_w, stride_w)
                else:
                    end_w = w
                if idx_h + 1 < num_chunks_in_h:
                    end_h = correct_end(end_h, kernel_size_h, stride_h)
                else:
                    end_h = h

                if idx_w > 0:
                    start_w = correct_start(start_w, stride_w)
                if idx_h > 0:
                    start_h = correct_start(start_h, stride_h)

                inner_output.append(
                    F.conv2d(
                        inner_input[:, :, start_h:end_h, start_w:end_w],
                        weight,
                        bias,
                        self.stride,
                        0,
                        self.dilation,
                        self.groups,
                    )
                )
            outputs.append(torch.cat(inner_output, dim=-1))
        return torch.cat(outputs, dim=-2)


class PatchConv2d:

    def __init__(self, chunk_size: int = -1):
        self.chunk_size = chunk_size

    def __call__(self, module: nn.Module) -> nn.Module:

        def replace_conv2d(module, name):
            if not isinstance(module, nn.Module):
                raise ValueError(f"module should be nn.Module, type: {type(module)}")

            for name, child_module in module.named_children():
                if isinstance(child_module, nn.Conv2d):
                    chunk_conv2d = ChunkConv2d(
                        child_module.in_channels,
                        child_module.out_channels,
                        child_module.kernel_size,
                        child_module.stride,
                        child_module.padding,
                        child_module.dilation,
                        child_module.groups,
                        child_module.bias is not None,
                        child_module.padding_mode,
                        child_module.weight.device,
                        child_module.weight.dtype,
                        self.chunk_size,
                    )
                    chunk_conv2d.weight.data = child_module.weight.data
                    chunk_conv2d.bias.data = child_module.bias.data
                    setattr(module, name, chunk_conv2d)
                    logger.info(f"{name} is replaced by ChunkConv2d")
                replace_conv2d(child_module, name)

        for attr_str in dir(module):
            target_attr = getattr(module, attr_str)
            # print(f"attr: {attr_str}, type: {type(target_attr)}")
            if isinstance(target_attr, nn.Module):
                replace_conv2d(target_attr, attr_str)
        return module
