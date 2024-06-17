# Copyright (c) 2024.6.12, PC Yang. All rights reserved.
import torch
import torch.distributed as dist
from torch import nn

from .parallel_state import (init_patch_parallel,
                             get_patch_parallel_next_group,
                             get_patch_parallel_previous_group)
from .patch_parallelism_conv2d import (PatchParallelismConv2d,
                                       PatchParallelismConv2dFirst,
                                       PatchParallelismConv2dLast)

import logging

logger = logging.getLogger(__name__)


class ReplaceConv2dWithPatchParallelismConv2d:

    def __init__(
        self,
        first_conv2d_name: str,
        last_conv2d_name: str,
    ):
        self.first_conv2d_name = first_conv2d_name
        self.last_conv2d_name = last_conv2d_name
        self.order_idx = 0

    def __call__(self, module: nn.Module) -> nn.Module:

        def replace_conv2d(module, name):
            if not isinstance(module, nn.Module):
                raise ValueError(
                    f"module should be nn.Module, type: {type(module)}")

            def _replace(module, child_module, name, kind, PatchClass):
                kwargs = {}
                if kind in {
                        "PatchParallelismConv2dLast", "PatchParallelismConv2d"
                }:
                    kwargs[
                        "previous_group"] = get_patch_parallel_previous_group()
                    kwargs["next_group"] = get_patch_parallel_next_group()
                    self.order_idx += 1
                    kwargs["order_idx"] = self.order_idx
                chunk_conv2d = PatchClass(
                    child_module.in_channels, child_module.out_channels,
                    child_module.kernel_size, child_module.stride,
                    child_module.padding, child_module.dilation,
                    child_module.groups, child_module.bias is not None,
                    child_module.padding_mode, child_module.weight.device,
                    child_module.weight.dtype, **kwargs)
                chunk_conv2d.weight.data = child_module.weight.data
                chunk_conv2d.bias.data = child_module.bias.data
                setattr(module, name, chunk_conv2d)
                logger.info(f"{name} is replaced by {kind}")

            for name, child_module in module.named_children():
                if isinstance(child_module, nn.Conv2d):
                    if name == self.first_conv2d_name:
                        _replace(module, child_module, name,
                                 "PatchParallelismConv2dFirst",
                                 PatchParallelismConv2dFirst)
                    elif name == self.last_conv2d_name:
                        _replace(module, child_module, name,
                                 "PatchParallelismConv2dLast",
                                 PatchParallelismConv2dLast)
                    else:
                        _replace(module, child_module, name,
                                 "PatchParallelismConv2d",
                                 PatchParallelismConv2d)
                replace_conv2d(child_module, name)

        for attr_str in dir(module):
            target_attr = getattr(module, attr_str)
            if isinstance(target_attr, nn.Module):
                replace_conv2d(target_attr, attr_str)
        return module
