import os
import torch
import torch.nn as nn
import torch.distributed as dist
import unittest

from lagecy.pipefuser.modules.conv_parallel.patch_parallel_conv2d import (
    PatchParallelismConv2dFirst,
    PatchParallelismConv2d,
    PatchParallelismConv2dLast,
)
from lagecy.pipefuser.modules.conv_parallel.parallel_state import (
    get_patch_parallel_next_group,
    get_patch_parallel_previous_group,
    init_patch_parallel,
)

STRIDE = 2
PADDING = (1, 1)
SEED = 42


class TestPatchParallelConv2d(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(SEED)
        conv_list = []
        parallel_conv_list = []
        self.num_layers = 3
        for i in range(self.num_layers):
            conv_list.append(
                nn.Conv2d(
                    in_channels=3,
                    out_channels=3,
                    kernel_size=3,
                    stride=STRIDE,
                    padding=PADDING,
                    dilation=1,
                    groups=1,
                    bias=True,
                    padding_mode="zeros",
                    device=None,
                    dtype=None,
                )
            )
        dist.init_process_group(backend="nccl")
        init_patch_parallel()
        self.rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        self.conv = nn.ModuleList(conv_list).cuda()
        conv0 = PatchParallelismConv2dFirst(
            in_channels=3,
            out_channels=3,
            kernel_size=3,
            stride=STRIDE,
            padding=PADDING,
            dilation=1,
            groups=1,
            bias=True,
            padding_mode="zeros",
            device=None,
            dtype=None,
        )
        conv0.weight.data = self.conv[0].weight.data
        conv0.bias.data = self.conv[0].bias.data
        parallel_conv_list.append(conv0)
        conv1 = PatchParallelismConv2d(
            in_channels=3,
            out_channels=3,
            kernel_size=3,
            stride=STRIDE,
            padding=PADDING,
            dilation=1,
            groups=1,
            bias=True,
            padding_mode="zeros",
            device=None,
            dtype=None,
            previous_group=get_patch_parallel_previous_group(),
            next_group=get_patch_parallel_next_group(),
            order_idx=0,
        )
        conv1.weight.data = self.conv[1].weight.data
        conv1.bias.data = self.conv[1].bias.data
        parallel_conv_list.append(conv1)
        conv2 = PatchParallelismConv2dLast(
            in_channels=3,
            out_channels=3,
            kernel_size=3,
            stride=STRIDE,
            padding=PADDING,
            dilation=1,
            groups=1,
            bias=True,
            padding_mode="zeros",
            device=None,
            dtype=None,
            previous_group=get_patch_parallel_previous_group(),
            next_group=get_patch_parallel_next_group(),
            order_idx=1,
        )
        conv2.weight.data = self.conv[2].weight.data
        conv2.bias.data = self.conv[2].bias.data
        parallel_conv_list.append(conv2)
        self.parallel_conv = nn.ModuleList(parallel_conv_list).cuda()
        self.input = torch.rand(1, 3, 8192, 8192).cuda()

    def test_parallel_conv(self):
        output = self.input
        for layer in self.conv:
            output = layer(output)
        output_parallel = self.input
        for i, layer in enumerate(self.parallel_conv):
            output_parallel = layer(output_parallel)
        self.assertTrue(torch.allclose(output, output_parallel, atol=1e-3, rtol=1e-5))

    def tearDown(self):
        del self.conv
        del self.parallel_conv


if __name__ == "__main__":
    unittest.main()
