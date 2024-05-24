import torch
import torch.nn as nn
import unittest

from pipefuser.modules.opt.chunk_conv2d import ChunkConv2d


class TestChunkConv(unittest.TestCase):
    def setUp(self):
        self.conv = nn.Conv2d(
            in_channels=3,
            out_channels=3,
            kernel_size=3,
            stride=1,
            padding=(1, 1),
            dilation=1,
            groups=1,
            bias=True,
            padding_mode="zeros",
            device=None,
            dtype=None,
        )
        self.chunk_conv = ChunkConv2d(
            in_channels=3,
            out_channels=3,
            kernel_size=3,
            stride=1,
            padding=(1, 1),
            dilation=1,
            groups=1,
            bias=True,
            padding_mode="zeros",
            device=None,
            dtype=None,
            chunk_size=1024,
        )
        self.chunk_conv.weight.data = self.conv.weight.data
        self.chunk_conv.bias.data = self.conv.bias.data
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=3,
            kernel_size=3,
            stride=2,
            padding=1,
            dilation=1,
            groups=1,
            bias=True,
            padding_mode="zeros",
            device=None,
            dtype=None,
        )
        self.chunk_conv1 = ChunkConv2d(
            in_channels=3,
            out_channels=3,
            kernel_size=3,
            stride=2,
            padding=1,
            dilation=1,
            groups=1,
            bias=True,
            padding_mode="zeros",
            device=None,
            dtype=None,
            chunk_size=1024,
        )
        self.chunk_conv1.weight.data = self.conv1.weight.data
        self.chunk_conv1.bias.data = self.conv1.bias.data
        self.conv2 = nn.Conv2d(
            in_channels=3,
            out_channels=3,
            kernel_size=2,
            stride=2,
            padding=0,
            dilation=1,
            groups=1,
            bias=True,
            padding_mode="zeros",
            device=None,
            dtype=None,
        )
        self.chunk_conv2 = ChunkConv2d(
            in_channels=3,
            out_channels=3,
            kernel_size=2,
            stride=2,
            padding=0,
            dilation=1,
            groups=1,
            bias=True,
            padding_mode="zeros",
            device=None,
            dtype=None,
            chunk_size=1024,
        )
        self.chunk_conv2.weight.data = self.conv2.weight.data
        self.chunk_conv2.bias.data = self.conv2.bias.data
        self.input = torch.rand(1, 3, 2058, 2058)

    def test_chunk_conv(self):
        output_chunk = self.chunk_conv(self.input)
        output_normal = self.conv(self.input)
        self.assertTrue(torch.allclose(output_chunk, output_normal))
        output_chunk1 = self.chunk_conv1(self.input)
        output_normal1 = self.conv1(self.input)
        self.assertTrue(torch.allclose(output_chunk1, output_normal1))
        output_chunk2 = self.chunk_conv2(self.input)
        output_normal2 = self.conv2(self.input)
        self.assertTrue(torch.allclose(output_chunk2, output_normal2))

    def tearDown(self):
        del self.conv
        del self.chunk_conv
        del self.input
        del self.conv1
        del self.chunk_conv1
        del self.conv2
        del self.chunk_conv2


if __name__ == "__main__":
    unittest.main()
