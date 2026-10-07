# Adapted from Tencent-Hunyuan/Prism (https://github.com/Tencent-Hunyuan/Prism),
# hymm/models/modules/dac_vae.py, itself derived from Descript's
# DAC (https://github.com/descriptinc/descript-audio-codec).
# Modified for xDiT: continuous, weight-norm-free decoder only;
# quantizer and audiotools helpers removed.
#
# Copyright (c) 2023-present, Descript
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Copyright (C) 2026 Tencent. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice (including the next
# paragraph) shall be included in all copies or substantial portions of the
# Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""DAC audio autoencoder in the continuous-latent form Prism's audio tower uses.

Vendored from Tencent-Hunyuan/Prism ``hymm/models/modules/dac_vae.py`` (itself
Descript's DAC). Only the continuous, weight-norm-free configuration is kept,
which is what the MOVA checkpoint ships; the residual vector quantizer and the
``audiotools`` file helpers are not needed for generation. Parameter names match
the checkpoint.
"""

import math
from typing import List

import numpy as np
import torch
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from diffusers.utils.accelerate_utils import apply_forward_hook
from torch import nn


def snake(x, alpha):
    shape = x.shape
    x = x.reshape(shape[0], shape[1], -1)
    x = x + (alpha + 1e-9).reciprocal() * torch.sin(alpha * x).pow(2)
    return x.reshape(shape)


class Snake1d(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1))

    def forward(self, x):
        return snake(x, self.alpha)


class ResidualUnit(nn.Module):
    def __init__(self, dim: int = 16, dilation: int = 1):
        super().__init__()
        pad = ((7 - 1) * dilation) // 2
        self.block = nn.Sequential(
            Snake1d(dim),
            nn.Conv1d(dim, dim, kernel_size=7, dilation=dilation, padding=pad),
            Snake1d(dim),
            nn.Conv1d(dim, dim, kernel_size=1),
        )

    def forward(self, x):
        y = self.block(x)
        pad = (x.shape[-1] - y.shape[-1]) // 2
        if pad > 0:
            x = x[..., pad:-pad]
        return x + y


class EncoderBlock(nn.Module):
    def __init__(self, dim: int = 16, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            ResidualUnit(dim // 2, dilation=1),
            ResidualUnit(dim // 2, dilation=3),
            ResidualUnit(dim // 2, dilation=9),
            Snake1d(dim // 2),
            nn.Conv1d(dim // 2, dim, kernel_size=2 * stride, stride=stride, padding=math.ceil(stride / 2)),
        )

    def forward(self, x):
        return self.block(x)


class Encoder(nn.Module):
    def __init__(self, d_model: int, strides: List[int], d_latent: int):
        super().__init__()
        block = [nn.Conv1d(1, d_model, kernel_size=7, padding=3)]
        for stride in strides:
            d_model *= 2
            block.append(EncoderBlock(d_model, stride=stride))
        block += [Snake1d(d_model), nn.Conv1d(d_model, d_latent, kernel_size=3, padding=1)]
        self.block = nn.Sequential(*block)

    def forward(self, x):
        return self.block(x)


class DecoderBlock(nn.Module):
    def __init__(self, input_dim: int = 16, output_dim: int = 8, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            Snake1d(input_dim),
            nn.ConvTranspose1d(
                input_dim,
                output_dim,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
                output_padding=stride % 2,
            ),
            ResidualUnit(output_dim, dilation=1),
            ResidualUnit(output_dim, dilation=3),
            ResidualUnit(output_dim, dilation=9),
        )

    def forward(self, x):
        return self.block(x)


class Decoder(nn.Module):
    def __init__(self, input_channel: int, channels: int, rates: List[int], d_out: int = 1):
        super().__init__()
        layers = [nn.Conv1d(input_channel, channels, kernel_size=7, padding=3)]
        for i, stride in enumerate(rates):
            layers.append(DecoderBlock(channels // 2**i, channels // 2 ** (i + 1), stride))
        output_dim = channels // 2 ** len(rates)
        layers += [Snake1d(output_dim), nn.Conv1d(output_dim, d_out, kernel_size=7, padding=3), nn.Tanh()]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class DAC(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(
        self,
        encoder_dim: int = 64,
        encoder_rates: List[int] = (2, 4, 8, 8),
        latent_dim: int = None,
        decoder_dim: int = 1536,
        decoder_rates: List[int] = (8, 8, 4, 2),
        n_codebooks: int = 9,
        codebook_size: int = 1024,
        codebook_dim: int = 8,
        quantizer_dropout: bool = False,
        sample_rate: int = 44100,
        continuous: bool = False,
        use_weight_norm: bool = True,
    ):
        super().__init__()
        if not continuous or use_weight_norm:
            raise NotImplementedError(
                "Prism's audio VAE is the continuous DAC with weight norm folded in "
                "(continuous=True, use_weight_norm=False)."
            )
        if latent_dim is None:
            latent_dim = encoder_dim * (2 ** len(encoder_rates))
        self.sample_rate = sample_rate
        self.latent_dim = latent_dim
        self.hop_length = int(np.prod(encoder_rates))

        self.encoder = Encoder(encoder_dim, list(encoder_rates), latent_dim)
        self.quant_conv = nn.Conv1d(latent_dim, 2 * latent_dim, 1)
        self.post_quant_conv = nn.Conv1d(latent_dim, latent_dim, 1)
        self.decoder = Decoder(latent_dim, decoder_dim, list(decoder_rates))

    @apply_forward_hook
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """``[B, latent_dim, T]`` latents to ``[B, 1, T * hop_length]`` waveform."""
        return self.decoder(self.post_quant_conv(z))
