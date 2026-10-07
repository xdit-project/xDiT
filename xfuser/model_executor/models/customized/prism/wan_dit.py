"""Prism's video and audio DiT towers (MOVA ``WanModel`` / ``WanAudioModel``).

Vendored from Tencent-Hunyuan/Prism ``hymm/models/modules/wan_{video,audio}_dit.py``
for inference only: training, gradient checkpointing, the camera-control adapter,
CLIP image input and the block-sparse-attention variants are left out. Parameter
names match the Prism checkpoint. Attention runs through xDiT's sequence-parallel
helpers in :mod:`.sp`.
"""

import math
from typing import Tuple

import torch
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from torch.nn import RMSNorm

from .sp import PrismSeqInfo, gathered_kv_attention, local_attention, usp_attention

VIDEO = "video"
AUDIO = "audio"


def sinusoidal_embedding_1d(dim, position):
    sinusoid = torch.outer(
        position.type(torch.float64),
        torch.pow(10000, -torch.arange(dim // 2, dtype=torch.float64, device=position.device).div(dim // 2)),
    )
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x.to(position.dtype)


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].double() / dim))
    freqs = torch.outer(torch.arange(end, dtype=torch.float64), freqs)
    return torch.polar(torch.ones_like(freqs), freqs)


def precompute_freqs_cis_3d(dim: int, end: int = 1024):
    return (
        precompute_freqs_cis(dim - 2 * (dim // 3), end),
        precompute_freqs_cis(dim // 3, end),
        precompute_freqs_cis(dim // 3, end),
    )


def precompute_freqs_cis_1d(dim: int, end: int = 16384):
    return precompute_freqs_cis(dim, end).chunk(3, dim=-1)


def video_freqs(freqs, f: int, h: int, w: int, device) -> torch.Tensor:
    """3D RoPE table for an ``f x h x w`` token grid, ``[f*h*w, 1, head_dim // 2]``."""
    return (
        torch.cat(
            [
                freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
                freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
            ],
            dim=-1,
        )
        .reshape(f * h * w, 1, -1)
        .to(device)
    )


def audio_freqs(freqs, f: int, device) -> torch.Tensor:
    """1D RoPE table for ``f`` audio tokens, ``[f, 1, head_dim // 2]``."""
    return torch.cat([part[:f].view(f, -1) for part in freqs], dim=-1).reshape(f, 1, -1).to(device)


@torch.amp.autocast("cuda", enabled=False)
def rope_apply(x, freqs, head_dim):
    """Rotate ``[B, S, H*D]`` by complex ``freqs`` in float64, as the reference does."""
    b, s, _ = x.shape
    x_out = torch.view_as_complex(x.to(torch.float64).reshape(b, s, -1, head_dim // 2, 2))
    x_out = torch.view_as_real(x_out * freqs).flatten(2)
    return x_out.to(x.dtype)


def modulate(x, shift, scale):
    # The reference runs this under torch.compile, which computes in float32 and
    # rounds once; matching that keeps the two within bf16 rounding of each other.
    return (x.float() * (1 + scale.float()) + shift.float()).to(x.dtype)


class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, eps: float, stream: str):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.stream = stream

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)

    def forward(self, x, freqs, seq: PrismSeqInfo):
        q = rope_apply(self.norm_q(self.q(x)), freqs, self.head_dim)
        k = rope_apply(self.norm_k(self.k(x)), freqs, self.head_dim)
        v = self.v(x)
        if self.stream == VIDEO:
            x = usp_attention(q, k, v, self.num_heads, seq.video)
        else:
            x = gathered_kv_attention(q, k, v, self.num_heads, seq.audio)
        return self.o(x)


class CrossAttention(nn.Module):
    """Attention from either tower's tokens to the (replicated) text embedding."""

    def __init__(self, dim: int, num_heads: int, eps: float):
        super().__init__()
        self.num_heads = num_heads

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)

    def forward(self, x, context):
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(context))
        v = self.v(context)
        return self.o(local_attention(q, k, v, self.num_heads))


class DiTBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, ffn_dim: int, eps: float, stream: str):
        super().__init__()
        self.self_attn = SelfAttention(dim, num_heads, eps, stream)
        self.cross_attn = CrossAttention(dim, num_heads, eps)
        self.norm1 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(dim, eps=eps)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn_dim), nn.GELU(approximate="tanh"), nn.Linear(ffn_dim, dim))
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(self, x, context, t_mod, freqs, seq: PrismSeqInfo):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod
        ).chunk(6, dim=1)
        x = x + gate_msa * self.self_attn(modulate(self.norm1(x), shift_msa, scale_msa), freqs, seq)
        x = x + self.cross_attn(self.norm3(x), context)
        x = x + gate_mlp * self.ffn(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class Head(nn.Module):
    def __init__(self, dim: int, out_dim: int, patch_size: Tuple[int, ...], eps: float):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.head = nn.Linear(dim, out_dim * math.prod(patch_size))
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, t):
        shift, scale = (self.modulation.to(dtype=t.dtype, device=t.device) + t.unsqueeze(1)).chunk(2, dim=1)
        return self.head(self.norm(x) * (1 + scale) + shift)


class _TowerMixin:
    """Embeddings shared by both towers; the blocks are driven by ``MOVABridge``."""

    def _build_embeddings(self, dim, text_dim, freq_dim):
        self.text_embedding = nn.Sequential(nn.Linear(text_dim, dim), nn.GELU(approximate="tanh"), nn.Linear(dim, dim))
        self.time_embedding = nn.Sequential(nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))

    def time_embed(self, timestep):
        """Timestep embedding and its six modulation vectors, both float32."""
        t = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, timestep))
        return t, self.time_projection(t).unflatten(1, (6, self.dim))


class WanModel(_TowerMixin, ModelMixin, ConfigMixin):
    """MOVA video tower (one Wan2.2 expert)."""

    _no_split_modules = ["DiTBlock"]

    @register_to_config
    def __init__(
        self,
        dim: int,
        in_dim: int,
        ffn_dim: int,
        out_dim: int,
        text_dim: int,
        freq_dim: int,
        eps: float,
        patch_size: Tuple[int, int, int],
        num_heads: int,
        num_layers: int,
        has_image_input: bool = False,
        has_image_pos_emb: bool = False,
        has_ref_conv: bool = False,
        add_control_adapter: bool = False,
        in_dim_control_adapter: int = 24,
        seperated_timestep: bool = False,
        require_vae_embedding: bool = True,
        require_clip_embedding: bool = False,
        fuse_vae_embedding_in_latents: bool = False,
    ):
        super().__init__()
        if has_image_input or has_ref_conv or add_control_adapter:
            raise NotImplementedError("Prism's video tower takes no CLIP image, reference conv or control input.")
        self.dim = dim
        self.freq_dim = freq_dim
        self.patch_size = tuple(patch_size)

        self.patch_embedding = nn.Conv3d(in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self._build_embeddings(dim, text_dim, freq_dim)
        self.blocks = nn.ModuleList([DiTBlock(dim, num_heads, ffn_dim, eps, VIDEO) for _ in range(num_layers)])
        self.head = Head(dim, out_dim, patch_size, eps)
        self.freqs = precompute_freqs_cis_3d(dim // num_heads)

    def patchify(self, x):
        x = self.patch_embedding(x.contiguous(memory_format=torch.channels_last_3d))
        f, h, w = x.shape[2:]
        return x.flatten(2).transpose(1, 2).contiguous(), (f, h, w)

    def unpatchify(self, x, grid_size):
        f, h, w = grid_size
        pf, ph, pw = self.patch_size
        x = x.view(x.shape[0], f, h, w, pf, ph, pw, -1)
        return x.permute(0, 7, 1, 4, 2, 5, 3, 6).reshape(x.shape[0], -1, f * pf, h * ph, w * pw)


class WanAudioModel(_TowerMixin, ModelMixin, ConfigMixin):
    """MOVA audio tower over DAC latents."""

    _no_split_modules = ["DiTBlock"]

    @register_to_config
    def __init__(
        self,
        dim: int,
        in_dim: int,
        ffn_dim: int,
        out_dim: int,
        text_dim: int,
        freq_dim: int,
        eps: float,
        patch_size: Tuple[int, ...],
        num_heads: int,
        num_layers: int,
        has_image_input: bool = False,
        has_image_pos_emb: bool = False,
        has_ref_conv: bool = False,
        add_control_adapter: bool = False,
        in_dim_control_adapter: int = 24,
        seperated_timestep: bool = False,
        require_vae_embedding: bool = True,
        require_clip_embedding: bool = True,
        fuse_vae_embedding_in_latents: bool = False,
        vae_type: str = "dac",
    ):
        super().__init__()
        if has_image_input or has_ref_conv or add_control_adapter:
            raise NotImplementedError("Prism's audio tower takes no CLIP image, reference conv or control input.")
        if vae_type != "dac":
            raise NotImplementedError(f"Prism's audio tower expects DAC latents, got vae_type={vae_type!r}.")
        self.dim = dim
        self.freq_dim = freq_dim
        self.patch_size = tuple(patch_size)

        self.patch_embedding = nn.Conv1d(in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self._build_embeddings(dim, text_dim, freq_dim)
        self.blocks = nn.ModuleList([DiTBlock(dim, num_heads, ffn_dim, eps, AUDIO) for _ in range(num_layers)])
        self.head = Head(dim, out_dim, patch_size, eps)
        self.freqs = precompute_freqs_cis_1d(dim // num_heads)

    def patchify(self, x):
        x = self.patch_embedding(x)
        return x.transpose(1, 2).contiguous(), x.shape[2]

    def unpatchify(self, x, f: int):
        (p,) = self.patch_size
        return x.view(x.shape[0], f, p, -1).permute(0, 3, 1, 2).reshape(x.shape[0], -1, f * p)
