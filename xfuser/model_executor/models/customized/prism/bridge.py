"""Prism's dual-tower bridge: cross-attention between the video and audio towers.

Vendored from Tencent-Hunyuan/Prism ``hymm/models/modules/interactionv2.py`` for
inference only (no pooled AdaLN, no block-sparse cross-attention). Parameter
names match the Prism checkpoint.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from torch.nn import RMSNorm

from .sp import PrismSeqInfo, gathered_kv_attention, usp_attention
from .wan_dit import AUDIO, VIDEO


class RotaryEmbedding(nn.Module):
    def __init__(self, base: float, dim: int):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids):
        inv_freq = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        with torch.autocast(device_type=x.device.type, enabled=False):
            freqs = (inv_freq @ position_ids[:, None, :].float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(x.dtype), emb.sin().to(x.dtype)


def _rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_emb(x, cos, sin):
    """Rotate ``[B, S, H, D]`` by ``[B, S, D]`` cos/sin.

    The reference compiles this, which computes in float32 and rounds once.
    """
    cos, sin = cos.unsqueeze(2).float(), sin.unsqueeze(2).float()
    xf = x.float()
    return (xf * cos + _rotate_half(xf) * sin).to(x.dtype)


def interaction_layers(strategy: str, min_layers: int):
    if strategy == "shallow_focus":
        return list(range(min(10, min_layers // 3)))
    if strategy == "distributed":
        return list(range(0, min_layers, 3))
    if strategy == "progressive":
        return list(range(min(8, min_layers))) + list(range(8, min_layers, 3))
    if strategy == "custom":
        return [i for i in (0, 2, 4, 6, 8, 12, 16, 20) if i < min_layers]
    if strategy == "full":
        return list(range(min_layers))
    raise ValueError(f"Unknown interaction strategy: {strategy}")


class ConditionalCrossAttention(nn.Module):
    """Queries from one tower attending to keys and values from the other."""

    def __init__(self, dim: int, kv_dim: int, num_heads: int, eps: float = 1e-6):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        # Set by MOVABridge: which stream the queries come from.
        self.q_stream = VIDEO

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(kv_dim, dim)
        self.v = nn.Linear(kv_dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)

    def forward(self, x, y, x_freqs, y_freqs, seq: PrismSeqInfo):
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(y))
        v = self.v(y)
        if x_freqs is not None:
            q = apply_rotary_emb(q.unflatten(-1, (-1, self.head_dim)), *x_freqs).flatten(2)
        if y_freqs is not None:
            k = apply_rotary_emb(k.unflatten(-1, (-1, self.head_dim)), *y_freqs).flatten(2)
        if self.q_stream == VIDEO:
            # Video queries over the short audio stream: gather it everywhere.
            out = gathered_kv_attention(q, k, v, self.num_heads, seq.audio)
        else:
            # Audio queries over the long video stream: Ulysses.
            out = usp_attention(q, k, v, self.num_heads, seq.video)
        return self.o(out)


class ConditionalCrossAttentionBlock(nn.Module):
    def __init__(self, dim: int, kv_dim: int, num_heads: int, eps: float = 1e-6):
        super().__init__()
        self.y_norm = nn.LayerNorm(kv_dim, eps=eps)
        self.inner = ConditionalCrossAttention(dim=dim, kv_dim=kv_dim, num_heads=num_heads, eps=eps)

    def forward(self, x, y, x_freqs, y_freqs, seq: PrismSeqInfo):
        return self.inner(x, self.y_norm(y), x_freqs, y_freqs, seq)


class DualTowerConditionalBridge(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(
        self,
        visual_layers: int = 30,
        audio_layers: int = 30,
        visual_hidden_dim: int = 3072,
        audio_hidden_dim: int = 1536,
        audio_fps: float = 44100.0 / 2048.0,
        head_dim: int = 128,
        interaction_strategy: str = "shallow_focus",
        apply_cross_rope: bool = False,
        apply_first_frame_bias_in_rope: bool = False,
        trainable_condition_scale: bool = False,
        pooled_adaln: bool = False,
    ):
        super().__init__()
        if pooled_adaln:
            raise NotImplementedError(
                "Pooled AdaLN needs the full video sequence on every rank; Prism does not use it."
            )
        if trainable_condition_scale:
            raise NotImplementedError("Prism's bridge uses a fixed condition scale.")
        self.audio_fps = audio_fps
        self.head_dim = head_dim
        self.apply_cross_rope = apply_cross_rope
        self.apply_first_frame_bias_in_rope = apply_first_frame_bias_in_rope

        layers = interaction_layers(interaction_strategy, min(visual_layers, audio_layers))
        self.rotary = RotaryEmbedding(base=10000.0, dim=head_dim)
        self.audio_to_video_conditioners = nn.ModuleDict(
            {
                str(i): ConditionalCrossAttentionBlock(
                    visual_hidden_dim, audio_hidden_dim, visual_hidden_dim // head_dim
                )
                for i in layers
            }
        )
        self.video_to_audio_conditioners = nn.ModuleDict(
            {
                str(i): ConditionalCrossAttentionBlock(
                    audio_hidden_dim, visual_hidden_dim, audio_hidden_dim // head_dim
                )
                for i in layers
            }
        )
        for block in self.video_to_audio_conditioners.values():
            block.inner.q_stream = AUDIO

    @torch.no_grad()
    def build_aligned_freqs(
        self,
        video_fps: float,
        grid_size: Tuple[int, int, int],
        audio_steps: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        """Cross-modal RoPE with video frames placed on the audio-step time axis.

        Returns ``((cos_v, sin_v), (cos_a, sin_a))`` shaped ``[1, L, head_dim]``.
        """
        f_v, h, w = grid_size
        # The video VAE's temporal stride (4) is hard-coded upstream as well.
        if self.apply_first_frame_bias_in_rope:
            t_starts = torch.zeros((f_v,), device=device, dtype=torch.float32)
            if f_v > 1:
                t_starts[1:] = (1.0 / float(video_fps)) + torch.arange(f_v - 1, device=device, dtype=torch.float32) * (
                    4.0 / float(video_fps)
                )
            video_pos_per_frame = t_starts * float(self.audio_fps)
        else:
            scale = float(self.audio_fps) / float(video_fps / 4.0)
            video_pos_per_frame = torch.arange(f_v, device=device, dtype=torch.float32) * scale
        video_pos = video_pos_per_frame.repeat_interleave(h * w).unsqueeze(0)
        audio_pos = torch.arange(int(audio_steps), device=device, dtype=torch.float32).unsqueeze(0)

        dummy = torch.empty(0, device=device, dtype=dtype)
        return self.rotary(dummy, video_pos), self.rotary(dummy, audio_pos)
