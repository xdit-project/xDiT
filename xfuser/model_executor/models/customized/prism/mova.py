# Adapted from Tencent-Hunyuan/Prism (https://github.com/Tencent-Hunyuan/Prism),
# hymm/models/modules/mova.py.
# Modified for xDiT: inference only; sequences sharded through xDiT
# sequence parallelism; built on meta from configs; training and FSDP helpers removed.
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
#
# Prism builds on MOVA (https://github.com/OpenMOSS/MOVA), Copyright (c) the MOVA
# authors, licensed under the Apache License, Version 2.0 (the "License"); you may
# not use this file except in compliance with the License. You may obtain a copy
# of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

"""``MOVABridge``: Prism's two video experts, audio tower and bridge as one module.

Vendored from Tencent-Hunyuan/Prism ``hymm/models/modules/mova.py`` for inference
only. The module tree, and with it every state-dict key, matches the Prism
checkpoint: the first ``min(video_layers, audio_layers)`` layers of the
high-noise expert are fused with the audio layers and bridge conditioners into
``fusion_blocks``; the rest of that expert is ``remaining_video_blocks``; the
low-noise expert keeps its own ``blocks`` and borrows the fused audio/bridge
parts while it is active.
"""

import json
import os

import torch
import torch.nn as nn
from diffusers import ConfigMixin, ModelMixin

from .bridge import DualTowerConditionalBridge
from .sp import PrismSeqInfo, SeqShard
from .wan_dit import WanAudioModel, WanModel, audio_freqs, video_freqs


class FusedMOVABlock(nn.Module):
    def __init__(self, video_block, audio_block, a2v_conditioner=None, v2a_conditioner=None):
        super().__init__()
        self.video_block = video_block
        self.audio_block = audio_block
        self.a2v_conditioner = a2v_conditioner
        self.v2a_conditioner = v2a_conditioner

    def forward(
        self,
        visual_x,
        audio_x,
        visual_context,
        audio_context,
        visual_t_mod,
        audio_t_mod,
        visual_freqs,
        audio_freqs,
        visual_rope,
        audio_rope,
        seq: PrismSeqInfo,
        video_block=None,
    ):
        # Both bridge directions read the towers' states from before this layer.
        visual_in = visual_x
        if self.a2v_conditioner is not None:
            visual_x = visual_x + self.a2v_conditioner(visual_x, audio_x, visual_rope, audio_rope, seq)
        if self.v2a_conditioner is not None:
            audio_x = audio_x + self.v2a_conditioner(audio_x, visual_in, audio_rope, visual_rope, seq)

        video_block = video_block if video_block is not None else self.video_block
        visual_x = video_block(visual_x, visual_context, visual_t_mod, visual_freqs, seq)
        audio_x = self.audio_block(audio_x, audio_context, audio_t_mod, audio_freqs, seq)
        return visual_x, audio_x


class MOVABridge(ModelMixin, ConfigMixin):
    def __init__(self, video_dit, video_dit_2, audio_dit, dual_tower_bridge, boundary_ratio: float = 0.9):
        super().__init__()
        self.boundary_ratio = boundary_ratio

        num_fused = min(len(video_dit.blocks), len(audio_dit.blocks))
        a2v = dual_tower_bridge.audio_to_video_conditioners
        v2a = dual_tower_bridge.video_to_audio_conditioners
        self.fusion_blocks = nn.ModuleList(
            FusedMOVABlock(
                video_dit.blocks[i],
                audio_dit.blocks[i],
                a2v[str(i)] if str(i) in a2v else None,
                v2a[str(i)] if str(i) in v2a else None,
            )
            for i in range(num_fused)
        )
        self.remaining_video_blocks = nn.ModuleList(video_dit.blocks[num_fused:])

        # The blocks now live above; keep the towers for their embeddings and heads.
        video_dit.blocks = nn.ModuleList()
        audio_dit.blocks = nn.ModuleList()
        dual_tower_bridge.audio_to_video_conditioners = nn.ModuleDict()
        dual_tower_bridge.video_to_audio_conditioners = nn.ModuleDict()
        self.video_dit = video_dit
        self.video_dit_2 = video_dit_2
        self.audio_dit = audio_dit
        self.dual_tower_bridge = dual_tower_bridge

        # Options for the video self-attention backend, e.g. block-sparse
        # settings; the token grid is added per call.
        self.video_attention_kwargs = {}

    @classmethod
    def from_mova_config(cls, model_dir: str) -> "MOVABridge":
        """Build the module tree from a MOVA checkpoint folder's configs, without weights.

        Parameters are allocated on the meta device; fill them with
        ``load_state_dict(..., assign=True)``. Buffers and RoPE tables are real.
        """
        from accelerate import init_empty_weights

        def config(subfolder):
            with open(os.path.join(model_dir, subfolder, "config.json")) as f:
                return json.load(f)

        with init_empty_weights(include_buffers=False):
            video_dit = WanModel.from_config(config("video_dit"))
            video_dit_2 = (
                WanModel.from_config(config("video_dit_2"))
                if os.path.isdir(os.path.join(model_dir, "video_dit_2"))
                else None
            )
            audio_dit = WanAudioModel.from_config(config("audio_dit"))
            bridge = DualTowerConditionalBridge.from_config(config("dual_tower_bridge"))

        boundary_ratio = 0.9
        index_path = os.path.join(model_dir, "model_index.json")
        if os.path.exists(index_path):
            with open(index_path) as f:
                boundary_ratio = json.load(f).get("boundary_ratio", boundary_ratio)
        return cls(video_dit, video_dit_2, audio_dit, bridge, boundary_ratio=boundary_ratio)

    @property
    def num_attention_heads(self) -> int:
        return self.video_dit.config.num_heads

    def forward(
        self,
        visual_latents,
        audio_latents,
        context,
        timestep,
        audio_context=None,
        audio_timestep=None,
        video_fps: float = 24.0,
        use_video_dit_2: bool = False,
    ):
        """One denoising step: ``(video velocity, audio velocity)`` for full latents."""
        visual_dit = self.video_dit_2 if use_video_dit_2 and self.video_dit_2 is not None else self.video_dit
        if audio_context is None:
            audio_context = context
        if audio_timestep is None:
            audio_timestep = timestep

        # The reference embeds timesteps in float32 under autocast, as MOVA does.
        with torch.autocast("cuda", dtype=torch.float32):
            visual_t, visual_t_mod = visual_dit.time_embed(timestep)
            audio_t, audio_t_mod = self.audio_dit.time_embed(audio_timestep)
        dtype = visual_dit.dtype
        visual_t, visual_t_mod = visual_t.to(dtype), visual_t_mod.to(dtype)
        audio_t, audio_t_mod = audio_t.to(dtype), audio_t_mod.to(dtype)

        visual_context = visual_dit.text_embedding(context)
        audio_context = self.audio_dit.text_embedding(audio_context)

        visual_x, grid_size = visual_dit.patchify(visual_latents.to(dtype))
        audio_x, audio_len = self.audio_dit.patchify(audio_latents.to(dtype))
        visual_rope_table = video_freqs(visual_dit.freqs, *grid_size, visual_x.device)
        audio_rope_table = audio_freqs(self.audio_dit.freqs, audio_len, audio_x.device)

        visual_rope = audio_rope = None
        if self.dual_tower_bridge.apply_cross_rope:
            visual_rope, audio_rope = self.dual_tower_bridge.build_aligned_freqs(
                video_fps=video_fps,
                grid_size=grid_size,
                audio_steps=audio_len,
                device=visual_x.device,
                dtype=visual_x.dtype,
            )

        seq = PrismSeqInfo(
            video=SeqShard.for_length(visual_x.shape[1]),
            audio=SeqShard.for_length(audio_len),
            video_attention_kwargs={**self.video_attention_kwargs, "bsa_thw": tuple(grid_size)},
        )
        visual_x = seq.video.split(visual_x, dim=1)
        audio_x = seq.audio.split(audio_x, dim=1)
        visual_rope_table = seq.video.split(visual_rope_table, dim=0)
        audio_rope_table = seq.audio.split(audio_rope_table, dim=0)
        if visual_rope is not None:
            visual_rope = tuple(seq.video.split(t, dim=1) for t in visual_rope)
            audio_rope = tuple(seq.audio.split(t, dim=1) for t in audio_rope)

        for i, fused in enumerate(self.fusion_blocks):
            visual_x, audio_x = fused(
                visual_x,
                audio_x,
                visual_context,
                audio_context,
                visual_t_mod,
                audio_t_mod,
                visual_rope_table,
                audio_rope_table,
                visual_rope,
                audio_rope,
                seq,
                video_block=visual_dit.blocks[i] if visual_dit is not self.video_dit else None,
            )
        remaining = (
            self.remaining_video_blocks
            if visual_dit is self.video_dit
            else visual_dit.blocks[len(self.fusion_blocks) :]
        )
        for block in remaining:
            visual_x = block(visual_x, visual_context, visual_t_mod, visual_rope_table, seq)

        # The heads are per-token, so they run on the shards before the gather.
        visual_out = seq.video.gather(visual_dit.head(visual_x, visual_t), dim=1)
        audio_out = seq.audio.gather(self.audio_dit.head(audio_x, audio_t), dim=1)
        return visual_dit.unpatchify(visual_out, grid_size), self.audio_dit.unpatchify(audio_out, audio_len)
