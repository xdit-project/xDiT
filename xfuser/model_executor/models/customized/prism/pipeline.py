# Adapted from Tencent-Hunyuan/Prism (https://github.com/Tencent-Hunyuan/Prism),
# hymm/diffusion/pipelines/mova_pipeline.py and hymm/sample/sample_mova_single.py.
# Modified for xDiT: xDiT pipeline interface (components, device
# placement, caller-supplied generator); CPU offload and VAE tiling knobs removed.
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

"""Image + text to video + audio sampling loop for Prism.

Vendored from Tencent-Hunyuan/Prism ``hymm/diffusion/pipelines/mova_pipeline.py``.
The denoising arithmetic is unchanged. What differs is the plumbing xDiT
expects of a pipeline: components are registered by name (the video VAE is
``vae``, so xDiT's VAE tiling, slicing and parallel decode apply to it), ``to``
moves them, and noise comes from a caller-supplied generator.
"""

import html
import re
from typing import Optional

import ftfy
import PIL.Image
import torch
from diffusers.utils.torch_utils import randn_tensor
from diffusers.video_processor import VideoProcessor
from tqdm import tqdm

from xfuser.core.distributed import get_world_group


def _prompt_clean(text):
    text = html.unescape(html.unescape(ftfy.fix_text(text))).strip()
    return re.sub(r"\s+", " ", text).strip()


def center_crop_resize(image: PIL.Image.Image, height: int, width: int) -> PIL.Image.Image:
    """Crop the reference image to the target aspect ratio, then resize, as Prism's sampler does."""
    w, h = image.size
    target_ratio = width / height
    if w / h > target_ratio:
        new_w = int(h * target_ratio)
        left = (w - new_w) // 2
        image = image.crop((left, 0, left + new_w, h))
    elif w / h < target_ratio:
        new_h = int(w / target_ratio)
        top = (h - new_h) // 2
        image = image.crop((0, top, w, top + new_h))
    return image.convert("RGB").resize((width, height), PIL.Image.LANCZOS)


class PrismPipeline:
    _component_names = ("transformer", "vae", "audio_vae", "text_encoder", "tokenizer", "scheduler")

    def __init__(self, transformer, vae, audio_vae, text_encoder, tokenizer, scheduler):
        self.transformer = transformer
        self.vae = vae
        self.audio_vae = audio_vae
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.scheduler = scheduler

        self.vae_scale_factor_spatial = vae.config.scale_factor_spatial
        self.vae_scale_factor_temporal = vae.config.scale_factor_temporal
        self.video_processor = VideoProcessor(vae_scale_factor=self.vae_scale_factor_spatial)

    @property
    def components(self):
        return {name: getattr(self, name) for name in self._component_names}

    @property
    def _execution_device(self):
        return next(self.transformer.parameters()).device

    @property
    def audio_sample_rate(self) -> int:
        return self.audio_vae.sample_rate

    def to(self, device):
        for component in self.components.values():
            if isinstance(component, torch.nn.Module):
                component.to(device)
        return self

    def _normalize(self, latents, inverse=False):
        shape = (1, self.vae.config.z_dim, 1, 1, 1)
        mean = torch.tensor(self.vae.config.latents_mean, device=latents.device, dtype=latents.dtype).view(shape)
        std = torch.tensor(self.vae.config.latents_std, device=latents.device, dtype=latents.dtype).view(shape)
        return latents * std + mean if inverse else (latents - mean) * (1.0 / std)

    def encode_prompt(self, prompt: str, max_sequence_length: int = 512):
        device = self._execution_device
        inputs = self.tokenizer(
            [_prompt_clean(prompt)],
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        mask = inputs.attention_mask
        embeds = self.text_encoder(inputs.input_ids.to(device), mask.to(device)).last_hidden_state
        embeds = embeds.to(dtype=self.text_encoder.dtype)
        # Zero the padded positions; the transformer attends to all of them.
        seq_len = int(mask.gt(0).sum())
        embeds[:, seq_len:] = 0
        return embeds

    def prepare_latents(self, image, height, width, num_frames, generator):
        device = self._execution_device
        num_latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
        latent_height = height // self.vae_scale_factor_spatial
        latent_width = width // self.vae_scale_factor_spatial
        shape = (1, self.vae.config.z_dim, num_latent_frames, latent_height, latent_width)
        latents = randn_tensor(shape, generator=generator, device=device, dtype=torch.float32)

        image = center_crop_resize(image, height, width)
        image = self.video_processor.preprocess(image, height=height, width=width).to(device, dtype=torch.float32)
        image = image.unsqueeze(2)
        video_condition = torch.cat(
            [image, image.new_zeros(image.shape[0], image.shape[1], num_frames - 1, height, width)], dim=2
        ).to(dtype=self.vae.dtype)
        latent_condition = self.vae.encode(video_condition).latent_dist.mode().to(torch.float32)
        latent_condition = self._normalize(latent_condition)

        # Mask of known frames: the first pixel frame, folded to latent frames.
        t = self.vae_scale_factor_temporal
        mask = torch.zeros(1, 1, num_frames, latent_height, latent_width, device=device)
        mask[:, :, 0] = 1
        mask = torch.cat([mask[:, :, :1].repeat_interleave(t, dim=2), mask[:, :, 1:]], dim=2)
        mask = mask.view(1, -1, t, latent_height, latent_width).transpose(1, 2)
        return latents, torch.cat([mask, latent_condition], dim=1)

    def prepare_audio_latents(self, num_frames, video_fps, generator):
        num_samples = int(self.audio_sample_rate * num_frames / video_fps)
        latent_t = (num_samples - 1) // self.audio_vae.hop_length + 1
        shape = (1, self.audio_vae.latent_dim, latent_t)
        return randn_tensor(shape, generator=generator, device=self._execution_device, dtype=torch.float32)

    @torch.no_grad()
    def __call__(
        self,
        prompt: str,
        image,
        audio_prompt: Optional[str] = None,
        negative_prompt: str = "",
        height: int = 480,
        width: int = 848,
        num_frames: int = 205,
        video_fps: float = 24.0,
        num_inference_steps: int = 50,
        visual_shift: float = 9.0,
        audio_shift: float = 7.0,
        cfg_scale: float = 5.0,
        generator: Optional[torch.Generator] = None,
        output_type: str = "np",
    ):
        """Returns ``(video, audio)``.

        ``video`` is ``[F, H, W, C]`` in ``[0, 1]`` for ``output_type="np"``; with
        ``"latent"`` both are returned as final latents instead of being decoded.
        """
        divisor = self.vae_scale_factor_spatial * 2
        if height % divisor or width % divisor:
            raise ValueError(f"height and width must be divisible by {divisor}, got {height}x{width}.")
        if num_frames % self.vae_scale_factor_temporal != 1:
            raise ValueError(f"num_frames - 1 must be divisible by {self.vae_scale_factor_temporal}, got {num_frames}.")

        prompt_embeds = self.encode_prompt(prompt)
        negative_prompt_embeds = self.encode_prompt(negative_prompt)
        # The audio tower falls back to the video prompt. Separate copies keep every call
        # free of aliased inputs, which a compiled transformer would otherwise trace apart.
        if audio_prompt is not None:
            audio_prompt_embeds = self.encode_prompt(audio_prompt)
        else:
            audio_prompt_embeds = prompt_embeds.clone()
        negative_audio_embeds = negative_prompt_embeds.clone()

        latents, condition = self.prepare_latents(image, height, width, num_frames, generator)
        audio_latents = self.prepare_audio_latents(num_frames, video_fps, generator)

        pairs = self.scheduler.set_pair_timesteps(num_inference_steps, visual_shift, audio_shift)
        boundary = self.transformer.boundary_ratio * self.scheduler.config.num_train_timesteps
        use_video_dit_2 = False
        device = self._execution_device
        for i in tqdm(range(len(pairs)), disable=get_world_group().rank != 0):
            timestep, audio_timestep = pairs[i]
            # Once the video timestep crosses the boundary, the low-noise expert takes over.
            use_video_dit_2 = use_video_dit_2 or timestep.item() < boundary
            call = dict(
                visual_latents=torch.cat([latents, condition], dim=1),
                audio_latents=audio_latents,
                timestep=timestep.view(1).to(device=device, dtype=torch.float32),
                audio_timestep=audio_timestep.view(1).to(device=device, dtype=torch.float32),
                video_fps=video_fps,
                use_video_dit_2=use_video_dit_2,
            )
            video_pred, audio_pred = self.transformer(context=prompt_embeds, audio_context=audio_prompt_embeds, **call)
            video_pred, audio_pred = video_pred.float(), audio_pred.float()
            if cfg_scale != 1.0:
                video_neg, audio_neg = self.transformer(
                    context=negative_prompt_embeds, audio_context=negative_audio_embeds, **call
                )
                video_pred = video_neg.float() + cfg_scale * (video_pred - video_neg.float())
                audio_pred = audio_neg.float() + cfg_scale * (audio_pred - audio_neg.float())

            next_pair = pairs[i + 1] if i + 1 < len(pairs) else (None, None)
            latents = self.scheduler.step_from_to(video_pred, timestep, next_pair[0], latents)
            audio_latents = self.scheduler.step_from_to(audio_pred, audio_timestep, next_pair[1], audio_latents)

        if output_type == "latent":
            return latents, audio_latents

        with torch.autocast("cuda", dtype=torch.bfloat16):
            video = self.vae.decode(self._normalize(latents, inverse=True)).sample
        video = self.video_processor.postprocess_video(video, output_type=output_type)[0]
        with torch.autocast("cuda", dtype=torch.float32):
            audio = self.audio_vae.decode(audio_latents)
        return video, audio[0]
