"""Prism (Tencent-Hunyuan): image + text to synchronized video + audio.

Inference-only port of https://github.com/Tencent-Hunyuan/Prism, a MOVA
derivative: two Wan2.2-style video experts, an audio DiT over DAC latents and a
bridge of cross-attention between the towers.
"""

from .dac_vae import DAC
from .mova import MOVABridge
from .pipeline import PrismPipeline
from .scheduler import FlowMatchPairScheduler

__all__ = ["DAC", "FlowMatchPairScheduler", "MOVABridge", "PrismPipeline"]
