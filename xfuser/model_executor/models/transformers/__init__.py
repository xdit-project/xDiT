"""Wrappers whose registration has to happen before any pipeline is built.

Importing a wrapper module registers it against the diffusers class it wraps, and the
pipelines below look their backbone up in that registry, so the import has to have
happened by then. Newer wrappers are absent from this list on purpose: their runners and
pipelines import them by module path at the point of use.
"""

from .register import xFuserTransformerWrappersRegister
from .base_transformer import xFuserTransformerBaseWrapper
from .pixart_transformer_2d import xFuserPixArtTransformer2DWrapper
from .transformer_sd3 import xFuserSD3Transformer2DWrapper
from .latte_transformer_3d import xFuserLatteTransformer3DWrapper
from .hunyuan_transformer_2d import xFuserHunyuanDiT2DWrapper
from .cogvideox_transformer_3d import xFuserCogVideoXTransformer3DWrapper
from .consisid_transformer_3d import xFuserConsisIDTransformer3DWrapper
from .sana_transformer_2d import xFuserSanaTransformer2DWrapper

__all__ = [
    "xFuserTransformerWrappersRegister",
    "xFuserTransformerBaseWrapper",
    "xFuserPixArtTransformer2DWrapper",
    "xFuserSD3Transformer2DWrapper",
    "xFuserLatteTransformer3DWrapper",
    "xFuserCogVideoXTransformer3DWrapper",
    "xFuserHunyuanDiT2DWrapper",
    "xFuserConsisIDTransformer3DWrapper",
    "xFuserSanaTransformer2DWrapper",
]
