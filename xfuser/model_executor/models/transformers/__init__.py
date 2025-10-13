from xfuser.config.diffusers import has_valid_diffusers_version
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
    "xFuserSanaTransformer2DWrapper"
    "xFuserWanTransformer3DWrapper",

# Gating some imports based on diffusers version, as they import part of diffusers
if has_valid_diffusers_version("flux"):
    from .transformer_flux import xFuserFluxTransformer2DWrapper
    __all__.append("xFuserFluxTransformer2DWrapper")
