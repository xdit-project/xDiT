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

# These wrappers import diffusers pipeline symbols that only exist on newer
# diffusers versions; skip them when unavailable instead of crashing the package.
try:
    from .transformer_flux import xFuserFluxTransformer2DWrapper  # noqa: F401

    __all__.append("xFuserFluxTransformer2DWrapper")
except ImportError:
    pass


try:
    from .transformer_z_image import xFuserZImageTransformer2DWrapper  # noqa: F401

    __all__.append("xFuserZImageTransformer2DWrapper")
except ImportError:
    pass


try:
    from .transformer_krea2 import xFuserKrea2Transformer2DWrapper  # noqa: F401

    __all__.append("xFuserKrea2Transformer2DWrapper")
except ImportError:
    pass
