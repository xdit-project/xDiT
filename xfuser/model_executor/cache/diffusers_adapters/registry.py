"""
adapted from https://github.com/ali-vilab/TeaCache.git
adapted from https://github.com/chengzeyi/ParaAttention.git
"""
import diffusers
from packaging import version
from typing import Type, Dict

TRANSFORMER_ADAPTER_REGISTRY: Dict[Type, str] = {}

def register_transformer_adapter(transformer_class: Type, adapter_name: str):
    TRANSFORMER_ADAPTER_REGISTRY[transformer_class] = adapter_name

if version.parse(diffusers.__version__) >= version.parse("0.35.2"):
    from diffusers.models.transformers.transformer_flux import FluxTransformer2DModel
    from xfuser.model_executor.models.transformers.transformer_flux import xFuserFluxTransformer2DWrapper
    register_transformer_adapter(FluxTransformer2DModel, "flux")
    register_transformer_adapter(xFuserFluxTransformer2DWrapper, "flux")

