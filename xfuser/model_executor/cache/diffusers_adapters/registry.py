"""
adapted from https://github.com/ali-vilab/TeaCache.git
adapted from https://github.com/chengzeyi/ParaAttention.git
"""
from typing import Type, Dict
from diffusers.models.transformers.transformer_flux import FluxTransformer2DModel
from xfuser.model_executor.models.transformers.transformer_flux import xFuserFluxTransformer2DWrapper

TRANSFORMER_ADAPTER_REGISTRY: Dict[Type, str] = {}

def register_transformer_adapter(transformer_class: Type, adapter_name: str):
    TRANSFORMER_ADAPTER_REGISTRY[transformer_class] = adapter_name

register_transformer_adapter(FluxTransformer2DModel, "flux")
register_transformer_adapter(xFuserFluxTransformer2DWrapper, "flux")

