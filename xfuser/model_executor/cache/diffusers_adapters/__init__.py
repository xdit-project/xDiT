"""
adapted from https://github.com/ali-vilab/TeaCache.git
adapted from https://github.com/chengzeyi/ParaAttention.git
"""
import importlib
from typing import Type, Dict, TypeVar
from xfuser.model_executor.cache.diffusers_adapters.registry import TRANSFORMER_ADAPTER_REGISTRY


def apply_cache_on_transformer(transformer, *args, **kwargs):
    adapter_name = TRANSFORMER_ADAPTER_REGISTRY.get(type(transformer))
    if not adapter_name:
        raise ValueError(f"Unknown transformer class: {transformer.__class__.__name__}")

    adapter_module = importlib.import_module(f".{adapter_name}", __package__)
    apply_cache_on_transformer_fn = getattr(adapter_module, "apply_cache_on_transformer")
    return apply_cache_on_transformer_fn(transformer, *args, **kwargs)
