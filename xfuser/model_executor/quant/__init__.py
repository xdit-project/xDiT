"""Streaming FP8 quantize-on-load adapters.

Importing this package pulls in the diffusers and transformers quantizer bases, so it is
imported lazily from the call sites that need it rather than at runner-package import time.
"""

from xfuser.model_executor.quant.aiter_fp8_quantizer import (
    AiterFp8BlockScaleConfig,
    AiterFp8BlockScaleQuantizer,
    AiterFp8BlockScaleTEConfig,
    AiterFp8BlockScaleTEQuantizer,
    AITER_FP8_BLOCKSCALE_QUANT_METHOD,
    AITER_FP8_BLOCKSCALE_TE_QUANT_METHOD,
    register_diffusers_fp8_quantizer,
    register_transformers_fp8_quantizer,
)

__all__ = [
    "AiterFp8BlockScaleConfig",
    "AiterFp8BlockScaleQuantizer",
    "AiterFp8BlockScaleTEConfig",
    "AiterFp8BlockScaleTEQuantizer",
    "AITER_FP8_BLOCKSCALE_QUANT_METHOD",
    "AITER_FP8_BLOCKSCALE_TE_QUANT_METHOD",
    "register_diffusers_fp8_quantizer",
    "register_transformers_fp8_quantizer",
]
