"""
FlashAttention, v2 through v4, plus the fp8/fp4 recipes.
"""

from xfuser.core.attention.constraints import NO_VARLEN
from xfuser.core.attention.requirements import CUDA_CAPABILITY, PLATFORM, SYMBOL
from xfuser.core.attention.spec import AttentionBackendType, Impl, Spec

_FA2 = "flash_attn:flash_attn_func"
_FA2_VARLEN = "flash_attn:flash_attn_varlen_func"
_FA3 = "flash_attn_interface:flash_attn_func"
_FA3_VARLEN = "flash_attn_interface:flash_attn_varlen_func"
_FA4 = "flash_attn.cute.interface:flash_attn_func"
_FA4_VARLEN = "flash_attn.cute.interface:flash_attn_varlen_func"
_FP4_QUANT = "xfuser.core.distributed.fp4_quantize:quantize_qk_to_fp4"

SPECS = [
    Spec(AttentionBackendType.FLASH, impl=Impl("kernel:flash_2"), returns_lse=True,
         requires=SYMBOL(_FA2) & SYMBOL(_FA2_VARLEN)),

    Spec(AttentionBackendType.FLASH_3, impl=Impl("kernel:flash_3"), returns_lse=True,
         requires=PLATFORM("cuda") & SYMBOL(_FA3) & SYMBOL(_FA3_VARLEN)),

    Spec(AttentionBackendType.FLASH_3_FP8, impl=Impl("kernel:flash_3_fp8"),
         returns_lse=True, low_precision=True, accepts=NO_VARLEN,
         requires=PLATFORM("cuda") & SYMBOL(_FA3)),

    Spec(AttentionBackendType.FLASH_4, impl=Impl("kernel:flash_4"),
         requires=PLATFORM("cuda") & SYMBOL(_FA4) & SYMBOL(_FA4_VARLEN)),

    Spec(AttentionBackendType.FLASH_4_FP4, impl=Impl("kernel:flash_4_fp4"),
         low_precision=True, accepts=NO_VARLEN,
         requires=PLATFORM("cuda") & CUDA_CAPABILITY((10, 0))
                & SYMBOL(_FA4) & SYMBOL(_FP4_QUANT)),
]
