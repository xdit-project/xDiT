"""AITER per-tensor FP8 attention.

Three paths, picked by what the call carries:

  pre-quantised   Q/K/V already fp8 from fp8 comms, descales in the kwargs
  MHA v4          dense, head_dim 128, non-causal -- the kernel owns rotation
                  and quantisation
  legacy          everything else: rotate Q/K here, quantise, dense or varlen
"""

from xfuser.core.attention.numerics import hadamard
from xfuser.core.attention.constraints import NO_DROPOUT
from xfuser.core.attention.requirements import SYMBOL
from xfuser.core.attention.spec import AttentionBackendType, Impl, Spec

SPECS = [
    Spec(
        AttentionBackendType.AITER_FP8,
        impl=Impl("kernel:aiter_fp8"),
        low_precision=True,
        accepts=NO_DROPOUT,
        accepts_prequantized=True,
        prequant_rotate=hadamard.rotate_qk,
        requires=SYMBOL("aiter:flash_attn_fp8_pertensor_func")
               & SYMBOL("aiter:per_tensor_quant")
               & hadamard.CREATE_HADAMARD,
    ),
]
