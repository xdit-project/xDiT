"""AITER per-tensor FP8 attention.

Three paths, picked by what the call carries:

  pre-quantised   Q/K/V already fp8 from fp8 comms, descales in the kwargs
  MHA v4          dense, head_dim 128, non-causal -- the kernel owns rotation
                  and quantisation
  rotate here     everything else: rotate Q/K, quantise, then dense or varlen
"""

from xfuser.core.attention.numerics import hadamard
from xfuser.core.attention.constraints import NO_DROPOUT, PACKED_KEYS
from xfuser.core.attention.requirements import NEVER, SYMBOL
from xfuser.core.attention.spec import AttentionBackendType, Impl, Spec

SPECS = [
    Spec(
        AttentionBackendType.AITER_FP8,
        impl=Impl("kernel:aiter_fp8"),
        low_precision=True,
        accepts=NO_DROPOUT & PACKED_KEYS,
        # Dual-path: MHA v4 for some shapes and v3 otherwise, so it would yield
        # no LSE for the rest.
        ring=NEVER,
        accepts_prequantized=True,
        prequant_rotate=hadamard.rotate_qk,
        initializers=(hadamard.prepare,),
        # Both entry points: accepts does not refuse packed keys, so a build
        # with only the dense one would report itself available and then raise
        # on the first model that packs.
        requires=SYMBOL("aiter:flash_attn_fp8_pertensor_func")
        & SYMBOL("aiter:flash_attn_varlen_fp8_pertensor_func")
        & SYMBOL("aiter:per_tensor_quant")
        & hadamard.CREATE_HADAMARD,
    ),
]
