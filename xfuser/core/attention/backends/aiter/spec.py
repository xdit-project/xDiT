"""AITER's flash attention: the dense bf16 backend."""

from xfuser.core.attention.constraints import ANY_CALL
from xfuser.core.attention.requirements import ALWAYS, ARCH, SYMBOL
from xfuser.core.attention.spec import AttentionBackendType, Impl, Spec

AITER_ARCH = ARCH("gfx942", "gfx950", "gfx1200", "gfx1201", "gfx1250")

SPECS = [
    Spec(
        AttentionBackendType.AITER,
        impl=Impl("kernel:aiter_attention"),
        ring=ALWAYS,
        reads_strided_bshd=True,
        requires=SYMBOL("aiter:flash_attn_func") & SYMBOL("aiter:flash_attn_varlen_func") & AITER_ARCH,
        accepts=ANY_CALL,
    ),
]
