"""AITER's flash attention: the dense bf16 backend."""

from xfuser.core.attention.requirements import ARCH, SYMBOL
from xfuser.core.attention.spec import AttentionBackendType, Impl, Spec

# Mirrors envs._on_mi3xx() and envs._on_rdna4(), which is what check_aiter
# gates on. ARCH matches by substring, so the RDNA4 pair must be spelled out:
# a "gfx12" prefix would also match gfx1250, a distinct architecture the
# runner docs treat separately.
AITER_ARCH = ARCH("gfx942", "gfx950", "gfx1200", "gfx1201")

SPECS = [
    Spec(
        AttentionBackendType.AITER,
        impl=Impl("kernel:aiter_attention"),
        returns_lse=True,
        requires=SYMBOL("aiter:flash_attn_func")
               & SYMBOL("aiter:flash_attn_varlen_func")
               & AITER_ARCH,
    ),
]
