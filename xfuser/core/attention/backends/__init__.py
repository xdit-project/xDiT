"""Backend modules: one per family, kernel code and specs together.

Each module exposes a module-level SPECS list. They are imported explicitly
below rather than auto-discovered, so grep finds every backend and a forgotten
import is a loud failure rather than a silently absent backend.

Migration status: modules listed here are served by the new registry; the rest
are still served by xfuser/core/distributed/attention_backend.py. The count is
visible via registry.missing_specs().
"""

from xfuser.core.attention.backends import (
    aiter,
    aiter_flydsl,
    aiter_fp8,
    aiter_mha_v4,
    aiter_sage,
    aiter_vsa,
    flash_attn,
    flex,
    npu,
    nvte,
    sage,
    sdpa,
)

MODULES = [
    sdpa,
    flash_attn,
    aiter,
    aiter_fp8,
    aiter_mha_v4,
    aiter_sage,
    aiter_vsa,
    aiter_flydsl,
    flex,
    sage,
    nvte,
    npu,
]

# AITER_MLA is deliberately not migrated: it is marked deprecated in favour of
# AITER_FP8 and the current AITER breaks it outright (mla_reduce_v1 signature
# change). It stays served by the legacy module pending removal.

__all__ = ["MODULES"]
