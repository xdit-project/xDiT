"""Backend packages: one per family, each split in two.

    <family>/spec.py     declaration -- imports no vendor library, so the
                         registry loads on any machine
    <family>/kernel.py   implementation -- imports its vendor library at module
                         level, and is imported only when the backend is
                         selected

Splitting them is what lets kernel modules use ordinary top-of-file imports:
Dynamo refuses to trace importlib, so resolving an implementation inside a
compiled region is a hard failure under fullgraph=True. Resolution happens in
runtime_state's compatibility check instead, which runs for the attention
backend, the cross-attention backend, and every backend in a hybrid schedule.

Packages are listed explicitly rather than auto-discovered, so grep finds every
backend and a forgotten entry is a loud failure rather than a silently absent
one.
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
    vsa_h3,
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
    vsa_h3,
    sage,
    nvte,
    npu,
]

__all__ = ["MODULES"]
