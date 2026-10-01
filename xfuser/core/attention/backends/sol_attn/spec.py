"""Sol-Attn, the Sana on-the-fly sparse attention kernel.

The package is an optional install (`sol_attn`). A checkout of Sana next to
xDiT is placed first on the import path so it wins over the copy baked into
the xDiT image.
"""

import os
import sys
from pathlib import Path

from xfuser.core.attention.constraints import (
    BF16,
    HEAD_DIM,
    NO_DROPOUT,
    NO_VARLEN,
    NON_CAUSAL,
    SAME_QKV,
)
from xfuser.core.attention.requirements import CUDA_CAPABILITY, NEVER, PLATFORM, SYMBOL
from xfuser.core.attention.spec import AttentionBackendType, Impl, Spec

_SOL_ATTN = "sol_attn:sol_attn"


def _local_package_root() -> Path | None:
    """Directory that contains the ``sol_attn`` package, if one is nearby."""

    env = os.environ.get("SOL_ATTN_PATH")
    if env:
        candidate = Path(env)
        if (candidate / "sol_attn" / "__init__.py").is_file():
            return candidate
        if candidate.name == "sol_attn" and (candidate / "__init__.py").is_file():
            return candidate.parent
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "Sana" / "techniques" / "sparse_backends"
        if (candidate / "sol_attn" / "__init__.py").is_file():
            return candidate
    return None


def _expose_local_sol_attn() -> None:
    """Prefer a sibling Sana checkout over whatever ``sol_attn`` is installed.

    The xDiT image ships a copy built against an older CUTLASS DSL. That copy
    is importable, so leaving it in front of the checkout keeps the broken
    kernels selected. Inserting the checkout first makes the next import use
    the source next to xDiT.
    """

    root = _local_package_root()
    if root is None:
        return
    entry = str(root)
    if entry in sys.path:
        sys.path.remove(entry)
    sys.path.insert(0, entry)


_expose_local_sol_attn()

# Calls this kernel cannot serve, including cross-attention, other dtypes, and
# other head widths, fall through to SDPA. The kernel returns no log-sumexp,
# so ring attention cannot merge it.
SPECS = [
    Spec(
        AttentionBackendType.SOL_ATTN,
        impl=Impl("kernel:sol_attention"),
        ring=NEVER,
        accepts=SAME_QKV & HEAD_DIM(128) & BF16 & NON_CAUSAL & NO_DROPOUT & NO_VARLEN,
        fallback=AttentionBackendType.SDPA,
        requires=PLATFORM("cuda") & CUDA_CAPABILITY((8, 0)) & SYMBOL(_SOL_ATTN),
    ),
]
