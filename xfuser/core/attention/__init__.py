"""Attention backends.

Framework:  spec.py  requirements.py  constraints.py  registry.py  layout.py
Backends:   backends/ -- one module per family, kernel and specs together.

The framework knows a callable and six facts about it. Everything vendor- or
kernel-specific lives in the backend module that owns it.
"""

from xfuser.core.attention.spec import (
    AttentionBackendType,
    AttnCall,
    ParallelContext,
    Spec,
)
from xfuser.core.attention import registry
from xfuser.core.attention.backends import MODULES as _BACKEND_MODULES

for _module in _BACKEND_MODULES:
    registry.register(_module.SPECS, package=_module.__name__)

__all__ = [
    "AttentionBackendType",
    "AttnCall",
    "ParallelContext",
    "Spec",
    "registry",
]
