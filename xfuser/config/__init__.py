"""xDiT configuration.

The argument parser loads without the runtime. The runtime configuration
classes in ``xfuser.config.config`` load on first access, because that module
checks the installed packages and devices when it loads.
"""

from typing import Any, List

from xfuser.compat import import_submodule

from .args import FlexibleArgumentParser, xFuserArgs
from .attention_a2a import AttentionA2AConfig
from .gemm import GemmQuantizationSpec

_RUNTIME_EXPORTS = (
    "EngineConfig",
    "ParallelConfig",
    "TensorParallelConfig",
    "PipeFusionParallelConfig",
    "SequenceParallelConfig",
    "DataParallelConfig",
    "ModelConfig",
    "InputConfig",
    "RuntimeConfig",
)

__all__ = [
    "FlexibleArgumentParser",
    "xFuserArgs",
    "AttentionA2AConfig",
    "GemmQuantizationSpec",
    *_RUNTIME_EXPORTS,
]


def __getattr__(name: str) -> Any:
    if name not in _RUNTIME_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_submodule(".config", __name__), name)
    globals()[name] = value
    return value


def __dir__() -> List[str]:
    return sorted(set(globals()) | set(_RUNTIME_EXPORTS))
