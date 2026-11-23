"""xDiT package exports.

The exports load on first access. Thus a light submodule, for example
``xfuser.cli``, does not load the pipelines and the runtime that they need.
"""

import importlib
from typing import Any, List

_PIPELINES = ".model_executor.pipelines"

# Exported name -> submodule that supplies it.
_EXPORTS = {
    "xFuserPixArtAlphaPipeline": _PIPELINES,
    "xFuserPixArtSigmaPipeline": _PIPELINES,
    "xFuserStableDiffusion3Pipeline": _PIPELINES,
    "xFuserFluxPipeline": _PIPELINES,
    "xFuserLattePipeline": _PIPELINES,
    "xFuserHunyuanDiTPipeline": _PIPELINES,
    "xFuserCogVideoXPipeline": _PIPELINES,
    "xFuserConsisIDPipeline": _PIPELINES,
    "xFuserStableDiffusionXLPipeline": _PIPELINES,
    "xFuserSanaPipeline": _PIPELINES,
    "xFuserSanaSprintPipeline": _PIPELINES,
    "xFuserLumina2Pipeline": _PIPELINES,
    "xFuserArgs": ".config",
    "EngineConfig": ".config",
    "xDiTParallel": ".parallel",
}

# Exports that the installed diffusers can be too old for. The pipelines
# package decides if they exist; this package exports the ones it supplies.
_OPTIONAL_EXPORTS = {
    "xFuserFlux2Pipeline": _PIPELINES,
    "xFuserFlux2KleinPipeline": _PIPELINES,
}


def _available_exports() -> List[str]:
    pipelines = importlib.import_module(_PIPELINES, __name__)
    optional = [name for name in _OPTIONAL_EXPORTS if hasattr(pipelines, name)]
    return list(_EXPORTS) + optional


def __getattr__(name: str) -> Any:
    if name == "__all__":
        value = _available_exports()
    else:
        module = _EXPORTS.get(name) or _OPTIONAL_EXPORTS.get(name)
        if module is None:
            raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
        try:
            value = getattr(importlib.import_module(module, __name__), name)
        except AttributeError:
            # An optional export that this install does not supply.
            raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    globals()[name] = value
    return value


def __dir__() -> List[str]:
    return sorted(set(globals()) | set(_EXPORTS) | set(_OPTIONAL_EXPORTS))
