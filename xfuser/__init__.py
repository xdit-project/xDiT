from xfuser.compat import optional_exporter
from xfuser.model_executor.pipelines import (
    xFuserPixArtAlphaPipeline,
    xFuserPixArtSigmaPipeline,
    xFuserStableDiffusion3Pipeline,
    xFuserFluxPipeline,
    xFuserLattePipeline,
    xFuserHunyuanDiTPipeline,
    xFuserCogVideoXPipeline,
    xFuserConsisIDPipeline,
    xFuserStableDiffusionXLPipeline,
    xFuserSanaPipeline,
    xFuserSanaSprintPipeline,
)
from xfuser.config import xFuserArgs, EngineConfig
from xfuser.parallel import xDiTParallel

__all__ = [
    "xFuserPixArtAlphaPipeline",
    "xFuserPixArtSigmaPipeline",
    "xFuserStableDiffusion3Pipeline",
    "xFuserFluxPipeline",
    "xFuserLattePipeline",
    "xFuserHunyuanDiTPipeline",
    "xFuserCogVideoXPipeline",
    "xFuserConsisIDPipeline",
    "xFuserStableDiffusionXLPipeline",
    "xFuserSanaPipeline",
    "xFuserSanaSprintPipeline",
    "xFuserArgs",
    "EngineConfig",
    "xDiTParallel",
]

# Whether these exist was already decided by the pipelines package; take whichever
# of them it was able to build.
_optional = optional_exporter(globals())
_optional(
    ".model_executor.pipelines", "xFuserFlux2Pipeline", "xFuserFlux2KleinPipeline"
)
