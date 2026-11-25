"""xDiT package exports.

The exports load on first access. Thus a light submodule, for example
``xfuser.cli``, does not load the pipelines and the runtime that they need.
"""

from xfuser.compat import lazy_exports

__getattr__, __dir__ = lazy_exports(
    globals(),
    {
        ".model_executor.pipelines": (
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
            "xFuserLumina2Pipeline",
        ),
        ".config": ("xFuserArgs", "EngineConfig"),
        ".parallel": ("xDiTParallel",),
    },
    # The installed diffusers can be too old for these. The pipelines package
    # decides if they exist.
    optional={".model_executor.pipelines": ("xFuserFlux2Pipeline", "xFuserFlux2KleinPipeline")},
)
