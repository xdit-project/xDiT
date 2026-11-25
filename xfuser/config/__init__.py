"""xDiT configuration.

The exports load on first access. Thus the argument parser loads without the
runtime configuration classes in ``xfuser.config.config``, which check the
installed packages and devices when they load.
"""

from xfuser.compat import lazy_exports

__getattr__, __dir__ = lazy_exports(
    globals(),
    {
        ".args": ("FlexibleArgumentParser", "xFuserArgs"),
        ".attention_a2a": ("AttentionA2AConfig",),
        ".gemm": ("GemmQuantizationSpec",),
        ".config": (
            "EngineConfig",
            "ParallelConfig",
            "TensorParallelConfig",
            "PipeFusionParallelConfig",
            "SequenceParallelConfig",
            "DataParallelConfig",
            "ModelConfig",
            "InputConfig",
            "RuntimeConfig",
        ),
    },
)
