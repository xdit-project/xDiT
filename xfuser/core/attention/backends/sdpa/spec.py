"""PyTorch's own attention: the generic dispatcher, its three aten backends,
and cuDNN. Always present, no vendor library, no layout conversion."""

from xfuser.core.attention.constraints import NO_VARLEN
from xfuser.core.attention.requirements import PLATFORM
from xfuser.core.attention.spec import AttentionBackendType, Impl, Spec

SPECS = [
    Spec(AttentionBackendType.SDPA, impl=Impl("kernel:sdpa"), accepts=NO_VARLEN),

    Spec(AttentionBackendType.SDPA_FLASH, impl=Impl("kernel:sdpa_flash"),
         returns_lse=True, accepts=NO_VARLEN),

    Spec(AttentionBackendType.SDPA_MATH, impl=Impl("kernel:sdpa_math"),
         returns_lse=False, accepts=NO_VARLEN),

    Spec(AttentionBackendType.SDPA_EFFICIENT, impl=Impl("kernel:sdpa_efficient"),
         returns_lse=True, accepts=NO_VARLEN),

    Spec(AttentionBackendType.CUDNN, impl=Impl("kernel:cudnn"),
         returns_lse=True, accepts=NO_VARLEN, requires=PLATFORM("cuda")),
]
