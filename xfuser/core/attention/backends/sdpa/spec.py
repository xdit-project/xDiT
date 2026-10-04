"""PyTorch's own attention: the generic dispatcher, its three aten backends,
and cuDNN. Always present, no vendor library, no layout conversion."""

from xfuser.core.attention.constraints import MASKED_VARLEN, NO_VARLEN
from xfuser.core.attention.requirements import ALWAYS, NEVER, PLATFORM
from xfuser.core.attention.spec import AttentionBackendType, Impl, Spec

SPECS = [
    Spec(AttentionBackendType.SDPA, impl=Impl("kernel:sdpa"), ring=NEVER, accepts=NO_VARLEN, requires=ALWAYS),
    Spec(
        AttentionBackendType.SDPA_FLASH, impl=Impl("kernel:sdpa_flash"), ring=ALWAYS, accepts=NO_VARLEN, requires=ALWAYS
    ),
    Spec(AttentionBackendType.SDPA_MATH, impl=Impl("kernel:sdpa_math"), ring=NEVER, accepts=NO_VARLEN, requires=ALWAYS),
    Spec(
        AttentionBackendType.SDPA_EFFICIENT,
        impl=Impl("kernel:sdpa_efficient"),
        ring=ALWAYS,
        accepts=NO_VARLEN,
        requires=ALWAYS,
    ),
    # cuDNN applies attn_mask as its additive bias, so it serves a padded call
    # through the mask and ignores the packing.
    Spec(
        AttentionBackendType.CUDNN,
        impl=Impl("kernel:cudnn"),
        ring=ALWAYS,
        accepts=MASKED_VARLEN,
        requires=PLATFORM("cuda"),
    ),
]
