"""PyTorch's own attention: the generic dispatcher, its three aten backends,
and cuDNN. Always present, no vendor library, no layout conversion."""

from xfuser.core.attention.constraints import HALF_PRECISION, MASKED_VARLEN, NO_VARLEN
from xfuser.core.attention.requirements import ALWAYS, NEVER, PLATFORM
from xfuser.core.attention.spec import AttentionBackendType, Impl, Spec

SPECS = [
    # SDPA and SDPA_MATH pass attn_mask, so they serve a padded call through the
    # mask and ignore its packing.
    Spec(AttentionBackendType.SDPA, impl=Impl("kernel:sdpa"), ring=NEVER, accepts=MASKED_VARLEN, requires=ALWAYS),
    Spec(
        AttentionBackendType.SDPA_FLASH, impl=Impl("kernel:sdpa_flash"), ring=ALWAYS, accepts=NO_VARLEN, requires=ALWAYS
    ),
    Spec(
        AttentionBackendType.SDPA_MATH,
        impl=Impl("kernel:sdpa_math"),
        ring=NEVER,
        accepts=MASKED_VARLEN,
        requires=ALWAYS,
    ),
    Spec(
        AttentionBackendType.SDPA_EFFICIENT,
        impl=Impl("kernel:sdpa_efficient"),
        ring=ALWAYS,
        accepts=MASKED_VARLEN,
        requires=ALWAYS,
    ),
    # cuDNN, like SDPA_EFFICIENT above, applies attn_mask as its additive bias, so it
    # serves a padded call through the mask and ignores the packing.
    Spec(
        AttentionBackendType.CUDNN,
        impl=Impl("kernel:cudnn"),
        ring=ALWAYS,
        # The aten cuDNN kernel returns NaN rather than raising for float32
        # inputs, which PyTorch's own dispatcher never sends it. Hand those to
        # the memory-efficient kernel, which serves float32 and returns the
        # query-length LSE ring attention needs.
        accepts=MASKED_VARLEN & HALF_PRECISION,
        fallback=AttentionBackendType.SDPA_EFFICIENT,
        requires=PLATFORM("cuda"),
    ),
]
