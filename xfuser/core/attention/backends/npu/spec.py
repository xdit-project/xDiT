"""Ascend NPU fused attention."""

from xfuser.core.attention.constraints import NO_VARLEN
from xfuser.core.attention.requirements import PLATFORM, SYMBOL
from xfuser.core.attention.spec import AttentionBackendType, Impl, Spec

SPECS = [
    Spec(AttentionBackendType.NPU, impl=Impl("kernel:npu_attention"),
         returns_lse=True, accepts=NO_VARLEN,
         requires=PLATFORM("npu")
                & SYMBOL("torch_npu:npu_fused_infer_attention_score")),
]
