"""Ascend NPU fused attention."""

from xfuser.core.attention.constraints import NO_VARLEN
from xfuser.core.attention.requirements import ALWAYS, PLATFORM, SYMBOL
from xfuser.core.attention.spec import AttentionBackendType, Impl, Spec

SPECS = [
    Spec(
        AttentionBackendType.NPU,
        impl=Impl("kernel:npu_attention"),
        ring=ALWAYS,
        accepts=NO_VARLEN,
        requires=PLATFORM("npu") & SYMBOL("torch_npu:npu_fused_infer_attention_score"),
    ),
]
