"""Ascend NPU fused attention."""

from xfuser.core.attention.requirements import PLATFORM, SYMBOL
from xfuser.core.attention.constraints import NO_VARLEN
from xfuser.core.attention.spec import AttentionBackendType, AttnCall, Spec


def npu_attention(query, key, value, call: AttnCall):
    import torch_npu

    q, k, v = (t.transpose(1, 2) for t in (query, key, value))
    out, lse = torch_npu.npu_fused_infer_attention_score(
        q, k, v,
        num_heads=q.shape[2],
        input_layout="BSND",
        scale=q.shape[-1] ** -0.5,
        softmax_lse_flag=True,
        pre_tokens=65535,
        next_tokens=65535,
    )
    return out.transpose(1, 2), lse.squeeze(-1)


SPECS = [
    Spec(AttentionBackendType.NPU, impl=npu_attention, returns_lse=True, accepts=NO_VARLEN,
         requires=PLATFORM("npu")
              & SYMBOL("torch_npu:npu_fused_infer_attention_score")),
]
