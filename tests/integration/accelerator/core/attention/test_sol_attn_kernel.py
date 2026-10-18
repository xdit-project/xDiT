"""Sol-Attn kernel smoke test. The kernel is NVIDIA-only."""

from types import SimpleNamespace

import pytest
import torch

from xfuser.core.attention import registry
from xfuser.core.attention.spec import AttentionBackendType, AttnCall

pytestmark = pytest.mark.nvidia


def test_kernel_returns_the_query_shape(monkeypatch):
    spec = registry.get(AttentionBackendType.SOL_ATTN)
    reason = spec.unavailable()
    if reason is not None:
        pytest.skip(reason)

    from xfuser.core.attention.backends.sol_attn import kernel

    runtime_config = SimpleNamespace(
        sol_attn_tau=1.0,
        sol_attn_thresh_type="diag",
        sol_attn_kv_splits="1",
        sol_attn_sink_tokens=0,
        sol_attn_sink_start=None,
    )
    monkeypatch.setattr(kernel, "get_runtime_state", lambda: SimpleNamespace(runtime_config=runtime_config))
    torch.manual_seed(0)
    shape = (1, 4, 128, 128)
    query, key, value = (torch.randn(shape, dtype=torch.bfloat16, device="cuda") for _ in range(3))
    output, lse = spec.run(query, key, value, AttnCall())
    assert output.shape == query.shape
    assert output.dtype == torch.bfloat16
    assert lse is None
    assert torch.isfinite(output).all()
