"""Runtime-only norm buffers must survive replicated meta materialization."""

import pytest

torch = pytest.importorskip("torch")

from accelerate import init_empty_weights
from xfuser.core.distributed.sharding import _restore_nonpersistent_buffers, _save_nonpersistent_buffers
from xfuser.core.utils.dtype_policy import cast_preserving_fp32_modules
from xfuser.model_executor.layers import norms

pytestmark = [pytest.mark.accelerator, pytest.mark.rocm]


@pytest.mark.skipif(not norms._HAS_AITER, reason="AITER is required")
@pytest.mark.parametrize("compiled", [False, True])
def test_non_affine_meta_norm_matches_eager_bf16_norm(compiled):
    eager = norms._AITERRMSNorm(None, eps=1e-6, dim=128).to(device="cuda", dtype=torch.bfloat16)
    with init_empty_weights():
        replicated = norms._AITERRMSNorm(None, eps=1e-6, dim=128)
    cast_preserving_fp32_modules(replicated, torch.bfloat16)
    saved = _save_nonpersistent_buffers(replicated, "cuda")
    replicated.to_empty(device="cuda")
    _restore_nonpersistent_buffers(saved)
    generator = torch.Generator(device="cuda").manual_seed(42)
    x = torch.randn(8, 128, generator=generator, device="cuda", dtype=torch.bfloat16)
    run = torch.compile(replicated, fullgraph=True) if compiled else replicated
    expected = eager(x)
    torch.testing.assert_close(run(x), expected, rtol=0, atol=0)
    torch.testing.assert_close(run(x), expected, rtol=0, atol=0)
    assert "_ones_weight" not in replicated.state_dict()
