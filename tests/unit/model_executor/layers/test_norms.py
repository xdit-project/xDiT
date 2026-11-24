"""AITER's synthetic RMSNorm weight is runtime state, not a checkpoint tensor."""

import pytest

torch = pytest.importorskip("torch")

from xfuser.model_executor.layers import norms


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_non_affine_norm_loads_empty_checkpoint_and_preserves_runtime_weight(monkeypatch, dtype):
    norm = norms._AITERRMSNorm(None, eps=1e-6, dim=4)
    # A non-affine Diffusers RMSNorm has no saved tensors.
    norm.load_state_dict({}, strict=True)
    received = []

    def kernel(x, weight, eps):
        received.append(weight.clone())
        return x

    monkeypatch.setattr(norms, "_aiter_rms_norm", kernel, raising=False)
    norm(torch.zeros(2, 4, dtype=dtype))
    torch.testing.assert_close(received[0], torch.ones(4, dtype=dtype), rtol=0, atol=0)
