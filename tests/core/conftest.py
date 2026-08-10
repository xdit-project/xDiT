"""Shared setup for the core tests."""

import pytest

from xfuser.envs import restore_torch_group_norm_for_distvae


@pytest.fixture(scope="session", autouse=True)
def torch_group_norm():
    """Give the session torch's own GroupNorm, not the one AITER leaves in its place

    On ROCm, importing xfuser swaps torch.nn.GroupNorm for AITER's, which several tests here
    cannot live with: a VAE built under the swap carries norms DistVAE will not shard, and
    decoding one on the CPU reaches a kernel that only exists on the GPU. Production reverts
    before it builds a VAE it means to shard, and nothing here is testing AITER's norm itself,
    so revert once for the session and let each test see what it expects.
    """
    restore_torch_group_norm_for_distvae()
