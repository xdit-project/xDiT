"""Shared setup for the core tests."""

import pytest

from xfuser.envs import restore_torch_group_norm_for_distvae


@pytest.fixture(scope="session", autouse=True)
def torch_group_norm():
    """Restore torch.nn.GroupNorm for this test session.

    On ROCm, importing xfuser may replace it with AITER GroupNorm, which DistVAE cannot shard
    and which cannot execute these CPU tests.
    """
    restore_torch_group_norm_for_distvae()
