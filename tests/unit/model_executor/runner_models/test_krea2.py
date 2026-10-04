"""Krea-2 accepts exactly the attention backends that exclude its padded keys."""

import pytest

from xfuser.config.args import xFuserArgs
from xfuser.model_executor.models.runner_models.krea2 import xFuserKrea2TurboModel


@pytest.fixture(autouse=True)
def _single_rank(monkeypatch):
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")


def test_cudnn_is_accepted():
    # cuDNN is what xDiT picks by default on NVIDIA without FlashAttention.
    config = xFuserArgs(model="krea/krea-2-turbo", attention_backend="CUDNN")
    xFuserKrea2TurboModel(config)


def test_a_backend_without_a_mask_is_still_refused():
    with pytest.raises(ValueError, match="does not support attention backend"):
        xFuserKrea2TurboModel(xFuserArgs(model="krea/krea-2-turbo", attention_backend="SDPA_FLASH"))
