"""xFuserLongContextAttention picks an installed attention kernel up front."""

import pytest

AttnType = pytest.importorskip("yunchang.kernels").AttnType

from xfuser.core.long_ctx_attention.hybrid import attn_layer  # noqa: E402


@pytest.fixture
def without_flash_attn(monkeypatch):
    # The module binds AttnType only when an accelerator is present.
    monkeypatch.setattr(attn_layer, "AttnType", AttnType)
    monkeypatch.setattr(attn_layer, "HAS_FLASH_ATTN", False)
    monkeypatch.setattr(attn_layer, "HAS_FLASH_ATTN_HOPPER", False)


def test_default_falls_back_to_torch_flash_without_flash_attn(without_flash_attn):
    assert attn_layer._resolve_attn_type(None) == AttnType.TORCH_FLASH


def test_default_uses_flash_attn_when_installed(without_flash_attn, monkeypatch):
    monkeypatch.setattr(attn_layer, "HAS_FLASH_ATTN", True)
    assert attn_layer._resolve_attn_type(None) == AttnType.FA


@pytest.mark.parametrize("requested", ["FA", "FA3"])
def test_requested_flash_attention_that_is_not_installed_fails_at_construction(without_flash_attn, requested):
    with pytest.raises(ImportError, match="not installed"):
        attn_layer._resolve_attn_type(getattr(AttnType, requested))
