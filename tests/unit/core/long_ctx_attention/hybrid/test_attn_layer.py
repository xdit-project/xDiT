"""xFuserLongContextAttention picks an available attention kernel up front."""

import enum

import pytest

from xfuser.core.long_ctx_attention.hybrid import attn_layer


class AttnType(enum.Enum):
    """The members of yunchang's AttnType that the resolution looks at.

    yunchang probes the GPU on import, so it is not imported here.
    """

    FA = "fa"
    FA3 = "fa3"
    TORCH_FLASH = "torch_flash"
    NPU = "npu"


@pytest.fixture
def host(monkeypatch):
    """A host without flash-attn or an NPU; tests switch on what they need."""
    monkeypatch.setattr(attn_layer, "AttnType", AttnType)
    env_info = {**attn_layer.env_info, "has_flash_attn": False, "has_flash_attn_3": False}
    monkeypatch.setattr(attn_layer, "env_info", env_info)
    monkeypatch.setattr(attn_layer.envs, "_is_npu", lambda: False)
    return env_info


def test_default_falls_back_to_torch_flash_without_flash_attn(host):
    assert attn_layer._resolve_attn_type(None) == AttnType.TORCH_FLASH


def test_default_uses_flash_attn_when_available(host):
    host["has_flash_attn"] = True
    assert attn_layer._resolve_attn_type(None) == AttnType.FA


def test_default_on_npu_is_the_npu_kernel(host, monkeypatch):
    monkeypatch.setattr(attn_layer.envs, "_is_npu", lambda: True)
    host["has_flash_attn"] = True
    assert attn_layer._resolve_attn_type(None) == AttnType.NPU


@pytest.mark.parametrize("requested", ["FA", "FA3"])
def test_requested_flash_attention_that_is_not_available_fails_at_construction(host, requested):
    with pytest.raises(ImportError, match="not available"):
        attn_layer._resolve_attn_type(getattr(AttnType, requested))


def test_requested_kernel_is_kept_when_available(host):
    host["has_flash_attn_3"] = True
    assert attn_layer._resolve_attn_type(AttnType.FA3) == AttnType.FA3
    assert attn_layer._resolve_attn_type(AttnType.TORCH_FLASH) == AttnType.TORCH_FLASH
