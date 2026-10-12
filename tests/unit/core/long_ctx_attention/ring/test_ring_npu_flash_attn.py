"""The ring package must import with a PyPI release of yunchang.

``update_npu_out`` exists only in unreleased yunchang commits, so importing it
on CUDA hosts made ``xfuser.core.long_ctx_attention.ring`` unusable for anyone
who installed yunchang from PyPI. Without an accelerator, yunchang is not
imported at all, so the package must not evaluate yunchang enums at import.
"""

import enum
import sys
from types import ModuleType

import pytest
import torch

import xfuser.core
import xfuser.envs as envs

_PACKAGE = "xfuser.core.long_ctx_attention"


def _module(name, **attrs):
    module = ModuleType(name)
    module.__path__ = []
    module.__dict__.update(attrs)
    return module


def _released_yunchang_modules():
    """The yunchang 0.6.x surface xDiT imports; it has no ``update_npu_out``."""

    class AttnType(enum.Enum):
        FA = "fa"
        FA3 = "fa3"
        NPU = "npu"
        SPARSE_SAGE = "sparse_sage"

    def unused(*args, **kwargs):
        raise AssertionError("importing the ring package must not call yunchang")

    modules = [
        _module("yunchang", LongContextAttention=type("LongContextAttention", (), {})),
        _module("yunchang.kernels", AttnType=AttnType, select_flash_attn_impl=unused),
        _module("yunchang.comm"),
        _module("yunchang.comm.all_to_all", SeqAllToAll4D=object),
        _module("yunchang.globals", HAS_SPARSE_SAGE_ATTENTION=False),
        _module("yunchang.ring"),
        _module("yunchang.ring.utils", RingComm=object, update_out_and_lse=unused),
        _module("yunchang.ring.ring_flash_attn", RingFlashAttnFunc=object),
        _module("yunchang.ring.ring_npu_flash_attn", RingNpuFlashAttnFunc=object),
    ]
    return {module.__name__: module for module in modules}


@pytest.fixture
def released_yunchang(monkeypatch):
    """Import the long-context package afresh against released yunchang."""
    for name, module in _released_yunchang_modules().items():
        monkeypatch.setitem(sys.modules, name, module)

    before = set(sys.modules)
    for name in [m for m in sys.modules if m == _PACKAGE or m.startswith(_PACKAGE + ".")]:
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.setattr(xfuser.core, "long_ctx_attention", None, raising=False)
    yield
    for name in set(sys.modules) - before:
        del sys.modules[name]


def test_ring_package_imports_on_cuda_with_released_yunchang(released_yunchang, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(envs, "_is_npu", lambda: False)

    from xfuser.core.long_ctx_attention.ring import (
        xdit_ring_flash_attn_func,
        xdit_ring_npu_flash_attn_func,
    )

    assert callable(xdit_ring_flash_attn_func)
    assert callable(xdit_ring_npu_flash_attn_func)


def test_ring_package_imports_without_an_accelerator(released_yunchang, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(envs, "_is_npu", lambda: False)

    from xfuser.core.long_ctx_attention.ring import (
        xdit_ring_flash_attn_func,
        xdit_ring_npu_flash_attn_func,
        xdit_sana_ring_flash_attn_func,
    )

    assert callable(xdit_ring_flash_attn_func)
    assert callable(xdit_ring_npu_flash_attn_func)
    assert callable(xdit_sana_ring_flash_attn_func)


def test_npu_with_released_yunchang_names_the_missing_requirement(released_yunchang, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(envs, "_is_npu", lambda: True)

    with pytest.raises(ImportError, match="update_npu_out"):
        import xfuser.core.long_ctx_attention.ring  # noqa: F401
