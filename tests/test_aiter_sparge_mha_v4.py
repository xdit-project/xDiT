"""MHA v4 Sparge: the block mask must reach the kernel, at the right tile."""

from types import SimpleNamespace

import pytest
import torch

from xfuser.core.attention import registry
from xfuser.core.attention.spec import AttentionBackendType, AttnCall

_MHA_V4_SPARGE_BACKENDS = (
    "AITER_I8FP8_SPARGE",
    "AITER_FP8_SPARGE",
    "AITER_MXFP8_SPARGE",
    "AITER_F8F6_SPARGE",
    "AITER_MXFP6_SPARGE",
    "AITER_F6F4_SPARGE",
    "AITER_MXFP4_SPARGE",
    "AITER_F4F4_SPARGE",
)


def _spec(name):
    return registry.get(AttentionBackendType[name])


def _require(name):
    """Skip unless this machine can run the backend, per its own spec."""
    if not torch.cuda.is_available():
        pytest.skip("requires a GPU")
    unavailable = _spec(name).unavailable()
    if unavailable is not None:
        pytest.skip(f"{name}: {unavailable}")


def _run(name, query, key, value, **kwargs):
    spec = _spec(name)
    spec.resolved()          # as backend selection does, before any compile
    return spec.run(query, key, value, AttnCall(**kwargs))


# ---------------------------------------------------------------------------
# what the table declares
# ---------------------------------------------------------------------------

def test_every_mha_v4_sparge_row_is_registered():
    for name in _MHA_V4_SPARGE_BACKENDS:
        spec = _spec(name)
        assert spec.sparsity == "sparge"
        assert spec.head_balanced
        assert spec.impl.target == "kernel:mha_v4_sparge"


def test_triton_sparge_backends_are_separate_from_mha_v4():
    """AITER_SPARGE/_V2 are the Sage-kernel sparge path, not MHA v4."""
    for name in ("AITER_SPARGE", "AITER_SPARGE_V2"):
        spec = _spec(name)
        assert spec.sparsity == "sparge"
        assert spec.impl.target.startswith("kernel:sparge")
        assert "aiter_sage" in spec.package


def test_mxfp8_sparge_is_gfx950_only():
    """Block-scaled Q/K has no gfx942 kernel, so the row declares the arch it
    needs rather than being refused once the launch fails."""
    from xfuser.core.attention.backends.aiter_mha_v4.spec import FORMATS

    mxfp8 = next(f for f in FORMATS if f.name == "MXFP8")
    assert mxfp8.sparge_on.names == ("gfx950",)

    fp8 = next(f for f in FORMATS if f.name == "FP8")
    assert fp8.sparge_on.names == ("gfx950", "gfx942")


@pytest.mark.parametrize("backend_name", _MHA_V4_SPARGE_BACKENDS)
def test_sparge_rejects_causal_and_dropout(backend_name):
    """Declared in `accepts`, so the refusal happens before the kernel runs."""
    spec = _spec(backend_name)
    tensor = torch.empty((1, 1, 1, 128))

    assert spec.rejects(tensor, tensor, tensor, AttnCall(is_causal=True)) is not None
    assert spec.rejects(tensor, tensor, tensor, AttnCall(dropout_p=0.1)) is not None


# ---------------------------------------------------------------------------
# what the kernel does with the mask
# ---------------------------------------------------------------------------

def test_sparge_passes_the_block_mask_and_tile_to_the_kernel(monkeypatch):
    from xfuser.core.attention.backends.aiter_mha_v4 import kernel

    _require("AITER_FP8_SPARGE")
    captured = {}

    def fake_build(query, key, value, *, is_causal, config, block_m, block_n,
                   ulysses_world_size, cost_sink, pad_block_divisible=False):
        captured["tile"] = (block_m, block_n)
        captured["pad_block_divisible"] = pad_block_divisible
        mask = torch.ones((query.shape[0], query.shape[1], 2, 4), dtype=torch.bool)
        return query, key, value, SimpleNamespace(), mask

    def fake_mha_v4(query, key, value, *formats, block_mask=None, **kwargs):
        captured["layout"] = tuple(query.shape)
        captured["block_mask"] = block_mask
        return torch.zeros_like(query)

    monkeypatch.setattr(kernel, "build_block_mask", fake_build)
    monkeypatch.setattr(kernel, "restore_sparge_output", lambda output, state: output)
    monkeypatch.setattr(kernel, "mha_v4", fake_mha_v4)

    query = torch.zeros((1, 2, 512, 128), device="cuda", dtype=torch.bfloat16)
    output, lse = _run("AITER_FP8_SPARGE", query, query, query)

    assert lse is None
    assert output.shape == query.shape
    assert captured["tile"] == (256, kernel.KV_TILE)
    assert captured["pad_block_divisible"] is True
    assert captured["layout"] == (1, 512, 2, 128)      # BSHD for the kernel
    assert tuple(captured["block_mask"].shape) == (1, 2, 2, 4)


def test_sparge_tile_follows_the_kernel_kv_tile(monkeypatch):
    """The mask is built at the kernel's own sparse geometry; a mismatch would
    mask the wrong keys rather than fail."""
    from xfuser.core.attention.backends.aiter_mha_v4 import kernel

    _require("AITER_FP8_SPARGE")
    captured = {}

    def fake_build(query, key, value, *, block_m, block_n, **kwargs):
        captured["tile"] = (block_m, block_n)
        mask = torch.ones((query.shape[0], query.shape[1], 2, 4), dtype=torch.bool)
        return query, key, value, SimpleNamespace(), mask

    monkeypatch.setattr(kernel, "KV_TILE", 64)
    monkeypatch.setattr(kernel, "build_block_mask", fake_build)
    monkeypatch.setattr(kernel, "restore_sparge_output", lambda output, state: output)
    monkeypatch.setattr(kernel, "mha_v4", lambda q, k, v, *a, **kw: torch.zeros_like(q))

    query = torch.zeros((1, 2, 512, 128), device="cuda", dtype=torch.bfloat16)
    _run("AITER_FP8_SPARGE", query, query, query)

    assert captured["tile"] == (256, 64)
