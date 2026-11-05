"""--mha_v4_kv_splits: what reaches AITER's mha_v4, and that a split call
attends as the unsplit one does.

Few heads per rank (high Ulysses degrees) leave most of the GPU idle, which
AITER fills by splitting the keys into ranges merged by their LSE.
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from xfuser.core.attention import registry
from xfuser.core.attention.spec import AttentionBackendType, AttnCall

pytestmark = pytest.mark.rocm

# Flux at Ulysses 8: 3 heads leave MI355X mostly idle, so "auto" splits.
FEW_HEADS = (1, 3, 4352, 128)
SPLIT_CAPABLE = ("AITER_BF16", "AITER_BF16FP8", "AITER_I8FP8", "AITER_FP8")
MX = ("AITER_MXFP8", "AITER_F8F6", "AITER_MXFP6", "AITER_MXFP4")


@pytest.fixture
def mha_v4_kernel():
    return pytest.importorskip("xfuser.core.attention.backends.aiter_mha_v4.kernel", reason="MHA v4 needs aiter")


@pytest.fixture
def kv_splits(monkeypatch):
    """Set --mha_v4_kv_splits as the runtime config carries it."""

    def set_splits(splits):
        runtime = SimpleNamespace(runtime_config=SimpleNamespace(mha_v4_kv_splits=splits))
        monkeypatch.setattr("xfuser.core.distributed.runtime_state._RUNTIME", runtime)

    return set_splits


def _spec(name):
    spec = registry.get(AttentionBackendType[name])
    if not torch.cuda.is_available():
        pytest.skip("requires an accelerator")
    reason = spec.unavailable()
    if reason is not None:
        pytest.skip(f"{name}: {reason}")
    spec.resolved()  # as backend selection does, before any compile
    return spec


def _fake_mha_v4(calls):
    def mha_v4(query, key, value, *formats, **kwargs):
        calls.append(kwargs)
        return torch.zeros_like(query)

    return mha_v4


# ---------------------------------------------------------------------------
# what reaches the kernel
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("splits", ["auto", 1, 4])
def test_dense_calls_hand_the_configured_splits_to_mha_v4(mha_v4_kernel, kv_splits, monkeypatch, splits):
    calls = []
    monkeypatch.setattr(mha_v4_kernel, "mha_v4", _fake_mha_v4(calls))
    kv_splits(splits)

    query = torch.zeros(FEW_HEADS[:2] + (512, 128), device="cuda", dtype=torch.bfloat16)
    _spec("AITER_BF16").run(query, query, query, AttnCall())

    assert calls[0]["kv_splits"] == splits


def test_dense_calls_split_automatically_before_the_runtime_is_configured(mha_v4_kernel, monkeypatch):
    calls = []
    monkeypatch.setattr(mha_v4_kernel, "mha_v4", _fake_mha_v4(calls))
    monkeypatch.setattr("xfuser.core.distributed.runtime_state._RUNTIME", None)

    query = torch.zeros(FEW_HEADS[:2] + (512, 128), device="cuda", dtype=torch.bfloat16)
    _spec("AITER_BF16").run(query, query, query, AttnCall())

    assert calls[0]["kv_splits"] == "auto"


def test_sparge_calls_never_split(mha_v4_kernel, kv_splits, monkeypatch):
    """The block mask indexes whole key tiles, which a split would reorder."""
    calls = []

    def fake_build(query, key, value, **kwargs):
        mask = torch.ones((query.shape[0], query.shape[1], 2, 4), dtype=torch.bool)
        return query, key, value, SimpleNamespace(), mask

    monkeypatch.setattr(mha_v4_kernel, "build_block_mask", fake_build)
    monkeypatch.setattr(mha_v4_kernel, "restore_sparge_output", lambda output, state: output)
    monkeypatch.setattr(mha_v4_kernel, "mha_v4", _fake_mha_v4(calls))
    kv_splits(4)

    query = torch.zeros(FEW_HEADS[:2] + (512, 128), device="cuda", dtype=torch.bfloat16)
    _spec("AITER_FP8_SPARGE").run(query, query, query, AttnCall())

    assert "kv_splits" not in calls[0]


def test_aiter_fp8_hands_the_configured_splits_to_mha_v4(kv_splits, monkeypatch):
    fp8_kernel = pytest.importorskip("xfuser.core.attention.backends.aiter_fp8.kernel", reason="AITER_FP8 needs aiter")
    if not fp8_kernel._USE_MHA_V4:
        pytest.skip("this AITER_FP8 build does not route through MHA v4")
    calls = []
    monkeypatch.setattr(fp8_kernel, "mha_v4", _fake_mha_v4(calls))
    kv_splits(2)

    query = torch.zeros(FEW_HEADS[:2] + (512, 128), device="cuda", dtype=torch.bfloat16)
    fp8_kernel.aiter_fp8(query, query, query, AttnCall())

    assert calls[0]["kv_splits"] == 2


@pytest.mark.parametrize("splits", ["auto", 1])
def test_an_aiter_without_split_kv_runs_unsplit(mha_v4_kernel, kv_splits, monkeypatch, splits):
    monkeypatch.setattr(mha_v4_kernel, "HAS_KV_SPLITS", False)
    kv_splits(splits)

    assert mha_v4_kernel.kv_splits_kwargs() == {}


def test_an_aiter_without_split_kv_refuses_an_explicit_count(mha_v4_kernel, kv_splits, monkeypatch):
    monkeypatch.setattr(mha_v4_kernel, "HAS_KV_SPLITS", False)
    kv_splits(4)

    with pytest.raises(NotImplementedError, match="update AITER"):
        mha_v4_kernel.kv_splits_kwargs()


# ---------------------------------------------------------------------------
# what the kernel computes
# ---------------------------------------------------------------------------


def _qkv(shape=FEW_HEADS):
    generator = torch.Generator().manual_seed(0)
    return [torch.randn(*shape, generator=generator).to(device="cuda", dtype=torch.bfloat16) for _ in range(3)]


def _error(output, reference):
    return (output.float() - reference).abs().max().item()


def _auto_splits_at(shape):
    from aiter.ops.mha_v4 import mha_v4_kv_splits

    batch, heads, seq_len, _ = shape
    return mha_v4_kv_splits(batch, seq_len, heads, seq_len, torch.device("cuda"))


@pytest.mark.parametrize("splits", ["auto", 2, 4])
@pytest.mark.parametrize("backend", SPLIT_CAPABLE)
def test_split_calls_attend_as_the_unsplit_call(backend, splits, kv_splits):
    spec = _spec(backend)
    if splits == "auto" and _auto_splits_at(FEW_HEADS) == 1:
        pytest.skip("AITER does not split this shape on this device")
    query, key, value = _qkv()
    reference = F.scaled_dot_product_attention(query.float(), key.float(), value.float())

    kv_splits(1)
    unsplit, _ = spec.run(query, key, value, AttnCall())
    kv_splits(splits)
    split, _ = spec.run(query, key, value, AttnCall())

    unsplit_error = _error(unsplit, reference)
    assert _error(split, reference) <= 1.25 * unsplit_error + 1e-3, (
        f"{backend} at kv_splits={splits}: {_error(split, reference):.3e}; unsplit {unsplit_error:.3e}"
    )


@pytest.mark.parametrize("backend", MX)
def test_auto_runs_the_recipes_aiter_cannot_split_unsplit(backend, kv_splits):
    spec = _spec(backend)
    query, key, value = _qkv()

    kv_splits(1)
    unsplit, _ = spec.run(query, key, value, AttnCall())
    kv_splits("auto")
    auto, _ = spec.run(query, key, value, AttnCall())

    assert torch.equal(auto, unsplit)


def test_an_explicit_count_aiter_cannot_serve_is_refused(kv_splits):
    spec = _spec("AITER_MXFP8")
    query, key, value = _qkv()
    kv_splits(4)

    with pytest.raises(NotImplementedError):
        spec.run(query, key, value, AttnCall())


@pytest.mark.parametrize("backend", ("AITER_BF16", "AITER_FP8"))
def test_compiled_split_calls_match_eager(backend, kv_splits):
    spec = _spec(backend)
    query, key, value = _qkv()
    kv_splits("auto")

    eager, _ = spec.run(query, key, value, AttnCall())
    torch._dynamo.reset()
    compiled = torch.compile(lambda q, k, v: spec.run(q, k, v, AttnCall())[0], fullgraph=True)

    assert torch.equal(compiled(query, key, value), eager)
