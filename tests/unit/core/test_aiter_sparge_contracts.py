"""MHA v4 Sparge: the block mask must reach the kernel, at the right tile."""

from types import SimpleNamespace

import pytest
import torch

from xfuser.core.attention import registry
from xfuser.core.attention.spec import AttentionBackendType, AttnCall, Sparsity

_MHA_V4_SPARGE_BACKENDS = (
    "AITER_BF16_SPARGE",
    "AITER_BF16FP8_SPARGE",
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
    spec.resolved()  # as backend selection does, before any compile
    return spec.run(query, key, value, AttnCall(**kwargs))


# ---------------------------------------------------------------------------
# what the table declares
# ---------------------------------------------------------------------------


def test_every_mha_v4_sparge_row_is_registered():
    """A row dropped from the MHA v4 table must fail here, not disappear quietly."""
    for name in _MHA_V4_SPARGE_BACKENDS:
        spec = _spec(name)
        assert spec.sparsity is Sparsity.SPARGE
        assert spec.head_balanced
        assert spec.impl.target == "kernel:mha_v4_sparge"


def test_triton_sparge_backends_are_separate_from_mha_v4():
    """AITER_SPARGE/_V2 are the Sage-kernel sparge path, not MHA v4."""
    for name in ("AITER_SPARGE", "AITER_SPARGE_V2"):
        spec = _spec(name)
        assert spec.sparsity is Sparsity.SPARGE
        assert spec.impl.target.startswith("kernel:sparge")
        assert "aiter_sage" in spec.package


def _architecture_gate(spec):
    from xfuser.core.attention.requirements import ARCH, All

    requirement = spec.requires
    parts = requirement._flat() if isinstance(requirement, All) else (requirement,)
    gates = [part for part in parts if isinstance(part, ARCH)]
    assert len(gates) == 1, f"{spec.type.name} architecture gates: {gates}"
    return gates[0]


def test_mxfp8_sparge_refuses_gfx942_while_fp8_sparge_accepts_it():
    """Block-scaled Q/K has no gfx942 kernel, so that arch must fail the
    requirement the registered backend actually consults."""
    from unittest import mock

    from xfuser.core.attention import requirements

    class _Props:
        gcnArchName = "gfx942"

    requirements._arch_of.cache_clear()
    try:
        with (
            mock.patch.object(torch.cuda, "is_available", return_value=True),
            mock.patch.object(torch.cuda, "get_device_properties", lambda index: _Props()),
            mock.patch.object(torch.cuda, "current_device", return_value=0),
        ):
            assert _architecture_gate(_spec("AITER_MXFP8_SPARGE")).unmet() is not None
            assert _architecture_gate(_spec("AITER_FP8_SPARGE")).unmet() is None
    finally:
        requirements._arch_of.cache_clear()


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


@pytest.mark.accelerator
@pytest.mark.rocm
def test_sparge_passes_the_block_mask_and_tile_to_the_kernel(monkeypatch):
    _require("AITER_FP8_SPARGE")
    from xfuser.core.attention.backends.aiter_mha_v4 import kernel

    captured = {}

    def fake_build(
        query,
        key,
        value,
        *,
        is_causal,
        config,
        block_m,
        block_n,
        ulysses_world_size,
        cost_sink,
        pad_block_divisible=False,
    ):
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
    geometry = kernel.SPARSE_GEOMETRY["FP8"][None]
    assert captured["tile"] == geometry.block_tile
    assert captured["pad_block_divisible"] is not geometry.ragged_kv
    assert captured["layout"] == (1, 512, 2, 128)  # BSHD for the kernel
    assert tuple(captured["block_mask"].shape) == (1, 2, 2, 4)


def _capture_sparge_launch(monkeypatch, kernel):
    """Record the tile and padding the mask is built with, and what the launch is told."""
    captured = {}

    def fake_build(query, key, value, *, block_m, block_n, pad_block_divisible=False, **kwargs):
        captured["tile"] = (block_m, block_n)
        captured["pad_block_divisible"] = pad_block_divisible
        mask = torch.ones((query.shape[0], query.shape[1], 2, 4), dtype=torch.bool)
        return query, key, value, SimpleNamespace(), mask

    def fake_mha_v4(query, key, value, *formats, block_tile=None, **kwargs):
        captured["block_tile"] = block_tile
        return torch.zeros_like(query)

    monkeypatch.setattr(kernel, "build_block_mask", fake_build)
    monkeypatch.setattr(kernel, "restore_sparge_output", lambda output, state: output)
    monkeypatch.setattr(kernel, "mha_v4", fake_mha_v4)
    return captured


def _configure_block_size(monkeypatch, kernel, block_size):
    runtime_config = SimpleNamespace(sparseattn_block_size=block_size)
    monkeypatch.setattr(kernel, "runtime_state_is_initialized", lambda: True)
    monkeypatch.setattr(kernel, "get_runtime_state", lambda: SimpleNamespace(runtime_config=runtime_config))


@pytest.mark.accelerator
@pytest.mark.rocm
@pytest.mark.parametrize("ragged_kv", [False, True])
def test_sparge_tile_and_pad_follow_the_rows_geometry(monkeypatch, ragged_kv):
    """The mask is built at the row's own sparse geometry; a mismatch would
    mask the wrong keys rather than fail. Only a row that cannot bound a short
    last block has its keys padded."""
    _require("AITER_FP8_SPARGE")
    from xfuser.core.attention.backends.aiter_mha_v4 import kernel

    geometry = kernel.SparseGeometry((256, 64), ragged_kv)
    monkeypatch.setattr(kernel, "SPARSE_GEOMETRY", {**kernel.SPARSE_GEOMETRY, "FP8": {None: geometry}})
    captured = _capture_sparge_launch(monkeypatch, kernel)

    query = torch.zeros((1, 2, 512, 128), device="cuda", dtype=torch.bfloat16)
    _run("AITER_FP8_SPARGE", query, query, query)

    assert captured["tile"] == (256, 64)
    assert captured["pad_block_divisible"] is not ragged_kv


@pytest.mark.accelerator
@pytest.mark.rocm
@pytest.mark.parametrize(
    "backend_name, expected_kv_tile",
    [("AITER_BF16_SPARGE", 64), ("AITER_BF16FP8_SPARGE", 64), ("AITER_FP8_SPARGE", 128)],
)
def test_sparge_cuts_its_mask_at_its_own_recipes_kv_tile(monkeypatch, backend_name, expected_kv_tile):
    """gfx950 routes BF16 Q/K on 64-key blocks and the rest on 128, so one tile
    cannot serve all. A mask cut at the other recipe's tile has the wrong number
    of KV columns and AITER refuses it."""
    _require(backend_name)
    from xfuser.core.attention.backends.aiter_mha_v4 import kernel
    from xfuser.core.attention.requirements import device_arch

    if "gfx950" not in device_arch():
        pytest.skip("the per-recipe KV tiles differ on gfx950 only")
    captured = _capture_sparge_launch(monkeypatch, kernel)

    query = torch.zeros((1, 2, 512, 128), device="cuda", dtype=torch.bfloat16)
    _run(backend_name, query, query, query)

    assert captured["tile"] == (256, expected_kv_tile)


def _require_tile(fmt_name, block_size):
    from xfuser.core.attention.backends.aiter_mha_v4 import kernel

    if not kernel.HAS_BLOCK_TILE or block_size not in kernel.SPARSE_GEOMETRY[fmt_name]:
        pytest.skip(f"no {block_size[0]}x{block_size[1]} {fmt_name} sparse row on this GPU")


@pytest.mark.accelerator
@pytest.mark.rocm
def test_a_configured_block_size_reaches_the_mask_and_the_launch(monkeypatch):
    """--sparseattn_block_size replaces the recipe's default tile on both sides:
    a mask cut at one tile and launched at another selects the wrong keys."""
    _require("AITER_FP8_SPARGE")
    _require_tile("FP8", (64, 64))
    from xfuser.core.attention.backends.aiter_mha_v4 import kernel

    _configure_block_size(monkeypatch, kernel, (64, 64))
    captured = _capture_sparge_launch(monkeypatch, kernel)

    query = torch.zeros((1, 2, 512, 128), device="cuda", dtype=torch.bfloat16)
    _run("AITER_FP8_SPARGE", query, query, query)

    assert captured["tile"] == (64, 64)
    assert captured["block_tile"] == (64, 64)
    assert captured["pad_block_divisible"] is not kernel.SPARSE_GEOMETRY["FP8"][(64, 64)].ragged_kv


@pytest.mark.accelerator
@pytest.mark.rocm
def test_an_unserved_block_size_is_refused_with_the_tiles_that_are():
    _require("AITER_FP8_SPARGE")
    from xfuser.core.attention.backends.aiter_mha_v4 import kernel
    from xfuser.core.attention.backends.aiter_mha_v4.spec import FORMATS

    fmt = next(fmt for fmt in FORMATS if fmt.name == "FP8")
    default = kernel.SPARSE_GEOMETRY["FP8"][None].block_tile
    if kernel.HAS_BLOCK_TILE:
        kernel.check_sparse_block_size(fmt, default)

    with pytest.raises(ValueError, match="--sparseattn_block_size 32x32 has no AITER_FP8_SPARGE kernel") as error:
        kernel.check_sparse_block_size(fmt, (32, 32))
    if kernel.HAS_BLOCK_TILE:
        assert f"{default[0]}x{default[1]}" in str(error.value)


# Every block selected and nothing reordered, so the launch should be exact
# attention up to its own quantisation.
_DENSE_SPARGE = {
    "spargeattn_simthreshold": 2.0,
    "spargeattn_cdfthreshold": 1.0,
    "spargeattn_reorder_sequence": False,
    "use_spargeattn_static_block_mask": False,
}


def _ragged_qkv(seq_len=1030):
    """A length no block size divides, with a short last block on every tile."""
    torch.manual_seed(0)
    return [torch.randn((1, 2, seq_len, 128), device="cuda", dtype=torch.bfloat16) for _ in range(3)]


@pytest.mark.accelerator
@pytest.mark.rocm
@pytest.mark.parametrize("backend_name", _MHA_V4_SPARGE_BACKENDS)
def test_sparge_runs_at_a_ragged_length(backend_name):
    """Each row is cut at its own KV tile. gfx950's rows differ (BF16 on 64
    keys, the rest on 128), and AITER refuses a mask cut at another row's."""
    _require(backend_name)
    query, key, value = _ragged_qkv()
    expected = torch.nn.functional.scaled_dot_product_attention(query.float(), key.float(), value.float())

    output, _ = _run(backend_name, query, key, value, attention_kwargs=dict(_DENSE_SPARGE))

    assert output.shape == query.shape
    cosine = torch.nn.functional.cosine_similarity(output.float().flatten(), expected.flatten(), dim=0)
    assert cosine > 0.97


@pytest.mark.accelerator
@pytest.mark.rocm
@pytest.mark.parametrize("backend_name, fmt_name", [("AITER_BF16_SPARGE", "BF16"), ("AITER_FP8_SPARGE", "FP8")])
def test_sparge_at_64x64_matches_exact_attention(monkeypatch, backend_name, fmt_name):
    """The 64x64 rows end to end: mask, launch and output, at a ragged length."""
    _require(backend_name)
    _require_tile(fmt_name, (64, 64))
    from xfuser.core.attention.backends.aiter_mha_v4 import kernel

    _configure_block_size(monkeypatch, kernel, (64, 64))
    query, key, value = _ragged_qkv()
    expected = torch.nn.functional.scaled_dot_product_attention(query.float(), key.float(), value.float())

    output, _ = _run(backend_name, query, key, value, attention_kwargs=dict(_DENSE_SPARGE))

    cosine = torch.nn.functional.cosine_similarity(output.float().flatten(), expected.flatten(), dim=0)
    assert cosine > 0.97


@pytest.mark.accelerator
@pytest.mark.rocm
def test_ragged_sparge_attends_no_padding_keys():
    """A zero pad key scores 0 and takes softmax mass. BF16 has no quantisation
    to hide that behind, so its output matches exact attention closely only
    when the tail goes unpadded."""
    _require("AITER_BF16_SPARGE")
    from xfuser.core.attention.backends.aiter_mha_v4 import kernel

    if not kernel.SPARSE_GEOMETRY["BF16"][None].ragged_kv:
        pytest.skip("this GPU's BF16 sparse row needs its keys padded")
    query, key, value = _ragged_qkv()
    expected = torch.nn.functional.scaled_dot_product_attention(query.float(), key.float(), value.float())

    output, _ = _run("AITER_BF16_SPARGE", query, key, value, attention_kwargs=dict(_DENSE_SPARGE))

    # Padded to the tile, this is about 3e-2.
    assert (output.float() - expected).norm() / expected.norm() < 1e-2
