from pathlib import Path
from types import SimpleNamespace
import inspect

import pytest
import torch
import torch.nn.functional as F


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


def _require_mha_v4_sparge_aiter(backend_name):
    if not torch.cuda.is_available() or torch.version.hip is None:
        pytest.skip("AITER MHA v4 Sparge requires a ROCm GPU.")

    arch_name = getattr(torch.cuda.get_device_properties(0), "gcnArchName", "")
    if "gfx950" not in arch_name:
        pytest.skip(f"AITER MHA v4 Sparge requires gfx950, got {arch_name}.")

    try:
        import aiter
        from aiter.ops.mha_v4 import mha_v4
    except ImportError:
        pytest.skip("AITER does not expose the MHA v4 API.")

    if "block_mask" not in inspect.signature(mha_v4).parameters:
        pytest.skip("AITER mha_v4 does not accept block_mask.")

    kernel_name = backend_name.removeprefix("AITER_").removesuffix("_SPARGE").lower()
    kernel_dir = (
        Path(aiter.__file__).resolve().parent.parent / "hsa" / "gfx950" / "fmha_v4_fwd"
    )
    if not (kernel_dir / f"fwd_hd128_{kernel_name}_sparse.co").exists():
        pytest.skip(f"AITER does not include the gfx950 {kernel_name} sparse FMHA kernel.")


def test_mha_v4_sparge_backends_are_registered():
    from xfuser.core.distributed.attention_backend import (
        AITER_MHA_V4_SPARGE_BACKENDS,
        ATTENTION_FUNCTION_REGISTRY,
        AttentionBackendType,
    )

    assert len(AITER_MHA_V4_SPARGE_BACKENDS) == 8
    for name in _MHA_V4_SPARGE_BACKENDS:
        backend = AttentionBackendType[name]
        assert backend in ATTENTION_FUNCTION_REGISTRY
        assert backend in AITER_MHA_V4_SPARGE_BACKENDS


def test_triton_sparge_backends_remain_registered():
    from xfuser.core.distributed.attention_backend import (
        ATTENTION_FUNCTION_REGISTRY,
        AttentionBackendType,
        _aiter_sparge_attn_call,
        _aiter_sparge_v2_attn_call,
    )

    assert (
        ATTENTION_FUNCTION_REGISTRY[AttentionBackendType.AITER_SPARGE]
        is _aiter_sparge_attn_call
    )
    assert (
        ATTENTION_FUNCTION_REGISTRY[AttentionBackendType.AITER_SPARGE_V2]
        is _aiter_sparge_v2_attn_call
    )


def test_fp8_sparge_passes_block_mask_to_mha_v4(monkeypatch):
    from xfuser.core.distributed import attention_backend as ab
    from xfuser.core.distributed.attention_backend import AttentionBackendType

    captured = {}

    def fake_build(query, key, value, is_causal, attention_kwargs, config, pad_block_divisible=False):
        captured["config"] = dict(config)
        captured["pad_block_divisible"] = pad_block_divisible
        mask = torch.ones((query.shape[0], query.shape[1], 2, 4), dtype=torch.bool)
        return query, key, value, SimpleNamespace(), mask, query.shape[1]

    def fake_mha_v4(query, key, value, q_format, k_format, v_format, block_mask=None):
        captured["layout"] = tuple(query.shape)
        captured["block_mask"] = block_mask
        captured["used_packed"] = False
        return torch.zeros_like(query)

    monkeypatch.setattr(ab, "_AITER_MHA_V4_HAS_BLOCK_MASK", True)
    monkeypatch.setattr(ab, "_AITER_MHA_V4_SPARSE_AVAILABLE", True)
    monkeypatch.setattr(ab, "_build_sparge_block_mask", fake_build)
    monkeypatch.setattr(ab, "restore_sparge_output", lambda output, state: output)
    monkeypatch.setattr(ab, "_aiter_mha_v4", fake_mha_v4)

    query = torch.zeros((1, 2, 512, 128), dtype=torch.bfloat16)
    output, lse = ab.ATTENTION_FUNCTION_REGISTRY[AttentionBackendType.AITER_FP8_SPARGE](
        query, query, query, dropout_p=0.0, is_causal=False
    )

    assert lse is None
    assert output.shape == query.shape
    assert captured["config"] == {"BLOCK_M": 256, "BLOCK_N": 128}
    assert captured["pad_block_divisible"] is True
    assert captured["layout"] == (1, 512, 2, 128)
    assert captured["block_mask"] is not None
    assert tuple(captured["block_mask"].shape) == (1, 2, 2, 4)
    assert captured["used_packed"] is False


def test_mxfp8_sparge_passes_block_mask_to_mha_v4_mxfp8(monkeypatch):
    from xfuser.core.distributed import attention_backend as ab
    from xfuser.core.distributed.attention_backend import AttentionBackendType

    captured = {}

    def fake_build(query, key, value, is_causal, attention_kwargs, config, pad_block_divisible=False):
        mask = torch.ones((query.shape[0], query.shape[1], 1, 1), dtype=torch.bool)
        return query, key, value, SimpleNamespace(), mask, query.shape[1]

    def fake_mxfp8(query, key, value, block_mask=None):
        captured["layout"] = tuple(query.shape)
        captured["block_mask"] = block_mask
        return torch.zeros_like(query)

    monkeypatch.setattr(ab, "_AITER_MHA_V4_HAS_BLOCK_MASK", True)
    monkeypatch.setattr(ab, "_AITER_MHA_V4_SPARSE_AVAILABLE", True)
    monkeypatch.setattr(ab, "_AITER_MHA_V4_MXFP8_HAS_BLOCK_MASK", True)
    monkeypatch.setattr(ab, "_build_sparge_block_mask", fake_build)
    monkeypatch.setattr(ab, "restore_sparge_output", lambda output, state: output)
    monkeypatch.setattr(ab, "_aiter_mha_v4_mxfp8", fake_mxfp8)
    monkeypatch.setattr(
        ab,
        "_aiter_launch_mxfp8_sparse",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("packed fallback")),
    )
    monkeypatch.setattr(
        ab,
        "_aiter_mha_v4",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("raw mha_v4")),
    )

    query = torch.zeros((1, 2, 128, 128), dtype=torch.bfloat16)
    output, _ = ab.ATTENTION_FUNCTION_REGISTRY[AttentionBackendType.AITER_MXFP8_SPARGE](
        query, query, query, dropout_p=0.0, is_causal=False
    )
    assert output.shape == query.shape
    assert captured["layout"] == (1, 128, 2, 128)
    assert captured["block_mask"] is not None
    assert tuple(captured["block_mask"].shape) == (1, 2, 1, 1)


@pytest.mark.parametrize("backend_name", _MHA_V4_SPARGE_BACKENDS)
def test_mha_v4_sparge_rejects_causal_and_dropout(backend_name):
    from xfuser.core.distributed.attention_backend import (
        ATTENTION_FUNCTION_REGISTRY,
        AttentionBackendType,
    )

    backend = AttentionBackendType[backend_name]
    tensor = torch.empty((1, 1, 128, 128), dtype=torch.bfloat16)
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    with pytest.raises(NotImplementedError, match="does not support causal masking"):
        ATTENTION_FUNCTION_REGISTRY[backend](
            tensor, tensor, tensor, dropout_p=0.0, is_causal=True
        )
    with pytest.raises(NotImplementedError, match="does not support dropout"):
        ATTENTION_FUNCTION_REGISTRY[backend](
            tensor, tensor, tensor, dropout_p=0.1, is_causal=False
        )


@pytest.mark.parametrize("backend_name", _MHA_V4_SPARGE_BACKENDS)
def test_mha_v4_sparge_matches_dense_sibling(backend_name, monkeypatch):
    _require_mha_v4_sparge_aiter(backend_name)

    from xfuser.core.distributed import attention_backend as ab
    from xfuser.core.distributed.attention_backend import (
        ATTENTION_FUNCTION_REGISTRY,
        AttentionBackendType,
    )

    monkeypatch.setattr(ab, "get_ulysses_parallel_world_size", lambda: 1)

    dense_name = backend_name.removesuffix("_SPARGE")
    torch.manual_seed(1234)
    shape = (1, 2, 512, 128)
    query = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    key = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    value = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    attention_kwargs = {
        "spargeattn_simthreshold": 1.0,
        "spargeattn_cdfthreshold": 1.0,
        "spargeattn_reorder_sequence": False,
        "use_spargeattn_static_block_mask": False,
    }

    with torch.no_grad():
        sparse, sparse_lse = ATTENTION_FUNCTION_REGISTRY[AttentionBackendType[backend_name]](
            query, key, value, dropout_p=0.0, is_causal=False, attention_kwargs=attention_kwargs
        )
        dense, dense_lse = ATTENTION_FUNCTION_REGISTRY[AttentionBackendType[dense_name]](
            query, key, value, dropout_p=0.0, is_causal=False
        )

    assert sparse.shape == query.shape
    assert torch.isfinite(sparse).all()
    assert sparse_lse is None and dense_lse is None
    cosine = F.cosine_similarity(
        sparse.float().flatten(), dense.float().flatten(), dim=0
    ).item()
    assert cosine > 0.95
