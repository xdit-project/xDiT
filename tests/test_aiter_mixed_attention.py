from pathlib import Path

import pytest
import torch
import torch.nn.functional as F


def _require_mha_v4_aiter(backend_name, supported_arches=("gfx950",)):
    if not torch.cuda.is_available() or torch.version.hip is None:
        pytest.skip("AITER mixed-precision attention requires a ROCm GPU.")

    arch_name = getattr(torch.cuda.get_device_properties(0), "gcnArchName", "")
    arch = next((name for name in supported_arches if name in arch_name), None)
    if arch is None:
        pytest.skip(
            f"AITER {backend_name} attention requires {supported_arches}, got {arch_name}."
        )

    try:
        import aiter
        from aiter.ops.mha_v4 import mha_v4
    except ImportError:
        pytest.skip("AITER does not expose the MHA v4 API.")

    if backend_name == "AITER_MXFP8":
        try:
            from aiter.ops.mha_v4 import mha_v4_mxfp8
        except ImportError:
            pytest.skip("AITER does not expose the MHA v4 MXFP8 raw API.")
        del mha_v4_mxfp8

    del mha_v4
    kernel_dir = (
        Path(aiter.__file__).resolve().parent.parent / "hsa" / arch / "fmha_v4_fwd"
    )
    kernel_name = backend_name.removeprefix("AITER_").lower()
    candidates = [kernel_dir / f"fwd_hd128_{kernel_name}.co"]
    if arch == "gfx942":
        candidates.append(kernel_dir / "MI300" / f"fwd_hd128_{kernel_name}.co")
    if not any(path.exists() for path in candidates):
        pytest.skip(f"AITER does not include the {arch} {kernel_name} FMHA kernel.")


@pytest.mark.parametrize(
    "backend_name",
    ["AITER_MXFP8", "AITER_F8F6", "AITER_F6F4", "AITER_MXFP4", "AITER_F4F4"],
)
@pytest.mark.parametrize("sequence_length", [128, 257])
def test_aiter_mixed_attention_matches_sdpa(backend_name, sequence_length):
    _require_mha_v4_aiter(backend_name)

    from xfuser.core.distributed.attention_backend import (
        ATTENTION_FUNCTION_REGISTRY,
        AttentionBackendType,
    )

    torch.manual_seed(1234)
    shape = (1, 5, sequence_length, 128)
    query = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    key = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    value = torch.randn(shape, device="cuda", dtype=torch.bfloat16)

    with torch.no_grad():
        reference = F.scaled_dot_product_attention(query, key, value)
        output, lse = ATTENTION_FUNCTION_REGISTRY[AttentionBackendType[backend_name]](
            query, key, value, dropout_p=0.0, is_causal=False
        )

    output_float = output.float()
    reference_float = reference.float()
    cosine_similarity = F.cosine_similarity(
        output_float.flatten(), reference_float.flatten(), dim=0
    ).item()

    assert output.shape == reference.shape
    assert torch.isfinite(output).all()
    assert lse is None
    assert cosine_similarity > 0.95


@pytest.mark.parametrize(
    "backend_name",
    ["AITER_MXFP8", "AITER_F8F6", "AITER_F6F4", "AITER_MXFP4", "AITER_F4F4"],
)
def test_aiter_mixed_attention_compiles_fullgraph(backend_name):
    _require_mha_v4_aiter(backend_name)

    from xfuser.core.distributed.attention_backend import (
        ATTENTION_FUNCTION_REGISTRY,
        AttentionBackendType,
    )

    attention_function = ATTENTION_FUNCTION_REGISTRY[AttentionBackendType[backend_name]]
    shape = (1, 5, 128, 128)
    query = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    key = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    value = torch.randn(shape, device="cuda", dtype=torch.bfloat16)

    def attention(query, key, value):
        return attention_function(query, key, value, dropout_p=0.0, is_causal=False)[0]

    output = torch.compile(attention, fullgraph=True)(query, key, value)
    assert output.shape == query.shape
    assert torch.isfinite(output).all()


def test_aiter_mxfp8_gqa_compiles_and_matches_sdpa():
    _require_mha_v4_aiter("AITER_MXFP8")

    from xfuser.core.distributed.attention_backend import (
        ATTENTION_FUNCTION_REGISTRY,
        AttentionBackendType,
    )

    torch.manual_seed(1234)
    query = torch.randn((1, 64, 128, 128), device="cuda", dtype=torch.bfloat16)
    key = torch.randn((1, 4, 128, 128), device="cuda", dtype=torch.bfloat16)
    value = torch.randn_like(key)
    attention_function = ATTENTION_FUNCTION_REGISTRY[AttentionBackendType.AITER_MXFP8]

    def attention(query, key, value):
        return attention_function(query, key, value, dropout_p=0.0, is_causal=False)[0]

    reference = F.scaled_dot_product_attention(query, key, value, enable_gqa=True)
    output = torch.compile(attention, fullgraph=True)(query, key, value)

    assert output.shape == query.shape
    assert torch.isfinite(output).all()
    assert (
        F.cosine_similarity(
            output.float().flatten(), reference.float().flatten(), dim=0
        ).item()
        > 0.95
    )


@pytest.mark.parametrize(
    "backend_name",
    ["AITER_MXFP8", "AITER_F8F6", "AITER_F6F4", "AITER_MXFP4", "AITER_F4F4"],
)
def test_aiter_mixed_attention_unequal_sequence_lengths(backend_name):
    _require_mha_v4_aiter(backend_name)

    from xfuser.core.distributed.attention_backend import (
        ATTENTION_FUNCTION_REGISTRY,
        AttentionBackendType,
    )

    query = torch.randn((2, 5, 128, 128), device="cuda", dtype=torch.bfloat16)
    key = torch.randn((2, 5, 257, 128), device="cuda", dtype=torch.bfloat16)
    value = torch.randn_like(key)
    output, _ = ATTENTION_FUNCTION_REGISTRY[AttentionBackendType[backend_name]](
        query, key, value, dropout_p=0.0, is_causal=False
    )

    assert output.shape == query.shape
    assert torch.isfinite(output).all()


@pytest.mark.parametrize(
    "backend_name", ["AITER_MXFP8", "AITER_F8F6", "AITER_MXFP6", "AITER_MXFP4"]
)
def test_aiter_mixed_cross_attention_compiles_fullgraph(backend_name):
    _require_mha_v4_aiter(backend_name)

    from xfuser.core.distributed.attention_backend import (
        ATTENTION_FUNCTION_REGISTRY,
        AttentionBackendType,
    )

    query = torch.randn((1, 5, 129, 128), device="cuda", dtype=torch.bfloat16)
    key = torch.randn((1, 5, 128, 128), device="cuda", dtype=torch.bfloat16)
    value = torch.randn_like(key)
    attention_function = ATTENTION_FUNCTION_REGISTRY[AttentionBackendType[backend_name]]

    def attention(query, key, value):
        return attention_function(query, key, value, dropout_p=0.0, is_causal=False)[0]

    output = torch.compile(attention, fullgraph=True)(query, key, value)
    assert output.shape == query.shape
    assert torch.isfinite(output).all()


def test_aiter_i8fp8_attention_compiles_fullgraph():
    _require_mha_v4_aiter("AITER_I8FP8", supported_arches=("gfx942", "gfx950"))

    from xfuser.core.distributed.attention_backend import (
        ATTENTION_FUNCTION_REGISTRY,
        AttentionBackendType,
    )

    query = torch.randn((1, 5, 128, 128), device="cuda", dtype=torch.bfloat16)
    key = torch.randn((1, 5, 128, 128), device="cuda", dtype=torch.bfloat16)
    value = torch.randn_like(key)
    attention_function = ATTENTION_FUNCTION_REGISTRY[AttentionBackendType.AITER_I8FP8]

    def attention(query, key, value):
        return attention_function(
            query, key, value, dropout_p=0.0, is_causal=False
        )[0]

    output = torch.compile(attention, fullgraph=True)(query, key, value)
    assert output.shape == query.shape
    assert torch.isfinite(output).all()


def test_aiter_fp8_attention_compiles_fullgraph_with_mha_v4():
    _require_mha_v4_aiter("AITER_FP8", supported_arches=("gfx942", "gfx950"))

    from xfuser.core.distributed.attention_backend import (
        ATTENTION_FUNCTION_REGISTRY,
        AttentionBackendType,
    )

    query = torch.randn((1, 5, 257, 128), device="cuda", dtype=torch.bfloat16)
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    reference = F.scaled_dot_product_attention(query, key, value)
    attention_function = ATTENTION_FUNCTION_REGISTRY[AttentionBackendType.AITER_FP8]

    def attention(query, key, value):
        return attention_function(
            query, key, value, dropout_p=0.0, is_causal=False
        )[0]

    output = torch.compile(attention, fullgraph=True)(query, key, value)
    assert output.shape == query.shape
    assert torch.isfinite(output).all()
    assert F.cosine_similarity(
        output.float().flatten(), reference.float().flatten(), dim=0
    ).item() > 0.995


def test_aiter_mha_v4_rejects_causal_attention():
    from xfuser.core.distributed.attention_backend import (
        AITER_MHA_V4_ONLY_BACKENDS,
        ATTENTION_FUNCTION_REGISTRY,
    )

    tensor = torch.empty((1, 1, 1, 128), device="cuda", dtype=torch.bfloat16)
    for backend in AITER_MHA_V4_ONLY_BACKENDS:
        _require_mha_v4_aiter(backend.name)
        with pytest.raises(
            NotImplementedError,
            match="does not support causal masking",
        ):
            ATTENTION_FUNCTION_REGISTRY[backend](
                tensor, tensor, tensor, dropout_p=0.0, is_causal=True
            )


def test_aiter_low_precision_attention_rejects_dropout():
    from xfuser.core.distributed.attention_backend import (
        AITER_LOW_PRECISION_BACKENDS,
        ATTENTION_FUNCTION_REGISTRY,
    )

    tensor = torch.empty((1, 1, 1, 128), device="cuda", dtype=torch.bfloat16)
    for backend in AITER_LOW_PRECISION_BACKENDS:
        _require_mha_v4_aiter(backend.name)
        with pytest.raises(NotImplementedError, match="does not support dropout"):
            ATTENTION_FUNCTION_REGISTRY[backend](
                tensor, tensor, tensor, dropout_p=0.1, is_causal=False
            )
