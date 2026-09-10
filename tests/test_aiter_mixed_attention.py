from pathlib import Path

import pytest
import torch
import torch.nn.functional as F


def test_aiter_bf16_backends_use_mha_v4_while_aiter_remains_mha_v3(monkeypatch):
    from xfuser.core.distributed import attention_backend

    calls = []

    class AttentionFormat:
        BF16 = "bf16"
        FP8 = "fp8"

    def mha_v4(query, key, value, *formats):
        calls.append(("mha_v4", formats))
        return torch.empty_like(query)

    def mha_v3(query, key, value, **kwargs):
        calls.append(("mha_v3", kwargs))
        return torch.empty_like(query), None

    monkeypatch.setattr(attention_backend, "_AiterAttentionFormat", AttentionFormat)
    monkeypatch.setattr(attention_backend, "_aiter_mha_v4", mha_v4)
    monkeypatch.setattr(attention_backend, "_aiter_native_fp8_format", lambda: AttentionFormat.FP8)
    monkeypatch.setattr(attention_backend, "flash_attn_func_aiter", mha_v3)
    monkeypatch.setattr(attention_backend, "AITER_HAS_ROUND_MODE", False)

    query = torch.empty((1, 2, 4, 128), dtype=torch.bfloat16)
    key = torch.empty_like(query)
    value = torch.empty_like(query)

    bf16_output, bf16_lse = attention_backend.ATTENTION_FUNCTION_REGISTRY[
        attention_backend.AttentionBackendType.AITER_BF16
    ](query, key, value, dropout_p=0.0, is_causal=False)
    bf16fp8_output, bf16fp8_lse = attention_backend.ATTENTION_FUNCTION_REGISTRY[
        attention_backend.AttentionBackendType.AITER_BF16FP8
    ](query, key, value, dropout_p=0.0, is_causal=False)
    legacy_output, legacy_lse = attention_backend.ATTENTION_FUNCTION_REGISTRY[
        attention_backend.AttentionBackendType.AITER
    ](query, key, value, dropout_p=0.0, is_causal=False)

    assert calls[0] == ("mha_v4", ("bf16", "bf16", "bf16"))
    assert calls[1] == ("mha_v4", ("bf16", "bf16", "fp8"))
    assert calls[2][0] == "mha_v3"
    assert bf16_output.shape == query.shape
    assert bf16fp8_output.shape == query.shape
    assert legacy_output.shape == query.shape
    assert bf16_lse is None
    assert bf16fp8_lse is None
    assert legacy_lse is None


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
            import inspect

            if inspect.signature(mha_v4).parameters.get("q_scale_mode") is None:
                pytest.skip("AITER does not expose the MHA v4 MXFP8 raw API.")
        else:
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


# AITER is mid-migration on the MXFP4 rows: dense moved to full MXFP4 Q/K/V while sparse kept
# MXFP4 Q/K + FP8 V, so aiter_mxfp4 resolves to no dense row on builds in between.
def _require_mha_v4_recipe(backend_name):
    from xfuser.core.distributed.attention_backend import (
        ATTENTION_FUNCTION_REGISTRY,
        AttentionBackendType,
    )

    probe = torch.zeros((1, 1, 128, 128), device="cuda", dtype=torch.bfloat16)
    try:
        with torch.no_grad():
            ATTENTION_FUNCTION_REGISTRY[AttentionBackendType[backend_name]](
                probe, probe, probe, dropout_p=0.0, is_causal=False
            )
    except NotImplementedError as exc:
        if "kernel row" not in str(exc):
            raise
        pytest.skip(f"Installed AITER has no kernel row for {backend_name}: {exc}")


# Dense MXFP4 V returns garbage at any sequence length that is not a multiple of 128 on AITER
# main: cosine against SDPA measures 0.040 at S=257 and -0.012 at S=129, while FP8 V and MXFP6 V
# stay correct. AITER's own unaligned-sequence test asserts only eager==compiled and isfinite,
# so it does not catch this.
_MXFP4_V_BACKENDS = ("AITER_F4F4", "AITER_F6F4")


def _xfail_broken_mxfp4_v(request, backend_name, sequence_length):
    if backend_name in _MXFP4_V_BACKENDS and sequence_length % 128:
        request.applymarker(
            pytest.mark.xfail(
                reason=(
                    f"AITER dense MXFP4 V is numerically wrong at S={sequence_length} "
                    "(S % 128 != 0); tracked upstream"
                ),
                strict=False,
            )
        )


@pytest.mark.parametrize(
    "backend_name",
    [
        "AITER_BF16",
        "AITER_BF16FP8",
        "AITER_MXFP8",
        "AITER_F8F6",
        "AITER_F6F4",
        "AITER_MXFP4",
        "AITER_F4F4",
    ],
)
@pytest.mark.parametrize("sequence_length", [128, 257])
def test_aiter_mixed_attention_matches_sdpa(backend_name, sequence_length, request):
    _require_mha_v4_aiter(backend_name)
    _require_mha_v4_recipe(backend_name)
    _xfail_broken_mxfp4_v(request, backend_name, sequence_length)

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
    [
        "AITER_BF16",
        "AITER_BF16FP8",
        "AITER_MXFP8",
        "AITER_F8F6",
        "AITER_F6F4",
        "AITER_MXFP4",
        "AITER_F4F4",
    ],
)
def test_aiter_mixed_attention_compiles_fullgraph(backend_name):
    _require_mha_v4_aiter(backend_name)
    _require_mha_v4_recipe(backend_name)

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
    [
        "AITER_BF16",
        "AITER_BF16FP8",
        "AITER_MXFP8",
        "AITER_F8F6",
        "AITER_F6F4",
        "AITER_MXFP4",
        "AITER_F4F4",
    ],
)
def test_aiter_mixed_attention_unequal_sequence_lengths(backend_name):
    _require_mha_v4_aiter(backend_name)
    _require_mha_v4_recipe(backend_name)

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
    "backend_name",
    [
        "AITER_BF16",
        "AITER_BF16FP8",
        "AITER_MXFP8",
        "AITER_F8F6",
        "AITER_MXFP6",
        "AITER_MXFP4",
    ],
)
def test_aiter_mixed_cross_attention_compiles_fullgraph(backend_name):
    _require_mha_v4_aiter(backend_name)
    _require_mha_v4_recipe(backend_name)

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


def test_aiter_mha_v4_rejects_multi_sequence_varlen_packed_keys():
    """MHA v4 has no key-padding mask, so several ragged sequences must fail loudly.

    Silently dropping attention_kwargs lets padded keys contribute to the softmax
    denominator, which is wrong rather than merely approximate. A single sequence is
    served densely instead (see test_aiter_mha_v4_serves_single_sequence_padding).
    """
    from xfuser.core.distributed.attention_backend import (
        AITER_MHA_V4_ONLY_BACKENDS,
        ATTENTION_FUNCTION_REGISTRY,
    )

    tensor = torch.empty((2, 1, 4, 128), device="cuda", dtype=torch.bfloat16)
    attention_kwargs = {
        "indices_k": torch.tensor([0, 4, 5], dtype=torch.int64, device="cuda"),
        "cu_seqlens_k": torch.tensor([0, 1, 3], dtype=torch.int32, device="cuda"),
        "max_seqlen_k": 2,
    }
    for backend in AITER_MHA_V4_ONLY_BACKENDS:
        _require_mha_v4_aiter(backend.name)
        with pytest.raises(
            NotImplementedError,
            match="batch size > 1",
        ):
            ATTENTION_FUNCTION_REGISTRY[backend](
                tensor,
                tensor,
                tensor,
                dropout_p=0.0,
                is_causal=False,
                attention_kwargs=attention_kwargs,
            )


def test_aiter_mha_v4_serves_single_sequence_padding():
    """One sequence needs no key-padding mask: the valid keys are just a shorter K/V."""
    from xfuser.core.distributed.attention_backend import (
        ATTENTION_FUNCTION_REGISTRY,
        AttentionBackendType,
    )

    _require_mha_v4_aiter(AttentionBackendType.AITER_BF16.name)
    _require_mha_v4_recipe(AttentionBackendType.AITER_BF16.name)

    torch.manual_seed(1234)
    valid, padded, heads = 300, 384, 4
    shape = (1, heads, padded, 128)
    query = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    key = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    value = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    attention_kwargs = {
        "indices_k": torch.arange(valid, dtype=torch.int64, device="cuda"),
        "cu_seqlens_k": torch.tensor([0, valid], dtype=torch.int32, device="cuda"),
        "max_seqlen_k": valid,
    }

    with torch.no_grad():
        reference = F.scaled_dot_product_attention(
            query, key[:, :, :valid], value[:, :, :valid]
        )
        output, _ = ATTENTION_FUNCTION_REGISTRY[AttentionBackendType.AITER_BF16](
            query,
            key,
            value,
            dropout_p=0.0,
            is_causal=False,
            attention_kwargs=attention_kwargs,
        )

    assert output.shape == reference.shape
    cosine = F.cosine_similarity(
        output.float().flatten(), reference.float().flatten(), dim=0
    )
    assert cosine > 0.99, f"cosine {cosine.item()}"


def test_aiter_mha_v4_falls_back_below_head_dim_128():
    """LTX-2 pairs 128-wide video blocks with 64-wide audio ones in a single backend selection."""
    from xfuser.core.distributed.attention_backend import (
        ATTENTION_FUNCTION_REGISTRY,
        AttentionBackendType,
    )

    _require_mha_v4_aiter(AttentionBackendType.AITER_BF16.name)

    torch.manual_seed(1234)
    shape = (1, 4, 256, 64)
    query = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    key = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    value = torch.randn(shape, device="cuda", dtype=torch.bfloat16)

    with torch.no_grad():
        reference = F.scaled_dot_product_attention(query, key, value)
        output, _ = ATTENTION_FUNCTION_REGISTRY[AttentionBackendType.AITER_BF16](
            query, key, value, dropout_p=0.0, is_causal=False
        )

    assert output.shape == reference.shape
    torch.testing.assert_close(output, reference, rtol=2e-2, atol=2e-2)
