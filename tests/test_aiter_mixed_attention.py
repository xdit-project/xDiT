import types
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
        # No seqlens_k parameter on purpose: a dense request must stay callable against an
        # AITER that predates per-batch key lengths.
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
def test_aiter_mixed_attention_matches_sdpa(backend_name, sequence_length):
    _require_mha_v4_aiter(backend_name)
    _require_mha_v4_recipe(backend_name)

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


def test_krea2_can_select_the_mha_v4_backends():
    """Krea-2 gates on varlen support, which MHA v4 now has for its shape.

    Its guidance runs the transformer once per branch rather than as one batched pair, so every
    call is a single padded sequence -- the case served by gathering the keys into a dense call.
    """
    from xfuser.core.distributed.attention_backend import AITER_MHA_V4_ONLY_BACKENDS
    from xfuser.model_executor.models.runner_models.krea2 import (
        _KREA2_SUPPORTED_ATTN_BACKENDS,
    )

    unsupported = [b.name for b in AITER_MHA_V4_ONLY_BACKENDS
                   if b not in _KREA2_SUPPORTED_ATTN_BACKENDS]
    assert not unsupported, unsupported


def test_ideogram4_refuses_the_mha_v4_backends():
    """Ideogram 4 runs head dimension 256, which MHA v4 does not serve.

    Selecting one used to be accepted and then quietly bypassed for every layer, so the choice
    read as applied while nothing about the run changed.
    """
    from xfuser.core.distributed.attention_backend import (
        AITER_MHA_V4_ONLY_BACKENDS,
        ATTENTION_BACKEND_HEAD_DIMS,
        AttentionBackendType,
    )
    from xfuser.model_executor.models.runner_models.base_model import (
        _validate_attention_head_dims,
    )
    from xfuser.model_executor.models.runner_models.ideogram4 import (
        xFuserIdeogram4Model,
    )
    from xfuser.model_executor.models.runner_models.ltx import (
        _xFuserLTX25VideoModelBase,
    )

    class _Config:
        use_hybrid_attn_schedule = False
        cross_attention_backend = None

        def __init__(self, backend):
            self.attention_backend = backend

    class _Model:
        def __init__(self, dims):
            self.attention_head_dims = dims
            self.settings = types.SimpleNamespace(model_name="model")

    ideogram4 = _Model(xFuserIdeogram4Model.attention_head_dims)
    for backend in AITER_MHA_V4_ONLY_BACKENDS:
        with pytest.raises(ValueError, match="head dimension"):
            _validate_attention_head_dims(ideogram4, _Config(backend.name))

    # v3 covers 256 and declares no constraint, so it stays selectable.
    assert AttentionBackendType.AITER not in ATTENTION_BACKEND_HEAD_DIMS
    _validate_attention_head_dims(ideogram4, _Config("AITER"))
    _validate_attention_head_dims(ideogram4, _Config("SDPA"))

    # LTX-2.5 mixes 128-wide video with 64-wide audio; one served width is enough to keep it.
    ltx25 = _Model(_xFuserLTX25VideoModelBase.attention_head_dims)
    assert 64 in ltx25.attention_head_dims
    _validate_attention_head_dims(ltx25, _Config("AITER_BF16"))

    # A model that declares nothing is never refused on head dimension.
    _validate_attention_head_dims(_Model(frozenset()), _Config("AITER_BF16"))


def test_aiter_mha_v4_serves_multi_sequence_varlen_packed_keys():
    """Several ragged sequences ride as a padded batch with their lengths in seqlens_k.

    The padded tail must not reach the softmax denominator, which a cosine against a
    per-sequence reference would catch: padding that contributed would pull every row.
    """
    from xfuser.core.distributed.attention_backend import (
        ATTENTION_FUNCTION_REGISTRY,
        AttentionBackendType,
    )

    _require_mha_v4_aiter(AttentionBackendType.AITER_BF16.name)

    torch.manual_seed(7)
    batch, heads, head_dim = 2, 4, 128
    padded, valid = 384, (300, 137)
    query = torch.randn(
        (batch, heads, padded, head_dim), device="cuda", dtype=torch.bfloat16
    )
    key = torch.randn_like(query)
    value = torch.randn_like(query)

    # indices_k names the surviving rows of the flattened (batch * padded) key tensor.
    indices_k = torch.cat(
        [torch.arange(b * padded, b * padded + n, device="cuda") for b, n in enumerate(valid)]
    )
    attention_kwargs = {
        "indices_k": indices_k,
        "cu_seqlens_k": torch.tensor(
            [0, valid[0], valid[0] + valid[1]], dtype=torch.int32, device="cuda"
        ),
        "max_seqlen_k": max(valid),
    }

    with torch.no_grad():
        output, _ = ATTENTION_FUNCTION_REGISTRY[AttentionBackendType.AITER_BF16](
            query,
            key,
            value,
            dropout_p=0.0,
            is_causal=False,
            attention_kwargs=attention_kwargs,
        )

    assert output.shape == query.shape
    for b, n in enumerate(valid):
        reference = F.scaled_dot_product_attention(
            query[b : b + 1], key[b : b + 1, :, :n], value[b : b + 1, :, :n]
        )
        cosine = F.cosine_similarity(
            output[b : b + 1].float().flatten(), reference.float().flatten(), dim=0
        )
        assert cosine > 0.99, f"sequence {b} of {valid}: {cosine}"


@pytest.mark.parametrize(
    "backend_name", ["AITER_I8FP8", "AITER_MXFP8", "AITER_MXFP6", "AITER_F8F6"]
)
def test_aiter_mha_v4_rejects_multi_sequence_padding_off_the_bf16_rows(backend_name):
    """Only the BF16 Q/K objects read seqlens_k, so the rest must say so rather than attend padding.

    One sequence stays served on every recipe: its keys are packed into a shorter dense K/V and
    no per-batch length is needed, so the restriction applies to a batch of several only.
    """
    from xfuser.core.distributed.attention_backend import (
        ATTENTION_FUNCTION_REGISTRY,
        AttentionBackendType,
    )

    _require_mha_v4_aiter(backend_name)
    backend = getattr(AttentionBackendType, backend_name)
    heads, head_dim, padded = 4, 128, 384

    def run(batch, valid):
        query = torch.randn(
            (batch, heads, padded, head_dim), device="cuda", dtype=torch.bfloat16
        )
        indices_k = torch.cat(
            [
                torch.arange(b * padded, b * padded + n, device="cuda")
                for b, n in enumerate(valid)
            ]
        )
        cumulative = torch.tensor(valid, device="cuda").cumsum(0)
        with torch.no_grad():
            return ATTENTION_FUNCTION_REGISTRY[backend](
                query,
                torch.randn_like(query),
                torch.randn_like(query),
                dropout_p=0.0,
                is_causal=False,
                attention_kwargs={
                    "indices_k": indices_k,
                    "cu_seqlens_k": torch.cat(
                        [torch.zeros(1, device="cuda"), cumulative]
                    ).to(torch.int32),
                    "max_seqlen_k": max(valid),
                },
            )

    output, _ = run(1, (300,))
    assert torch.isfinite(output.float()).all()

    with pytest.raises(NotImplementedError, match="BF16 Q/K rows"):
        run(2, (300, 137))


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


@pytest.mark.parametrize("head_dim", [64, 256])
def test_aiter_mha_v4_falls_back_to_v3_off_head_dim_128(head_dim):
    """LTX-2 pairs 128-wide video blocks with 64-wide audio ones in a single backend selection.

    The fallback goes to v3 rather than SDPA so it still returns an LSE: ring parallelism merges
    on it, and a None would reach the merge only for the odd-sized blocks.
    """
    from xfuser.core.distributed.attention_backend import (
        ATTENTION_FUNCTION_REGISTRY,
        AttentionBackendType,
    )

    _require_mha_v4_aiter(AttentionBackendType.AITER_BF16.name)

    torch.manual_seed(1234)
    shape = (1, 4, 256, head_dim)
    query = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    key = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    value = torch.randn(shape, device="cuda", dtype=torch.bfloat16)

    with torch.no_grad():
        reference = F.scaled_dot_product_attention(query, key, value)
        output, softmax_lse = ATTENTION_FUNCTION_REGISTRY[
            AttentionBackendType.AITER_BF16
        ](query, key, value, dropout_p=0.0, is_causal=False)

    assert output.shape == reference.shape
    torch.testing.assert_close(output, reference, rtol=2e-2, atol=2e-2)
    assert softmax_lse is not None
    assert softmax_lse.shape == shape[:3]


def test_aiter_mha_v4_fallback_lse_keeps_batch_major_layout_under_padding():
    """The v3 varlen kernel hands its LSE back flat as [heads, batch * padded].

    Everything the ring merge otherwise sees -- the v3 dense branch and MHA v4 alike -- is
    [batch, heads, sq], so the flat one has to be folded back before it leaves the backend.
    LTX-2 is what reaches this: its 64-wide audio blocks take the fallback while the 128-wide
    video blocks stay on v4, and the merge would receive two different layouts in one step.
    """
    from xfuser.core.distributed.attention_backend import (
        ATTENTION_FUNCTION_REGISTRY,
        AttentionBackendType,
    )

    _require_mha_v4_aiter(AttentionBackendType.AITER_BF16.name)

    torch.manual_seed(5)
    batch, heads, head_dim = 2, 4, 64
    padded, valid = 384, (300, 137)
    query = torch.randn(
        (batch, heads, padded, head_dim), device="cuda", dtype=torch.bfloat16
    )
    key = torch.randn_like(query)
    value = torch.randn_like(query)

    indices_k = torch.cat(
        [torch.arange(b * padded, b * padded + n, device="cuda") for b, n in enumerate(valid)]
    )
    attention_kwargs = {
        "indices_k": indices_k,
        "cu_seqlens_k": torch.tensor(
            [0, valid[0], valid[0] + valid[1]], dtype=torch.int32, device="cuda"
        ),
        "max_seqlen_k": max(valid),
    }

    attention = ATTENTION_FUNCTION_REGISTRY[AttentionBackendType.AITER_BF16]
    with torch.no_grad():
        _, padded_lse = attention(
            query,
            key,
            value,
            dropout_p=0.0,
            is_causal=False,
            attention_kwargs=attention_kwargs,
        )
        _, dense_lse = attention(query, key, value, dropout_p=0.0, is_causal=False)

    # A flat [heads, batch * padded] LSE happens to hold the same element count, so compare the
    # layout against the unpadded call rather than the size alone.
    assert padded_lse.shape == dense_lse.shape == (batch, heads, padded)

    # Fold-and-transpose alone would satisfy the shape while scrambling which row belongs to
    # which sequence, so pin the values to a per-sequence reference too.
    for b, n in enumerate(valid):
        scores = (
            query[b].float() @ key[b, :, :n].float().transpose(-1, -2)
        ) * head_dim**-0.5
        torch.testing.assert_close(
            padded_lse[b], torch.logsumexp(scores, dim=-1), rtol=1e-3, atol=1e-3
        )


def test_aiter_mha_v4_lse_capability_excludes_gfx942(monkeypatch):
    """The lse buffer exists in AITER's signature on gfx942 too, but AITER refuses to fill it.

    Probing the signature alone would report a capability that raises on the first ring step.
    """
    from xfuser.core.distributed import attention_backend

    monkeypatch.setattr(attention_backend.torch.cuda, "is_available", lambda: True)

    class _Props:
        gcnArchName = "gfx942:sramecc+:xnack-"

    monkeypatch.setattr(
        attention_backend.torch.cuda, "get_device_properties", lambda _=0: _Props()
    )

    def _mha_v4(query, key, value, block_mask=None, seqlens_k=None, q_scale_mode=None):
        raise AssertionError("probe must not call the kernel")

    caps = attention_backend._probe_aiter_mha_v4_capabilities(_mha_v4)

    assert caps.is_gfx942
    assert caps.enabled
    assert not caps.lse


def test_aiter_mha_v4_ring_is_refused_on_gfx942(monkeypatch):
    """Ring weights each chunk by exp(lse); an unmeasured LSE must fail before the run starts."""
    from xfuser.core.distributed import runtime_state
    from xfuser.core.distributed.attention_backend import AttentionBackendType

    monkeypatch.setattr(runtime_state, "aiter_mha_v4_is_gfx942", lambda: True)

    class _Parallel:
        ring_degree = 2

    state = object.__new__(runtime_state.DiTRuntimeState)
    state.parallel_config = _Parallel()

    with pytest.raises(RuntimeError, match="gfx942"):
        runtime_state.RuntimeState._check_if_backend_compatible_with_current_configuration(
            state, AttentionBackendType.AITER_BF16
        )


def test_aiter_mixed_attention_compiles_fullgraph_with_a_trailing_pad():
    """The trim is Python-level, so it must fold away rather than break the graph."""
    _require_mha_v4_aiter("AITER_BF16")
    _require_mha_v4_recipe("AITER_BF16")

    from xfuser.core.distributed.attention_backend import (
        ATTENTION_FUNCTION_REGISTRY,
        AttentionBackendType,
    )

    valid_length = 128
    shape = (1, 5, 192, 128)
    query = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    key = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    value = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    attention_function = ATTENTION_FUNCTION_REGISTRY[AttentionBackendType.AITER_BF16]
    attention_kwargs = {
        "indices_k": torch.arange(valid_length, device="cuda"),
        "cu_seqlens_k": torch.tensor(
            [0, valid_length], dtype=torch.int32, device="cuda"
        ),
        "max_seqlen_k": valid_length,
        "valid_kv_len": valid_length,
    }

    def attention(query, key, value):
        return attention_function(
            query,
            key,
            value,
            dropout_p=0.0,
            is_causal=False,
            attention_kwargs=attention_kwargs,
        )[0]

    output = torch.compile(attention, fullgraph=True)(query, key, value)
    assert output.shape == query.shape
    assert torch.isfinite(output).all()


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
def test_aiter_mixed_attention_serves_a_declared_trailing_pad(backend_name, request):
    """A declared trailing pad is served by slicing K/V, matching the same maths.

    Every query row is kept, including the pad rows: they are not packed, their
    outputs are discarded by the caller, and trimming them by a key-side length
    would be wrong wherever Q and K differ.
    """
    _require_mha_v4_aiter(backend_name)
    _require_mha_v4_recipe(backend_name)

    from xfuser.core.distributed.attention_backend import (
        ATTENTION_FUNCTION_REGISTRY,
        AttentionBackendType,
    )

    valid_length = 256
    padded_length = 384
    torch.manual_seed(1234)
    shape = (1, 5, padded_length, 128)
    query = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    key = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    value = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    # MiniMax-H3's pad rows are zero hidden states, so their projections are zero
    # vectors: unmasked they score exp(0) against every query.
    key[:, :, valid_length:] = 0
    value[:, :, valid_length:] = 0

    attention_kwargs = {
        "indices_k": torch.arange(valid_length, device="cuda"),
        "cu_seqlens_k": torch.tensor(
            [0, valid_length], dtype=torch.int32, device="cuda"
        ),
        "max_seqlen_k": valid_length,
        "valid_kv_len": valid_length,
    }

    with torch.no_grad():
        reference = F.scaled_dot_product_attention(
            query, key[:, :, :valid_length], value[:, :, :valid_length]
        )
        output, lse = ATTENTION_FUNCTION_REGISTRY[AttentionBackendType[backend_name]](
            query,
            key,
            value,
            dropout_p=0.0,
            is_causal=False,
            attention_kwargs=attention_kwargs,
        )

    cosine_similarity = F.cosine_similarity(
        output.float().flatten(), reference.float().flatten(), dim=0
    ).item()

    assert output.shape == query.shape
    assert torch.isfinite(output).all()
    assert lse is None
    assert cosine_similarity > 0.95


def test_aiter_mha_v4_gather_padded_keys_contract():
    """A declared trailing pad is sliced; anything else falls through to the gather."""
    from xfuser.core.distributed.attention_backend import (
        _aiter_mha_v4_gather_padded_keys,
    )

    # Post-permute layout, so the key length is dim 1.
    query = torch.randn(1, 14, 2, 8)
    key = torch.randn(1, 14, 2, 8)
    value = torch.randn_like(key)
    indices_k = torch.arange(13)

    for kwargs in (None, {"indices_k": None}):
        assert _aiter_mha_v4_gather_padded_keys(query, key, value, kwargs) == (
            query,
            key,
            value,
            None,
        )

    _q, trimmed_key, trimmed_value, seqlens_k = _aiter_mha_v4_gather_padded_keys(
        query,
        key,
        value,
        {"indices_k": indices_k, "max_seqlen_k": 13, "valid_kv_len": 13},
    )
    assert trimmed_key.shape == (1, 13, 2, 8)
    assert trimmed_value.shape == (1, 13, 2, 8)
    assert seqlens_k is None
    torch.testing.assert_close(trimmed_key, key[:, :13])

    with pytest.raises(ValueError, match="valid_kv_len must be in"):
        _aiter_mha_v4_gather_padded_keys(
            query, key, value, {"indices_k": indices_k, "valid_kv_len": 15}
        )
    with pytest.raises(ValueError, match="as many valid keys"):
        _aiter_mha_v4_gather_padded_keys(
            query,
            key,
            value,
            {"indices_k": indices_k, "max_seqlen_k": 12, "valid_kv_len": 13},
        )


def test_aiter_mha_v4_serves_an_undeclared_mask_with_interior_gaps():
    """Without valid_kv_len the keys are gathered, which a mask with interior gaps needs.

    Slicing to a trailing length is only correct when the producer promises the pad is one
    trailing block. Gathering by indices_k carries no such assumption, so the rows between
    the gaps are poisoned here: anything that sliced instead would attend over them.
    """
    from xfuser.core.distributed.attention_backend import (
        ATTENTION_FUNCTION_REGISTRY,
        AttentionBackendType,
    )

    _require_mha_v4_aiter(AttentionBackendType.AITER_BF16.name)
    _require_mha_v4_recipe(AttentionBackendType.AITER_BF16.name)

    torch.manual_seed(11)
    heads, padded = 4, 384
    shape = (1, heads, padded, 128)
    query = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    key = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    value = torch.randn(shape, device="cuda", dtype=torch.bfloat16)

    indices_k = torch.cat(
        [
            torch.arange(0, 100, device="cuda"),
            torch.arange(200, 300, device="cuda"),
        ]
    )
    key[:, :, 100:200] = 50.0
    value[:, :, 100:200] = 50.0
    key[:, :, 300:] = 50.0
    value[:, :, 300:] = 50.0

    attention_kwargs = {
        "indices_k": indices_k,
        "cu_seqlens_k": torch.tensor(
            [0, indices_k.numel()], dtype=torch.int32, device="cuda"
        ),
        "max_seqlen_k": indices_k.numel(),
    }

    with torch.no_grad():
        reference = F.scaled_dot_product_attention(
            query, key[:, :, indices_k], value[:, :, indices_k]
        )
        output, _ = ATTENTION_FUNCTION_REGISTRY[AttentionBackendType.AITER_BF16](
            query,
            key,
            value,
            dropout_p=0.0,
            is_causal=False,
            attention_kwargs=attention_kwargs,
        )

    assert output.shape == query.shape
    cosine = F.cosine_similarity(
        output.float().flatten(), reference.float().flatten(), dim=0
    ).item()
    assert cosine > 0.99, cosine
