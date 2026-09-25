from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from xfuser.core.attention import registry
from xfuser.core.attention.spec import (
    AttentionBackendType,
    AttnCall,
    VarlenPacking,
)

# Backends whose kernel is an MHA v4 launcher.
_MHA_V4_BACKENDS = frozenset(
    spec.type for spec in registry.REGISTRY.values()
    if spec.impl.target.startswith("kernel:mha_v4")
)


def _impl(backend):
    """The backend's kernel, with the call signature these tests use.

    Resolved here rather than on first dispatch, which is what runtime_state
    does when a backend is selected: importing a kernel module inside a
    compiled region is a fullgraph failure.
    """
    spec = registry.get(_as_type(backend))
    spec.resolved()

    def call(query, key, value, dropout_p=0.0, is_causal=False, attention_kwargs=None):
        return spec.run(query, key, value, AttnCall(
            dropout_p=dropout_p, is_causal=is_causal,
            attention_kwargs=attention_kwargs or {},
        ))

    return call


def _run(backend, query, key, value, **kwargs):
    return _impl(backend)(query, key, value, **kwargs)


def _as_type(backend):
    return backend if isinstance(backend, AttentionBackendType) \
        else AttentionBackendType[backend]



def test_bf16_rows_route_to_mha_v4_while_aiter_stays_on_mha_v3():
    """Which kernel a backend reaches is declared, not discovered: the BF16
    rows bind an MHA v4 launcher, AITER binds the v3 flash entry point."""
    from xfuser.core.attention import registry
    from xfuser.core.attention.backends.aiter_mha_v4.spec import Fmt
    from xfuser.core.attention.spec import AttentionBackendType

    bf16 = registry.get(AttentionBackendType.AITER_BF16)
    bf16fp8 = registry.get(AttentionBackendType.AITER_BF16FP8)
    aiter = registry.get(AttentionBackendType.AITER)

    assert bf16.impl.target == "kernel:mha_v4_dense"
    assert (bf16.impl.bound["fmt"].qk, bf16.impl.bound["fmt"].v) == (Fmt.BF16, Fmt.BF16)

    assert bf16fp8.impl.target == "kernel:mha_v4_dense"
    assert (bf16fp8.impl.bound["fmt"].qk, bf16fp8.impl.bound["fmt"].v) == (
        Fmt.BF16, Fmt.NATIVE_FP8
    )

    assert aiter.impl.target == "kernel:aiter_attention"
    assert aiter.impl.bound == {}


def _require_mha_v4_aiter(backend_name, supported_arches=("gfx950",)):
    """Skip unless this machine can actually run the backend.

    Two separate questions. The spec answers the first -- arch, symbols and
    signatures are all declared in `requires` -- so asking it covers every
    reason rather than just the arch. The second is whether AITER ships a
    precompiled kernel for this arch and format, which only the file tells us.
    """
    if not torch.cuda.is_available():
        pytest.skip("requires a GPU")

    unavailable = registry.get(_as_type(backend_name)).unavailable()
    if unavailable is not None:
        pytest.skip(f"{backend_name}: {unavailable}")

    arch_name = getattr(torch.cuda.get_device_properties(0), "gcnArchName", "")
    arch = next((name for name in supported_arches if name in arch_name), None)
    if arch is None:
        pytest.skip(f"{backend_name} requires {supported_arches}, got {arch_name}")

    import aiter

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
    probe = torch.zeros((1, 1, 128, 128), device="cuda", dtype=torch.bfloat16)
    try:
        with torch.no_grad():
            _run(backend_name, 
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

    torch.manual_seed(1234)
    shape = (1, 5, sequence_length, 128)
    query = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    key = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    value = torch.randn(shape, device="cuda", dtype=torch.bfloat16)

    with torch.no_grad():
        reference = F.scaled_dot_product_attention(query, key, value)
        output, lse = _run(backend_name, 
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

    attention_function = _impl(backend_name)
    shape = (1, 5, 128, 128)
    query = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    key = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    value = torch.randn(shape, device="cuda", dtype=torch.bfloat16)

    def attention(query, key, value):
        return attention_function(query, key, value, dropout_p=0.0, is_causal=False)[0]

    output = torch.compile(attention, fullgraph=True)(query, key, value)
    assert output.shape == query.shape
    assert torch.isfinite(output).all()


def test_aiter_mixed_attention_compiles_fullgraph_with_a_trailing_pad():
    """The trim is Python-level, so it must fold away rather than break the graph."""
    _require_mha_v4_aiter("AITER_BF16")
    _require_mha_v4_recipe("AITER_BF16")

    valid_length = 128
    shape = (1, 5, 192, 128)
    query = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    key = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    value = torch.randn(shape, device="cuda", dtype=torch.bfloat16)

    spec = registry.get(AttentionBackendType.AITER_BF16)
    spec.resolved()
    call = AttnCall(
        varlen=VarlenPacking(
            indices_k=torch.arange(valid_length, device="cuda"),
            cu_seqlens_k=torch.tensor(
                [0, valid_length], dtype=torch.int32, device="cuda"
            ),
            max_seqlen_k=valid_length,
        ),
        attention_kwargs={"valid_kv_len": valid_length},
    )

    def attention(query, key, value):
        return spec.run(query, key, value, call)[0]

    output = torch.compile(attention, fullgraph=True)(query, key, value)
    assert output.shape == query.shape
    assert torch.isfinite(output).all()


def test_aiter_mxfp8_gqa_compiles_and_matches_sdpa():
    _require_mha_v4_aiter("AITER_MXFP8")

    torch.manual_seed(1234)
    query = torch.randn((1, 64, 128, 128), device="cuda", dtype=torch.bfloat16)
    key = torch.randn((1, 4, 128, 128), device="cuda", dtype=torch.bfloat16)
    value = torch.randn_like(key)
    attention_function = _impl("AITER_MXFP8")

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

    query = torch.randn((2, 5, 128, 128), device="cuda", dtype=torch.bfloat16)
    key = torch.randn((2, 5, 257, 128), device="cuda", dtype=torch.bfloat16)
    value = torch.randn_like(key)
    output, _ = _run(backend_name, 
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

    query = torch.randn((1, 5, 129, 128), device="cuda", dtype=torch.bfloat16)
    key = torch.randn((1, 5, 128, 128), device="cuda", dtype=torch.bfloat16)
    value = torch.randn_like(key)
    attention_function = _impl(backend_name)

    def attention(query, key, value):
        return attention_function(query, key, value, dropout_p=0.0, is_causal=False)[0]

    output = torch.compile(attention, fullgraph=True)(query, key, value)
    assert output.shape == query.shape
    assert torch.isfinite(output).all()


def test_aiter_i8fp8_attention_compiles_fullgraph():
    _require_mha_v4_aiter("AITER_I8FP8", supported_arches=("gfx942", "gfx950"))

    query = torch.randn((1, 5, 128, 128), device="cuda", dtype=torch.bfloat16)
    key = torch.randn((1, 5, 128, 128), device="cuda", dtype=torch.bfloat16)
    value = torch.randn_like(key)
    attention_function = _impl("AITER_I8FP8")

    def attention(query, key, value):
        return attention_function(
            query, key, value, dropout_p=0.0, is_causal=False
        )[0]

    output = torch.compile(attention, fullgraph=True)(query, key, value)
    assert output.shape == query.shape
    assert torch.isfinite(output).all()


def test_aiter_fp8_attention_compiles_fullgraph_with_mha_v4():
    _require_mha_v4_aiter("AITER_FP8", supported_arches=("gfx942", "gfx950"))

    query = torch.randn((1, 5, 257, 128), device="cuda", dtype=torch.bfloat16)
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    reference = F.scaled_dot_product_attention(query, key, value)
    attention_function = _impl("AITER_FP8")

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


def _available(backends):
    """Deterministic order, and skipping one backend must not abort the test."""
    for backend in sorted(backends, key=lambda b: b.name):
        if registry.get(backend).unavailable() is None:
            yield backend


@pytest.mark.parametrize(
    "case, call",
    [
        ("causal masking", AttnCall(is_causal=True)),
        ("dropout", AttnCall(dropout_p=0.1)),
        ("varlen packed keys", AttnCall(varlen=VarlenPacking(
            indices_k=torch.zeros(1, dtype=torch.int64),
            cu_seqlens_k=torch.tensor([0, 1], dtype=torch.int32),
            max_seqlen_k=1,
        ))),
    ],
    ids=["causal", "dropout", "varlen"],
)
def test_mha_v4_refuses_calls_it_cannot_serve(case, call):
    """Dense MHA v4 has no causal mask, no dropout and no key-padding mask.
    Each refusal is declared in `accepts`, so it happens before the kernel
    runs -- silently dropping the packing would let padded keys contribute to
    the softmax denominator, which is wrong rather than approximate."""
    tensor = torch.empty((1, 1, 1, 128))
    checked = 0
    for backend in _available(_MHA_V4_BACKENDS):
        reason = registry.get(backend).rejects(tensor, tensor, tensor, call)
        assert reason is not None, f"{backend.name} accepts {case}"
        checked += 1
    assert checked, "no MHA v4 backend was available to check"


@pytest.mark.parametrize(
    "backend_name",
    [
        "AITER_BF16",
        "AITER_BF16FP8",
        "AITER_MXFP8",
        "AITER_F8F6",
        "AITER_F6F4",
        "AITER_MXFP4",
        pytest.param(
            "AITER_F4F4",
            marks=pytest.mark.skip(
                reason="faults the GPU once allocations accumulate; fixed in newer AITER"
            ),
        ),
    ],
)
def test_mha_v4_serves_a_declared_trailing_pad(backend_name, request):
    """A declared trailing pad is served by slicing K/V, matching the same maths.

    Every query row is kept, including the pad rows: they are not packed, their
    outputs are discarded by the caller, and trimming them by a key-side length
    would be wrong wherever Q and K differ.
    """
    _require_mha_v4_aiter(backend_name)
    _require_mha_v4_recipe(backend_name)

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

    call = AttnCall(
        varlen=VarlenPacking(
            indices_k=torch.arange(valid_length, device="cuda"),
            cu_seqlens_k=torch.tensor(
                [0, valid_length], dtype=torch.int32, device="cuda"
            ),
            max_seqlen_k=valid_length,
        ),
        attention_kwargs={"valid_kv_len": valid_length},
    )

    with torch.no_grad():
        reference = F.scaled_dot_product_attention(
            query, key[:, :, :valid_length], value[:, :, :valid_length]
        )
        spec = registry.get(AttentionBackendType[backend_name])
        spec.resolved()
        output, lse = spec.run(query, key, value, call)

    cosine_similarity = F.cosine_similarity(
        output.float().flatten(), reference.float().flatten(), dim=0
    ).item()

    assert output.shape == query.shape
    assert torch.isfinite(output).all()
    assert lse is None
    assert cosine_similarity > 0.95
