import itertools
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from xfuser.core.attention import registry
from xfuser.core.attention.spec import (
    AttentionBackendType,
    AttnCall,
    VarlenPacking,
)

pytestmark = [pytest.mark.accelerator, pytest.mark.rocm]

# Backends whose kernel is an MHA v4 launcher.
_MHA_V4_BACKENDS = frozenset(
    spec.type for spec in registry.REGISTRY.values() if spec.impl.target.startswith("kernel:mha_v4")
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
        kwargs = attention_kwargs or {}
        return spec.run(
            query,
            key,
            value,
            AttnCall(
                dropout_p=dropout_p,
                is_causal=is_causal,
                # Derived from the kwargs exactly as usp does. Without it a caller
                # can pass indices_k and still take the dense path, which is a test
                # that proves nothing about the packing it believes it sent.
                varlen=VarlenPacking.from_kwargs(kwargs),
                attention_kwargs=kwargs,
            ),
        )

    return call


def _run(backend, query, key, value, **kwargs):
    return _impl(backend)(query, key, value, **kwargs)


def _as_type(backend):
    return backend if isinstance(backend, AttentionBackendType) else AttentionBackendType[backend]


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
    assert (bf16fp8.impl.bound["fmt"].qk, bf16fp8.impl.bound["fmt"].v) == (Fmt.BF16, Fmt.NATIVE_FP8)

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

    kernel_dir = Path(aiter.__file__).resolve().parent.parent / "hsa" / arch / "fmha_v4_fwd"
    kernel_name = backend_name.removeprefix("AITER_").lower()
    candidates = [kernel_dir / f"fwd_hd128_{kernel_name}.co"]
    if arch == "gfx942":
        candidates.append(kernel_dir / "MI300" / f"fwd_hd128_{kernel_name}.co")
    if not any(path.exists() for path in candidates):
        pytest.skip(f"AITER does not include the {arch} {kernel_name} FMHA kernel.")


# AITER is mid-migration on the MXFP4 rows: dense moved to full MXFP4 Q/K/V while sparse kept
# MXFP4 Q/K + FP8 V, so aiter_mxfp4 resolves to no dense row on builds in between.
def _mha_v4_kernel():
    """The MHA v4 kernel module, for the capability flags it reads at import."""
    import importlib

    return importlib.import_module("xfuser.core.attention.backends.aiter_mha_v4.kernel")


def _require_mha_v4_recipe(backend_name):
    probe = torch.zeros((1, 1, 128, 128), device="cuda", dtype=torch.bfloat16)
    try:
        with torch.no_grad():
            _run(backend_name, probe, probe, probe, dropout_p=0.0, is_causal=False)
    except NotImplementedError as exc:
        if "kernel row" not in str(exc):
            raise
        pytest.skip(f"Installed AITER has no kernel row for {backend_name}: {exc}")


# Dense MXFP4 V returns garbage at any sequence length that is not a multiple of 128 on AITER
# main: cosine against SDPA measures 0.040 at S=257 and -0.012 at S=129, while FP8 V and MXFP6 V
# stay correct. AITER's own unaligned-sequence test asserts only eager==compiled and isfinite,
# so it does not catch this.
_MXFP4_V_BACKENDS = ("AITER_F4F4", "AITER_F6F4")


def _xfail_broken_mxfp4_v(backend_name, sequence_length):
    if backend_name in _MXFP4_V_BACKENDS and sequence_length % 128:
        pytest.xfail(
            f"AITER dense MXFP4 V is numerically wrong at S={sequence_length} (S % 128 != 0); tracked upstream"
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
def test_aiter_mixed_attention_matches_sdpa(backend_name, sequence_length):
    _require_mha_v4_aiter(backend_name)
    _require_mha_v4_recipe(backend_name)
    _xfail_broken_mxfp4_v(backend_name, sequence_length)

    torch.manual_seed(1234)
    shape = (1, 5, sequence_length, 128)
    query = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    key = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    value = torch.randn(shape, device="cuda", dtype=torch.bfloat16)

    with torch.no_grad():
        reference = F.scaled_dot_product_attention(query, key, value)
        output, lse = _run(backend_name, query, key, value, dropout_p=0.0, is_causal=False)

    cosine_similarity = F.cosine_similarity(output.float().flatten(), reference.float().flatten(), dim=0).item()

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
            cu_seqlens_k=torch.tensor([0, valid_length], dtype=torch.int32, device="cuda"),
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
    assert F.cosine_similarity(output.float().flatten(), reference.float().flatten(), dim=0).item() > 0.95


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

    # The MXFP4-V rows require a full V tile. Keep Q and K unequal while
    # respecting that kernel constraint; the other rows exercise a ragged K.
    key_length = 256 if backend_name in _MXFP4_V_BACKENDS else 257
    query = torch.randn((2, 5, 128, 128), device="cuda", dtype=torch.bfloat16)
    key = torch.randn((2, 5, key_length, 128), device="cuda", dtype=torch.bfloat16)
    value = torch.randn_like(key)
    output, _ = _run(backend_name, query, key, value, dropout_p=0.0, is_causal=False)

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
        return attention_function(query, key, value, dropout_p=0.0, is_causal=False)[0]

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
        return attention_function(query, key, value, dropout_p=0.0, is_causal=False)[0]

    output = torch.compile(attention, fullgraph=True)(query, key, value)
    assert output.shape == query.shape
    assert torch.isfinite(output).all()
    assert F.cosine_similarity(output.float().flatten(), reference.float().flatten(), dim=0).item() > 0.995


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
    ],
    ids=["causal", "dropout"],
)
def test_mha_v4_refuses_calls_it_cannot_serve(case, call):
    """Dense MHA v4 has no causal mask and no dropout. Each refusal is
    declared in `accepts`, so it happens before the kernel runs.

    Packed keys are no longer among them: the kernel shortens K/V instead --
    slicing a declared trailing pad or gathering the valid rows -- so the
    padding is not there to reach the softmax denominator. The one packed
    shape it still cannot serve is raised by the kernel rather than declared
    here; see test_mha_v4_refuses_several_packed_sequences for why."""
    tensor = torch.empty((1, 1, 1, 128))
    checked = 0
    for backend in _available(_MHA_V4_BACKENDS):
        reason = registry.get(backend).rejects(tensor, tensor, tensor, call)
        assert reason is not None, f"{backend.name} accepts {case}"
        checked += 1
    assert checked, "no MHA v4 backend was available to check"


def test_mha_v4_refuses_several_packed_sequences():
    """A batch of several packed sequences needs AITER's per-batch key
    lengths, which only the BF16 Q/K rows consume -- the others would attend
    over the padding, so AITER rejects them.

    Raised from the kernel rather than declared in `accepts` on purpose: a
    declared refusal sends the call to the v3 fallback, which would serve it
    correctly while leaving the chosen backend applied to only part of the
    run. A raise says which backends do work instead.

    F8F6 rather than BF16 so the assertion holds on either AITER: an older
    build has no seqlens_k at all and refuses every row, a newer one refuses
    this row specifically. Both messages name per-batch key lengths.
    """
    _require_mha_v4_aiter("AITER_F8F6")
    _require_mha_v4_recipe("AITER_F8F6")

    batch, heads, seq_len, head_dim = 2, 4, 128, 128
    shape = (batch, heads, seq_len, head_dim)
    query = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    key = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    value = torch.randn(shape, device="cuda", dtype=torch.bfloat16)

    # Two sequences of different valid lengths, so there is no single trailing
    # pad to slice and no valid_kv_len to declare.
    lengths = (96, 64)
    indices = torch.cat([torch.arange(b * seq_len, b * seq_len + n, device="cuda") for b, n in enumerate(lengths)])
    call = AttnCall(
        varlen=VarlenPacking(
            indices_k=indices,
            cu_seqlens_k=torch.tensor([0, lengths[0], sum(lengths)], dtype=torch.int32, device="cuda"),
            max_seqlen_k=max(lengths),
        ),
    )

    spec = registry.get(AttentionBackendType.AITER_F8F6)
    spec.resolved()
    assert spec.rejects(query, key, value, call) is None, (
        "accepts must not refuse this -- that would route it to the fallback"
    )
    with pytest.raises(NotImplementedError, match="per-batch key lengths"):
        spec.run(query, key, value, call)


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
            cu_seqlens_k=torch.tensor([0, valid_length], dtype=torch.int32, device="cuda"),
            max_seqlen_k=valid_length,
        ),
        attention_kwargs={"valid_kv_len": valid_length},
    )

    with torch.no_grad():
        reference = F.scaled_dot_product_attention(query, key[:, :, :valid_length], value[:, :, :valid_length])
        spec = registry.get(AttentionBackendType[backend_name])
        spec.resolved()
        output, lse = spec.run(query, key, value, call)

    cosine_similarity = F.cosine_similarity(output.float().flatten(), reference.float().flatten(), dim=0).item()

    assert output.shape == query.shape
    assert torch.isfinite(output).all()
    assert lse is None
    assert cosine_similarity > 0.95


def test_mha_v4_returns_an_lse_only_when_ring_asks_for_one():
    """Ring merges per-rank partials on the log-sumexp, so the dense rows
    return one when the call carries a ring degree and not otherwise -- asking
    for it buys a write the caller would discard.

    Pins the layout too. The kernel writes [batch, heads, Sq], which is what
    the merge expects, so mha_v4_dense permutes only O. That claim is a
    comment everywhere else; here it is an assertion.

    Single-rank on purpose: the degree is read off the call, so the plumbing
    is exercised without an initialised process group. A wrong LSE is
    invisible to output checks -- O never reads it -- so this pins shape and
    finiteness rather than values, and the numbers are #802's business.
    """
    _require_mha_v4_aiter("AITER_BF16")
    _require_mha_v4_recipe("AITER_BF16")

    spec = registry.get(AttentionBackendType.AITER_BF16)
    no_ring = spec.ring.unmet()
    if no_ring is not None:
        pytest.skip(f"AITER_BF16 cannot ring here: {no_ring}")

    batch, heads, seq_len, head_dim = 1, 5, 128, 128
    shape = (batch, heads, seq_len, head_dim)
    query = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    key = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    value = torch.randn(shape, device="cuda", dtype=torch.bfloat16)

    spec.resolved()
    with torch.no_grad():
        dense_out, dense_lse = spec.run(query, key, value, AttnCall())
        ring_out, ring_lse = spec.run(query, key, value, AttnCall(ring_world_size=2))

    assert dense_lse is None, "no ring degree, so no LSE is asked for"
    assert ring_lse is not None, "ring degree, so the kernel must return one"
    assert ring_lse.shape == (batch, heads, seq_len), "the ring merge reads [batch, heads, Sq]; only O is permuted"
    assert torch.isfinite(ring_lse).all()
    # Asking for the LSE must not change what O is.
    assert torch.equal(dense_out, ring_out)


# ---------------------------------------------------------------------------
# packed keys end to end
#
# Ported from the monolith's suite onto the spec API. The shapes, references
# and tolerances are its work; only the dispatch changed.
# ---------------------------------------------------------------------------


def _packed_kwargs(valid, padded, device="cuda"):
    """indices_k names the surviving rows of the flattened (batch * padded) K."""
    indices_k = torch.cat([torch.arange(b * padded, b * padded + n, device=device) for b, n in enumerate(valid)])
    return {
        "indices_k": indices_k,
        "cu_seqlens_k": torch.tensor([0, *itertools.accumulate(valid)], dtype=torch.int32, device=device),
        "max_seqlen_k": max(valid),
    }


def test_mha_v4_serves_single_sequence_padding_by_gathering():
    """One sequence needs no key-padding mask: its valid keys are a shorter
    K/V, so attending over them is exact rather than approximate.

    No valid_kv_len here, which is what separates this from the trailing-pad
    test above -- the kernel gathers rather than slices, and that is the path
    Krea-2 takes."""
    _require_mha_v4_aiter("AITER_BF16")
    _require_mha_v4_recipe("AITER_BF16")

    torch.manual_seed(1234)
    valid, padded, heads = 300, 384, 4
    shape = (1, heads, padded, 128)
    query = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    key = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    value = torch.randn(shape, device="cuda", dtype=torch.bfloat16)

    with torch.no_grad():
        reference = F.scaled_dot_product_attention(query, key[:, :, :valid], value[:, :, :valid])
        output, _ = _impl("AITER_BF16")(
            query,
            key,
            value,
            attention_kwargs=_packed_kwargs((valid,), padded),
        )

    assert output.shape == reference.shape
    cosine = F.cosine_similarity(output.float().flatten(), reference.float().flatten(), dim=0)
    assert cosine > 0.99, f"cosine {cosine.item()}"


def test_mha_v4_serves_an_undeclared_mask_with_interior_gaps():
    """Gathering carries no assumption that the pad is one trailing block, so
    it stays correct for a mask with interior gaps -- which a slice would
    silently mis-serve.

    The rows between the gaps are poisoned, so anything that sliced to a
    trailing length instead would attend over them and the cosine would fall
    apart."""
    _require_mha_v4_aiter("AITER_BF16")
    _require_mha_v4_recipe("AITER_BF16")

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
    for dead in (slice(100, 200), slice(300, None)):
        key[:, :, dead] = 50.0
        value[:, :, dead] = 50.0

    with torch.no_grad():
        reference = F.scaled_dot_product_attention(query, key[:, :, indices_k], value[:, :, indices_k])
        output, _ = _impl("AITER_BF16")(
            query,
            key,
            value,
            attention_kwargs={
                "indices_k": indices_k,
                "cu_seqlens_k": torch.tensor([0, indices_k.numel()], dtype=torch.int32, device="cuda"),
                "max_seqlen_k": indices_k.numel(),
            },
        )

    cosine = F.cosine_similarity(output.float().flatten(), reference.float().flatten(), dim=0)
    assert cosine > 0.99, f"cosine {cosine.item()}"


def test_mha_v4_serves_several_packed_sequences_on_the_bf16_rows():
    """Several ragged sequences ride as a padded batch with their true lengths
    in seqlens_k, so the padding is never visited.

    Compared per sequence rather than in aggregate: padding that did reach the
    softmax denominator would pull every row of the shorter one."""
    _require_mha_v4_aiter("AITER_BF16")
    _require_mha_v4_recipe("AITER_BF16")
    if not _mha_v4_kernel().HAS_SEQLENS_K:
        pytest.skip("this AITER cannot express per-batch key lengths")

    torch.manual_seed(7)
    batch, heads, head_dim = 2, 4, 128
    padded, valid = 384, (300, 137)
    query = torch.randn((batch, heads, padded, head_dim), device="cuda", dtype=torch.bfloat16)
    key = torch.randn_like(query)
    value = torch.randn_like(query)

    with torch.no_grad():
        output, _ = _impl("AITER_BF16")(
            query,
            key,
            value,
            attention_kwargs=_packed_kwargs(valid, padded),
        )

    assert output.shape == query.shape
    for b, n in enumerate(valid):
        reference = F.scaled_dot_product_attention(query[b : b + 1], key[b : b + 1, :, :n], value[b : b + 1, :, :n])
        cosine = F.cosine_similarity(output[b : b + 1].float().flatten(), reference.float().flatten(), dim=0)
        assert cosine > 0.99, f"sequence {b} of {valid}: {cosine}"


@pytest.mark.parametrize("head_dim", [64, 256])
def test_mha_v4_falls_back_to_v3_off_head_dim_128(head_dim):
    """LTX-2 pairs 128-wide video blocks with 64-wide audio ones under one
    backend selection, so refusing the odd widths would make the family
    unselectable for it.

    The fallback is v3 rather than SDPA so it still returns an LSE: ring
    merges on one, and a None would arrive from the odd-sized blocks only."""
    _require_mha_v4_aiter("AITER_BF16")

    torch.manual_seed(1234)
    shape = (1, 4, 256, head_dim)
    query = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    key = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    value = torch.randn(shape, device="cuda", dtype=torch.bfloat16)

    spec = registry.get(AttentionBackendType.AITER_BF16)
    assert spec.rejects(query, key, value, AttnCall()) is not None, (
        "accepts must refuse this width -- that is what routes it to the fallback"
    )

    with torch.no_grad():
        reference = F.scaled_dot_product_attention(query, key, value)
        output, softmax_lse = _impl("AITER_BF16")(query, key, value)

    torch.testing.assert_close(output, reference, rtol=2e-2, atol=2e-2)
    assert softmax_lse is not None, "the fallback must still produce an LSE"


def test_mha_v4_fallback_lse_keeps_batch_major_layout_under_padding():
    """The v3 varlen kernel hands its LSE back flat as [heads, batch*padded];
    the v3 dense branch and MHA v4 both give [batch, heads, sq], so the flat
    one is folded before it leaves the backend.

    LTX-2 reaches this: its 64-wide audio blocks take the fallback while the
    128-wide video ones stay on v4, and the ring merge would otherwise be
    handed two layouts in a single step. The element count matches either way,
    so the shape is what pins it -- a fold without the transpose would pass on
    count alone."""
    _require_mha_v4_aiter("AITER_BF16")

    torch.manual_seed(5)
    batch, heads, head_dim = 2, 4, 64
    padded, valid = 384, (300, 137)
    query = torch.randn((batch, heads, padded, head_dim), device="cuda", dtype=torch.bfloat16)
    key = torch.randn_like(query)
    value = torch.randn_like(query)

    attention = _impl("AITER_BF16")
    with torch.no_grad():
        _, padded_lse = attention(query, key, value, attention_kwargs=_packed_kwargs(valid, padded))
        _, dense_lse = attention(query, key, value)

    assert dense_lse is not None and padded_lse is not None
    assert padded_lse.shape == dense_lse.shape == (batch, heads, padded), (
        f"packed {tuple(padded_lse.shape)} vs dense {tuple(dense_lse.shape)}"
    )


# ---------------------------------------------------------------------------
# head dimension, checked at configuration rather than per call
# ---------------------------------------------------------------------------


class _HeadDimConfig:
    use_hybrid_attn_schedule = False
    cross_attention_backend = None

    def __init__(self, backend):
        self.attention_backend = backend


class _HeadDimModel:
    def __init__(self, dims):
        self.attention_head_dims = dims
        self.settings = SimpleNamespace(model_name="model")


def test_a_model_is_refused_a_backend_that_serves_none_of_its_head_dims():
    """Ideogram 4 runs head dimension 256, which MHA v4 does not serve.
    Selecting one used to be accepted and then bypassed for every layer, so
    the choice read as applied while nothing about the run changed.

    The widths come from each backend's own `accepts`, so this needs no list:
    a backend declaring no HEAD_DIM makes no claim and stays selectable."""
    from xfuser.core.attention.backends.aiter_mha_v4.spec import DENSE_BACKENDS
    from xfuser.model_executor.models.runner_models.base_model import (
        _validate_attention_head_dims,
    )
    from xfuser.model_executor.models.runner_models.ideogram4 import (
        xFuserIdeogram4Model,
    )
    from xfuser.model_executor.models.runner_models.ltx import (
        _xFuserLTX25VideoModelBase,
    )

    ideogram4 = _HeadDimModel(xFuserIdeogram4Model.attention_head_dims)
    for backend in DENSE_BACKENDS:
        with pytest.raises(ValueError, match="head dimension"):
            _validate_attention_head_dims(ideogram4, _HeadDimConfig(backend.name))

    # v3 covers 256 and constrains nothing, so it stays selectable.
    assert registry.get(AttentionBackendType.AITER).accepts.head_dims() is None
    _validate_attention_head_dims(ideogram4, _HeadDimConfig("AITER"))
    _validate_attention_head_dims(ideogram4, _HeadDimConfig("SDPA"))

    # LTX-2.5 mixes 128-wide video with 64-wide audio; one served width keeps it.
    ltx25 = _HeadDimModel(_xFuserLTX25VideoModelBase.attention_head_dims)
    assert 64 in ltx25.attention_head_dims
    _validate_attention_head_dims(ltx25, _HeadDimConfig("AITER_BF16"))

    # A model that declares nothing is never refused on head dimension.
    _validate_attention_head_dims(_HeadDimModel(frozenset()), _HeadDimConfig("AITER_BF16"))


def test_the_head_dim_refusal_says_what_would_actually_happen():
    """A dense row hands the call to its fallback, so the cost is a selection
    that does nothing. A sparge row has no fallback and raises, so the cost is
    a run that stops. Same refusal, different consequence, and the message has
    to name the right one or it sends the reader looking for a silent bypass
    that never happens."""
    from xfuser.model_executor.models.runner_models.base_model import (
        _validate_attention_head_dims,
    )

    model = _HeadDimModel(frozenset({256}))

    with pytest.raises(ValueError, match="fall through to AITER"):
        _validate_attention_head_dims(model, _HeadDimConfig("AITER_BF16"))

    with pytest.raises(ValueError, match="would be refused"):
        _validate_attention_head_dims(model, _HeadDimConfig("AITER_FP8_SPARGE"))


def test_mha_v4_gathers_a_multi_sequence_batch_that_declares_valid_kv_len():
    """valid_kv_len is one length for the whole call, so it cannot describe a
    batch. Declaring it alongside several ragged sequences used to take the
    trailing-pad slice, which cut every row to the longest segment and left the
    shorter rows attending over their own pad.

    Nothing rejects the declaration: it is within the padded length, and it
    does equal max_seqlen_k, because the longest segment genuinely is the valid
    count -- for one of the rows. Only the output shows it, which is why the
    pad is poisoned here and the comparison is per sequence. Upstream measured
    cosine [1.0, 0.0126] before the fix: the long row fine, the short one
    destroyed.
    """
    _require_mha_v4_aiter("AITER_BF16")
    _require_mha_v4_recipe("AITER_BF16")
    if not _mha_v4_kernel().HAS_SEQLENS_K:
        pytest.skip("this AITER cannot express per-batch key lengths")

    torch.manual_seed(7)
    heads, padded, valid = 4, 384, (300, 137)
    shape = (len(valid), heads, padded, 128)
    query = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    for b, n in enumerate(valid):
        key[b, :, n:] = 50.0
        value[b, :, n:] = 50.0

    kwargs = _packed_kwargs(valid, padded)
    kwargs["valid_kv_len"] = max(valid)  # the declaration that misleads

    with torch.no_grad():
        output, _ = _impl("AITER_BF16")(query, key, value, attention_kwargs=kwargs)

    for b, n in enumerate(valid):
        reference = F.scaled_dot_product_attention(query[b : b + 1], key[b : b + 1, :, :n], value[b : b + 1, :, :n])
        cosine = F.cosine_similarity(output[b : b + 1].float().flatten(), reference.float().flatten(), dim=0).item()
        assert cosine > 0.99, f"sequence {b} of {valid}: {cosine}"
