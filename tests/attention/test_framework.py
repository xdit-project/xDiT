"""The attention framework itself, with no backends registered.

These exercise the pieces every backend depends on -- predicates, call
constraints, the registry and the layout helpers -- without needing a GPU or
any vendor library.
"""

from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from xfuser.core.attention import registry
from xfuser.core.attention.numerics.layout import from_bshd, pack_kv, to_bshd
from xfuser.core.attention.requirements import (
    ALWAYS,
    ARCH,
    CUDA_CAPABILITY,
    NEVER,
    PARAM,
    PLATFORM,
    SYMBOL,
    Requirement,
    _resolve_with_reason,
)
from xfuser.core.attention.constraints import (
    ANY_CALL,
    HEAD_DIM,
    MHA_ONLY,
    NON_CAUSAL,
    NO_VARLEN,
)
from xfuser.core.attention.spec import (
    AttentionBackendType,
    AttnCall,
    Impl,
    Sparsity,
    Spec,
    VarlenPacking,
)


# --------------------------------------------------------------------------
# requirements
# --------------------------------------------------------------------------


def test_always_is_satisfied():
    assert ALWAYS.unmet() is None


def test_symbol_present_and_absent():
    assert SYMBOL("math:sqrt").unmet() is None
    assert SYMBOL("math:no_such_function").unmet() is not None
    assert SYMBOL("no_such_module_xyz:thing").unmet() is not None


def test_symbol_failure_names_the_symbol_without_prescribing_a_remedy():
    reason = SYMBOL("aiter.ops.mha_v4:mha_v4_does_not_exist").unmet()
    assert "aiter.ops.mha_v4.mha_v4_does_not_exist" in reason
    # Directional advice is wrong half the time for an unversioned upstream.
    assert "update" not in reason.lower()
    assert "upgrade" not in reason.lower()


def test_param_checks_signature():
    assert PARAM("json:dumps", "skipkeys").unmet() is None
    reason = PARAM("json:dumps", "no_such_param").unmet()
    assert "no_such_param" in reason


def test_param_distinguishes_unreadable_signature_from_absent_parameter():
    """Some C extensions expose no signature. Reporting that as "parameter
    absent" would silently disable a backend that actually works, so the two
    cases get different messages."""
    reason = PARAM("functools:reduce", "function").unmet()
    assert reason is not None
    assert "signature cannot be read" in reason
    assert "has no parameter" not in reason


def test_param_reports_import_failure_rather_than_missing_param():
    reason = PARAM("no_such_module_xyz:thing", "whatever").unmet()
    assert "not installed" in reason


def test_and_returns_first_failure_and_short_circuits():
    combined = SYMBOL("math:sqrt") & SYMBOL("math:nope") & SYMBOL("math:also_nope")
    assert "math.nope" in combined.unmet()


def test_and_flattens_and_passes_when_all_hold():
    combined = SYMBOL("math:sqrt") & SYMBOL("math:pi") & ALWAYS
    assert combined.unmet() is None


def test_requirement_is_not_truthy():
    with pytest.raises(TypeError):
        bool(ALWAYS)


def test_probes_are_memoised():
    _resolve_with_reason.cache_clear()
    SYMBOL("math:sqrt").unmet()
    SYMBOL("math:sqrt").unmet()
    SYMBOL("math:sqrt").unmet()
    assert _resolve_with_reason.cache_info().hits >= 2


def test_arch_reports_what_it_found():
    reason = ARCH("gfx_nonexistent").unmet()
    assert reason is not None and "gfx_nonexistent" in reason


def test_device_arch_follows_the_current_device():
    """A rank sets its device during distributed init, so reading device 0
    reads a different GPU unless the launcher also pinned the visible devices.
    On a mixed box that gates the backend on the wrong architecture, silently."""
    from xfuser.core.attention import requirements

    archs = {0: "gfx942", 1: "gfx950"}

    class _Props:
        def __init__(self, index):
            self.gcnArchName = archs[index]

    requirements._arch_of.cache_clear()
    try:
        with (
            mock.patch.object(torch.cuda, "is_available", return_value=True),
            mock.patch.object(torch.cuda, "get_device_properties", _Props),
            mock.patch.object(torch.cuda, "current_device") as current,
        ):
            current.return_value = 1
            assert requirements.device_arch() == "gfx950"
            current.return_value = 0
            assert requirements.device_arch() == "gfx942", "cached across devices"
            assert ARCH("gfx942").unmet() is None
            assert "found gfx942" in ARCH("gfx950").unmet()
    finally:
        requirements._arch_of.cache_clear()


def test_cuda_capability_reads_the_current_device():
    with (
        mock.patch.object(torch.cuda, "is_available", return_value=True),
        mock.patch.object(torch.cuda, "current_device", return_value=3),
        mock.patch.object(torch.cuda, "get_device_capability") as capability,
    ):
        capability.return_value = (9, 0)
        assert "found (9, 0)" in CUDA_CAPABILITY((10, 0)).unmet()
        capability.assert_called_once_with(3)
        capability.return_value = (10, 0)
        assert CUDA_CAPABILITY((10, 0)).unmet() is None


def test_capability_and_arch_report_the_absence_of_a_gpu():
    with mock.patch.object(torch.cuda, "is_available", return_value=False):
        assert "no GPU detected" in CUDA_CAPABILITY((10, 0)).unmet()
        assert "no GPU detected" in ARCH("gfx950").unmet()


# --------------------------------------------------------------------------
# call constraints
# --------------------------------------------------------------------------


def _qkv(batch=1, heads=4, q_len=16, kv_len=16, head_dim=128):
    return (
        torch.zeros(batch, heads, q_len, head_dim),
        torch.zeros(batch, heads, kv_len, head_dim),
        torch.zeros(batch, heads, kv_len, head_dim),
    )


def test_an_attn_call_built_inside_a_graph_can_have_its_degrees_read():
    """usp builds the AttnCall inside the traced region and kernels read the
    parallel degrees off it -- aiter_sage and MHA v4 both branch on
    ring_world_size.

    Held flat rather than in a nested dataclass for this reason alone: Dynamo
    has no source for an object a default_factory produced, and its sourceless
    builder fails on a user-defined class with "AttributeError: 'NoneType'
    object has no attribute 'name'". Two AttnCalls comparing equal would then
    compile differently depending only on whether the caller spelled out a
    value identical to the default."""

    def read_degrees(x):
        call = AttnCall(dropout_p=0.0, is_causal=False, attention_kwargs={})
        if call.ring_world_size > 1 or call.ulysses_world_size > 1:
            return x + 1.0
        return x * 2.0

    compiled = torch.compile(read_degrees, fullgraph=True)
    assert torch.allclose(compiled(torch.ones(4)), torch.full((4,), 2.0))


def test_any_shape_accepts_everything():
    q, k, v = _qkv(head_dim=64)
    assert ANY_CALL.unmet(q, k, v, AttnCall(is_causal=True)) is None


def test_head_dim_constraint():
    q, k, v = _qkv(head_dim=64)
    reason = HEAD_DIM(128).unmet(q, k, v, AttnCall())
    assert "head dimension 128" in reason and "64" in reason

    q, k, v = _qkv(head_dim=128)
    assert HEAD_DIM(128).unmet(q, k, v, AttnCall()) is None


def test_head_dim_is_readable_for_shape_selection():
    """The conformance suite reads this to avoid generating rejected shapes."""
    assert HEAD_DIM(128).head_dims() == (128,)
    assert (HEAD_DIM(128) & NON_CAUSAL).head_dims() == (128,)
    assert ANY_CALL.head_dims() is None


def test_non_causal_constraint():
    q, k, v = _qkv()
    assert NON_CAUSAL.unmet(q, k, v, AttnCall(is_causal=False)) is None
    assert NON_CAUSAL.unmet(q, k, v, AttnCall(is_causal=True)) is not None


def test_mha_only_constraint():
    q = torch.zeros(1, 8, 16, 128)
    k = torch.zeros(1, 2, 16, 128)
    assert MHA_ONLY.unmet(q, k, k, AttnCall()) is not None


def test_combined_constraint_reports_first_failure():
    q, k, v = _qkv(head_dim=64)
    combined = HEAD_DIM(128) & NON_CAUSAL & NO_VARLEN
    assert "head dimension" in combined.unmet(q, k, v, AttnCall(is_causal=True))


# --------------------------------------------------------------------------
# registry
# --------------------------------------------------------------------------


def _spec_kwargs(**kwargs):
    defaults = dict(impl=lambda q, k, v, c: (q, None), ring=ALWAYS, requires=ALWAYS, accepts=ANY_CALL)
    defaults.update(kwargs)
    return defaults


def _spec(backend, **kwargs):
    return Spec(backend, **_spec_kwargs(**kwargs))


def _module(name, *specs):
    """A stand-in for a backend package: build_registry wants SPECS and a
    __name__ to resolve Impl targets against."""
    return SimpleNamespace(__name__=name, SPECS=list(specs))


def test_using_swaps_the_registry_and_restores_it():
    spec = _spec(AttentionBackendType.SDPA)
    before = dict(registry.REGISTRY)

    with registry.using([spec]):
        assert registry.get(AttentionBackendType.SDPA) is spec
        assert len(registry.REGISTRY) == 1

    assert registry.REGISTRY == before


def test_using_restores_the_registry_after_a_failure():
    """Otherwise one failing test leaves every later one querying a registry
    with three backends in it."""
    before = dict(registry.REGISTRY)

    with pytest.raises(RuntimeError):
        with registry.using([_spec(AttentionBackendType.SDPA)]):
            raise RuntimeError("boom")

    assert registry.REGISTRY == before


def test_the_registry_cannot_be_written_through():
    """A consumer holds REGISTRY to query it. The derived sets other
    subsystems keep would all shift under a stray write."""
    with pytest.raises(TypeError):
        registry.REGISTRY[AttentionBackendType.SDPA] = _spec(AttentionBackendType.SDPA)
    with pytest.raises(AttributeError):
        registry.REGISTRY.clear()


def test_duplicate_registration_is_rejected():
    """Two modules claiming one backend: whichever imported last would win,
    silently, and which that is depends on the order of the MODULES list."""
    duplicate = _module("pkg.b", _spec(AttentionBackendType.SDPA))
    with pytest.raises(ValueError, match="already registered"):
        registry.build_registry(
            [
                _module("pkg.a", _spec(AttentionBackendType.SDPA)),
                duplicate,
            ]
        )


def test_build_registry_stamps_each_spec_with_its_package():
    built = registry.build_registry([_module("pkg.a", _spec(AttentionBackendType.SDPA))])
    assert built[AttentionBackendType.SDPA].package == "pkg.a"


def test_build_registry_does_not_install():
    """Pure: a registry can be built and inspected without the real one
    moving under everything that already queried it."""
    before = dict(registry.REGISTRY)
    registry.build_registry([_module("pkg.a", _spec(AttentionBackendType.SDPA))])
    assert registry.REGISTRY == before


def test_install_refuses_a_registry_missing_a_backend():
    """An enum member with no spec is selectable on the command line and
    resolves to nothing, so the package must fail to import rather than wait
    for a run to pick it."""
    with pytest.raises(ValueError, match="AITER_F4F4"):
        registry.install([_module("pkg.a", _spec(AttentionBackendType.SDPA))])
    assert registry.missing_specs() == [], "a refused install must not be applied"


# --------------------------------------------------------------------------
# fallback
# --------------------------------------------------------------------------


def _pair(**kwargs):
    """A spec that falls back to SDPA, and the SDPA spec it falls back to."""
    served = []

    def fallback_impl(q, k, v, c):
        served.append("fallback")
        return q, None

    primary = _spec(
        AttentionBackendType.AITER_BF16,
        accepts=HEAD_DIM(128),
        fallback=AttentionBackendType.SDPA,
        **kwargs,
    )
    return primary, _spec(AttentionBackendType.SDPA, impl=fallback_impl), served


def test_a_rejected_call_goes_to_the_fallback_rather_than_raising():
    """MHA v4 runs head dim 128, and LTX-2 pairs 128-wide video blocks with
    64-wide audio ones under one backend choice. Raising would make the
    backend unselectable for that model."""
    primary, fallback, served = _pair()
    with registry.using([primary, fallback]):
        spec = registry.get(AttentionBackendType.AITER_BF16)
        spec.run(*_qkv(head_dim=128), AttnCall())
        assert served == [], "a call it serves must not reach the fallback"
        spec.run(*_qkv(head_dim=64), AttnCall())
        assert served == ["fallback"]


def test_a_rejected_call_still_raises_without_a_fallback():
    spec = _spec(AttentionBackendType.SDPA, accepts=HEAD_DIM(128))
    with pytest.raises(NotImplementedError, match="head dimension"):
        spec.run(*_qkv(head_dim=64), AttnCall())


def test_resolving_a_spec_resolves_its_fallback():
    """The fallback must be imported at selection, not on the first call that
    needs it: by then we are inside the traced region, where Dynamo refuses
    importlib. Resolving through _fallback rather than the registry is what
    makes the instance we import for the instance we dispatch to."""
    primary, fallback, _ = _pair()
    with registry.using([primary, fallback]):
        spec = registry.get(AttentionBackendType.AITER_BF16)
        assert spec._fallback is registry.get(AttentionBackendType.SDPA), (
            "the stamped fallback must be the registry's own object"
        )
        spec.resolved()
        assert spec._fallback._resolved is not None


def test_a_warm_spec_does_not_rewalk_its_fallback_chain():
    """The chain is walked after the cache check, so resolving twice costs one
    attribute read. Safe because _resolved is set only after the fallback is,
    making a warm spec proof that the chain behind it is warm too."""
    walked = []

    class CountingSpec(Spec):
        def resolved(self):
            walked.append(self.type.name)
            return super().resolved()

    fallback = CountingSpec(AttentionBackendType.SDPA, **_spec_kwargs())
    primary = CountingSpec(
        AttentionBackendType.AITER_BF16,
        fallback=AttentionBackendType.SDPA,
        **_spec_kwargs(),
    )
    with registry.using([primary, fallback]):
        spec = registry.get(AttentionBackendType.AITER_BF16)
        spec.resolved()
        assert walked == ["AITER_BF16", "SDPA"]
        spec.resolved()
        assert walked == ["AITER_BF16", "SDPA", "AITER_BF16"]


def test_the_mha_v4_dense_rows_ring_only_where_the_lse_is_measured():
    """AITER refuses the LSE on gfx942 until its value has been checked
    against a reference, and a wrong one is invisible to every output test
    because O never reads it. Stated as gfx950 rather than "not gfx942" so a
    new architecture opts in rather than inheriting the claim."""
    from xfuser.core.attention import registry
    from xfuser.core.attention.backends.aiter_mha_v4.spec import DENSE_BACKENDS

    for backend in DENSE_BACKENDS:
        spec = registry.get(backend)
        assert spec.ring is not NEVER, f"{backend.name}: dense rows can ring"
        reason = spec.ring.unmet()
        # On a machine without AITER or a GPU this is unmet, and saying why is
        # the whole point -- it must not raise while working that out.
        assert reason is None or isinstance(reason, str)

    # The sparse rows export no LSE at all, whatever the device.
    sparge = registry.get(AttentionBackendType.AITER_FP8_SPARGE)
    assert sparge.ring is NEVER


def test_a_fallback_must_ring_wherever_the_backend_it_serves_does():
    """A mixed-width model on a ring run sends some blocks to the fallback.
    If that one returns no LSE the merge gets a partial from the blocks the
    selection served and nothing from the rest -- and no output check can see
    it, because O never reads the LSE."""
    primary = _spec(
        AttentionBackendType.AITER_BF16, ring=ALWAYS, accepts=HEAD_DIM(128), fallback=AttentionBackendType.SDPA
    )
    fallback = _spec(AttentionBackendType.SDPA, ring=NEVER)

    with registry.using([primary, fallback]):
        spec = registry.get(AttentionBackendType.AITER_BF16)
        assert spec.ring.unmet() is None, "the selection itself can ring"
        assert spec._fallback.ring.unmet() is not None, "but the fallback cannot, which runtime_state must refuse"


def test_a_fallback_must_name_a_registered_backend():
    spec = _spec(AttentionBackendType.AITER_BF16, fallback=AttentionBackendType.AITER_F4F4)
    with pytest.raises(ValueError, match="AITER_F4F4"):
        registry.build_registry([_module("pkg.a", spec)])


def test_a_fallback_cycle_is_refused():
    """resolved() walks the chain and run() dispatches down it, so a cycle is
    an infinite recursion rather than a wrong answer."""
    a = _spec(AttentionBackendType.SDPA, fallback=AttentionBackendType.FLASH)
    b = _spec(AttentionBackendType.FLASH, fallback=AttentionBackendType.SDPA)
    with pytest.raises(ValueError, match="fallback cycle"):
        registry.build_registry([_module("pkg.a", a, b)])


@pytest.mark.parametrize(
    "field,value",
    [("sparsity", Sparsity.SPARGE), ("head_balanced", True), ("accepts_prequantized", True)],
)
def test_a_fallback_cannot_sit_beside_a_fact_it_would_route_around(field, value):
    """These are read from the *selected* spec elsewhere -- base_model gates on
    sparsity, usp on head_balanced, fp8_comms hands the kernel pre-quantized
    Q/K/V on accepts_prequantized. A fallback kernel honours none of them, so
    the combination is refused rather than documented."""
    primary, fallback, _ = _pair(**{field: value})
    with pytest.raises(ValueError, match=field):
        registry.build_registry([_module("pkg.a", primary, fallback)])


def test_get_unregistered_names_the_backend():
    with registry.using([_spec(AttentionBackendType.SDPA)]):
        with pytest.raises(KeyError, match="AITER_F4F4"):
            registry.get(AttentionBackendType.AITER_F4F4)


def test_find_returns_none_rather_than_raising():
    """usp and runtime_state look a backend up before knowing it is
    registered, and do so inside a traced region -- so this reads the dict,
    not the REGISTRY proxy, which Dynamo cannot subscript by a non-constant
    key. tests/test_minimax_h3.py's fullgraph cases are what catch a
    regression here."""
    spec = _spec(AttentionBackendType.SDPA)
    with registry.using([spec]):
        assert registry.find(AttentionBackendType.SDPA) is spec
        assert registry.find(AttentionBackendType.AITER_F4F4) is None


def test_queries_select_by_field():
    with registry.using(
        [
            _spec(
                AttentionBackendType.AITER_MXFP4_SPARGE,
                sparsity=Sparsity.SPARGE,
                head_balanced=True,
                low_precision=True,
                ring=NEVER,
            ),
            _spec(AttentionBackendType.AITER_MXFP4, low_precision=True, ring=NEVER),
            _spec(AttentionBackendType.SDPA),
        ]
    ):
        assert registry.types_where(is_sparse=True) == frozenset({AttentionBackendType.AITER_MXFP4_SPARGE})
        assert registry.types_where(low_precision=True) == frozenset(
            {
                AttentionBackendType.AITER_MXFP4_SPARGE,
                AttentionBackendType.AITER_MXFP4,
            }
        )
        assert registry.types_where(ring=ALWAYS) == frozenset({AttentionBackendType.SDPA})


def test_missing_specs_lists_members_without_a_spec():
    with registry.using([]):
        assert len(registry.missing_specs()) == len(list(AttentionBackendType))
    with registry.using([_spec(AttentionBackendType.SDPA)]):
        assert AttentionBackendType.SDPA not in registry.missing_specs()


def test_manifest_renders_from_specs():
    with registry.using([_spec(AttentionBackendType.SDPA)]):
        text = registry.manifest()
    assert "SDPA" in text and "BACKEND" in text


def test_spec_defaults_are_conservative():
    """A spec that names no sparsity strategy carries none, and one whose
    requirement is met is available."""
    spec = _spec(AttentionBackendType.SDPA, ring=NEVER)
    assert spec.sparsity is None and spec.is_sparse is False
    assert spec.unavailable() is None
    assert spec.rejects(*_qkv(), AttnCall()) is None


def test_a_spec_must_state_what_it_requires_accepts_and_rings():
    """None has a default, because the only values that could be one are the
    permissive ones: an omitted ``accepts`` would read as "serves every call",
    which is exactly the claim a new backend is least entitled to make by
    accident, and an omitted ``ring`` would claim an LSE the kernel may not
    produce. Saying ALWAYS, ANY_CALL or NEVER is no more work than saying
    nothing, and it distinguishes decided from forgotten."""

    def impl(query, key, value, call):
        return query, None

    stated = dict(requires=ALWAYS, accepts=ANY_CALL, ring=NEVER)

    for omitted in stated:
        with pytest.raises(TypeError, match=omitted):
            Spec(AttentionBackendType.SDPA, impl=impl, **{k: v for k, v in stated.items() if k != omitted})

    spec = Spec(AttentionBackendType.SDPA, impl=impl, **stated)
    assert spec.requires is ALWAYS
    assert spec.accepts is ANY_CALL
    assert spec.ring is NEVER


def test_spec_unavailable_surfaces_the_requirement_reason():
    spec = _spec(AttentionBackendType.SDPA, requires=SYMBOL("no_such_module_xyz:x"))
    assert "not installed" in spec.unavailable()


def test_spec_rejects_unacceptable_calls():
    spec = _spec(AttentionBackendType.SDPA, accepts=HEAD_DIM(128) & NON_CAUSAL)
    q, k, v = _qkv(head_dim=128)
    assert spec.rejects(q, k, v, AttnCall()) is None
    assert spec.rejects(q, k, v, AttnCall(is_causal=True)) is not None


# --------------------------------------------------------------------------
# layout
# --------------------------------------------------------------------------


def test_bshd_roundtrip():
    x = torch.randn(2, 4, 16, 64)
    assert to_bshd(x).shape == (2, 16, 4, 64)
    assert torch.equal(from_bshd(to_bshd(x)), x)


def test_to_bshd_contiguity_is_explicit():
    x = torch.randn(2, 4, 16, 64)
    assert not to_bshd(x).is_contiguous()
    assert to_bshd(x, contiguous=True).is_contiguous()


def test_to_bshd_handles_multiple_tensors():
    q, k, v = (torch.randn(1, 2, 8, 32) for _ in range(3))
    out = to_bshd(q, k, v)
    assert len(out) == 3 and all(t.shape == (1, 8, 2, 32) for t in out)


def test_varlen_packing_absent_without_indices():
    assert VarlenPacking.from_kwargs(None) is None
    assert VarlenPacking.from_kwargs({}) is None


def test_pack_kv_keeps_every_query_and_gathers_kv():
    """Q is never filtered; K/V are gathered by the packing indices."""
    batch, seq_len, heads, head_dim = 2, 4, 3, 8
    q = torch.randn(batch, seq_len, heads, head_dim)
    k = torch.randn(batch, seq_len, heads, head_dim)
    v = torch.randn(batch, seq_len, heads, head_dim)

    indices = torch.tensor([0, 1, 4, 5, 6], dtype=torch.long)
    packing = VarlenPacking(
        indices_k=indices,
        cu_seqlens_k=torch.tensor([0, 2, 5], dtype=torch.int32),
        max_seqlen_k=3,
    )
    packed = pack_kv(q, k, v, packing)

    assert packed.q.shape == (batch * seq_len, heads, head_dim)
    assert packed.k.shape == (len(indices), heads, head_dim)
    assert torch.equal(packed.k, k.reshape(-1, heads, head_dim)[indices])
    assert torch.equal(packed.cu_seqlens_q, torch.tensor([0, 4, 8], dtype=torch.int32))
    assert packed.max_seqlen_q == seq_len

    out = torch.randn(batch * seq_len, heads, head_dim)
    assert packed.unflatten(out).shape == (batch, seq_len, heads, head_dim)


def test_pack_kv_flattens_keys_against_their_own_length():
    """Cross attention: K is longer than Q, and indices_k indexes the key side.

    Flattening K against the query's length either raises -- LTX-2.5 hit
    "shape '[768, 32, 128]' is invalid for input of size 4194304" on its
    cross-attention blocks -- or, where the counts happen to agree, silently
    gathers the wrong rows. The output still unflattens to the query's shape,
    because that is what attention returns.
    """
    batch, q_len, kv_len, heads, head_dim = 2, 3, 5, 3, 8
    q = torch.randn(batch, q_len, heads, head_dim)
    k = torch.randn(batch, kv_len, heads, head_dim)
    v = torch.randn(batch, kv_len, heads, head_dim)

    # Two sequences of 4 and 2 valid keys out of 5 padded rows each.
    indices = torch.tensor([0, 1, 2, 3, 5, 6], dtype=torch.long)
    packed = pack_kv(
        q,
        k,
        v,
        VarlenPacking(
            indices_k=indices,
            cu_seqlens_k=torch.tensor([0, 4, 6], dtype=torch.int32),
            max_seqlen_k=4,
        ),
    )

    assert packed.q.shape == (batch * q_len, heads, head_dim)
    assert packed.k.shape == (len(indices), heads, head_dim)
    assert torch.equal(packed.k, k.reshape(-1, heads, head_dim)[indices])
    assert torch.equal(packed.v, v.reshape(-1, heads, head_dim)[indices])
    # Q side still describes the query, which is what the kernel is asked for.
    assert packed.max_seqlen_q == q_len
    assert torch.equal(packed.cu_seqlens_q, torch.tensor([0, 3, 6], dtype=torch.int32))
    out = torch.randn(batch * q_len, heads, head_dim)
    assert packed.unflatten(out).shape == (batch, q_len, heads, head_dim)


# --------------------------------------------------------------------------
# the enum move
# --------------------------------------------------------------------------


def test_enum_still_has_every_member():
    """Guards the refactor against dropping one. 45 came over from the
    monolith; AITER_BF16_SPARGE and AITER_BF16FP8_SPARGE joined with the MHA
    v4 Sparge rows that serve them, so a further change to this number should
    be a deliberate new backend rather than a casualty."""
    assert len(list(AttentionBackendType)) == 47


# --------------------------------------------------------------------------
# the framework must import anywhere
# --------------------------------------------------------------------------

VENDOR_MODULES = {
    "aiter",
    "flash_attn",
    "flash_attn_interface",
    "sageattention",
    "transformer_engine",
    "torch_npu",
    "flex_block_attn",
    "yunchang",
    "distvae",
}

FRAMEWORK_MODULES = ["spec", "requirements", "constraints", "registry", "numerics/layout", "numerics/hadamard"]


def _imported_top_level_modules(path):
    import ast

    names = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            names.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module.split(".")[0])
    return names


def test_framework_has_no_vendor_dependencies():
    """Backends may need aiter or flash-attn; the framework never may. A user
    on any machine must be able to import the registry and be told which
    backends are unavailable, rather than hitting an ImportError."""
    from pathlib import Path
    import xfuser.core.attention as package

    root = Path(package.__file__).parent
    for name in FRAMEWORK_MODULES:
        imported = _imported_top_level_modules(root / f"{name}.py")
        leaked = imported & VENDOR_MODULES
        assert not leaked, f"{name}.py imports vendor module(s): {sorted(leaked)}"


def test_framework_only_depends_on_torch_stdlib_and_itself():
    from pathlib import Path
    import xfuser.core.attention as package

    allowed = {
        "torch",
        "xfuser",
        "__future__",
        "dataclasses",
        "enum",
        "typing",
        "functools",
        "importlib",
        "inspect",
        "ast",
        "pathlib",
        "contextlib",
        "types",
    }
    root = Path(package.__file__).parent
    for name in FRAMEWORK_MODULES:
        unexpected = _imported_top_level_modules(root / f"{name}.py") - allowed
        assert not unexpected, f"{name}.py imports {sorted(unexpected)}"


def test_missing_module_is_reported_not_raised():
    """A backend whose library is absent must yield a reason, never an
    exception -- that is what lets the registry load on any machine."""
    spec = Spec(
        AttentionBackendType.AITER_F4F4,
        impl=lambda q, k, v, c: None,
        ring=NEVER,
        requires=SYMBOL("aiter.ops.definitely_not_here:kernel"),
        accepts=ANY_CALL,
    )
    assert "not installed" in spec.unavailable()


def test_or_is_satisfied_when_any_branch_holds():
    assert (SYMBOL("math:nope") | SYMBOL("math:sqrt")).unmet() is None
    assert (SYMBOL("math:sqrt") | SYMBOL("math:nope")).unmet() is None


def test_or_reports_every_branch_when_all_fail():
    reason = (SYMBOL("math:nope_a") | SYMBOL("math:nope_b")).unmet()
    assert "math.nope_a" in reason and "math.nope_b" in reason


def test_or_flattens_and_composes_with_and():
    combined = SYMBOL("math:sqrt") & (SYMBOL("math:nope") | SYMBOL("math:pi"))
    assert combined.unmet() is None
    combined = SYMBOL("math:sqrt") & (SYMBOL("math:nope_a") | SYMBOL("math:nope_b"))
    assert "none of:" in combined.unmet()


def test_arch_varargs_give_a_single_combined_reason():
    """ARCH("a", "b") rather than ARCH("a") | ARCH("b"): one reason naming both,
    instead of one branch reason each."""
    reason = ARCH("gfx_nope_a", "gfx_nope_b").unmet()
    assert "gfx_nope_a or gfx_nope_b" in reason
    assert "none of:" not in reason


def test_run_enforces_accepts_before_dispatching():
    """The constraint is declared once, on the spec, and checked in one place.
    A kernel function never repeats it."""
    called = []

    def impl(q, k, v, c):
        called.append(True)
        return q, None

    spec = Spec(
        AttentionBackendType.SDPA,
        impl=impl,
        accepts=HEAD_DIM(128) & NON_CAUSAL,
        requires=ALWAYS,
        ring=NEVER,
    )
    q, k, v = _qkv(head_dim=128)

    spec.run(q, k, v, AttnCall())
    assert called == [True]

    with pytest.raises(NotImplementedError, match="causal"):
        spec.run(q, k, v, AttnCall(is_causal=True))
    with pytest.raises(NotImplementedError, match="head dimension"):
        spec.run(*_qkv(head_dim=64), AttnCall())
    assert called == [True], "impl must not be reached for a rejected call"


def test_resolved_hands_out_the_same_callable_every_time():
    """An Impl with bound arguments builds a functools.partial, and two
    partials over the same function compare unequal. Resolving twice would
    hand dispatch a callable Dynamo has not seen before, so resolve() caches
    -- for a plain callable too, so run() takes one path for both."""
    spec = Spec(
        AttentionBackendType.SDPA,
        # Any pure-python target with a keyword to bind; the point is the
        # partial, not the function.
        impl=Impl("layout:to_bshd", bound={"contiguous": True}),
        package="xfuser.core.attention.numerics",
        requires=ALWAYS,
        accepts=ANY_CALL,
        ring=NEVER,
    )
    assert spec.resolved() is spec.resolved()

    plain = _spec(AttentionBackendType.FLASH)
    assert plain.resolved() is plain.impl
    assert plain._resolved is plain.impl, "a plain callable must be cached too"


def test_run_does_not_re_resolve_per_call():
    """resolved() is a module import; run() is the hot path."""
    resolutions = []

    class CountingImpl(Impl):
        def resolve(self, package):
            resolutions.append(package)
            return lambda q, k, v, c: (q, None)

    spec = Spec(
        AttentionBackendType.SDPA,
        impl=CountingImpl("kernel:whatever"),
        package="pkg",
        requires=ALWAYS,
        accepts=ANY_CALL,
        ring=NEVER,
    )
    q, k, v = _qkv()
    for _ in range(3):
        spec.run(q, k, v, AttnCall())
    assert resolutions == ["pkg"]


def test_platform_is_what_the_process_can_reach_not_what_it_was_built_for():
    """torch.version.hip is None on a CPU-only build exactly as it is on an
    NVIDIA one, so a build-only answer calls a GPU-less machine "cuda" and
    every PLATFORM("cuda")-gated backend reports itself available there."""
    from xfuser.core.attention import requirements

    requirements._platform.cache_clear()
    try:
        with mock.patch.object(torch.cuda, "is_available", return_value=False):
            assert requirements._platform() == "cpu"
            assert "found cpu" in PLATFORM("cuda").unmet()
    finally:
        requirements._platform.cache_clear()


def test_backends_without_an_lse_cannot_join_ring():
    """Ring attention merges a softmax log-sumexp across ranks, so a backend
    that produces none cannot join. Every spec must answer that question, and
    the answer is a predicate because on MHA v4 it depends on the AITER build
    and the device."""
    from xfuser.core.attention import registry

    for spec in registry.REGISTRY.values():
        assert isinstance(spec.ring, Requirement), f"{spec.type.name}: ring must be a Requirement, got {spec.ring!r}"
        # Callable without a GPU or a vendor library: that is the point of
        # requirements being lazy and reporting rather than raising.
        reason = spec.ring.unmet()
        assert reason is None or isinstance(reason, str)


# --------------------------------------------------------------------------
# satisfied() and FIRST_OF
# --------------------------------------------------------------------------


def test_satisfied_is_the_boolean_view_of_unmet():
    """unmet() carries the reason, which gating and messages need; satisfied()
    is for branching on a capability without a double negative."""
    assert SYMBOL("math:sqrt").satisfied() is True
    assert SYMBOL("math:nope").satisfied() is False
    assert ALWAYS.satisfied() is True
    assert (SYMBOL("math:sqrt") & SYMBOL("math:nope")).satisfied() is False


def test_first_of_gates_on_any_path():
    from xfuser.core.attention.requirements import FIRST_OF

    assert FIRST_OF("math:nope", "math:sqrt").unmet() is None
    assert FIRST_OF("math:sqrt", "math:nope").unmet() is None
    reason = FIRST_OF("math:nope_a", "math:nope_b").unmet()
    assert "math.nope_a" in reason and "math.nope_b" in reason


def test_first_of_resolves_to_the_first_path_that_works():
    """The point of the type: the gate and the import are one declaration, so
    adding a path cannot update one and miss the other."""
    import math

    from xfuser.core.attention.requirements import FIRST_OF

    assert FIRST_OF("math:nope", "math:sqrt").resolve() is math.sqrt
    assert FIRST_OF("math:sqrt", "math:pow").resolve() is math.sqrt

    with pytest.raises(ImportError, match="none of these"):
        FIRST_OF("math:nope_a", "math:nope_b").resolve()


def test_first_of_composes_with_and():
    from xfuser.core.attention.requirements import FIRST_OF

    combined = SYMBOL("math:pi") & FIRST_OF("math:nope", "math:sqrt")
    assert combined.satisfied()


def test_every_backend_that_can_reach_hadamard_declares_an_initializer():
    """hadamard.matrix() walks importlib on a cold cache, and Dynamo refuses to
    trace that -- so any backend reaching it per call must resolve it at
    selection, which is what `initializers` is for.

    Which backends those are is read off the specs rather than listed here: a
    row reaches hadamard exactly when it rotates with it, and it says so by
    naming CREATE_HADAMARD in `requires` or hadamard.rotate_qk in
    `prequant_rotate`. Sage v1 declares neither and does not rotate; v2
    declares the first; the fp8 rows declare the second.

    That is the whole point of moving the preparation onto the spec.
    AITER_FLYDSL_FP8 declared the rotation and prepared nothing, because the
    fact and the remedy lived in different files; now one implies the other
    and this test can check it without knowing any backend's name."""
    from xfuser.core.attention import registry
    from xfuser.core.attention.numerics import hadamard

    def names_hadamard(requirement):
        if requirement is hadamard.CREATE_HADAMARD:
            return True
        return any(names_hadamard(p) for p in getattr(requirement, "parts", ()))

    checked = 0
    for spec in registry.REGISTRY.values():
        rotates = spec.prequant_rotate is hadamard.rotate_qk or names_hadamard(spec.requires)
        if not rotates:
            continue
        assert spec.initializers, (
            f"{spec.type.name} rotates with hadamard but declares no initializer to resolve it at selection"
        )
        checked += 1
    assert checked >= 5, f"expected at least 5 rotating backends, found {checked}"


def test_hadamard_declares_its_symbol_once():
    """hadamard.matrix() resolves through the same object backends gate on."""
    from xfuser.core.attention.numerics import hadamard
    from xfuser.core.attention.requirements import FIRST_OF

    assert isinstance(hadamard.CREATE_HADAMARD, FIRST_OF)
    assert len(hadamard.CREATE_HADAMARD.targets) == 2


# --------------------------------------------------------------------------
# varlen packing
# --------------------------------------------------------------------------

# Backends whose kernel branches on call.varlen and honours the packing, by
# either of the two routes that exist:
#
#   - a varlen entry point, which takes the packed K/V and cu_seqlens as they
#     are (AITER v3, FlashAttention);
#   - shortening K/V so that the padding is not there to be attended over,
#     which is what MHA v4's dense rows do -- slicing a declared trailing pad,
#     gathering the valid rows, or scattering several sequences into a padded
#     batch whose true lengths travel in seqlens_k.
#
# Everything else must declare NO_VARLEN: accepting a packed call without
# honouring it runs dense attention over padded keys and returns wrong numbers
# rather than failing.
#
# Maintained by hand on purpose. Deriving it from the specs would compare the
# registry with itself; the point is that adding a name here is a claim someone
# makes deliberately, having checked the kernel.
VARLEN_CAPABLE = {
    "AITER",
    "AITER_FP8",
    "FLASH",
    "FLASH_3",
    "FLASH_4",
    # The MHA v4 dense rows. Its sparge rows are absent and must stay so: the
    # sorted-sparse launch needs the key length padded to its KV tile, which is
    # the alignment all three of those routes remove.
    "AITER_BF16",
    "AITER_BF16FP8",
    "AITER_I8FP8",
    "AITER_F8F6",
    "AITER_MXFP6",
    "AITER_F6F4",
    "AITER_MXFP4",
    "AITER_F4F4",
    "AITER_MXFP8",
}


def test_only_varlen_capable_backends_accept_packed_keys():
    import torch

    from xfuser.core.attention import registry
    from xfuser.core.attention.spec import VarlenPacking

    q = torch.zeros(1, 4, 8, 128)
    packed = AttnCall(
        varlen=VarlenPacking(
            indices_k=torch.tensor([0]),
            cu_seqlens_k=torch.tensor([0, 1]),
            max_seqlen_k=1,
        )
    )
    accepting = {backend.name for backend, spec in registry.REGISTRY.items() if spec.rejects(q, q, q, packed) is None}
    assert accepting == VARLEN_CAPABLE, (
        "varlen support disagrees with what the kernels implement; "
        f"unexpected: {sorted(accepting - VARLEN_CAPABLE)}, "
        f"missing: {sorted(VARLEN_CAPABLE - accepting)}"
    )


# --------------------------------------------------------------------------
# every spec points at code that exists
# --------------------------------------------------------------------------


def _impl_source(spec):
    """The file and function name an Impl target names, without importing it."""
    from pathlib import Path
    import importlib

    module_name, _, symbol = spec.impl.target.partition(":")
    package = importlib.import_module(spec.package)
    return Path(package.__file__).parent / f"{module_name}.py", symbol


def test_every_impl_target_names_a_function_that_exists():
    """Static: the target is a string, so a typo or a renamed function is
    invisible until that backend is selected on a machine that can run it.
    Parsing the file catches it anywhere, with no vendor library present."""
    import ast

    from xfuser.core.attention import registry

    for spec in registry.REGISTRY.values():
        path, symbol = _impl_source(spec)
        assert path.exists(), f"{spec.type.name}: no module at {path}"
        defined = {
            node.name
            for node in ast.parse(path.read_text()).body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        assert symbol in defined, f"{spec.type.name}: {path.name} defines no {symbol}"


def test_every_registered_backend_resolves_where_it_is_available():
    """A kernel module is only imported when its backend is selected, so a
    broken one -- a bad import, a name used before it is defined -- stays
    invisible until a run picks it. Import every module this machine can, and
    check the target really is callable."""
    from xfuser.core.attention import registry

    for spec in registry.REGISTRY.values():
        if spec.unavailable() is not None:
            continue
        assert callable(spec.resolved()), f"{spec.type.name}: impl is not callable"


def test_every_enum_member_has_a_spec():
    """A member with no spec cannot be selected, so a forgotten registration is
    a backend that silently does not exist."""
    from xfuser.core.attention import registry

    assert registry.missing_specs() == []


def test_sparsity_kinds_are_known():
    """Consumers select by exact match -- base_model groups SSTA apart from
    SPARGE and VSA -- so a strategy outside the closed set is a backend that
    quietly belongs to no group and is never gated. The enum is what enforces
    this; the test only confirms nothing reaches the registry around it."""
    from xfuser.core.attention import registry

    for spec in registry.REGISTRY.values():
        assert spec.sparsity is None or isinstance(spec.sparsity, Sparsity), (
            f"{spec.type.name}: unknown sparsity {spec.sparsity!r}"
        )


def test_impl_bound_arguments_are_accepted_by_the_target():
    """A generated family binds a table row to a shared launcher by keyword.
    Rename the launcher's parameter and every spec in the family breaks at its
    first call; the binding is a plain dict, so nothing else notices."""
    import ast

    from xfuser.core.attention import registry

    for spec in registry.REGISTRY.values():
        if not spec.impl.bound:
            continue
        path, symbol = _impl_source(spec)
        fn = next(
            node
            for node in ast.parse(path.read_text()).body
            if isinstance(node, ast.FunctionDef) and node.name == symbol
        )
        accepted = {a.arg for a in fn.args.args + fn.args.kwonlyargs}
        missing = set(spec.impl.bound) - accepted
        assert not missing, f"{spec.type.name}: {symbol} takes no {sorted(missing)}"


# --------------------------------------------------------------------------
# trailing-pad packing
# --------------------------------------------------------------------------


def _packed(max_seqlen_k, **kwargs):
    from xfuser.core.attention.spec import VarlenPacking

    return AttnCall(
        varlen=VarlenPacking(
            indices_k=torch.zeros(1, dtype=torch.int64),
            cu_seqlens_k=torch.tensor([0, max_seqlen_k], dtype=torch.int32),
            max_seqlen_k=max_seqlen_k,
        ),
        attention_kwargs=kwargs,
    )


def test_packed_keys_accepts_a_dense_call():
    from xfuser.core.attention.constraints import PACKED_KEYS

    t = torch.empty((1, 2, 8, 128))
    assert PACKED_KEYS.unmet(t, t, t, AttnCall()) is None


def test_packed_keys_accepts_an_undeclared_pack():
    """Gathering the valid rows attends over exactly those keys, so a pack
    needs no trailing-pad declaration to be served. Krea-2 passes a key
    padding mask and no valid_kv_len, and this is what lets it through."""
    from xfuser.core.attention.constraints import PACKED_KEYS

    t = torch.empty((1, 2, 8, 128))
    assert PACKED_KEYS.unmet(t, t, t, _packed(4)) is None


def test_packed_keys_accepts_a_declared_pad():
    from xfuser.core.attention.constraints import PACKED_KEYS

    t = torch.empty((1, 2, 8, 128))
    assert PACKED_KEYS.unmet(t, t, t, _packed(4, valid_kv_len=4)) is None


def test_packed_keys_checks_a_declared_length():
    """valid_kv_len licenses the cheaper route -- slice a trailing block
    rather than gather -- so a wrong one drops or invents keys."""
    from xfuser.core.attention.constraints import PACKED_KEYS

    t = torch.empty((1, 2, 8, 128))
    assert "valid_kv_len" in PACKED_KEYS.unmet(t, t, t, _packed(9, valid_kv_len=9))
    assert "valid_kv_len" in PACKED_KEYS.unmet(t, t, t, _packed(4, valid_kv_len=0))


def test_packed_keys_requires_the_longest_segment_to_be_the_valid_count():
    """A trailing pad has one run of real keys, so its longest segment is the
    valid count. A disagreement means the pad is not trailing, and slicing it
    would be wrong -- caught here rather than gathered as if declared."""
    from xfuser.core.attention.constraints import PACKED_KEYS

    t = torch.empty((1, 2, 8, 128))
    reason = PACKED_KEYS.unmet(t, t, t, _packed(6, valid_kv_len=4))
    assert reason is not None and "max_seqlen_k" in reason


def test_dense_mha_v4_serves_packed_keys_but_sparge_does_not():
    """Sparge needs the key length padded to its KV tile, which is exactly the
    alignment both a trailing-pad slice and a gather remove."""
    from xfuser.core.attention import registry

    t = torch.empty((1, 2, 8, 128))
    call = _packed(4, valid_kv_len=4)

    dense = registry.get(AttentionBackendType.AITER_BF16)
    assert dense.rejects(t, t, t, call) is None

    sparge = registry.get(AttentionBackendType.AITER_FP8_SPARGE)
    assert sparge.rejects(t, t, t, call) is not None
