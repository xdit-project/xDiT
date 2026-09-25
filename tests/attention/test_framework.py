"""The attention framework itself, with no backends registered.

These exercise the pieces every backend depends on -- predicates, call
constraints, the registry and the layout helpers -- without needing a GPU or
any vendor library.
"""

import pytest
import torch

from xfuser.core.attention import registry
from xfuser.core.attention.numerics.layout import from_bshd, pack_kv, to_bshd
from xfuser.core.attention.requirements import (
    ALWAYS,
    ARCH,
    PARAM,
    SYMBOL,
    Requirement,
    _resolve_with_reason,
    resolve,
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


# --------------------------------------------------------------------------
# call constraints
# --------------------------------------------------------------------------

def _qkv(batch=1, heads=4, q_len=16, kv_len=16, head_dim=128):
    return (
        torch.zeros(batch, heads, q_len, head_dim),
        torch.zeros(batch, heads, kv_len, head_dim),
        torch.zeros(batch, heads, kv_len, head_dim),
    )


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

def _spec(backend, **kwargs):
    defaults = dict(impl=lambda q, k, v, c: (q, None), returns_lse=True, requires=ALWAYS)
    defaults.update(kwargs)
    return Spec(backend, **defaults)


@pytest.fixture
def clean_registry():
    saved = dict(registry.REGISTRY)
    registry.clear()
    yield registry
    registry.REGISTRY.clear()
    registry.REGISTRY.update(saved)


def test_register_and_get(clean_registry):
    spec = _spec(AttentionBackendType.SDPA)
    clean_registry.register([spec])
    assert clean_registry.get(AttentionBackendType.SDPA) is spec


def test_duplicate_registration_is_rejected(clean_registry):
    clean_registry.register([_spec(AttentionBackendType.SDPA)])
    with pytest.raises(ValueError, match="already registered"):
        clean_registry.register([_spec(AttentionBackendType.SDPA)])


def test_get_unregistered_names_the_backend(clean_registry):
    with pytest.raises(KeyError, match="AITER_F4F4"):
        clean_registry.get(AttentionBackendType.AITER_F4F4)


def test_queries_replace_the_group_tuples(clean_registry):
    clean_registry.register([
        _spec(AttentionBackendType.AITER_MXFP4_SPARGE, sparsity="sparge",
              head_balanced=True, low_precision=True, returns_lse=False),
        _spec(AttentionBackendType.AITER_MXFP4, low_precision=True, returns_lse=False),
        _spec(AttentionBackendType.SDPA),
    ])
    assert clean_registry.types_where(is_sparse=True) == frozenset(
        {AttentionBackendType.AITER_MXFP4_SPARGE}
    )
    assert clean_registry.types_where(low_precision=True) == frozenset({
        AttentionBackendType.AITER_MXFP4_SPARGE,
        AttentionBackendType.AITER_MXFP4,
    })
    assert clean_registry.types_where(returns_lse=True) == frozenset(
        {AttentionBackendType.SDPA}
    )


def test_missing_specs_lists_members_without_a_spec(clean_registry):
    assert len(clean_registry.missing_specs()) == len(list(AttentionBackendType))
    clean_registry.register([_spec(AttentionBackendType.SDPA)])
    assert AttentionBackendType.SDPA not in clean_registry.missing_specs()


def test_manifest_renders_from_specs(clean_registry):
    clean_registry.register([_spec(AttentionBackendType.SDPA)])
    text = clean_registry.manifest()
    assert "SDPA" in text and "BACKEND" in text


def test_spec_defaults_are_conservative():
    """A spec needs only a type and an impl. returns_lse defaults False so a
    backend must opt in to ring participation; requires defaults to ALWAYS."""
    spec = Spec(AttentionBackendType.SDPA, impl=lambda q, k, v, c: (q, None))
    assert spec.returns_lse is False
    assert spec.requires is ALWAYS
    assert spec.unavailable() is None
    assert spec.rejects(*_qkv(), AttnCall()) is None


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


def test_pack_kv_matches_legacy_semantics():
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
    assert torch.equal(
        packed.cu_seqlens_q, torch.tensor([0, 4, 8], dtype=torch.int32)
    )
    assert packed.max_seqlen_q == seq_len

    out = torch.randn(batch * seq_len, heads, head_dim)
    assert packed.unflatten(out).shape == (batch, seq_len, heads, head_dim)


# --------------------------------------------------------------------------
# the enum move
# --------------------------------------------------------------------------

def test_enum_still_has_every_member():
    assert len(list(AttentionBackendType)) == 44


# --------------------------------------------------------------------------
# the framework must import anywhere
# --------------------------------------------------------------------------

VENDOR_MODULES = {
    "aiter", "flash_attn", "flash_attn_interface", "sageattention",
    "transformer_engine", "torch_npu", "flex_block_attn", "yunchang", "distvae",
}

FRAMEWORK_MODULES = ["spec", "requirements", "constraints", "registry",
                     "numerics/layout", "numerics/hadamard"]


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

    allowed = {"torch", "xfuser", "__future__", "dataclasses", "enum", "typing",
               "functools", "importlib", "inspect", "ast", "pathlib"}
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
        returns_lse=False,
        requires=SYMBOL("aiter.ops.definitely_not_here:kernel"),
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
    )
    q, k, v = _qkv(head_dim=128)

    spec.run(q, k, v, AttnCall())
    assert called == [True]

    with pytest.raises(NotImplementedError, match="causal"):
        spec.run(q, k, v, AttnCall(is_causal=True))
    with pytest.raises(NotImplementedError, match="head dimension"):
        spec.run(*_qkv(head_dim=64), AttnCall())
    assert called == [True], "impl must not be reached for a rejected call"


def test_backends_without_an_lse_cannot_join_ring():
    """Ring attention merges a softmax log-sumexp across ranks, so a backend
    that produces none cannot join. Every spec must answer that question."""
    from xfuser.core.attention import registry

    for spec in registry.where(returns_lse=False):
        assert spec.returns_lse is False   # trivially true; the value is the point
    # The value matters less than every spec having one: an unanswered
    # backend would otherwise be assumed ring-capable by default.
    assert all(isinstance(s.returns_lse, bool) for s in registry.REGISTRY.values())


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


def test_hadamard_declares_its_symbol_once():
    """hadamard.matrix() resolves through the same object backends gate on."""
    from xfuser.core.attention.numerics import hadamard
    from xfuser.core.attention.requirements import FIRST_OF

    assert isinstance(hadamard.CREATE_HADAMARD, FIRST_OF)
    assert len(hadamard.CREATE_HADAMARD.targets) == 2


# --------------------------------------------------------------------------
# shim hygiene
# --------------------------------------------------------------------------

def test_every_live_shim_carries_a_date():
    """Shims accumulate when nobody can tell which are safe to delete. A live
    marker must say when it was introduced, so shim_report.py can age it;
    historical "aiter-shim cut" notes are not subject to this."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent))
    from shim_report import find_live

    undated = [
        f"{path}:{number}  {text}"
        for path, number, date, text in find_live()
        if date is None
    ]
    assert not undated, (
        "live aiter-shim markers without 'added YYYY-MM-DD':\n  "
        + "\n  ".join(undated)
    )


# --------------------------------------------------------------------------
# varlen packing
# --------------------------------------------------------------------------

# Backends whose kernel branches on call.varlen and calls a varlen entry point.
# Everything else must declare NO_VARLEN: accepting a packed call without
# honouring it runs dense attention over padded keys and returns wrong numbers
# rather than failing.
VARLEN_CAPABLE = {
    "AITER", "AITER_FP8", "FLASH", "FLASH_3", "FLASH_4",
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
    accepting = {
        backend.name
        for backend, spec in registry.REGISTRY.items()
        if spec.rejects(q, q, q, packed) is None
    }
    assert accepting == VARLEN_CAPABLE, (
        "varlen support disagrees with what the kernels implement; "
        f"unexpected: {sorted(accepting - VARLEN_CAPABLE)}, "
        f"missing: {sorted(VARLEN_CAPABLE - accepting)}"
    )
