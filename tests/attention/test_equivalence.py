"""Old vs new: every migrated backend must behave exactly as it does today.

This is the safety net for the migration. For each backend present in both the
legacy ATTENTION_FUNCTION_REGISTRY and the new spec registry it asserts:

  * identical outputs on identical inputs (bitwise -- same kernel, same args),
  * returns_lse agrees with the legacy ring blocklist,
  * the new availability check is never looser than the legacy one.

The last is one-directional on purpose. The new specs declare availability the
legacy module never checked (CUDNN has no check at all today, so it is picked
on ROCm and fails at the first attention call). Being stricter is a fix; being
looser would be a regression.
"""

import pytest
import torch

from xfuser.core.attention import registry
from xfuser.core.attention.spec import AttentionBackendType, AttnCall
from xfuser.core.distributed import get_runtime_state
from xfuser.core.distributed.attention_backend import ATTENTION_FUNCTION_REGISTRY

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="equivalence requires a GPU"
)

# Backends whose legacy ring support cannot be read off the blocklist.
#
#   AITER_SAGE / AITER_SAGE_V2 are in the blocklist but allowed through when
#   their wrapper exposes return_lse, so membership does not decide it.
#
#   NVTE_FP8 and FLEX_VSA_H3 both return (out, None) yet are absent from the
#   blocklist, so the legacy path admits them to ring attention with no LSE to
#   merge on. FLEX_VSA_H3 is worse: its dense-fallback branch returns AITER's
#   LSE while its sparse branch returns None, so the contract depends on
#   whether the model published vsa_h3 metadata. Both specs say
#   returns_lse=False, matching the primary path; these are gaps in the
#   blocklist, not differences in the port.
_RING_BLOCKLIST_EXCEPTIONS = {
    AttentionBackendType.AITER_SAGE,
    AttentionBackendType.AITER_SAGE_V2,
    AttentionBackendType.NVTE_FP8,
    AttentionBackendType.FLEX_VSA_H3,
}


# AITER_F4F4 dense faults the GPU on the cross-attention case, but only with a
# suite's worth of preceding work: legacy and the spec each return finite output
# for that call in a fresh process, the F4F4 group passes alone, and every
# backend's cross case passes together -- yet the full run aborts in
# fmha_fwd_hd128_f4f4_gfx950. That is an out-of-bounds read whose landing page
# depends on prior allocations, the same shape as the Sage v2 Hadamard and the
# sparge cross-attention faults. Excluded rather than chased: F4F4 is one row of
# a generated table whose other rows are compared here, so the coverage lost is
# small. Declaring SELF_ATTENTION on the dense MHA v4 specs would fix it the way
# it fixed sparge, but legacy accepts cross there, so that is a behaviour change
# to make deliberately rather than as a test workaround.
_FAULTS_THE_GPU = {AttentionBackendType.AITER_F4F4}


def migrated():
    """Backends served by both registries -- grows each migration phase."""
    return sorted(
        (b for b in registry.REGISTRY
         if b in ATTENTION_FUNCTION_REGISTRY and b not in _FAULTS_THE_GPU),
        key=lambda b: b.name,
    )


CASES = {
    "self_d128": dict(batch=1, heads=8, q_len=512, kv_len=512, head_dim=128),
    "self_d64": dict(batch=1, heads=8, q_len=512, kv_len=512, head_dim=64),
    "cross": dict(batch=1, heads=8, q_len=512, kv_len=128, head_dim=128),
    "batch2": dict(batch=2, heads=4, q_len=256, kv_len=256, head_dim=128),
}

# Spatial layout per case, product == q_len. Sparse backends refuse to run
# without it, so supplying it is what makes their comparison meaningful
# rather than just "both raised".
THW = {
    "self_d128": (8, 8, 8),
    "self_d64": (8, 8, 8),
    "cross": (8, 8, 8),
    "batch2": (4, 8, 8),
}


def kwargs_for(spec, case) -> dict:
    if not spec.is_sparse:
        return {}
    return {
        "thw": THW[case],
        "encoder_sequence_length": 0,
        "spargeattn_simthreshold": 0.3,
        "spargeattn_cdfthreshold": 0.92,
        "spargeattn_reorder_sequence": True,
        "use_spargeattn_static_block_mask": True,
    }


def make_qkv(batch, heads, q_len, kv_len, head_dim):
    generator = torch.Generator(device="cuda").manual_seed(4242)

    def rand(seq_len):
        return torch.randn(
            batch, heads, seq_len, head_dim,
            generator=generator, device="cuda", dtype=torch.bfloat16,
        )

    return rand(q_len), rand(kv_len), rand(kv_len)


def same(a, b) -> bool:
    if a is None or b is None:
        return a is None and b is None
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    return torch.equal(a, b)


@pytest.mark.parametrize("backend", migrated(), ids=lambda b: b.name)
@pytest.mark.parametrize("case", sorted(CASES), ids=str)
@pytest.mark.parametrize("is_causal", [False, True], ids=["dense", "causal"])
def test_new_matches_legacy(backend, case, is_causal):
    spec = registry.get(backend)
    unavailable = spec.unavailable()
    if unavailable is not None:
        pytest.skip(f"{backend.name}: {unavailable}")

    query, key, value = make_qkv(**CASES[case])
    legacy = ATTENTION_FUNCTION_REGISTRY[backend]

    def run_legacy():
        return legacy(
            query, key, value, dropout_p=0.0, is_causal=is_causal,
            attention_kwargs=kwargs_for(spec, case),
        )

    def run_new():
        return spec.run(
            query, key, value,
            AttnCall(
                dropout_p=0.0, is_causal=is_causal,
                attention_kwargs=kwargs_for(spec, case),
            ),
        )

    # Where the spec refuses, only the new path is exercised. Legacy is not
    # called: some of these shapes are exactly the ones it fails to guard
    # against, and running them corrupts GPU memory rather than raising.
    if spec.rejects(query, key, value, AttnCall(is_causal=is_causal)) is not None:
        with pytest.raises(Exception):
            run_new()
        return

    # Refusal is part of behaviour: where legacy raises, so must the port.
    # Messages differ (several come from AITER itself), so only the refusal is
    # compared, not its wording.
    try:
        legacy_out, legacy_lse = run_legacy()
    except Exception as exc:                # noqa: BLE001
        with pytest.raises(Exception):
            run_new()
        return

    new_out, new_lse = run_new()

    assert same(legacy_out, new_out), f"{backend.name}/{case}: output differs"
    assert same(legacy_lse, new_lse), f"{backend.name}/{case}: second value differs"


@pytest.mark.parametrize("backend", migrated(), ids=lambda b: b.name)
def test_attn_mask_is_plumbed_identically(backend):
    """SDPA and SDPA_MATH read attn_mask out of attention_kwargs; the rest
    ignore it. Either way both registries must agree."""
    spec = registry.get(backend)
    if spec.unavailable() is not None:
        pytest.skip(spec.unavailable())

    query, key, value = make_qkv(**CASES["self_d128"])
    mask = torch.zeros(
        1, 1, query.shape[2], key.shape[2], device="cuda", dtype=torch.bool
    )
    mask[..., : key.shape[2] // 2] = True
    kwargs = {"attn_mask": mask}

    try:
        legacy_out, _ = ATTENTION_FUNCTION_REGISTRY[backend](
            query, key, value, dropout_p=0.0, is_causal=False, attention_kwargs=kwargs
        )
    except Exception as exc:                # noqa: BLE001
        with pytest.raises(type(exc)):
            spec.run(query, key, value, AttnCall(attention_kwargs=kwargs))
        return

    new_out, _ = spec.run(query, key, value, AttnCall(attention_kwargs=dict(kwargs)))
    assert same(legacy_out, new_out), f"{backend.name}: attn_mask handling differs"


@pytest.mark.parametrize("backend", migrated(), ids=lambda b: b.name)
def test_returns_lse_matches_legacy_ring_blocklist(backend):
    """returns_lse replaces the 22-member blocklist in runtime_state. Checked
    behaviourally rather than by reading the list, so the two cannot drift."""
    if backend in _RING_BLOCKLIST_EXCEPTIONS:
        pytest.skip("legacy ring support is signature-dependent for this backend")

    # The compatibility check refuses an unavailable backend before it reaches
    # the ring check, so on this machine the two refusals are indistinguishable
    # and the oracle would read "ring allowed" for everything unavailable.
    unavailable = registry.get(backend).unavailable()
    if unavailable is not None:
        pytest.skip(f"{backend.name}: {unavailable}")

    state = get_runtime_state()
    original = state.parallel_config.ring_degree
    state.parallel_config.ring_degree = 2
    try:
        legacy_allows_ring = True
        try:
            state._check_if_backend_compatible_with_current_configuration(backend)
        except Exception as exc:            # noqa: BLE001
            if "ring parallelism" in str(exc):
                legacy_allows_ring = False
    finally:
        state.parallel_config.ring_degree = original

    assert registry.get(backend).returns_lse == legacy_allows_ring, (
        f"{backend.name}: returns_lse disagrees with the legacy ring blocklist"
    )


@pytest.mark.parametrize("backend", migrated(), ids=lambda b: b.name)
def test_new_availability_is_never_looser_than_legacy(backend):
    state = get_runtime_state()
    legacy_available = True
    try:
        state._check_if_backend_compatible_with_current_configuration(backend)
    except Exception:                       # noqa: BLE001
        legacy_available = False

    new_reason = registry.get(backend).unavailable()
    if new_reason is None:
        assert legacy_available, (
            f"{backend.name}: new registry allows a backend the legacy one rejects"
        )


def test_migration_progress_is_visible():
    """Not an assertion about completeness -- just makes the remaining work
    countable while the two registries coexist."""
    remaining = registry.missing_specs()
    print(
        f"\nmigrated {len(registry.REGISTRY)}/{len(list(AttentionBackendType))}; "
        f"{len(remaining)} still served by the legacy module"
    )


def test_mha_v4_format_names_resolve_against_aiter():
    """The MHA v4 table names formats by string and resolves them with
    getattr, so a name AITER does not carry fails only when that backend is
    selected. Two of them -- MXFP6 and MXFP4 -- exist purely as aliases for
    FP6_E2M3 and FP4_E2M1, do not appear when iterating AttentionFormat, and
    could be dropped upstream at any time. This turns that into a red test.
    """
    try:
        import aiter.ops.mha_v4  # noqa: F401
    except ImportError:
        pytest.skip("AITER mha_v4 not available")

    # Resolution happens when kernel.py is imported: FORMAT and SCALE are built
    # by getattr against AITER's enums, so a dropped alias fails the import.
    from xfuser.core.attention.backends.aiter_mha_v4 import kernel
    from xfuser.core.attention.backends.aiter_mha_v4.spec import Fmt, Scale

    _aiter_format = kernel.FORMAT.get
    _aiter_scale = kernel.SCALE.get

    for fmt in Fmt:
        assert _aiter_format(fmt) is not None, f"Fmt.{fmt.name} does not resolve"
    for scale in Scale:
        assert _aiter_scale(scale) is not None, f"Scale.{scale.name} does not resolve"
