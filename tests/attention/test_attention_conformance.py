"""Characterise every registered attention backend.

A characterisation harness, not a correctness gate: for each backend it
records the deviation from an fp32 SDPA reference and writes a snapshot.
Sparse backends deviate a lot by design, so nothing here asserts a quality
threshold -- the snapshot is what later runs are compared against.

Run standalone for the report:

    python -m pytest tests/attention/test_attention_conformance.py -s
    python tests/attention/test_attention_conformance.py --update   # write snapshot
    python tests/attention/test_attention_conformance.py --check     # diff vs snapshot

Requires a GPU. Backends unavailable in this environment are skipped with the
reason reported by the runtime state's own compatibility check.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

from xfuser.config.args import xFuserArgs
from xfuser.core.distributed import (
    init_distributed_environment,
    initialize_model_parallel,
    initialize_runtime_state,
    get_runtime_state,
)
from xfuser.core.attention import registry
from xfuser.core.attention.spec import AttentionBackendType, AttnCall
SNAPSHOT = Path(__file__).with_name("conformance_snapshot.json")

# Relative-error drift above this between runs is worth a human look.
DRIFT_TOLERANCE = 0.05


# --------------------------------------------------------------------------
# environment
# --------------------------------------------------------------------------

def init_environment() -> None:
    """Single-rank distributed + runtime state. Idempotent, so this works both
    under pytest (where conftest may already have done it) and as a script."""
    from pathlib import Path
    import sys

    sys.path.insert(0, str(Path(__file__).parent))
    from conftest import ensure_attention_environment

    ensure_attention_environment()


def _arg_default(name: str, fallback):
    field = xFuserArgs.__dataclass_fields__.get(name)
    return fallback if field is None else field.default


# --------------------------------------------------------------------------
# cases
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Case:
    name: str
    batch: int
    heads: int
    q_len: int
    kv_len: int
    head_dim: int
    is_causal: bool = False
    thw: Optional[tuple] = None     # spatial layout; product must equal q_len

    @property
    def is_self_attention(self) -> bool:
        return self.q_len == self.kv_len


CASES = [
    # The workhorse shape: Wan-like self attention, head_dim 128.
    Case("self_d128", 1, 24, 4096, 4096, 128, thw=(16, 16, 16)),
    # head_dim 64 -- FlyDSL accepts it, MHA v4 Sparge does not.
    Case("self_d64", 1, 16, 4096, 4096, 64, thw=(16, 16, 16)),
    # Text cross attention: VSA and FlyDSL route around this.
    Case("cross_d128", 1, 24, 4096, 512, 128),
    # Causal: several backends refuse it outright.
    Case("causal_d128", 1, 16, 2048, 2048, 128, is_causal=True),
    # Below the FlyDSL fp8 sequence crossover (2560 / 3584).
    Case("short_d128", 1, 8, 1024, 1024, 128, thw=(4, 16, 16)),
    # Batch > 1.
    Case("batch2_d128", 2, 8, 2048, 2048, 128, thw=(8, 16, 16)),
]


def make_tensors(case: Case, device: str, dtype: torch.dtype):
    """BHSD, matching what USP hands the backends."""
    generator = torch.Generator(device=device).manual_seed(1234)

    def rand(seq_len):
        return torch.randn(
            case.batch, case.heads, seq_len, case.head_dim,
            generator=generator, device=device, dtype=dtype,
        )

    return rand(case.q_len), rand(case.kv_len), rand(case.kv_len)


# --------------------------------------------------------------------------
# per-family attention_kwargs
# --------------------------------------------------------------------------

_SPARGE_BACKENDS = frozenset({
    AttentionBackendType.AITER_SPARGE,
    AttentionBackendType.AITER_SPARGE_V2,
    AttentionBackendType.FLEX_BLOCK_SPARGE,
}) | registry.types_where(sparsity="sparge")

# SSTA kwargs are supplied by a model's downloaded sparse config
# (see hunyuan.py: sparse_config["attn_param"]), not by CLI args, so these
# cannot be synthesised standalone.
_SSTA_BACKENDS = frozenset({
    AttentionBackendType.FLEX_BLOCK_ATTN,
    AttentionBackendType.AITER_SPARSE_SAGE,
    AttentionBackendType.AITER_SPARSE_SAGE_V2,
})

_VSA_BACKENDS = frozenset({
    AttentionBackendType.AITER_VSA,
    AttentionBackendType.FLEX_VSA_H3,
})


def build_attention_kwargs(backend: AttentionBackendType, case: Case) -> Optional[dict]:
    """Kwargs for this backend family, or None when the case cannot be built."""
    if backend in _SSTA_BACKENDS:
        return None

    if backend in _SPARGE_BACKENDS:
        if case.thw is None:
            return None
        return {
            "thw": case.thw,
            "encoder_sequence_length": 0,
            "spargeattn_simthreshold": _arg_default("spargeattn_simthreshold", 0.3),
            "spargeattn_cdfthreshold": _arg_default("spargeattn_cdfthreshold", 0.92),
            "spargeattn_reorder_sequence": _arg_default("spargeattn_reorder_sequence", True),
            "use_spargeattn_static_block_mask": _arg_default(
                "use_spargeattn_static_block_mask", True
            ),
        }

    if backend in _VSA_BACKENDS:
        if case.thw is None:
            return None
        return {
            "thw": case.thw,
            "vsa_block_size": _arg_default("vsa_block_size", 128),
            "vsa_top_k": _arg_default("vsa_top_k", 1),
            "vsa_top_k_ratio": _arg_default("vsa_top_k_ratio", 0.0),
            "vsa_prob_threshold": _arg_default("vsa_prob_threshold", 0.9),
            "vsa_reorder_sequence": _arg_default("vsa_reorder_sequence", True),
            "use_vsa_static_block_mask": _arg_default("use_vsa_static_block_mask", True),
            "use_vsa_first_frame_mask": _arg_default("use_vsa_first_frame_mask", True),
            "vsa_collect_density": False,
            "vsa_drop_rates": None,
        }

    return {}


# --------------------------------------------------------------------------
# measurement
# --------------------------------------------------------------------------

def reference(query, key, value, is_causal: bool):
    """fp32 SDPA. Upcast so the reference is not itself the error source."""
    return F.scaled_dot_product_attention(
        query.float(), key.float(), value.float(), is_causal=is_causal
    )


def deviation(actual, expected) -> dict:
    actual = actual.float()
    diff = actual - expected
    denom = expected.norm().item() or 1.0
    return {
        "max_abs": diff.abs().max().item(),
        "rel_l2": (diff.norm().item() / denom),
    }


def availability_error(backend: AttentionBackendType) -> Optional[str]:
    """The runtime state's own compatibility check, reused as the skip reason."""
    state = get_runtime_state()
    try:
        state._check_if_backend_compatible_with_current_configuration(backend)
    except Exception as exc:                       # noqa: BLE001 - reporting only
        return f"{type(exc).__name__}: {exc}"
    return None


def run_case(backend: AttentionBackendType, case: Case, device: str, dtype) -> dict:
    attention_kwargs = build_attention_kwargs(backend, case)
    if attention_kwargs is None:
        return {"status": "skip", "reason": "no synthesisable attention_kwargs"}

    query, key, value = make_tensors(case, device, dtype)
    expected = reference(query, key, value, case.is_causal)

    spec = registry.get(backend)
    try:
        output, _lse = spec.run(
            query, key, value,
            AttnCall(
                dropout_p=0.0,
                is_causal=case.is_causal,
                attention_kwargs=attention_kwargs,
            ),
        )
    except Exception as exc:                       # noqa: BLE001 - reporting only
        return {"status": "error", "reason": f"{type(exc).__name__}: {exc}"}

    if output.shape != expected.shape:
        return {
            "status": "error",
            "reason": f"shape {tuple(output.shape)} != {tuple(expected.shape)}",
        }
    if not torch.isfinite(output).all():
        return {"status": "error", "reason": "non-finite output"}

    return {"status": "ok", **deviation(output, expected)}


def run_all(device: str = "cuda", dtype=torch.bfloat16) -> dict:
    results: dict = {}
    for backend in sorted(registry.REGISTRY, key=lambda b: b.name):
        unavailable = availability_error(backend)
        if unavailable is not None:
            results[backend.name] = {
                case.name: {"status": "skip", "reason": unavailable} for case in CASES
            }
            continue
        results[backend.name] = {
            case.name: run_case(backend, case, device, dtype) for case in CASES
        }
    return results


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------

def render(results: dict) -> str:
    lines = []
    width = max(len(name) for name in results) if results else 10
    header = "BACKEND".ljust(width) + "  " + "  ".join(c.name.rjust(12) for c in CASES)
    lines.append(header)
    lines.append("-" * len(header))

    for name, cases in results.items():
        cells = []
        for case in CASES:
            entry = cases.get(case.name, {})
            status = entry.get("status")
            if status == "ok":
                cells.append(f"{entry['rel_l2']:.2e}".rjust(12))
            elif status == "skip":
                cells.append("skip".rjust(12))
            else:
                cells.append("ERROR".rjust(12))
        lines.append(name.ljust(width) + "  " + "  ".join(cells))

    lines.append("")
    lines.append("rel_l2 vs fp32 SDPA. Sparse backends deviate by design.")

    notes = []
    for name, cases in results.items():
        for case_name, entry in cases.items():
            if entry.get("status") in ("skip", "error"):
                notes.append((name, case_name, entry["status"], entry.get("reason", "")))
    if notes:
        lines.append("")
        lines.append("Skips and errors:")
        seen = set()
        for name, case_name, status, reason in notes:
            key = (name, reason)
            if key in seen:
                continue
            seen.add(key)
            lines.append(f"  {status.upper():5}  {name} [{case_name}]: {reason}")
    return "\n".join(lines)


def compare(results: dict, snapshot: dict) -> list:
    drifts = []
    for name, cases in results.items():
        for case_name, entry in cases.items():
            before = snapshot.get(name, {}).get(case_name, {})
            if entry.get("status") != before.get("status"):
                drifts.append(
                    f"{name}[{case_name}]: status {before.get('status')} -> {entry.get('status')}"
                )
            elif entry.get("status") == "ok":
                old, new = before.get("rel_l2", 0.0), entry["rel_l2"]
                if abs(new - old) > DRIFT_TOLERANCE * max(old, 1e-6):
                    drifts.append(f"{name}[{case_name}]: rel_l2 {old:.3e} -> {new:.3e}")
    return drifts


# --------------------------------------------------------------------------
# entry points
# --------------------------------------------------------------------------

def test_attention_conformance_baseline():
    """Smoke test: every backend either runs, skips with a reason, or is reported."""
    if not torch.cuda.is_available():
        import pytest
        pytest.skip("conformance baseline requires a GPU")

    init_environment()
    results = run_all()
    print("\n" + render(results))

    ran = [
        (b, c) for b, cases in results.items()
        for c, e in cases.items() if e["status"] == "ok"
    ]
    assert ran, "no backend produced a result; the harness itself is broken"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--update", action="store_true", help="write the snapshot")
    parser.add_argument("--check", action="store_true", help="diff against the snapshot")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)

    init_environment()
    results = run_all(device=args.device)
    print(render(results))

    if args.update:
        SNAPSHOT.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
        print(f"\nwrote {SNAPSHOT}")

    from conftest import teardown_attention_environment

    if args.check:
        if not SNAPSHOT.exists():
            print(f"\nno snapshot at {SNAPSHOT}; run with --update first")
            return 1
        drifts = compare(results, json.loads(SNAPSHOT.read_text()))
        if drifts:
            print("\nDrift vs snapshot:")
            for line in drifts:
                print(f"  {line}")
            teardown_attention_environment()
            return 1
        print("\nno drift vs snapshot")
    teardown_attention_environment()
    return 0


if __name__ == "__main__":
    sys.exit(main())
