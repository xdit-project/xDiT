"""Generate the GPU validation matrix from what the runners declare.

Hand-authoring cases does not scale and never converges: fifty runners carry a usable load
declaration between them, covering 444 placement-and-quantization combinations before any
hardware profile or rank count multiplies that, against a few dozen cases anyone is willing
to write by hand. Worse, the knowledge is already in the repo, so each hand-written case
restates a runner's LoadDeclaration and can fall out of step with it.

So the enumeration is derived and only two things stay hand-written:

``profiles.json`` holds what cannot be derived, namely what a hardware profile may attempt
and how many ranks a placement gets.

``curated_cases.json`` holds cases whose expected outcome is a claim about the world rather
than about the code: that AITER ships no FP4 kernels for gfx942, that FSDP2 cannot shard
uint8 MXFP4 before torch 2.12. Generating those from the same probe functions the runner
calls would leave the suite asserting only that the code agrees with itself, so they are
written by a person and win any collision with a generated case, which also keeps their
case IDs, and therefore their result history, stable.

Generated cases only ever expect success. That is not circular: the declaration decides
what is worth attempting and the GPU decides whether it works.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROFILES = ROOT / "tests/gpu_validation/profiles.json"
DEFAULT_CURATED = ROOT / "tests/gpu_validation/curated_cases.json"
DEFAULT_OUTPUT = ROOT / "tests/gpu_validation/matrix.json"

PLACEMENT_FOR_MODE = {
    "eager": "eager",
    "fsdp_meta": "fsdp_blockwise",
    "replicated_meta": "replicated",
}
QUANTIZATION_FOR_FORMAT = {
    "none": "none",
    "fp8": "fp8",
    "fp4": "fp4",
    "fp8_fp4": "hybrid_fp8_fp4",
    "int8": "int8",
}
PLACEMENT_SLUG = {
    "eager": "eager",
    "replicated": "replicated",
    "fsdp_blockwise": "fsdp",
}
QUANTIZATION_SLUG = {"none": "bf16"}
# Fields that make two cases the same test, so a curated case can be recognised as already
# covering a generated one. Tags, notes and IDs are deliberately absent. So is the
# accelerator: two tokens that admit any arch in common would both run on that machine, which
# makes the pair redundant rather than distinct, so overlap is compared separately.
IDENTITY_FIELDS = (
    "model",
    "placement",
    "quantization",
    "world_size",
    "te_fp8",
    "offload",
    "transformers",
)
_RDNA4 = frozenset({"gfx1200", "gfx1201"})
_MI3XX = frozenset({"gfx942", "gfx950"})
# Only the archs that appear in the matrix need naming; non_rdna4_rocm is deliberately open
# ended, so it is represented by everything it is known to admit.
_ARCHS_FOR_TOKEN = {
    "gfx942": frozenset({"gfx942"}),
    "gfx950": frozenset({"gfx950"}),
    "gfx942_or_gfx950": _MI3XX,
    "gfx1200_or_gfx1201": _RDNA4,
    "non_rdna4_rocm": _MI3XX | frozenset({"gfx90a"}),
}


def _slug(text: str) -> str:
    return re.sub(r"-+", "-", re.sub(r"[^a-z0-9]+", "-", text.lower())).strip("-")


def load_registry():
    sys.path.insert(0, str(ROOT))
    from xfuser.model_executor.models.runner_models.base_model import MODEL_REGISTRY

    return MODEL_REGISTRY


def declared_variants(declaration) -> list[tuple[str, str]]:
    """Placement and quantization pairs a runner says it can load."""
    placements = sorted(
        PLACEMENT_FOR_MODE[mode.value]
        for mode in declaration.materialization_modes
        if mode.value in PLACEMENT_FOR_MODE
    )
    quantizations = sorted(
        QUANTIZATION_FOR_FORMAT[fmt.value]
        for fmt in declaration.quantization_formats
        if fmt.value in QUANTIZATION_FOR_FORMAT
    )
    return [(p, q) for p in placements for q in quantizations]


def quantizes_its_text_encoder(cls) -> bool:
    """Whether --use_fp8_text_encoder would do anything for this runner.

    A runner with no text-encoder FP8 targets accepts the flag and logs that it has no effect, so
    emitting a te_fp8 case for it would duplicate the plain FP8 case under a name claiming to test
    something it does not.
    """
    settings = getattr(cls, "settings", None)
    return bool(getattr(settings, "fp8_text_encoder_module_list", None))


def screened_variants(plan, declaration, cls) -> list[dict]:
    """The plan's cases this runner declares support for, with te_fp8 dropped where it is inert."""
    declared = set(declared_variants(declaration))
    quantizes_te = quantizes_its_text_encoder(cls)
    chosen: list[dict] = []
    seen: set[tuple[str, str, bool]] = set()
    for entry in plan["cases"]:
        placement, quantization = entry["placement"], entry["quantization"]
        if (placement, quantization) not in declared:
            continue
        te_fp8 = bool(entry["te_fp8"]) and quantizes_te
        identity = (placement, quantization, te_fp8)
        if identity in seen:
            continue
        seen.add(identity)
        chosen.append(
            {"placement": placement, "quantization": quantization, "te_fp8": te_fp8}
        )
    return chosen


def canonical_runners(registry) -> list[tuple[str, type]]:
    """One alias per runner class, matching how cache_inventory reports them.

    The registry maps 83 names onto far fewer classes because checkpoint ids are registered
    as aliases beside the display name, and generating a case per alias would emit several
    identical tests for one runner.
    """
    by_class: dict[str, tuple[str, type]] = {}
    for alias, cls in registry.items():
        by_class.setdefault(cls.__name__, (alias, cls))
    return [by_class[name] for name in sorted(by_class)]


def resolve_checkpoint(alias: str, cls) -> str | None:
    """The repo this runner would load, whether or not it is on this disk.

    Deliberately blind to the cache. The matrix is a plan, and which weights a given machine
    happens to hold is an execution-time fact: filtering on it here would make the generated
    file differ per node, so regenerating anywhere else would churn the diff, invalidate
    --check and make case IDs come and go, taking their result history with them.
    """
    sys.path.insert(0, str(ROOT / "tools"))
    import cache_inventory

    candidates = cache_inventory.rank_for_alias(
        cache_inventory.candidate_repos(cls), alias
    )
    return candidates[0] if candidates else None


def cache_state(alias: str, cls, hub: str, cached: dict[str, str]) -> str | None:
    """The cached repo with real weights for this runner, or None. Reporting only."""
    import os

    sys.path.insert(0, str(ROOT / "tools"))
    import cache_inventory

    for candidate in cache_inventory.rank_for_alias(
        cache_inventory.candidate_repos(cls), alias
    ):
        entry = cached.get(candidate)
        if entry is None:
            continue
        # A metadata-only fetch sits in the cache without weights, which is how the
        # FLUX.2-klein-4B stub looks; sizing it apart keeps it from counting as present.
        if cache_inventory.directory_size(os.path.join(hub, entry)).endswith(("K", "M")):
            continue
        return candidate
    return None


def case_identity(case: dict) -> tuple:
    return tuple(case[field] for field in IDENTITY_FIELDS)


def accelerators_overlap(left: str, right: str) -> bool:
    """Whether two accelerator tokens admit any architecture in common.

    CUDA tokens resolve to nothing here, so they never collide with a ROCm profile.
    """
    return bool(
        _ARCHS_FOR_TOKEN.get(left, frozenset())
        & _ARCHS_FOR_TOKEN.get(right, frozenset())
    )


def generate_cases(registry, profiles: dict) -> tuple[list, list]:
    """Cases the profiles permit, plus a reason for every runner left out."""
    needs_input = set(profiles["requires_input_image"]["models"])
    cases: list[dict] = []
    skipped: list[dict] = []

    for profile in profiles["profiles"]:
        allowed = set(profile["quantizations"])
        # A profile's own rank counts win, so hardware that ships eight devices is not
        # measured at a rank count nobody deploys.
        policy = profile.get("world_size_policy", profiles["world_size_policy"])
        for alias, cls in canonical_runners(registry):
            declaration = getattr(cls, "load_declaration", None)
            if declaration is None:
                skipped.append({"model": alias, "reason": "no load declaration"})
                continue
            if getattr(declaration, "unsupported_reason", None):
                skipped.append(
                    {
                        "model": alias,
                        "reason": f"declared unsupported: {declaration.unsupported_reason}",
                    }
                )
                continue
            if alias in needs_input:
                skipped.append({"model": alias, "reason": "needs an input image"})
                continue
            checkpoint = resolve_checkpoint(alias, cls)
            if checkpoint is None:
                skipped.append({"model": alias, "reason": "no checkpoint id in the source"})
                continue

            variants = [
                variant
                for variant in screened_variants(
                    profile["screening_plan"], declaration, cls
                )
                if variant["quantization"] in allowed and variant["placement"] in policy
            ]
            if not variants:
                skipped.append(
                    {
                        "model": alias,
                        "reason": "declares nothing this profile may attempt",
                    }
                )
                continue
            family = _slug(cls.__module__.rsplit(".", 1)[-1])
            for variant in variants:
                placement = variant["placement"]
                quantization = variant["quantization"]
                te_fp8 = variant["te_fp8"]
                world_size = policy[placement]
                quant_slug = QUANTIZATION_SLUG.get(quantization, quantization)
                case_id = "-".join(
                    [
                        "gen",
                        profile["id"],
                        _slug(alias),
                        quant_slug,
                        *(["te"] if te_fp8 else []),
                        PLACEMENT_SLUG[placement],
                        f"w{world_size}",
                    ]
                )
                cases.append(
                    {
                        "id": case_id,
                        "tags": sorted(
                            {
                                *profile["tags"],
                                quant_slug,
                                PLACEMENT_SLUG[placement],
                                *(["te-fp8"] if te_fp8 else []),
                            }
                        ),
                        "model": alias,
                        "model_family": family,
                        "hardware": {
                            "backend": profile["backend"],
                            "accelerator": profile["accelerator"],
                        },
                        "placement": placement,
                        "quantization": quantization,
                        "te_fp8": te_fp8,
                        "offload": "none",
                        "transformers": profile["transformers"],
                        "checkpoint": {"source": "hub", "value": checkpoint},
                        "world_size": world_size,
                        "args": list(profile.get("runtime_args", [])),
                        "expected": {"outcome": "inference_success"},
                        "quality_notes": (
                            f"Generated from the {alias} load declaration. Compare the "
                            "output against the same model's eager bf16 case at the same "
                            "seed before trusting the quantized or sharded result."
                        ),
                    }
                )
    return cases, skipped


def merge_cases(curated: list[dict], generated: list[dict]) -> tuple[list[dict], list[dict]]:
    """Curated cases win, so their IDs and result history survive regeneration."""
    taken: dict[tuple, list[str]] = {}
    for case in curated:
        taken.setdefault(case_identity(case), []).append(
            case["hardware"]["accelerator"]
        )
    kept, superseded = [], []
    for case in generated:
        identity = case_identity(case)
        token = case["hardware"]["accelerator"]
        if any(
            accelerators_overlap(token, existing)
            for existing in taken.get(identity, ())
        ):
            superseded.append(case)
            continue
        taken.setdefault(identity, []).append(token)
        kept.append(case)
    return kept, superseded


def build_matrix(profiles_path: Path, curated_path: Path) -> tuple[dict, dict]:
    profiles = json.loads(profiles_path.read_text())
    curated_doc = json.loads(curated_path.read_text())
    registry = load_registry()
    generated, skipped = generate_cases(registry, profiles)
    kept, superseded = merge_cases(curated_doc["cases"], generated)

    matrix = {
        "schema_version": 2,
        "validation_status": "NOT RUN",
        "description": curated_doc["description"],
        "generated_by": "tools/generate_validation_matrix.py",
        "defaults": curated_doc["defaults"],
        "cases": curated_doc["cases"] + kept,
    }
    sys.path.insert(0, str(ROOT / "tools"))
    import cache_inventory

    hub = cache_inventory.default_hub()
    cached_repos = cache_inventory.cached_repos(hub)
    by_alias = {alias: cls for alias, cls in canonical_runners(registry)}
    planned_models = sorted({case["model"] for case in kept})
    absent = [
        model
        for model in planned_models
        if model in by_alias
        and cache_state(model, by_alias[model], hub, cached_repos) is None
    ]
    return matrix, {
        "generated": len(generated),
        "kept": len(kept),
        "superseded_by_curated": [case["id"] for case in superseded],
        "curated": len(curated_doc["cases"]),
        "skipped": skipped,
        "planned_models": planned_models,
        "models_without_local_weights": absent,
    }


def render_summary(report: dict) -> str:
    lines = [
        f"curated cases kept: {report['curated']}",
        f"generated cases added: {report['kept']} of {report['generated']} produced",
    ]
    if report["superseded_by_curated"]:
        lines.append(
            f"{len(report['superseded_by_curated'])} generated case(s) already covered "
            "by a curated case"
        )
    absent = report["models_without_local_weights"]
    lines.append(
        f"{len(report['planned_models']) - len(absent)} of "
        f"{len(report['planned_models'])} planned models have weights on this machine"
        + (f"; missing: {', '.join(absent)}" if absent else "")
    )
    reasons: dict[str, list[str]] = {}
    for entry in report["skipped"]:
        reasons.setdefault(entry["reason"], []).append(entry["model"])
    if reasons:
        lines.append("runners with no generated case:")
        for reason, models in sorted(reasons.items()):
            lines.append(f"  {reason}: {len(models)}")
            for model in sorted(models):
                lines.append(f"    {model}")
    return "\n".join(lines)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--profiles", type=Path, default=DEFAULT_PROFILES)
    parser.add_argument("--curated", type=Path, default=DEFAULT_CURATED)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail if the checked-in matrix differs from what the declarations imply",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    matrix, report = build_matrix(args.profiles, args.curated)

    sys.path.insert(0, str(ROOT / "tools"))
    import gpu_validation

    gpu_validation.validate_matrix(matrix)
    text = json.dumps(matrix, indent=2) + "\n"

    if args.check:
        if not args.output.exists() or args.output.read_text() != text:
            print(
                f"{args.output} is stale; regenerate with "
                "python tools/generate_validation_matrix.py",
                file=sys.stderr,
            )
            return 1
        print("matrix is up to date")
        return 0

    args.output.write_text(text)
    print(render_summary(report))
    print(f"wrote {len(matrix['cases'])} cases to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
