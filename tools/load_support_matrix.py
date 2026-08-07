#!/usr/bin/env python3
"""Report which runner models support which load features, and where testing does not reach.

Three facts decide whether a model's memory-efficient load is trustworthy, and they are kept in
three different places:

* what the runner *declares* it can construct before weights are allocated (``LoadCapability`` on
  the runner class),
* what the validation matrix *plans* to exercise (``tests/gpu_validation/matrix.json``),
* what has actually *run* on hardware (results recorded by ``tools/gpu_validation.py``).

A model can declare a capability nothing ever exercises, or carry a matrix case nobody has run, and
either way it looks supported from one angle and untested from another. This joins the three so the
gap is a report rather than something to reconstruct by hand.

Capability is read by importing the runners rather than by parsing them: quantization support is
derived from each runner's ModelCapabilities at class-decoration time, so it does not exist in the
source text. That makes this a heavier tool than its dependency-light sibling; run it where xfuser
imports.
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MATRIX = ROOT / "tests/gpu_validation/matrix.json"


def load_runners() -> dict[type, list[str]]:
    """Registered names grouped by the runner class that serves them."""
    from xfuser.model_executor.models.runner_models.base_model import MODEL_REGISTRY

    names_by_class: dict[type, list[str]] = collections.defaultdict(list)
    for name, runner in MODEL_REGISTRY.items():
        names_by_class[runner].append(name)
    return names_by_class


def capability_row(runner, names: list[str]) -> dict[str, Any]:
    from xfuser.model_executor.models.runner_models.loading.contracts import (
        MaterializationMode,
    )

    capability = runner.load_capability
    modes = capability.materialization_modes
    formats = sorted({fmt.value for fmt, _ in capability.quantization_contracts} - {"none"})
    backends = sorted(
        {backend.value for _, backend in capability.quantization_contracts} - {"none"}
    )
    return {
        "runner": runner.__name__,
        # Shortest registered alias: the matrix and CI configs use the bare name, not the hub id.
        "model": min(names, key=len),
        "aliases": sorted(names),
        "eager": MaterializationMode.EAGER in modes,
        "fsdp_meta": MaterializationMode.FSDP_META in modes,
        "replicated_meta": MaterializationMode.REPLICATED_META in modes,
        "loader_adapter": capability.loader_adapter.value,
        "quantization_formats": formats,
        "quantization_backends": backends,
        "excluded_components": {
            exclusion.component: exclusion.reason
            for exclusion in capability.component_exclusions
        },
        "withheld_reason": capability.unsupported_reason,
    }


def matrix_coverage(matrix_path: Path) -> dict[str, list[dict]]:
    """Matrix cases grouped by the model name they name."""
    cases = json.loads(matrix_path.read_text())["cases"]
    by_model: dict[str, list[dict]] = collections.defaultdict(list)
    for case in cases:
        by_model[case["model"]].append(case)
    return by_model


def executed_cases(results_path: Path | None) -> dict[str, str]:
    """Case id -> status of its most recent recorded run.

    Latest wins rather than any-failure-sticks: a case that failed, was fixed, and passed is
    passing, and reporting it as broken would send someone chasing a resolved failure.
    """
    if results_path is None or not results_path.exists():
        return {}
    latest: dict[str, tuple[str, str]] = {}
    for line in results_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        when = record.get("recorded_at", "")
        case_id = record["case_id"]
        if case_id not in latest or when >= latest[case_id][0]:
            latest[case_id] = (when, record.get("status", "unknown"))
    return {case_id: status for case_id, (_, status) in latest.items()}


def build_report(
    matrix_path: Path,
    results_path: Path | None,
    *,
    capability_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Join declared capability, planned matrix cases, and recorded runs.

    capability_rows is injectable so the join can be exercised without importing the runners.
    """
    if capability_rows is None:
        names_by_class = load_runners()
        capability_rows = [
            capability_row(runner, names)
            for runner, names in sorted(
                names_by_class.items(), key=lambda kv: kv[0].__name__
            )
        ]
    by_model = matrix_coverage(matrix_path)
    statuses = executed_cases(results_path)

    rows = []
    for row in capability_rows:
        names = row["aliases"]
        cases = [case for name in names for case in by_model.get(name, ())]
        row["placements"] = sorted({case["placement"] for case in cases})
        row["case_ids"] = sorted(case["id"] for case in cases)
        row["executed"] = sorted(
            case_id for case_id in row["case_ids"] if case_id in statuses
        )
        # gpu_validation records success as "passed" and "passed_expected_rejection"; a case whose
        # whole point is to be refused passes by being refused.
        row["failed"] = sorted(
            case_id
            for case_id in row["executed"]
            if not statuses[case_id].startswith("passed")
        )
        rows.append(row)

    unregistered = sorted(
        model
        for model in by_model
        if not any(model in row["aliases"] for row in rows)
    )
    return {"models": rows, "matrix_models_not_registered": unregistered}


def _memory_efficient(row) -> bool:
    return row["fsdp_meta"] or row["replicated_meta"]


def render_markdown(report: dict[str, Any]) -> str:
    rows = report["models"]
    lines = [
        "# Runner model load support",
        "",
        "Generated by `tools/load_support_matrix.py`. Declared columns come from each runner's",
        "`LoadCapability`; coverage columns come from the GPU validation matrix and recorded runs.",
        "",
        "| model | eager | FSDP meta | replicated meta | quantization | placements tested | run |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    mark = {True: "yes", False: "no"}
    for row in rows:
        placements = ", ".join(row["placements"]) or "none"
        run = f"{len(row['executed'])}/{len(row['case_ids'])}" if row["case_ids"] else "none"
        lines.append(
            f"| {row['model']} | {mark[row['eager']]} | {mark[row['fsdp_meta']]} | "
            f"{mark[row['replicated_meta']]} | "
            f"{', '.join(row['quantization_formats']) or 'none'} | {placements} | {run} |"
        )

    withheld = [row for row in rows if not _memory_efficient(row)]
    if withheld:
        lines += ["", "## Withheld from memory-efficient load", ""]
        for row in withheld:
            lines.append(f"- **{row['model']}** ({row['loader_adapter']}): {row['withheld_reason']}")

    excluded = [row for row in rows if row["excluded_components"]]
    if excluded:
        lines += ["", "## Components excluded from meta construction", ""]
        for row in excluded:
            for component, reason in sorted(row["excluded_components"].items()):
                lines.append(f"- **{row['model']}** `{component}`: {reason}")
    return "\n".join(lines) + "\n"


def render_gaps(report: dict[str, Any]) -> str:
    rows = report["models"]
    capable = [row for row in rows if _memory_efficient(row)]
    uncovered = [row for row in capable if not row["case_ids"]]
    eager_only = [
        row
        for row in capable
        if row["case_ids"] and row["placements"] == ["eager"]
    ]
    unrun = [row for row in capable if row["case_ids"] and not row["executed"]]
    unasserted = [
        row for row in rows if not _memory_efficient(row) and not row["case_ids"]
    ]
    failed = [row for row in rows if row["failed"]]

    lines = [
        f"{len(rows)} runner models, {len(capable)} declaring a memory-efficient load.",
        "",
    ]

    def section(title, entries, render):
        lines.append(f"{len(entries)} {title}")
        for row in entries:
            lines.append(f"  {row['model']:26s} {render(row)}")
        lines.append("")

    section(
        "capable models with no matrix case at all:",
        uncovered,
        lambda row: f"({row['runner']})",
    )
    section(
        "capable models exercised only in eager placement:",
        eager_only,
        lambda row: ", ".join(row["case_ids"]),
    )
    section(
        "capable models whose cases have never been run:",
        unrun,
        lambda row: ", ".join(row["case_ids"]),
    )
    section(
        "withheld models with no case asserting the rejection:",
        unasserted,
        lambda row: f"({row['loader_adapter']})",
    )
    if failed:
        section("models with a recorded failure:", failed, lambda row: ", ".join(row["failed"]))
    if report["matrix_models_not_registered"]:
        lines.append(
            f"matrix names absent from the registry: "
            f"{', '.join(report['matrix_models_not_registered'])}"
        )
    return "\n".join(lines)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument(
        "--results",
        type=Path,
        default=None,
        help="JSONL written by tools/gpu_validation.py, to mark cases as actually run",
    )
    parser.add_argument(
        "--format",
        choices=("markdown", "json", "gaps"),
        default="gaps",
        help="gaps (default) lists what is untested; markdown emits the support table",
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = build_report(args.matrix, args.results)
    if args.format == "json":
        text = json.dumps(report, indent=2) + "\n"
    elif args.format == "markdown":
        text = render_markdown(report)
    else:
        text = render_gaps(report) + "\n"
    if args.output:
        args.output.write_text(text)
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
