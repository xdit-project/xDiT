"""Render GPU validation results as a report a person can read at a glance.

``load_support_matrix.py`` answers what each runner *declares* and how many planned cases
exist. This answers what actually happened on one machine: did it pass, how long did the
load take, how much host and device memory did it peak at, and which combinations for a
model are still untested on this hardware.

Reads the JSONL written by ``tools/gpu_validation.py``. Every number here is recorded by
that runner; nothing is inferred except the post-load column, which is wall minus load and
therefore covers inference plus VAE decode, saving and teardown rather than inference alone.
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MATRIX = ROOT / "tests/gpu_validation/matrix.json"

GREEN_STATUSES = {"passed", "passed_expected_rejection"}
NOT_RUN_STATUSES = {"environment_mismatch"}


def load_records(paths: list[Path]) -> list[dict]:
    """Latest record per case id, so a re-run supersedes an earlier attempt."""
    by_case: dict[str, dict] = {}
    for path in paths:
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            case_id = record.get("case_id")
            if not case_id:
                continue
            previous = by_case.get(case_id)
            if previous is None or record.get("recorded_at", "") >= previous.get(
                "recorded_at", ""
            ):
                by_case[case_id] = record
    return list(by_case.values())


def _seconds(value) -> str:
    """One unit for the whole column, so loads of very different cost stay comparable."""
    if value is None:
        return "-"
    return f"{value:.1f}"


def _gib(value) -> str:
    if value in (None, 0):
        return "-"
    return f"{value / (1024 ** 3):.1f}G"


def _combination(case: dict) -> str:
    """Label carrying every dimension that distinguishes cases for the same model.

    Placement, quantization and world size are not enough on their own: Wan2.2-I2V has
    three eager/fp8 cases separated only by rank count and whether the text encoder is
    quantized too, and two RDNA4 FLUX.2-dev cases differ only in offload and transformers
    major. Anything that varies has to appear, or a row silently stands in for another.
    """
    label = f"{case['placement']}/{case['quantization']}/w{case['world_size']}"
    extras = []
    if case.get("te_fp8"):
        extras.append("te")
    if case.get("offload", "none") != "none":
        extras.append(f"offload={case['offload']}")
    if case.get("transformers") != "5.x":
        extras.append(f"tf{case['transformers']}")
    return "+".join([label, *extras])


def _post_load(metrics: dict):
    wall = metrics.get("wall_duration_seconds")
    load = metrics.get("load_duration_seconds")
    if wall is None or load is None:
        return None
    return max(wall - load, 0.0)


def _table(rows: list[list[str]], headers: list[str]) -> list[str]:
    widths = [
        max(len(headers[i]), max((len(r[i]) for r in rows), default=0))
        for i in range(len(headers))
    ]
    numeric = set(range(4, len(headers)))

    def fmt(cells: list[str]) -> str:
        out = []
        for i, cell in enumerate(cells):
            out.append(cell.rjust(widths[i]) if i in numeric else cell.ljust(widths[i]))
        return "  ".join(out).rstrip()

    lines = [fmt(headers), "  ".join("-" * w for w in widths)]
    lines.extend(fmt(row) for row in rows)
    return lines


def describe_environment(records: list[dict]) -> list[str]:
    lines = []
    environments = {
        json.dumps(
            {
                "platform": r.get("environment", {}).get("platform"),
                "accelerators": sorted(
                    set(r.get("environment", {}).get("accelerators") or [])
                ),
                "versions": r.get("environment", {}).get("versions"),
            },
            sort_keys=True,
        )
        for r in records
    }
    for blob in sorted(environments):
        env = json.loads(blob)
        accelerators = ", ".join(env["accelerators"]) or "unknown"
        lines.append(f"  platform {env['platform'] or 'unknown'} on {accelerators}")
        versions = env.get("versions") or {}
        if versions:
            shown = ", ".join(
                f"{name} {value}"
                for name, value in sorted(versions.items())
                if value is not None
            )
            if shown:
                lines.append(f"  {shown}")
    return lines


def build_report(results: list[Path], matrix_path: Path) -> dict:
    records = load_records(results)
    matrix = json.loads(matrix_path.read_text())
    planned = {case["id"]: case for case in matrix["cases"]}

    ran = [r for r in records if r.get("status") not in NOT_RUN_STATUSES]
    skipped = [r for r in records if r.get("status") in NOT_RUN_STATUSES]
    green = [r for r in ran if r["status"] in GREEN_STATUSES]
    failed = [r for r in ran if r["status"] not in GREEN_STATUSES]

    accelerators = sorted(
        {
            accelerator
            for r in records
            for accelerator in (r.get("environment", {}).get("accelerators") or [])
        }
    )
    executed = {r["case_id"] for r in ran}
    relevant = [
        case
        for case in planned.values()
        if _case_targets(case, set(accelerators)) or case["id"] in executed
    ]
    return {
        "records": records,
        "ran": ran,
        "skipped": skipped,
        "green": green,
        "failed": failed,
        "planned": planned,
        "relevant": relevant,
        "accelerators": accelerators,
    }


def render(report: dict, *, markdown: bool = False) -> str:
    ran, failed, green = report["ran"], report["failed"], report["green"]
    planned, skipped = report["planned"], report["skipped"]
    fence = "```" if markdown else None
    lines: list[str] = []
    title = "# GPU validation results" if markdown else "GPU validation results"
    lines.append(title)
    lines.append("")

    if not ran:
        lines.append(
            "No case has been run. Point --results at the JSONL that "
            "tools/gpu_validation.py --execute writes."
        )
        return "\n".join(lines) + "\n"

    verdict = (
        f"GREEN: all {len(green)} of {len(ran)} executed cases met their expectation"
        if not failed
        else f"NOT GREEN: {len(failed)} of {len(ran)} executed cases failed"
    )
    lines.append(f"**{verdict}**" if markdown else verdict)
    relevant = report["relevant"]
    here = ", ".join(report["accelerators"]) or "this machine"
    elsewhere = len(planned) - len(relevant)
    lines.append(
        f"{len(ran)} of {len(relevant)} cases planned for {here} have run; "
        f"the other {elsewhere} in the matrix target hardware this machine does not have."
    )
    if skipped:
        lines.append(
            f"{len(skipped)} case(s) were asked for but skipped as a hardware mismatch."
        )
    lines.append("")

    lines.append("## Environment" if markdown else "Environment")
    lines.extend(describe_environment(ran))
    lines.append("")

    rows = []
    for record in sorted(ran, key=lambda r: r["case_id"]):
        case, metrics = record["case"], record.get("metrics", {})
        mark = {"passed": "ok", "passed_expected_rejection": "rej"}.get(
            record["status"], "FAIL"
        )
        rows.append(
            [
                mark,
                record["case_id"],
                case["model"],
                _combination(case),
                _seconds(metrics.get("load_duration_seconds")),
                _seconds(_post_load(metrics)),
                _seconds(metrics.get("wall_duration_seconds")),
                _gib(metrics.get("peak_gpu_memory_bytes")),
                _gib(metrics.get("peak_host_rss_bytes")),
            ]
        )
    headers = [
        "",
        "case",
        "model",
        "placement/quant/world",
        "load s",
        "post-load s",
        "wall s",
        "peak vram",
        "peak host",
    ]
    lines.append("## Per case" if markdown else "Per case")
    if fence:
        lines.append(fence)
    lines.extend(_table(rows, headers))
    if fence:
        lines.append(fence)
    lines.append("")
    lines.append(
        "rej means the case was refused as the matrix expects, so a guard fired before "
        "the load and there are no timings to report."
    )
    lines.append(
        "post-load s is wall minus load, so it covers inference, VAE decode, saving and "
        "teardown rather than inference alone."
    )

    scopes = {
        r.get("metrics", {}).get("gpu_memory_scope")
        for r in ran
        if r.get("metrics", {}).get("peak_gpu_memory_bytes")
    }
    # Anything that is not explicitly process-local is treated as contaminated, so a
    # renamed scope errs towards warning rather than silently dropping the caveat.
    if any(scope != "process_tree" for scope in scopes if scope):
        lines.append(
            "peak vram was sampled device-globally on this platform, so on a shared node "
            "it includes other tenants and is an upper bound, not this run's usage."
        )
    lines.append("")

    if failed:
        lines.append("## Failures" if markdown else "Failures")
        for record in sorted(failed, key=lambda r: r["case_id"]):
            expected = record.get("expected", {}).get("outcome")
            lines.append(
                f"  {record['case_id']}: {record['status']} "
                f"(expected {expected}, exit {record.get('exit_status')})"
            )
            notes = (record.get("quality", {}) or {}).get("matrix_notes")
            if notes:
                lines.append(f"    {notes[:160]}")
        lines.append("")

    lines.append("## Coverage by model" if markdown else "Coverage by model")
    lines.append(
        "Planned combinations for the hardware these results came from, and whether each ran."
    )
    lines.append("")
    by_model: dict[str, list[dict]] = collections.defaultdict(list)
    for case in report["relevant"]:
        by_model[case["model"]].append(case)
    status_by_id = {r["case_id"]: r["status"] for r in ran}
    rows = []
    for model in sorted(by_model):
        shown = model
        for case in sorted(by_model[model], key=_combination):
            status = status_by_id.get(case["id"])
            if status is None:
                outcome = "not run"
            elif status == "passed":
                outcome = "ok"
            elif status == "passed_expected_rejection":
                outcome = "rej"
            else:
                outcome = "FAIL"
            rows.append([shown, _combination(case), outcome])
            shown = ""
    if fence:
        lines.append(fence)
    lines.extend(_table(rows, ["model", "placement/quant/world", "outcome"]))
    if fence:
        lines.append(fence)
    lines.append("")
    lines.append(
        "rej is a combination this hardware and torch build are expected to refuse, "
        "so it is covered but is not usable."
    )
    return "\n".join(lines) + "\n"


def _case_targets(case: dict, accelerators: set[str]) -> bool:
    """Whether a planned case is meant for the accelerators these results came from."""
    if not accelerators:
        return False
    token = case["hardware"]["accelerator"]
    rdna4 = {"gfx1200", "gfx1201"}
    if token == "non_rdna4_rocm":
        return all(a.startswith("gfx") and a not in rdna4 for a in accelerators)
    if token == "gfx1200_or_gfx1201":
        return all(a in rdna4 for a in accelerators)
    if token == "gfx942_or_gfx950":
        return all(a in {"gfx942", "gfx950"} for a in accelerators)
    if token.startswith("gfx"):
        return all(a == token for a in accelerators)
    return False


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--results",
        type=Path,
        nargs="+",
        required=True,
        help="JSONL written by tools/gpu_validation.py --execute",
    )
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--format", choices=("text", "markdown"), default="text")
    parser.add_argument("--output", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = build_report(args.results, args.matrix)
    text = render(report, markdown=args.format == "markdown")
    if args.output:
        args.output.write_text(text)
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
