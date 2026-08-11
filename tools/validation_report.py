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
# Mirrors gpu_validation.GPU_METRICS_VERSION; a record below it holds memory figures this report
# cannot put in the same column as the rest.
CURRENT_METRICS_VERSION = 4


def _is_scoring_run(record: dict) -> bool:
    """Whether this run measured something other than the case as the matrix declares it.

    Trusts the marker when it is there, and otherwise compares the command against the case's own
    arguments: a run missing compile that the case asks for is not that case's performance, whoever
    stripped it. That covers records written before the marker existed, which would otherwise keep
    reporting compile-free timings in a column of compiled ones.
    """
    quality = record.get("quality") or {}
    if "scoring_run" in quality:
        return bool(quality["scoring_run"])
    declared = (record.get("case") or {}).get("args") or []
    command = record.get("command") or []
    return "--use_torch_compile" in declared and "--use_torch_compile" not in command


def load_records(paths: list[Path]) -> list[dict]:
    """Latest record per case id, so a re-run supersedes an earlier attempt.

    Quality scoring runs are kept apart from that. They disable torch.compile to make the comparison
    deterministic, so their timings are not the case as declared, and letting one supersede a
    performance run put compile-free numbers in the same column as compiled ones with nothing saying
    so. A case therefore reports its newest real run, carrying the verdict from its newest scored
    run; a case that has only ever been scored still reports, since that beats reporting nothing.
    """
    performance: dict[str, dict] = {}
    scored: dict[str, dict] = {}
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
            latest = scored if _is_scoring_run(record) else performance
            previous = latest.get(case_id)
            if previous is None or record.get("recorded_at", "") >= previous.get(
                "recorded_at", ""
            ):
                latest[case_id] = record
    records = []
    for case_id in performance.keys() | scored.keys():
        record = performance.get(case_id) or scored[case_id]
        verdict = ((scored.get(case_id) or {}).get("quality") or {}).get("reference")
        if verdict is not None and record is not scored.get(case_id):
            record = {
                **record,
                "quality": {**(record.get("quality") or {}), "reference": verdict},
            }
        records.append(record)
    return records


def _seconds(value) -> str:
    """One unit for the whole column, so loads of very different cost stay comparable."""
    if value is None:
        return "-"
    return f"{value:.1f}"


def _gib(value) -> str:
    if value in (None, 0):
        return "-"
    return f"{value / (1024 ** 3):.1f}G"


def _quality(record: dict) -> str:
    """How the case's image compared against its reference, when it was scored.

    Shows the SSIM rather than only pass or fail, because the number is what tells you whether a
    case is comfortably fine or sitting near the floor. A dash means nobody scored this run: most
    records predate scoring, and the reference case has nothing to compare itself against.
    """
    reference = (record.get("quality") or {}).get("reference")
    if not isinstance(reference, dict):
        return "-"
    scores = reference.get("scores") or {}
    if not scores.get("comparable"):
        return "n/c"
    score = f"{scores['ssim']:.3f}"
    # A model that does not reproduce its sample under a numeric change cannot be judged by this
    # number, so the score is shown as an observation and never as a failure.
    if not reference.get("gated", True):
        return f"{score} info"
    return score + ("" if reference.get("verdict") == "pass" else " FAIL")


def _drew_something(record: dict) -> str:
    """Whether the artifact has a picture in it, which no reference is needed to answer.

    Its own column because it is the one quality statement that holds for every model, and a run
    that saved a flat frame otherwise reads as a pass with an unremarkable timing row.
    """
    content = (record.get("quality") or {}).get("content")
    if not isinstance(content, dict):
        return "-"
    if not content.get("measured"):
        return "n/m"
    if content.get("verdict") == "fail":
        return f"BLANK {content['std']:.3f}"
    return f"{content['std']:.2f}"


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


def _text_encoder_path(record: dict) -> str:
    """Whether the text encoder streamed its quantization or quantized after loading.

    Its own column because it is not decided by the case's flags but at runtime, from what
    the installed libraries expose and which load path the denoiser took. Two rows asking
    for the same thing can therefore have measured different things, and the memory columns
    beside this one are where that shows up.
    """
    case = record["case"]
    if not case.get("te_fp8"):
        return "-"
    paths = (record.get("metrics") or {}).get("quantization_paths")
    if paths is None:
        return "?"
    modes = {
        detail["materialization"]
        for name, detail in paths.items()
        if name.startswith("text_encoder")
    }
    if not modes:
        return "none"
    short = {"streaming": "stream", "post_load": "post"}
    return "+".join(sorted(short.get(mode, mode) for mode in modes))


def _post_load(metrics: dict):
    """Wall minus the two phases the runner times, so the columns account for the whole run.

    Compile has to come out too. It sits between the load and the first forward, so charging it to
    post-load would move tens of seconds from one unexplained column into another.
    """
    wall = metrics.get("wall_duration_seconds")
    load = metrics.get("load_duration_seconds")
    if wall is None or load is None:
        return None
    return max(wall - load - (metrics.get("compile_duration_seconds") or 0.0), 0.0)


def _table(rows: list[list[str]], headers: list[str]) -> list[str]:
    widths = [
        max(len(headers[i]), max((len(r[i]) for r in rows), default=0))
        for i in range(len(headers))
    ]
    numeric = set(range(5, len(headers)))

    def fmt(cells: list[str]) -> str:
        out = []
        for i, cell in enumerate(cells):
            out.append(cell.rjust(widths[i]) if i in numeric else cell.ljust(widths[i]))
        return "  ".join(out).rstrip()

    lines = [fmt(headers), "  ".join("-" * w for w in widths)]
    lines.extend(fmt(row) for row in rows)
    return lines


def accelerators_of(record: dict) -> list[str]:
    """The archs a record ran on.

    Read through validation_probe because the environment's own ``platform`` is the OS
    string; the probe holds what the case was matched against.
    """
    probe = record.get("environment", {}).get("validation_probe", {})
    return probe.get("accelerators") or []


def _environment_facts(record: dict) -> dict:
    environment = record.get("environment", {})
    probe = environment.get("validation_probe", {})
    return {
        "platform": probe.get("platform"),
        "accelerators": sorted(set(accelerators_of(record))),
        "versions": environment.get("packages"),
    }


def describe_environment(records: list[dict]) -> list[str]:
    lines = []
    environments = {
        json.dumps(_environment_facts(r), sort_keys=True) for r in records
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

    # A record whose case the matrix no longer plans describes a configuration nobody asked about
    # any more — a renamed case, or one dropped when a profile changed its rank counts. Judging the
    # run by it would report a failure that cannot be reproduced or fixed, so it is set aside and
    # listed rather than counted.
    stale = [r for r in records if r["case_id"] not in planned]
    current = [r for r in records if r["case_id"] in planned]

    ran = [r for r in current if r.get("status") not in NOT_RUN_STATUSES]
    skipped = [r for r in current if r.get("status") in NOT_RUN_STATUSES]
    green = [r for r in ran if r["status"] in GREEN_STATUSES]
    failed = [r for r in ran if r["status"] not in GREEN_STATUSES]

    accelerators = sorted(
        {accelerator for r in records for accelerator in accelerators_of(r)}
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
        "stale": stale,
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
                _text_encoder_path(record),
                _seconds(metrics.get("load_duration_seconds")),
                _seconds(metrics.get("compile_duration_seconds")),
                _seconds(_post_load(metrics)),
                _seconds(metrics.get("wall_duration_seconds")),
                _gib(metrics.get("peak_load_gpu_memory_bytes")),
                _gib(metrics.get("peak_gpu_memory_bytes"))
                if _memory_is_comparable(metrics)
                else "stale",
                _gib(metrics.get("peak_host_anon_bytes"))
                if _memory_is_comparable(metrics)
                else "stale",
                _gib(metrics.get("peak_host_file_cache_bytes")),
                _drew_something(record),
                _quality(record),
            ]
        )
    headers = [
        "",
        "case",
        "model",
        "placement/quant/world",
        "te quant",
        "load s",
        "compile s",
        "post-load s",
        "wall s",
        "load vram",
        "peak vram",
        "host anon",
        "host cache",
        "spread",
        "vs ref",
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
        "te quant is which path the quantized text encoder took: stream quantizes each "
        "parameter as it is read, post materializes the encoder first and quantizes after. "
        "The runner chooses at runtime, so it is not read off the case; ? means the record "
        "predates this being captured and only its log can say."
    )
    lines.append(
        "vs ref is SSIM against the same model's eager bf16 case. It comes from a separate run made "
        "with --score-quality, which disables torch.compile so the comparison is deterministic, and "
        "so it does not describe the run the timings on this row came from. It is a "
        "gross-correctness check: quantization moves an image about as much as compile picking a "
        "different kernel does, so a passing score means the model still drew the picture rather "
        "than that quality is unchanged."
    )
    lines.append(
        "A score marked info was not allowed to fail its case, because that model does not reproduce "
        "its sample when the numerics change. Base Z-Image scores 0.625 between two renders that are "
        "both good pictures of the prompt, so on models like it a low score says a different sample "
        "came out and not that the load was wrong. Only a model measured to hold its sample, so far "
        "only Z-Image-Turbo, is gated on this number."
    )
    lines.append(
        "load s depends on how much of the checkpoint was already in page cache, and the same case "
        "has measured 56.7s cold against 18.2s warm. Rows are only comparable to each other when "
        "their runs shared that state, so a row re-run on its own can look better than its "
        "neighbours for no other reason."
    )
    lines.append(
        "load s ends where compile warmup begins, so it measures loading rather than loading plus "
        "compiling; compile s is that warmup, which is near-constant for a model and swamps the "
        "difference between load strategies when the two are added together."
    )
    lines.append(
        "post-load s is wall minus load and compile, so it covers inference, VAE decode, saving and "
        "teardown rather than inference alone."
    )
    lines.append(
        "host anon is the container's anonymous pages, which is what a load actually allocated and "
        "what the OOM killer watches; host cache is mmap'd checkpoint page cache, which the kernel "
        "reclaims under pressure and so is not a cost. Neither is summed over ranks, so they do not "
        "grow just because a run used more of them."
    )
    lines.append(
        "vram figures are the busiest single device, not a sum over the run's devices, so they "
        "are comparable across rank counts. load vram is the peak while the load was in flight, "
        "which is the figure a memory-efficient load is meant to move; peak vram covers the whole "
        "run and is usually dominated by inference activations."
    )

    if any(not _memory_is_comparable(r.get("metrics", {})) for r in ran):
        lines.append(
            "stale in the vram column means the record predates the current memory definition, "
            "when the figure was summed over the node's devices rather than taken per device. "
            "Those numbers are not comparable with the rest, so re-run the case to replace them."
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

    stale = report["stale"]
    if stale:
        lines.append("## Stale records" if markdown else "Stale records")
        lines.append(
            "The matrix no longer plans these cases, so they are excluded from the verdict above. "
            "Re-run the case that replaced them to get a current result."
        )
        for record in sorted(stale, key=lambda r: r["case_id"]):
            lines.append(f"  {record['case_id']}: {record.get('status')}")
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


def _memory_is_comparable(metrics: dict) -> bool:
    """Whether this record's GPU memory means what the current column heading says.

    Records written before the definition changed carry a node-wide sum, which is a different
    quantity and larger the more ranks the run used. Showing it beside a per-device peak would
    invite exactly the comparison the change was made to enable.
    """
    if not metrics.get("peak_gpu_memory_bytes"):
        return True
    return metrics.get("metrics_version", 1) >= CURRENT_METRICS_VERSION


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
