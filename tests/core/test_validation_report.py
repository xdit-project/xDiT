"""The results dashboard has to stay honest about what ran and what merely passed."""

import importlib.util
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
REPORT_PATH = ROOT / "tools/validation_report.py"
MATRIX_PATH = ROOT / "tests/gpu_validation/matrix.json"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def report_tool():
    return _load("validation_report", REPORT_PATH)


@pytest.fixture(scope="module")
def matrix():
    return json.loads(MATRIX_PATH.read_text())


def _record(case, status, **metrics):
    base = {
        "wall_duration_seconds": None,
        "load_duration_seconds": None,
        "first_forward": "not_reached",
        "peak_host_rss_bytes": None,
        "peak_cgroup_memory_bytes": None,
        "peak_gpu_memory_bytes": None,
        "gpu_memory_scope": None,
    }
    base.update(metrics)
    return {
        "schema_version": 2,
        "recorded_at": "2026-01-01T00:00:00+00:00",
        "case_id": case["id"],
        "case": case,
        "status": status,
        "execution": "NOT RUN" if status == "environment_mismatch" else "RAN",
        "expected": case["expected"],
        "command": ["torchrun"],
        # Mirrors gpu_validation.collect_environment: the top-level platform is the OS
        # string and the arch the case was matched against lives under validation_probe.
        "environment": {
            "platform": "Linux-5.15.0-x86_64",
            "packages": {"torch": "2.9.1"},
            "validation_probe": {"platform": "rocm", "accelerators": ["gfx950"]},
        },
        "exit_status": 0 if status == "passed" else 1,
        "metrics": base,
        "output": {"files": [{"bytes": 8, "sha256": "ab"}]},
        "quality": {"matrix_notes": case.get("quality_notes", "")},
    }


def _cases_for(matrix, accelerator):
    return [c for c in matrix["cases"] if c["hardware"]["accelerator"] == accelerator]


def _write(tmp_path, records):
    path = tmp_path / "results.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
    return path


def test_a_rerun_supersedes_the_earlier_attempt(report_tool, matrix, tmp_path):
    case = _cases_for(matrix, "gfx950")[0]
    first = _record(case, "failed_inference")
    second = _record(case, "passed", wall_duration_seconds=10.0)
    second["recorded_at"] = "2026-02-01T00:00:00+00:00"
    records = report_tool.load_records([_write(tmp_path, [first, second])])
    assert [r["status"] for r in records] == ["passed"]


def test_an_expected_rejection_is_not_shown_as_a_working_combination(
    report_tool, matrix, tmp_path
):
    """A guard firing on purpose is a pass, but the combination still does not work."""
    case = next(
        c
        for c in _cases_for(matrix, "gfx950")
        if c["expected"]["outcome"] != "inference_success"
    )
    path = _write(tmp_path, [_record(case, "passed_expected_rejection")])
    text = report_tool.render(report_tool.build_report([path], MATRIX_PATH))
    combination = report_tool._combination(case)
    coverage_row = next(
        line
        for line in text.splitlines()
        if combination in line and line.split()[-1] in {"ok", "rej", "FAIL", "run"}
    )
    assert coverage_row.endswith("rej")
    assert "GREEN" in text and "NOT GREEN" not in text


def test_the_denominator_counts_only_cases_this_hardware_can_run(
    report_tool, matrix, tmp_path
):
    case = _cases_for(matrix, "gfx950")[0]
    path = _write(tmp_path, [_record(case, "passed", wall_duration_seconds=1.0)])
    report = report_tool.build_report([path], MATRIX_PATH)
    relevant = {c["id"] for c in report["relevant"]}
    assert case["id"] in relevant
    for other in _cases_for(matrix, "gfx1200_or_gfx1201"):
        assert other["id"] not in relevant
    assert len(relevant) < len(matrix["cases"])


def test_a_failure_makes_the_whole_report_not_green(report_tool, matrix, tmp_path):
    passing, failing = _cases_for(matrix, "gfx950")[:2]
    path = _write(
        tmp_path,
        [
            _record(passing, "passed", wall_duration_seconds=1.0),
            _record(failing, "failed_inference"),
        ],
    )
    text = report_tool.render(report_tool.build_report([path], MATRIX_PATH))
    assert "NOT GREEN" in text
    assert failing["id"] in text.split("Failures", 1)[1]


@pytest.mark.parametrize(
    "scope, flagged",
    [("device_global", True), ("process_tree", False)],
)
def test_device_global_vram_is_flagged_as_an_upper_bound(
    report_tool, matrix, tmp_path, scope, flagged
):
    """On ROCm the sampler reads whole-device usage, so a shared node inflates it.

    The scope strings have to be the ones the runner actually writes; a caveat keyed to a
    value it never emits would silently stop warning.
    """
    case = _cases_for(matrix, "gfx950")[0]
    path = _write(
        tmp_path,
        [
            _record(
                case,
                "passed",
                wall_duration_seconds=1.0,
                peak_gpu_memory_bytes=4 * 1024**3,
                gpu_memory_scope=scope,
            )
        ],
    )
    text = report_tool.render(report_tool.build_report([path], MATRIX_PATH))
    assert ("upper bound" in text) is flagged


def test_post_load_time_is_never_negative(report_tool, matrix, tmp_path):
    case = _cases_for(matrix, "gfx950")[0]
    metrics = {"wall_duration_seconds": 5.0, "load_duration_seconds": 9.0}
    assert report_tool._post_load(metrics) == 0.0


def test_every_case_for_one_model_on_one_arch_gets_a_distinct_label(
    report_tool, matrix
):
    """Coverage rows are keyed by this label, so a collision hides an untested case."""
    seen = {}
    for case in matrix["cases"]:
        key = (case["model"], case["hardware"]["accelerator"])
        label = report_tool._combination(case)
        collision = seen.setdefault((key, label), case["id"])
        assert collision == case["id"], (
            f"{case['id']} and {collision} both render as {label} for {key}"
        )


def test_the_report_reads_the_keys_the_runner_actually_writes(report_tool):
    """Guards against reading a shape the runner never produces.

    Looking for accelerators at the top level found nothing, which showed the environment
    as unknown and, worse, silently narrowed the planned set to whatever had already run.
    """
    runner = _load("gpu_validation_runner_env", ROOT / "tools/gpu_validation.py")
    environment = runner.collect_environment(
        {"platform": "rocm", "accelerators": ["gfx950"]}
    )
    record = {"environment": environment}

    assert report_tool.accelerators_of(record) == ["gfx950"]
    facts = report_tool._environment_facts(record)
    assert facts["platform"] == "rocm"
    assert facts["versions"], "package versions should be found"
    assert "torch" in facts["versions"]


def test_missing_results_file_reports_nothing_ran(report_tool, tmp_path):
    report = report_tool.build_report([tmp_path / "absent.jsonl"], MATRIX_PATH)
    text = report_tool.render(report)
    assert "No case has been run" in text
