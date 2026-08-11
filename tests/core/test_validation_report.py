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
def runner():
    return _load("gpu_validation", ROOT / "tools/gpu_validation.py")


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
        # What the runner writes today; tests for older records drop it deliberately.
        "metrics_version": 4,
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


def test_a_quantized_text_encoder_reports_which_path_it_took(
    report_tool, matrix, tmp_path
):
    """Two runs of one case can differ here, so the row has to say which it measured."""
    case = next(c for c in _cases_for(matrix, "gfx942_or_gfx950") if c.get("te_fp8"))
    streamed = _record(
        case,
        "passed",
        wall_duration_seconds=1.0,
        quantization_paths={
            "transformer": {"materialization": "streaming", "fallback": None},
            "text_encoder": {"materialization": "streaming", "fallback": None},
        },
    )
    fell_back = _record(
        case,
        "passed",
        wall_duration_seconds=1.0,
        quantization_paths={
            "text_encoder": {
                "materialization": "post_load",
                "fallback": "streaming disabled by the runner",
            }
        },
    )

    assert report_tool._text_encoder_path(streamed) == "stream"
    assert report_tool._text_encoder_path(fell_back) == "post"


def test_a_record_written_before_the_path_was_captured_says_so(
    report_tool, matrix, tmp_path
):
    """Blank would read as a case that quantized nothing, which is a different claim."""
    case = next(c for c in _cases_for(matrix, "gfx942_or_gfx950") if c.get("te_fp8"))
    older = _record(case, "passed", wall_duration_seconds=1.0)

    assert report_tool._text_encoder_path(older) == "?"


def test_a_case_with_no_quantized_text_encoder_claims_nothing_about_one(
    report_tool, matrix
):
    case = next(
        c for c in _cases_for(matrix, "gfx942_or_gfx950") if not c.get("te_fp8")
    )
    record = _record(case, "passed", wall_duration_seconds=1.0, quantization_paths={})

    assert report_tool._text_encoder_path(record) == "-"


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


def test_memory_from_before_the_definition_changed_is_not_shown_as_comparable(
    report_tool, matrix, tmp_path
):
    """The old figure summed the node's devices, so it grew with the rank count.

    Printing it in the same column as a per-device peak would invite the comparison the change was
    made to enable, and would make sharding look like it cost eight times the memory.
    """
    case = _cases_for(matrix, "gfx950")[0]
    old = _record(case, "passed", wall_duration_seconds=1.0, peak_gpu_memory_bytes=280 * 1024**3)
    old["metrics"].pop("metrics_version", None)
    path = _write(tmp_path, [old])

    text = report_tool.render(report_tool.build_report([path], MATRIX_PATH))

    assert "280" not in text
    assert "stale" in text


def test_the_report_and_the_runner_agree_on_the_metrics_version(report_tool, runner):
    """Two constants naming one definition drift apart silently."""
    assert report_tool.CURRENT_METRICS_VERSION == runner.GPU_METRICS_VERSION


def test_a_record_for_a_case_the_matrix_dropped_does_not_make_the_report_red(
    report_tool, matrix, tmp_path
):
    """Profiles change their rank counts, which renames generated cases.

    A leftover failure under the old name describes a case nobody can re-run, so counting it would
    leave the report permanently red with no action available.
    """
    passing = _cases_for(matrix, "gfx950")[0]
    dropped = _record(passing, "failed_inference")
    dropped["case_id"] = f"{passing['id']}-removed-when-the-profile-changed"
    path = _write(
        tmp_path,
        [_record(passing, "passed", wall_duration_seconds=1.0), dropped],
    )

    report = report_tool.build_report([path], MATRIX_PATH)
    text = report_tool.render(report)

    assert [r["case_id"] for r in report["stale"]] == [dropped["case_id"]]
    assert not report["failed"]
    assert "GREEN" in text and "NOT GREEN" not in text
    assert dropped["case_id"] in text.split("Stale records", 1)[1]


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


def test_compile_time_is_not_charged_to_the_work_after_the_load(report_tool):
    """Compile sits between the load and the first forward, so it belongs to neither neighbour.

    Leaving it in post-load would move tens of seconds out of one unexplained column into another.
    """
    metrics = {
        "wall_duration_seconds": 100.0,
        "load_duration_seconds": 30.0,
        "compile_duration_seconds": 22.0,
    }

    assert report_tool._post_load(metrics) == 48.0


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


def test_a_scored_case_shows_how_close_it_came_to_its_reference(report_tool, matrix, tmp_path):
    """The number matters, not just the verdict: it says whether a case is fine or near the floor."""
    case = _cases_for(matrix, "gfx950")[0]
    record = _record(case, "passed", wall_duration_seconds=10.0)
    record["quality"] = {
        "matrix_notes": "",
        "reference": {
            "case_id": "some-eager-bf16-case",
            "verdict": "pass",
            "scores": {"comparable": True, "ssim": 0.982, "psnr": 24.9, "mse": 3.25e-3},
        },
    }

    text = report_tool.render(report_tool.build_report([_write(tmp_path, [record])], MATRIX_PATH))

    assert "vs ref" in text
    assert "0.982" in text


def test_a_failed_comparison_is_marked_and_not_just_printed(report_tool, matrix, tmp_path):
    """A score below the floor has to be visible as a failure in the row, not only in the number."""
    case = _cases_for(matrix, "gfx950")[0]
    record = _record(case, "failed_quality", wall_duration_seconds=10.0)
    record["quality"] = {
        "matrix_notes": "",
        "reference": {
            "case_id": "some-eager-bf16-case",
            "verdict": "fail",
            "scores": {"comparable": True, "ssim": 0.412, "psnr": 9.1, "mse": 0.12},
        },
    }

    text = report_tool.render(report_tool.build_report([_write(tmp_path, [record])], MATRIX_PATH))

    assert "0.412 FAIL" in text
    assert "FAIL" in text.split("0.412")[0], "the row itself has to read as failed"


def test_a_score_the_model_cannot_support_reads_as_an_observation(report_tool, matrix, tmp_path):
    """Marking it FAIL would send someone chasing a defect in a run whose image is fine.

    Base Z-Image renders a different sample under different numerics, so it scores 0.625 against its
    own bf16 reference with both images clean. The number is still shown, because a collapse would be
    visible in it, but the row must not claim the case failed.
    """
    case = _cases_for(matrix, "gfx950")[0]
    record = _record(case, "passed", wall_duration_seconds=10.0)
    record["quality"] = {
        "matrix_notes": "",
        "reference": {
            "case_id": "some-eager-bf16-case",
            "verdict": "fail",
            "gated": False,
            "scores": {"comparable": True, "ssim": 0.625, "psnr": 11.7, "mse": 0.07},
        },
    }

    text = report_tool.render(report_tool.build_report([_write(tmp_path, [record])], MATRIX_PATH))

    assert "0.625 info" in text
    assert "0.625 FAIL" not in text
    assert "FAIL" not in text.split("0.625")[0], "the row must not read as failed"
    assert "does not reproduce its sample" in text, "the reader needs to know why it is not gated"


def test_the_report_warns_that_load_times_depend_on_page_cache(report_tool, matrix, tmp_path):
    """The same case measured 56.7s cold and 18.2s warm, and nothing in a row says which it was.

    Two separate readings of this dashboard drew the wrong conclusion from rows whose runs did not
    share cache state, so the caveat is part of the report rather than tribal knowledge.
    """
    case = _cases_for(matrix, "gfx950")[0]
    record = _record(case, "passed", load_duration_seconds=40.0, wall_duration_seconds=80.0)

    text = report_tool.render(report_tool.build_report([_write(tmp_path, [record])], MATRIX_PATH))

    assert "page cache" in text
    assert "comparable" in text


def test_a_scoring_run_does_not_replace_the_timings_of_a_real_one(report_tool, matrix, tmp_path):
    """Scoring runs disable compile, so their timings are not the case as declared.

    Letting one supersede put compile-free numbers in the same column as compiled ones with nothing
    saying so, which is exactly the silent mixing the columns exist to prevent.
    """
    case = _cases_for(matrix, "gfx950")[0]
    real = _record(case, "passed", wall_duration_seconds=80.0, load_duration_seconds=50.0,
                   compile_duration_seconds=17.6)
    scoring = _record(case, "passed", wall_duration_seconds=66.0, load_duration_seconds=34.4)
    scoring["recorded_at"] = "2026-03-01T00:00:00+00:00"
    scoring["quality"] = {
        "matrix_notes": "",
        "scoring_run": True,
        "reference": {
            "case_id": "ref",
            "verdict": "pass",
            "scores": {"comparable": True, "ssim": 0.923, "psnr": 24.9, "mse": 3e-3},
        },
    }

    text = report_tool.render(
        report_tool.build_report([_write(tmp_path, [real, scoring])], MATRIX_PATH)
    )

    assert "50.0" in text and "17.6" in text, "the compiled run's timings have to survive"
    assert "34.4" not in text, "the scoring run's load time must not take over the column"
    assert "0.923" in text, "its verdict still has to reach the row"


def test_a_scoring_run_recorded_before_the_marker_existed_is_still_recognised(
    report_tool, matrix, tmp_path
):
    """Records already on disk have no marker, and would otherwise keep polluting the columns.

    A run missing compile that its own case asks for is not that case's performance, marker or not.
    """
    case = {**_cases_for(matrix, "gfx950")[0], "args": ["--use_torch_compile"]}
    real = _record(case, "passed", load_duration_seconds=50.0, compile_duration_seconds=17.6)
    real["command"] = ["torchrun", "--use_torch_compile"]
    scoring = _record(case, "passed", load_duration_seconds=34.4)
    scoring["recorded_at"] = "2026-03-01T00:00:00+00:00"
    scoring["command"] = ["torchrun"]
    scoring["quality"] = {"matrix_notes": ""}

    text = report_tool.render(
        report_tool.build_report([_write(tmp_path, [real, scoring])], MATRIX_PATH)
    )

    assert "50.0" in text and "34.4" not in text


def test_a_case_only_ever_scored_still_reports(report_tool, matrix, tmp_path):
    """Reporting nothing for it would be worse than reporting a run whose compile column is empty."""
    case = _cases_for(matrix, "gfx950")[0]
    scoring = _record(case, "passed", wall_duration_seconds=66.0, load_duration_seconds=34.4)
    scoring["quality"] = {"matrix_notes": "", "scoring_run": True, "reference": None}

    text = report_tool.render(report_tool.build_report([_write(tmp_path, [scoring])], MATRIX_PATH))

    assert "34.4" in text


def test_an_unscored_run_leaves_the_column_empty_rather_than_implying_a_pass(
    report_tool, matrix, tmp_path
):
    """Most records predate scoring, and the reference has nothing to compare itself against."""
    case = _cases_for(matrix, "gfx950")[0]
    record = _record(case, "passed", wall_duration_seconds=10.0)

    text = report_tool.render(report_tool.build_report([_write(tmp_path, [record])], MATRIX_PATH))

    assert "vs ref" in text
    assert "FAIL" not in text
