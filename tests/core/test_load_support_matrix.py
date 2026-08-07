"""Joining declared load capability with planned and executed GPU validation.

The report decides where testing does not reach, so its classifications have to be right: a case
that was refused on purpose is not a failure, a case fixed after a failure is not still broken, and
a model whose only coverage is eager has never exercised the memory-efficient path at all.
"""

import importlib.util
import json
from pathlib import Path

import pytest

TOOL = Path(__file__).resolve().parents[2] / "tools/load_support_matrix.py"


@pytest.fixture(scope="module")
def tool():
    """Loaded by path: the tool imports xfuser lazily, so this stays a plain unit test."""
    spec = importlib.util.spec_from_file_location("load_support_matrix", TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def capability(model, *, fsdp=True, aliases=None):
    return {
        "runner": f"xFuser{model}Model",
        "model": model,
        "aliases": aliases or [model],
        "eager": True,
        "fsdp_meta": fsdp,
        "replicated_meta": fsdp,
        "loader_adapter": "standard_transformer",
        "quantization_formats": ["fp8"],
        "quantization_backends": ["aiter"],
        "excluded_components": {},
        "withheld_reason": None if fsdp else "not verified",
    }


def write(tmp_path, cases, results=None):
    matrix = tmp_path / "matrix.json"
    matrix.write_text(json.dumps({"cases": cases}))
    if results is None:
        return matrix, None
    path = tmp_path / "results.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in results))
    return matrix, path


def case(case_id, model, placement="fsdp_blockwise"):
    return {"id": case_id, "model": model, "placement": placement}


def test_a_case_refused_on_purpose_counts_as_run_not_failed(tool, tmp_path):
    matrix, results = write(
        tmp_path,
        [case("a-rejected", "Demo")],
        [{"case_id": "a-rejected", "status": "passed_expected_rejection", "recorded_at": "1"}],
    )

    report = tool.build_report(matrix, results, capability_rows=[capability("Demo")])

    row = report["models"][0]
    assert row["executed"] == ["a-rejected"]
    assert row["failed"] == []


def test_a_case_fixed_after_failing_is_not_reported_as_broken(tool, tmp_path):
    matrix, results = write(
        tmp_path,
        [case("a-case", "Demo")],
        [
            {"case_id": "a-case", "status": "failed_inference", "recorded_at": "2026-01-01"},
            {"case_id": "a-case", "status": "passed", "recorded_at": "2026-06-01"},
        ],
    )

    report = tool.build_report(matrix, results, capability_rows=[capability("Demo")])

    assert report["models"][0]["failed"] == []


def test_a_case_that_regressed_is_reported_as_broken(tool, tmp_path):
    matrix, results = write(
        tmp_path,
        [case("a-case", "Demo")],
        [
            {"case_id": "a-case", "status": "passed", "recorded_at": "2026-01-01"},
            {"case_id": "a-case", "status": "failed_inference", "recorded_at": "2026-06-01"},
        ],
    )

    report = tool.build_report(matrix, results, capability_rows=[capability("Demo")])

    assert report["models"][0]["failed"] == ["a-case"]


def test_coverage_follows_every_alias_a_runner_answers_to(tool, tmp_path):
    """The matrix names a model by one alias; the registry knows several."""
    matrix, _ = write(tmp_path, [case("a-case", "demo/Demo-v1")])

    report = tool.build_report(
        matrix, None, capability_rows=[capability("Demo", aliases=["Demo", "demo/Demo-v1"])]
    )

    assert report["models"][0]["case_ids"] == ["a-case"]
    assert report["matrix_models_not_registered"] == []


def test_a_matrix_name_no_runner_answers_to_is_reported(tool, tmp_path):
    matrix, _ = write(tmp_path, [case("a-case", "Ghost")])

    report = tool.build_report(matrix, None, capability_rows=[capability("Demo")])

    assert report["matrix_models_not_registered"] == ["Ghost"]


def test_gaps_separate_the_uncovered_from_the_merely_unrun(tool, tmp_path):
    matrix, results = write(
        tmp_path,
        [case("ran", "Ran"), case("planned", "Planned")],
        [{"case_id": "ran", "status": "passed", "recorded_at": "1"}],
    )

    report = tool.build_report(
        matrix,
        results,
        capability_rows=[
            capability("Ran"),
            capability("Planned"),
            capability("Absent"),
        ],
    )
    gaps = tool.render_gaps(report)

    assert "1 capable models with no matrix case at all:" in gaps
    assert "Absent" in gaps.split("no matrix case at all:")[1].split("\n\n")[0]
    assert "Planned" in gaps.split("never been run:")[1].split("\n\n")[0]
    assert "Ran" not in gaps.split("never been run:")[1].split("\n\n")[0]


def test_a_model_tested_only_in_eager_is_called_out(tool, tmp_path):
    """Eager coverage says nothing about the memory-efficient path."""
    matrix, results = write(
        tmp_path,
        [case("eager-only", "Demo", placement="eager")],
        [{"case_id": "eager-only", "status": "passed", "recorded_at": "1"}],
    )

    report = tool.build_report(matrix, results, capability_rows=[capability("Demo")])
    gaps = tool.render_gaps(report)

    section = gaps.split("exercised only in eager placement:")[1].split("\n\n")[0]
    assert "Demo" in section


def test_a_withheld_model_with_no_case_leaves_its_refusal_unasserted(tool, tmp_path):
    matrix, _ = write(tmp_path, [])

    report = tool.build_report(
        matrix, None, capability_rows=[capability("Withheld", fsdp=False)]
    )
    gaps = tool.render_gaps(report)

    section = gaps.split("no case asserting the rejection:")[1].split("\n\n")[0]
    assert "Withheld" in section


def test_the_markdown_table_lists_every_model_once(tool, tmp_path):
    matrix, _ = write(tmp_path, [case("a-case", "Demo")])

    report = tool.build_report(
        matrix, None, capability_rows=[capability("Demo"), capability("Other", fsdp=False)]
    )
    markdown = tool.render_markdown(report)

    assert markdown.count("\n| Demo |") == 1
    assert "| Other | yes | no | no |" in markdown
    assert "not verified" in markdown
