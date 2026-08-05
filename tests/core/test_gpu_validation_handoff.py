"""Dependency-light contracts for the external GPU validation handoff."""

import importlib.util
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
MATRIX_PATH = ROOT / "tests/gpu_validation/matrix.json"
RUNNER_PATH = ROOT / "tools/gpu_validation.py"
GUIDE_PATH = ROOT / "docs/runner/gpu_validation_handoff.md"


@pytest.fixture(scope="module")
def runner():
    spec = importlib.util.spec_from_file_location(
        "gpu_validation_runner", RUNNER_PATH
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def matrix(runner):
    return runner.load_matrix(MATRIX_PATH)


def test_matrix_schema_and_case_ids_are_valid(runner, matrix):
    runner.validate_matrix(matrix)
    ids = [case["id"] for case in matrix["cases"]]

    assert matrix["schema_version"] == 2
    assert matrix["validation_status"] == "NOT RUN"
    assert len(ids) == len(set(ids))
    assert all(case_id == case_id.lower() for case_id in ids)
    assert all(" " not in case_id for case_id in ids)


def test_operator_guide_is_linked_and_marks_results_not_run():
    guide = GUIDE_PATH.read_text()
    runner_guide = (ROOT / "docs/runner/runner.md").read_text()

    assert "GPU Validation Handoff" in guide
    assert "NOT RUN" in guide
    assert "gpu_validation_handoff.md" in runner_guide


def test_matrix_covers_required_validation_dimensions(matrix):
    cases = matrix["cases"]

    assert {
        case["hardware"]["backend"] for case in cases
    } >= {
        "rdna4_aiter",
        "rocm_torchao",
        "cuda_ada_torchao",
        "cuda_hopper_torchao",
        "cuda_blackwell_torchao",
    }
    assert {case["placement"] for case in cases} >= {
        "eager",
        "replicated",
        "fsdp_blockwise",
    }
    assert {case["te_fp8"] for case in cases} == {True, False}
    assert {case["offload"] for case in cases} >= {
        "none",
        "model",
        "sequential",
        "group",
        "group_low_cpu_mem",
    }
    assert {case["transformers"] for case in cases} >= {"4.x", "5.x"}
    assert {
        case["checkpoint"]["source"] for case in cases
    } == {"hub", "local"}
    assert any("dual-transformer" in case["tags"] for case in cases)
    assert any("custom-exclusion" in case["tags"] for case in cases)
    assert {
        case["expected"]["outcome"] for case in cases
    } == {"inference_success", "preflight_failure"}


def test_command_generation_uses_xdit_and_torchrun(runner, matrix):
    eager = next(
        case for case in matrix["cases"] if case["placement"] == "eager"
    )
    distributed = next(
        case
        for case in matrix["cases"]
        if case["placement"] == "fsdp_blockwise"
    )

    eager_command = runner.build_command(
        eager, matrix["defaults"], run_id="test-run"
    )
    distributed_command = runner.build_command(
        distributed, matrix["defaults"], run_id="test-run"
    )

    assert eager_command[0] == "xdit"
    assert "--model" in eager_command
    assert distributed_command[:4] == [
        "torchrun",
        f"--nproc_per_node={distributed['world_size']}",
        "-m",
        "xfuser.runner",
    ]
    assert "--memory_efficient_sharding" in distributed_command
    assert "--fully_shard_degree" in distributed_command
    output_index = eager_command.index("--output_directory")
    assert eager_command[output_index + 1].endswith(
        f"{eager['id']}/test-run"
    )


def test_local_checkpoint_command_keeps_env_placeholder(runner, matrix):
    case = next(
        case
        for case in matrix["cases"]
        if case["checkpoint"]["source"] == "local"
    )

    command = runner.build_command(
        case, matrix["defaults"], run_id="test-run"
    )

    placeholder = f"${{{case['checkpoint']['env']}}}"
    assert any(placeholder in argument for argument in command)
    rendered = runner.format_command(command)
    assert f'HF_HOME="{placeholder}"' in rendered
    assert f"'{placeholder}'" not in rendered


def test_filters_compose(runner, matrix):
    selected = runner.select_cases(
        matrix["cases"],
        tags=["expected-failure"],
        models=["Wan2.2-Distilled-I2V"],
        backends=["rdna4_aiter"],
        case_ids=[],
    )

    assert selected
    assert all("expected-failure" in case["tags"] for case in selected)
    assert all(
        case["model"] == "Wan2.2-Distilled-I2V" for case in selected
    )
    assert all(
        case["hardware"]["backend"] == "rdna4_aiter"
        for case in selected
    )


@pytest.mark.parametrize(
    (
        "exit_status",
        "log",
        "first_forward",
        "expected",
        "output",
        "classification",
    ),
    [
        (
            0,
            "",
            "succeeded",
            {"outcome": "inference_success"},
            {
                "path": "out.png",
                "sha256": "deadbeef",
                "bytes": 12,
                "files": [
                    {
                        "path": "out.png",
                        "sha256": "deadbeef",
                        "bytes": 12,
                    }
                ],
            },
            "passed",
        ),
        (
            0,
            "",
            "succeeded",
            {"outcome": "inference_success"},
            {"path": None, "sha256": None, "files": []},
            "failed_missing_output",
        ),
        (
            2,
            "custom checkpoint semantics are not collective-safe",
            "not_reached",
            {
                "outcome": "preflight_failure",
                "error_pattern": "not collective-safe",
            },
            {"path": None, "sha256": None, "files": []},
            "passed_expected_rejection",
        ),
        (
            2,
            "different error",
            "not_reached",
            {
                "outcome": "preflight_failure",
                "error_pattern": "not collective-safe",
            },
            {"path": None, "sha256": None, "files": []},
            "failed_wrong_rejection",
        ),
        (
            0,
            "",
            "succeeded",
            {
                "outcome": "preflight_failure",
                "error_pattern": "not collective-safe",
            },
            {"path": None, "sha256": None, "files": []},
            "failed_missing_rejection",
        ),
    ],
)
def test_expected_failure_classification(
    runner,
    exit_status,
    log,
    first_forward,
    expected,
    output,
    classification,
):
    assert (
        runner.classify_outcome(
            exit_status, log, first_forward, expected, output
        )
        == classification
    )


def test_output_discovery_requires_new_nonempty_hashed_artifact(
    runner, tmp_path
):
    (tmp_path / "empty.png").touch()
    (tmp_path / "metadata.txt").write_text("not generated media")
    (tmp_path / "weights.safetensors").write_bytes(b"not output media")
    artifact = tmp_path / "result.png"
    artifact.write_bytes(b"new output")

    output = runner.hash_outputs(tmp_path)

    assert output["path"] == str(artifact)
    assert output["bytes"] == len(b"new output")
    assert output["sha256"]
    assert [item["path"] for item in output["files"]] == [str(artifact)]
    assert ".png" in runner.GENERATED_ARTIFACT_EXTENSIONS
    assert ".txt" not in runner.GENERATED_ARTIFACT_EXTENSIONS


def test_fresh_run_directory_is_unique_and_reuse_fails(runner, tmp_path):
    defaults = {
        "output_root": str(tmp_path),
        "prompt": "p",
        "seed": 1,
        "num_inference_steps": 1,
        "height": 64,
        "width": 64,
    }
    case = {
        "id": "case",
        "checkpoint": {"source": "hub"},
        "world_size": 1,
        "model": "FLUX.1-dev",
        "placement": "eager",
        "quantization": "none",
        "te_fp8": False,
        "offload": "none",
        "args": [],
    }
    first = runner.build_command(case, defaults, run_id="run-a")
    second = runner.build_command(case, defaults, run_id="run-b")
    first_dir = Path(first[first.index("--output_directory") + 1])
    second_dir = Path(second[second.index("--output_directory") + 1])

    assert first_dir != second_dir
    runner.reserve_output_directory(first_dir)
    with pytest.raises(FileExistsError):
        runner.reserve_output_directory(first_dir)


def test_output_directory_is_absolute_from_non_repo_cwd(
    runner, matrix, tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    case = matrix["cases"][0]
    defaults = {
        **matrix["defaults"],
        "output_root": "relative-output",
    }

    command = runner.build_command(case, defaults, run_id="outside-cwd")
    output_dir = Path(command[command.index("--output_directory") + 1])

    assert output_dir.is_absolute()
    assert output_dir == (
        tmp_path / "relative-output" / case["id"] / "outside-cwd"
    )
    runner.reserve_output_directory(output_dir)
    artifact = output_dir / "result.png"
    artifact.write_bytes(b"generated")
    assert runner.hash_outputs(output_dir)["path"] == str(artifact)


def test_environment_probe_is_injectable(runner):
    responses = {
        "rocminfo": "Name: gfx1201\nName: gfx1201",
        "rocm-smi": '{"card0": {"Card series": "AMD Radeon"}}',
    }

    def command_runner(command):
        return responses.get(command[0])

    versions = {"transformers": "5.1.0", "aiter": "0.1", "torchao": "0.15"}
    observed = runner.probe_environment(
        command_runner=command_runner,
        version_getter=versions.get,
        environ={},
    )

    assert observed["platform"] == "rocm"
    assert observed["accelerators"] == ["gfx1201", "gfx1201"]
    assert observed["device_count"] == 2
    assert observed["transformers_major"] == 5
    assert observed["aiter_available"] is True


def test_environment_probe_applies_cuda_visible_devices_mask(runner):
    responses = {
        "nvidia-smi": (
            "0, GPU-aaaa, 8.9\n"
            "1, GPU-bbbb, 8.9\n"
            "2, GPU-cccc, 9.0"
        )
    }

    observed = runner.probe_environment(
        command_runner=lambda command: responses.get(command[0]),
        version_getter=lambda name: {
            "transformers": "5.1.0",
            "torchao": "0.15",
        }.get(name),
        environ={"CUDA_VISIBLE_DEVICES": "1,2"},
    )

    assert observed["accelerators"] == ["sm89", "sm90"]
    assert observed["device_count"] == 2
    assert observed["visibility"]["variable"] == "CUDA_VISIBLE_DEVICES"
    assert observed["visibility"]["value"] == "1,2"


def test_environment_validation_rejects_backend_accelerator_and_major_mismatch(
    runner, matrix
):
    case = next(
        case
        for case in matrix["cases"]
        if case["hardware"]["backend"] == "cuda_ada_torchao"
        and case["transformers"] == "5.x"
    )
    matching = {
        "platform": "cuda",
        "accelerators": ["sm89"],
        "transformers_major": 5,
        "aiter_available": False,
        "torchao_available": True,
    }

    assert runner.environment_mismatches(case, matching) == []
    mismatches = runner.environment_mismatches(
        case,
        {
            **matching,
            "platform": "rocm",
            "accelerators": ["gfx1201"],
            "transformers_major": 4,
        },
    )

    assert any("CUDA" in mismatch for mismatch in mismatches)
    assert any("sm89" in mismatch for mismatch in mismatches)
    assert any("Transformers 5.x" in mismatch for mismatch in mismatches)


def test_environment_validation_rejects_insufficient_matching_devices(
    runner, matrix
):
    case = next(
        case
        for case in matrix["cases"]
        if case["hardware"]["backend"] == "cuda_hopper_torchao"
        and case["world_size"] == 4
    )
    observed = {
        "platform": "cuda",
        "accelerators": ["sm90", "sm90"],
        "device_count": 2,
        "transformers_version": "5.1.0",
        "transformers_major": 5,
        "aiter_available": False,
        "torchao_available": True,
    }

    mismatches = runner.environment_mismatches(case, observed)

    assert any(
        "world_size 4 requires 4 matching devices; observed 2" in mismatch
        for mismatch in mismatches
    )


def test_environment_mismatch_record_is_not_run(runner, matrix):
    record = runner.make_environment_mismatch_record(
        case=matrix["cases"][0],
        command=["xdit"],
        environment={"platform": "cuda"},
        mismatches=["requires ROCm"],
    )

    assert record["schema_version"] == 2
    assert record["status"] == "environment_mismatch"
    assert record["execution"] == "NOT RUN"
    assert record["exit_status"] is None
    assert record["output"]["files"] == []


@pytest.mark.parametrize(
    ("statuses", "expected"),
    [
        (["passed", "passed_expected_rejection"], 0),
        (["environment_mismatch"], 2),
        (["passed", "failed_inference"], 1),
        (["environment_mismatch", "failed_missing_rejection"], 1),
    ],
)
def test_aggregate_exit_code_preserves_batch_failures(
    runner, statuses, expected
):
    assert runner.aggregate_exit_code(statuses) == expected


def test_continue_on_error_runs_remaining_cases_but_returns_failure(
    runner, matrix, monkeypatch, tmp_path
):
    cases = [dict(matrix["cases"][0]), dict(matrix["cases"][0])]
    cases[0]["id"] = "first-case"
    cases[1]["id"] = "second-case"
    fake_matrix = {
        "schema_version": 2,
        "validation_status": "NOT RUN",
        "defaults": {
            **matrix["defaults"],
            "output_root": str(tmp_path / "outputs"),
        },
        "cases": cases,
    }
    observed = {
        "platform": "rocm",
        "accelerators": ["gfx1201"],
        "transformers_major": 5,
        "aiter_available": True,
        "torchao_available": True,
    }
    statuses = iter(["failed_inference", "passed"])
    executed = []

    monkeypatch.setattr(runner, "load_matrix", lambda path: fake_matrix)
    monkeypatch.setattr(runner, "probe_environment", lambda: observed)
    monkeypatch.setattr(
        runner,
        "collect_environment",
        lambda probe: {"validation_probe": probe},
    )

    def fake_execute(case, command, **kwargs):
        executed.append(case["id"])
        return {"status": next(statuses)}

    monkeypatch.setattr(runner, "execute_case", fake_execute)

    exit_code = runner.main(["--execute", "--continue-on-error"])

    assert executed == ["first-case", "second-case"]
    assert exit_code == 1


def test_environment_mismatch_does_not_execute_case(
    runner, matrix, monkeypatch, tmp_path
):
    fake_matrix = {
        "schema_version": 2,
        "validation_status": "NOT RUN",
        "defaults": matrix["defaults"],
        "cases": [matrix["cases"][0]],
    }
    observed = {
        "platform": "cuda",
        "accelerators": ["sm89"],
        "transformers_version": "5.1.0",
        "transformers_major": 5,
        "aiter_available": False,
        "torchao_available": True,
    }
    results = tmp_path / "results.jsonl"

    monkeypatch.setattr(runner, "load_matrix", lambda path: fake_matrix)
    monkeypatch.setattr(runner, "probe_environment", lambda: observed)
    monkeypatch.setattr(
        runner,
        "collect_environment",
        lambda probe: {"validation_probe": probe},
    )
    monkeypatch.setattr(
        runner,
        "execute_case",
        lambda *args, **kwargs: pytest.fail("case must not execute"),
    )

    exit_code = runner.main(
        ["--execute", "--results", str(results)]
    )
    record = json.loads(results.read_text())

    assert exit_code == 2
    assert record["status"] == "environment_mismatch"
    assert record["execution"] == "NOT RUN"


def test_missing_placeholder_continues_batch_without_logging_value(
    runner, matrix, monkeypatch, tmp_path
):
    cases = [dict(matrix["cases"][0]), dict(matrix["cases"][0])]
    cases[0]["id"] = "missing-placeholder"
    cases[0]["args"] = ["--input_images", "${XDIT_PRIVATE_INPUT}"]
    cases[1]["id"] = "runnable-case"
    fake_matrix = {
        "schema_version": 2,
        "validation_status": "NOT RUN",
        "defaults": {
            **matrix["defaults"],
            "output_root": str(tmp_path / "outputs"),
        },
        "cases": cases,
    }
    observed = {
        "platform": "rocm",
        "accelerators": ["gfx1201"],
        "device_count": 1,
        "transformers_version": "5.1.0",
        "transformers_major": 5,
        "aiter_available": True,
        "torchao_available": True,
    }
    results = tmp_path / "results.jsonl"
    executed = []

    monkeypatch.delenv("XDIT_PRIVATE_INPUT", raising=False)
    monkeypatch.setattr(runner, "load_matrix", lambda path: fake_matrix)
    monkeypatch.setattr(runner, "probe_environment", lambda: observed)
    monkeypatch.setattr(
        runner,
        "collect_environment",
        lambda probe: {"validation_probe": probe},
    )

    def fake_execute(case, command, **kwargs):
        executed.append(case["id"])
        return {"status": "passed"}

    monkeypatch.setattr(runner, "execute_case", fake_execute)

    exit_code = runner.main(
        [
            "--execute",
            "--continue-on-error",
            "--results",
            str(results),
        ]
    )
    records = [
        json.loads(line) for line in results.read_text().splitlines()
    ]

    assert exit_code == 2
    assert executed == ["runnable-case"]
    assert len(records) == 1
    assert records[0]["status"] == "environment_mismatch"
    assert records[0]["execution"] == "NOT RUN"
    assert records[0]["command"][-1] == "${XDIT_PRIVATE_INPUT}"
    assert records[0]["environment_mismatches"] == [
        "required environment variable XDIT_PRIVATE_INPUT is unset"
    ]


def test_placeholder_expansion_is_separate_from_recorded_command(
    runner, monkeypatch
):
    template = ["xdit", "--input_images", "${XDIT_SECRET_PATH}"]
    monkeypatch.setenv("XDIT_SECRET_PATH", "/secret/value")

    expanded = runner.expand_command(template)

    assert expanded[-1] == "/secret/value"
    record = runner.make_environment_mismatch_record(
        case={
            "id": "case",
            "expected": {"outcome": "inference_success"},
            "quality_notes": "",
        },
        command=template,
        environment={},
        mismatches=["other mismatch"],
    )
    serialized = json.dumps(record)
    assert "${XDIT_SECRET_PATH}" in serialized
    assert "/secret/value" not in serialized
    redacted = runner._redact(
        "loading /secret/value",
        runner._redactions(template),
    )
    assert redacted == "loading ${XDIT_SECRET_PATH}"


@pytest.mark.parametrize(
    "arguments",
    [
        ["--list", "--execute"],
        ["--list", "--dry-run"],
    ],
)
def test_list_is_mutually_exclusive_with_action_modes(runner, arguments):
    with pytest.raises(SystemExit) as exc:
        runner._parser().parse_args(arguments)

    assert exc.value.code == 2


def test_result_record_serializes_required_fields(runner, tmp_path, matrix):
    case = matrix["cases"][0]
    record = runner.make_result_record(
        case=case,
        command=["xdit", "--model", case["model"]],
        environment={"commit_sha": "abc123"},
        exit_status=0,
        metrics={
            "peak_host_rss_bytes": 1024,
            "peak_cgroup_memory_bytes": 2048,
            "peak_gpu_memory_bytes": 4096,
            "gpu_memory_scope": "process_tree",
            "load_duration_seconds": 1.5,
            "first_forward": "succeeded",
        },
        output={
            "path": "out.png",
            "sha256": "deadbeef",
            "bytes": 10,
            "files": [
                {"path": "out.png", "sha256": "deadbeef", "bytes": 10}
            ],
        },
        log="ok",
        quality_notes="matches reference",
        reference="reference.png",
    )
    result_path = tmp_path / "results.jsonl"

    runner.append_result(result_path, record)
    loaded = json.loads(result_path.read_text().strip())

    assert loaded["schema_version"] == 2
    assert loaded["case_id"] == case["id"]
    assert loaded["case"]["hardware"] == case["hardware"]
    assert loaded["status"] == "passed"
    assert loaded["environment"]["commit_sha"] == "abc123"
    assert loaded["metrics"]["first_forward"] == "succeeded"
    assert loaded["output"]["sha256"] == "deadbeef"
    assert loaded["quality"]["reference"] == "reference.png"
