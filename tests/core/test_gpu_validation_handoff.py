"""Dependency-light contracts for the external GPU validation handoff."""

import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

ROOT = Path(__file__).resolve().parents[2]
MATRIX_PATH = ROOT / "tests/gpu_validation/matrix.json"
RUNNER_PATH = ROOT / "tools/gpu_validation.py"
GUIDE_PATH = ROOT / "docs/runner/gpu_validation_handoff.md"


@pytest.fixture(scope="module")
def runner():
    spec = importlib.util.spec_from_file_location("gpu_validation_runner", RUNNER_PATH)
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


def test_case_timeout_uses_case_then_default_then_cli_override(runner):
    defaults = {"timeout_seconds": 120}

    parsed = runner._parser().parse_args(["--timeout-seconds", "5"])
    assert parsed.timeout_seconds == 5
    assert runner.resolve_timeout_seconds({}, defaults, None) == 120
    assert runner.resolve_timeout_seconds({"timeout_seconds": 30}, defaults, None) == 30
    assert runner.resolve_timeout_seconds({"timeout_seconds": 30}, defaults, 5) == 5


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_timeout_rejects_non_finite_values(runner, matrix, value):
    with pytest.raises(ValueError, match="finite positive number"):
        runner.resolve_timeout_seconds(
            {"id": "bad-timeout", "timeout_seconds": value}, {}, None
        )

    invalid_matrix = {
        **matrix,
        "defaults": {**matrix["defaults"], "timeout_seconds": value},
    }
    with pytest.raises(ValueError, match="finite positive number"):
        runner.validate_matrix(invalid_matrix)

    invalid_case_matrix = {
        **matrix,
        "cases": [
            {**matrix["cases"][0], "timeout_seconds": value},
            *matrix["cases"][1:],
        ],
    }
    with pytest.raises(ValueError, match="finite positive number"):
        runner.validate_matrix(invalid_case_matrix)

    with pytest.raises(SystemExit) as exc:
        runner._parser().parse_args(["--timeout-seconds", str(value)])
    assert exc.value.code == 2


def test_operator_guide_is_linked_and_marks_results_not_run():
    guide = GUIDE_PATH.read_text()
    runner_guide = (ROOT / "docs/runner/runner.md").read_text()

    assert "GPU Validation Handoff" in guide
    assert "NOT RUN" in guide
    assert "gpu_validation_handoff.md" in runner_guide


def test_matrix_covers_required_validation_dimensions(matrix):
    cases = matrix["cases"]

    assert {case["hardware"]["backend"] for case in cases} >= {
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
    assert {case["checkpoint"]["source"] for case in cases} == {"hub", "local"}
    assert any("dual-transformer" in case["tags"] for case in cases)
    assert any("custom-exclusion" in case["tags"] for case in cases)
    assert {case["expected"]["outcome"] for case in cases} == {
        "inference_success",
        "preflight_failure",
    }


def test_command_generation_uses_xdit_and_torchrun(runner, matrix):
    eager = next(case for case in matrix["cases"] if case["placement"] == "eager")
    distributed = next(
        case for case in matrix["cases"] if case["placement"] == "fsdp_blockwise"
    )

    eager_command = runner.build_command(eager, matrix["defaults"], run_id="test-run")
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
    assert eager_command[output_index + 1].endswith(f"{eager['id']}/test-run")


def test_the_control_shards_the_same_way_without_the_memory_efficient_fill(runner, matrix):
    """The control's whole purpose is to differ from its blockwise case in exactly one flag.

    If it dropped --fully_shard_degree too it would stop being a control, because the memory
    difference between the pair would then include not sharding rather than only the fill strategy.
    """
    control = next(
        case for case in matrix["cases"] if case["placement"] == "fsdp_eager_fill"
    )

    command = runner.build_command(control, matrix["defaults"], run_id="test-run")

    assert "--fully_shard_degree" in command
    assert "--memory_efficient_sharding" not in command
    assert "--memory_efficient_replicated_load" not in command


def test_distributed_commands_declare_a_parallel_degree(runner, matrix):
    """Every multi-rank command must satisfy dp*cfg*sp*tp*pp == dit_parallel_size.

    --fully_shard_degree does not contribute to that product, so an FSDP case
    that only sets it aborts in config validation before the model is built.
    """
    degree_flags = (
        "--ulysses_degree",
        "--ring_degree",
        "--data_parallel_degree",
        "--tensor_parallel_degree",
        "--pipefusion_parallel_degree",
        "--use_cfg_parallel",
    )
    for case in matrix["cases"]:
        if case["world_size"] < 2:
            continue
        command = runner.build_command(case, matrix["defaults"], run_id="test-run")
        declared = [flag for flag in degree_flags if flag in command]
        assert declared, (
            f"{case['id']}: multi-rank command declares no parallel degree; "
            f"the runner will reject it before model allocation"
        )
        for flag in declared:
            if flag == "--use_cfg_parallel":
                continue
            assert command[command.index(flag) + 1] == str(case["world_size"])


@pytest.mark.parametrize("placement", ["eager", "replicated", "fsdp_blockwise"])
def test_every_placement_declares_a_parallel_degree_when_distributed(
    runner, matrix, placement
):
    """A multi-rank eager case is the control for the memory-efficient loads."""

    case = dict(
        next(case for case in matrix["cases"] if case["world_size"] == 1),
        id="synthetic",
        placement=placement,
        world_size=4,
    )

    runner.validate_matrix({**matrix, "cases": [case]})
    command = runner.build_command(case, matrix["defaults"], run_id="test-run")

    assert command[command.index("--ulysses_degree") + 1] == "4"
    memory_efficient = ("--memory_efficient_replicated_load", "--fully_shard_degree")
    if placement == "eager":
        assert not any(flag in command for flag in memory_efficient)


def test_local_checkpoint_command_keeps_env_placeholder(runner, matrix):
    case = next(
        case for case in matrix["cases"] if case["checkpoint"]["source"] == "local"
    )

    command = runner.build_command(case, matrix["defaults"], run_id="test-run")

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
    assert all(case["model"] == "Wan2.2-Distilled-I2V" for case in selected)
    assert all(case["hardware"]["backend"] == "rdna4_aiter" for case in selected)


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
            "UnsupportedLoadContract: custom checkpoint semantics are not "
            "collective-safe",
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
            "UnsupportedLoadContract: different error",
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
        runner.classify_outcome(exit_status, log, first_forward, expected, output)
        == classification
    )


def test_expected_rejection_ignores_informational_output(runner):
    """A rejection pattern must not be satisfied by routine INFO logging.

    Observed on gfx942: rocm-flux2-fp4-rejected died on a gated-repo 401, but
    the descriptor line below matches "AITER.*FP4" while reporting that FP4 was
    accepted, so the case was recorded as passed_expected_rejection.
    """
    log = (
        "INFO 08-06 10:12:04 [runner_utils.py:30] transformer quantization: "
        "requested=fp4, backend=aiter, storage=aiter_mxfp4_per_1x32, "
        "materialization=blockwise\n"
        "[rank0]: huggingface_hub.errors.GatedRepoError: 401 Client Error.\n"
    )
    expected = {
        "outcome": "preflight_failure",
        "error_pattern": "FP4.*AITER|AITER.*FP4",
    }
    output = {"path": None, "sha256": None, "files": []}

    assert (
        runner.classify_outcome(1, log, "not_reached", expected, output)
        == "failed_wrong_rejection"
    )


def test_expected_rejection_matches_the_raised_error(runner):
    """The real rejection still passes: captured from rocm-zimage-int8-rejected."""
    log = (
        "INFO 08-06 09:48:11 [runner_utils.py:30] Initializing model: Z-Image-Turbo\n"
        '[rank0]:     raise ValueError("Int8 GEMMs on ROCm are not supported.")\n'
        "[rank0]: ValueError: Int8 GEMMs on ROCm are not supported.\n"
    )
    expected = {
        "outcome": "preflight_failure",
        "error_pattern": "INT8.*ROCm|ROCm.*INT8",
    }
    output = {"path": None, "sha256": None, "files": []}

    assert (
        runner.classify_outcome(1, log, "not_reached", expected, output)
        == "passed_expected_rejection"
    )


def test_failure_text_selects_only_failure_lines(runner):
    log = (
        "INFO 08-06 10:00:26 [runner_utils.py:30] Running model...\n"
        "WARNING 08-06 10:00:06 [runtime_state.py:129] Using AITER attention.\n"
        "[rank0]: Traceback (most recent call last):\n"
        "[rank0]: RuntimeError: kernel unavailable\n"
        "E0806 10:12:33.874000 28461 api.py:882] failed (exitcode: -6)\n"
    )

    selected = runner.failure_text(log)

    assert "RuntimeError: kernel unavailable" in selected
    assert "failed (exitcode: -6)" in selected
    assert "Running model" not in selected
    assert "Using AITER attention" not in selected


def test_output_discovery_requires_new_nonempty_hashed_artifact(runner, tmp_path):
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
    assert output_dir == (tmp_path / "relative-output" / case["id"] / "outside-cwd")
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


def test_probe_detects_dependencies_without_distribution_metadata(runner):
    """AITER vendored as a source checkout has no dist-info but still works.

    The amdsiloai/pytorch-xdit image ships AITER this way. Keying availability
    on importlib.metadata alone reported aiter_available False while the runner
    reported has_aiter True, which would refuse every rdna4_aiter case on
    genuine RDNA4 hardware.
    """
    observed = runner.probe_environment(
        command_runner=lambda command: (
            "Name: gfx1201" if command[0] == "rocminfo" else None
        ),
        version_getter={"transformers": "5.1.0"}.get,
        module_probe=lambda name: name in {"aiter", "torchao"},
        environ={},
    )

    assert observed["aiter_available"] is True
    assert observed["torchao_available"] is True


def test_probe_reports_missing_dependencies_as_unavailable(runner):
    observed = runner.probe_environment(
        command_runner=lambda command: (
            "Name: gfx1201" if command[0] == "rocminfo" else None
        ),
        version_getter={"transformers": "5.1.0"}.get,
        module_probe=lambda name: False,
        environ={},
    )

    assert observed["aiter_available"] is False
    assert observed["torchao_available"] is False


def test_environment_probe_applies_cuda_visible_devices_mask(runner):
    responses = {
        "nvidia-smi": ("0, GPU-aaaa, 8.9\n" "1, GPU-bbbb, 8.9\n" "2, GPU-cccc, 9.0")
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


def test_rocm_cases_are_pinned_to_the_arch_they_declare(runner, matrix):
    """A gfx942 case must not be selected on gfx950, and the reverse.

    The backend token only separates RDNA4 from the rest, so before this was enforced
    every gfx942 case also matched a gfx950 host. That silently mis-scores FP4: AITER
    builds no FP4 kernels for gfx942, so those cases assert a rejection that does not
    happen on gfx950, where the probe reports MXFP4 as available.
    """
    case = next(
        case
        for case in matrix["cases"]
        if case["hardware"]["accelerator"] == "gfx942" and case["world_size"] == 1
    )
    observed = {
        "platform": "rocm",
        "transformers_major": 5,
        "aiter_available": True,
        "torchao_available": True,
    }

    assert runner.environment_mismatches(
        case, {**observed, "accelerators": ["gfx942"]}
    ) == []
    assert any(
        "gfx942" in mismatch
        for mismatch in runner.environment_mismatches(
            case, {**observed, "accelerators": ["gfx950"]}
        )
    )

    gfx950_case = next(
        case
        for case in matrix["cases"]
        if case["hardware"]["accelerator"] == "gfx950" and case["world_size"] == 1
    )

    assert runner.environment_mismatches(
        gfx950_case, {**observed, "accelerators": ["gfx950"]}
    ) == []
    assert any(
        "gfx950" in mismatch
        for mismatch in runner.environment_mismatches(
            gfx950_case, {**observed, "accelerators": ["gfx942"]}
        )
    )


def test_non_rdna4_token_still_admits_any_non_rdna4_arch(runner, matrix):
    """Narrowing must not have turned the broad token into an exact match.

    FP8 and INT8 keep it deliberately: their gates are RDNA4-versus-rest and
    CUDA-versus-ROCm, neither of which distinguishes gfx942 from gfx950.
    """
    case = next(
        case
        for case in matrix["cases"]
        if case["hardware"]["accelerator"] == "non_rdna4_rocm"
        and case["world_size"] == 1
    )
    observed = {
        "platform": "rocm",
        "transformers_major": 5,
        "aiter_available": True,
        "torchao_available": True,
    }

    for accelerator in ("gfx942", "gfx950", "gfx90a"):
        assert runner.environment_mismatches(
            case, {**observed, "accelerators": [accelerator]}
        ) == [], accelerator
    assert any(
        "non_rdna4_rocm" in mismatch
        for mismatch in runner.environment_mismatches(
            case, {**observed, "accelerators": ["gfx1201"]}
        )
    )


def test_mi3xx_token_admits_both_datacentre_archs_but_not_older_rocm(runner, matrix):
    """The pair token exists so behaviour shared by gfx942 and gfx950 is not pinned to one.

    It must stay narrower than non_rdna4_rocm: cases carrying it assume the FP8 support
    that envs._on_mi3xx gates on, which gfx90a does not have.
    """
    case = next(
        case
        for case in matrix["cases"]
        if case["hardware"]["accelerator"] == "gfx942_or_gfx950"
        and case["world_size"] == 1
    )
    observed = {
        "platform": "rocm",
        "transformers_major": 5,
        "aiter_available": True,
        "torchao_available": True,
    }

    for accelerator in ("gfx942", "gfx950"):
        assert runner.environment_mismatches(
            case, {**observed, "accelerators": [accelerator]}
        ) == [], accelerator
    for accelerator in ("gfx90a", "gfx1201"):
        assert any(
            "gfx942_or_gfx950" in mismatch
            for mismatch in runner.environment_mismatches(
                case, {**observed, "accelerators": [accelerator]}
            )
        ), accelerator


def test_an_unknown_accelerator_token_is_rejected_not_silently_skipped(runner, matrix):
    """A token nothing recognises matches no device, so it must fail loudly."""
    case = dict(matrix["cases"][0])
    case["hardware"] = {**case["hardware"], "accelerator": "gfx9999"}
    broken = {**matrix, "cases": [case]}

    with pytest.raises(ValueError, match="invalid hardware declaration"):
        runner.validate_matrix(broken)


def test_environment_validation_rejects_insufficient_matching_devices(runner, matrix):
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
def test_aggregate_exit_code_preserves_batch_failures(runner, statuses, expected):
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
    statuses = iter(["timed_out", "passed"])
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


def test_execute_case_times_out_and_kills_isolated_process_group(
    runner, monkeypatch, tmp_path
):
    child_state = tmp_path / "child.json"
    child_ready = tmp_path / "child.ready"
    output_dir = tmp_path / "output"
    results = tmp_path / "results.jsonl"
    child_code = (
        "import pathlib,signal,sys,time;"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN);"
        "pathlib.Path(sys.argv[1]).write_text('ready');"
        "time.sleep(60)"
    )
    parent_code = (
        "import json,os,pathlib,signal,subprocess,sys,time;"
        f"child=subprocess.Popen([sys.executable,'-c',{child_code!r},sys.argv[2]]);"
        "ready=pathlib.Path(sys.argv[2]);"
        "\nwhile not ready.exists(): time.sleep(0.01)\n"
        "open(sys.argv[1],'w').write(json.dumps("
        "{'pid':child.pid,'pgrp':os.getpgrp()}));"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN);"
        "time.sleep(60)"
    )
    command = [
        sys.executable,
        "-c",
        parent_code,
        str(child_state),
        str(child_ready),
        "--output_directory",
        str(output_dir),
    ]
    case = {
        "id": "timeout-case",
        "expected": {"outcome": "inference_success"},
        "quality_notes": "",
    }

    class NullMonitor:
        peak_host_rss = 0
        peak_cgroup = 0
        peak_gpu = None
        gpu_scope = None
        peak_host_anon = 0
        peak_host_file_cache = 0

        def peak_gpu_between(self, start, end):
            return None

        def __init__(self, root_pid):
            pass

        def start(self):
            pass

        def stop(self):
            pass

    monkeypatch.setattr(runner, "ResourceMonitor", NullMonitor)
    monkeypatch.setattr(runner, "PROCESS_TERMINATE_GRACE_SECONDS", 0.1)
    real_popen = runner.subprocess.Popen
    real_killpg = runner.os.killpg
    killed = False
    wrapped_processes = []

    class DelayedReapProcess:
        def __init__(self, process):
            self.process = process
            self.pid = process.pid
            self.stdout = process.stdout
            self._returncode = None

        @property
        def returncode(self):
            return self._returncode

        def poll(self):
            if killed and self._returncode is None:
                return None
            self._returncode = self.process.poll()
            return self._returncode

        def wait(self, timeout=None):
            if killed and self._returncode is None:
                scheduling_interval = 0.03
                if timeout is not None and timeout < scheduling_interval:
                    raise runner.subprocess.TimeoutExpired(self.process.args, timeout)
                time.sleep(scheduling_interval)
                if timeout is not None:
                    timeout -= scheduling_interval
            self._returncode = self.process.wait(timeout=timeout)
            return self._returncode

    def popen_after_process_tree_is_ready(*args, **kwargs):
        process = real_popen(*args, **kwargs)
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and not child_state.exists():
            time.sleep(0.01)
        if not child_state.exists():
            os.killpg(process.pid, 9)
            pytest.fail("child process tree did not become ready")
        wrapped = DelayedReapProcess(process)
        wrapped_processes.append(wrapped)
        return wrapped

    def track_killpg(process_group, sig):
        nonlocal killed
        if sig == runner.signal.SIGKILL:
            killed = True
        return real_killpg(process_group, sig)

    monkeypatch.setattr(runner.subprocess, "Popen", popen_after_process_tree_is_ready)
    monkeypatch.setattr(runner.os, "killpg", track_killpg)

    record = runner.execute_case(
        case,
        command,
        results_path=results,
        quality_notes="",
        reference=None,
        environment={},
        timeout_seconds=0.05,
    )
    for wrapped in wrapped_processes:
        if wrapped.process.poll() is None:
            real_killpg(wrapped.pid, runner.signal.SIGKILL)
        wrapped.process.wait(timeout=2)

    child = json.loads(child_state.read_text())
    assert child["pgrp"] != os.getpgrp()
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        stat_path = Path(f"/proc/{child['pid']}/stat")
        if not stat_path.exists() or stat_path.read_text().split()[2] == "Z":
            break
        time.sleep(0.02)
    else:
        subprocess.run(
            ["kill", "-KILL", str(child["pid"])],
            check=False,
            capture_output=True,
        )
        pytest.fail("timed-out grandchild process remained alive")

    assert record["status"] == "timed_out"
    assert record["timed_out"] is True
    assert record["timeout_seconds"] == 0.05
    assert record["execution"] == "RAN"
    assert isinstance(record["exit_status"], int)
    assert json.loads(results.read_text())["status"] == "timed_out"


def test_term_grace_is_not_shortened_when_root_closes_stdout(
    runner, monkeypatch, tmp_path
):
    ready = tmp_path / "ready"
    term_complete = tmp_path / "term-complete"
    output_dir = tmp_path / "output"
    results = tmp_path / "results.jsonl"
    program = (
        "import os,pathlib,signal,sys,time;"
        "ready=pathlib.Path(sys.argv[1]);"
        "complete=pathlib.Path(sys.argv[2]);"
        "\ndef handle_term(signum,frame):\n"
        " time.sleep(0.12)\n"
        " complete.write_text('complete')\n"
        " raise SystemExit(0)\n"
        "signal.signal(signal.SIGTERM,handle_term);"
        "ready.write_text('ready');"
        "os.close(1);os.close(2);"
        "time.sleep(60)"
    )
    command = [
        sys.executable,
        "-c",
        program,
        str(ready),
        str(term_complete),
        "--output_directory",
        str(output_dir),
    ]
    case = {
        "id": "term-grace",
        "expected": {"outcome": "inference_success"},
        "quality_notes": "",
    }

    class NullMonitor:
        peak_host_rss = 0
        peak_cgroup = 0
        peak_gpu = None
        gpu_scope = None
        peak_host_anon = 0
        peak_host_file_cache = 0

        def peak_gpu_between(self, start, end):
            return None

        def __init__(self, root_pid):
            pass

        def start(self):
            pass

        def stop(self):
            pass

    monkeypatch.setattr(runner, "ResourceMonitor", NullMonitor)
    monkeypatch.setattr(runner, "PROCESS_TERMINATE_GRACE_SECONDS", 0.3)
    real_popen = runner.subprocess.Popen

    def popen_after_root_is_ready(*args, **kwargs):
        process = real_popen(*args, **kwargs)
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and not ready.exists():
            time.sleep(0.01)
        if not ready.exists():
            os.killpg(process.pid, 9)
            pytest.fail("root process did not become ready")
        return process

    monkeypatch.setattr(runner.subprocess, "Popen", popen_after_root_is_ready)

    record = runner.execute_case(
        case,
        command,
        results_path=results,
        quality_notes="",
        reference=None,
        environment={},
        timeout_seconds=0.05,
    )

    assert term_complete.read_text() == "complete"
    assert record["status"] == "timed_out"
    assert record["exit_status"] == 0


@pytest.mark.parametrize("race_signal", [signal.SIGTERM, signal.SIGKILL])
def test_process_lookup_race_still_reaps_root(runner, monkeypatch, race_signal):
    class Process:
        pid = 12345

        def poll(self):
            return None

        def wait(self, timeout=None):
            assert timeout is not None
            return -signal.SIGKILL

    class Reader:
        def join(self, timeout=None):
            assert timeout is not None

    def killpg(process_group, sent_signal):
        assert process_group == Process.pid
        if sent_signal == race_signal:
            raise ProcessLookupError

    monkeypatch.setattr(runner.os, "killpg", killpg)
    monkeypatch.setattr(
        runner,
        "_process_group_exists",
        lambda process_group: race_signal == signal.SIGKILL,
    )

    assert runner._terminate_process_group(Process(), Reader(), 0.01) == -signal.SIGKILL


def test_execute_case_bounds_drain_when_ready_descendant_retains_stdout(
    runner, monkeypatch, tmp_path
):
    ready = tmp_path / "ready"
    parent_ready = tmp_path / "parent-ready"
    output_dir = tmp_path / "output"
    results = tmp_path / "results.jsonl"
    child_code = (
        "import os,pathlib,signal,sys,time;"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN);"
        "print('grandchild-ready',flush=True);"
        "pathlib.Path(sys.argv[1]).write_text(str(os.getpid()));"
        "time.sleep(1)"
    )
    parent_code = (
        "import pathlib,subprocess,sys,time;"
        f"subprocess.Popen([sys.executable,'-c',{child_code!r},sys.argv[1]]);"
        "ready=pathlib.Path(sys.argv[1]);"
        "\nwhile not ready.exists(): time.sleep(0.01)\n"
        "print('parent-ready',flush=True);"
        "pathlib.Path(sys.argv[2]).write_text('ready')"
    )
    command = [
        sys.executable,
        "-c",
        parent_code,
        str(ready),
        str(parent_ready),
        "--output_directory",
        str(output_dir),
    ]
    case = {
        "id": "retained-stdout",
        "expected": {"outcome": "inference_success"},
        "quality_notes": "",
    }

    class NullMonitor:
        peak_host_rss = 0
        peak_cgroup = 0
        peak_gpu = None
        gpu_scope = None
        peak_host_anon = 0
        peak_host_file_cache = 0

        def peak_gpu_between(self, start, end):
            return None

        def __init__(self, root_pid):
            pass

        def start(self):
            pass

        def stop(self):
            pass

    monkeypatch.setattr(runner, "ResourceMonitor", NullMonitor)
    monkeypatch.setattr(runner, "PROCESS_TERMINATE_GRACE_SECONDS", 0.1)
    real_popen = runner.subprocess.Popen

    def popen_after_grandchild_is_ready(*args, **kwargs):
        process = real_popen(*args, **kwargs)
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and not parent_ready.exists():
            time.sleep(0.01)
        if not parent_ready.exists():
            os.killpg(process.pid, 9)
            pytest.fail("parent did not emit readiness output")
        return process

    monkeypatch.setattr(runner.subprocess, "Popen", popen_after_grandchild_is_ready)

    started = time.monotonic()
    record = runner.execute_case(
        case,
        command,
        results_path=results,
        quality_notes="",
        reference=None,
        environment={},
        timeout_seconds=0.05,
    )
    elapsed = time.monotonic() - started

    log = (output_dir / "validation.log").read_text()
    child_pid = int(ready.read_text())
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        stat_path = Path(f"/proc/{child_pid}/stat")
        if not stat_path.exists() or stat_path.read_text().split()[2] == "Z":
            break
        time.sleep(0.02)
    else:
        subprocess.run(
            ["kill", "-KILL", str(child_pid)],
            check=False,
            capture_output=True,
        )
        pytest.fail("stdout-retaining grandchild remained alive")

    assert elapsed < 0.7
    assert record["status"] == "timed_out"
    assert "parent-ready" in log
    assert "grandchild-ready" in log


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

    exit_code = runner.main(["--execute", "--results", str(results)])
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
    records = [json.loads(line) for line in results.read_text().splitlines()]

    assert exit_code == 2
    assert executed == ["runnable-case"]
    assert len(records) == 1
    assert records[0]["status"] == "environment_mismatch"
    assert records[0]["execution"] == "NOT RUN"
    assert records[0]["command"][-1] == "${XDIT_PRIVATE_INPUT}"
    assert records[0]["environment_mismatches"] == [
        "required environment variable XDIT_PRIVATE_INPUT is unset"
    ]


def test_placeholder_expansion_is_separate_from_recorded_command(runner, monkeypatch):
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


def test_vram_is_reported_per_device_so_rank_count_cannot_inflate_it(runner, monkeypatch):
    """Eight ranks holding a shard each must not read as eight times the memory.

    Summing devices made a sharded run look worse the wider it was spread, which is the opposite of
    what sharding does and made the dashboard unable to show the load feature working at all.
    """
    idle, busy = 300 * 1024**2, 25 * 1024**3
    monkeypatch.setattr(
        runner,
        "_run_text",
        lambda command: json.dumps(
            {
                f"card{index}": {"VRAM Total Used Memory (B)": str(busy if index else idle)}
                for index in range(8)
            }
        ),
    )

    by_device = runner._rocm_global_memory()

    assert len(by_device) == 8
    assert max(by_device.values()) == busy
    assert sum(by_device.values()) > busy, "the test data must distinguish max from sum"


def test_host_memory_separates_allocations_from_reclaimable_cache(runner, monkeypatch, tmp_path):
    """Summing RSS over ranks re-counted shared pages; the cgroup counts each page once.

    The split matters as much as the total: mmap'd checkpoint cache is reclaimable, so folding it
    into the reported figure would let page cache look like a load cost.
    """
    stat = tmp_path / "memory.stat"
    stat.write_text("anon 42000000000\nfile 31000000000\nkernel 5000\n")
    monkeypatch.setattr(runner, "Path", lambda p: stat if "memory.stat" in str(p) else Path(p))

    anon, file_cache = runner._cgroup_memory_breakdown()

    assert anon == 42_000_000_000
    assert file_cache == 31_000_000_000


def test_host_memory_breakdown_is_absent_rather_than_wrong_off_cgroups(runner, monkeypatch):
    monkeypatch.setattr(
        runner, "Path", lambda p: Path("/nonexistent/memory.stat")
    )

    assert runner._cgroup_memory_breakdown() == (None, None)


def test_the_load_phase_peak_is_taken_from_the_load_window_only(runner):
    """The whole-run peak is dominated by inference, so it cannot show what the load cost."""
    monitor = runner.ResourceMonitor(root_pid=os.getpid())
    monitor.gpu_samples = [
        (100.0, 3 * 1024**3),  # during load
        (110.0, 4 * 1024**3),  # during load
        (200.0, 90 * 1024**3),  # inference, long after the load finished
    ]

    assert monitor.peak_gpu_between(99.0, 120.0) == 4 * 1024**3
    assert monitor.peak_gpu_between(300.0, 400.0) is None


def test_compile_warmup_is_timed_separately_from_the_load(runner, monkeypatch, tmp_path):
    """Compile runs inside model initialization, so the load window used to swallow it.

    It is near-constant for a model while the load is what we are changing, so including it shrank
    the measured difference between load strategies towards nothing.
    """
    output_dir = tmp_path / "output"
    results = tmp_path / "results.jsonl"
    child_code = (
        "import time,sys;"
        "print('Initializing model: fake',flush=True);"
        "time.sleep(0.2);"
        "print('Torch.compile enabled. Warming up torch compiler ...',flush=True);"
        "time.sleep(0.6);"
        "print('Model initialization complete.',flush=True);"
        "print('Running model...',flush=True)"
    )
    command = [
        sys.executable,
        "-c",
        child_code,
        "--output_directory",
        str(output_dir),
    ]
    case = {
        "id": "compile-split",
        "expected": {"outcome": "inference_success"},
        "quality_notes": "",
    }

    class NullMonitor:
        peak_host_rss = 0
        peak_cgroup = 0
        peak_gpu = None
        gpu_scope = None
        peak_host_anon = 0
        peak_host_file_cache = 0

        def peak_gpu_between(self, start, end):
            return None

        def __init__(self, root_pid):
            pass

        def start(self):
            pass

        def stop(self):
            pass

    monkeypatch.setattr(runner, "ResourceMonitor", NullMonitor)

    record = runner.execute_case(
        case,
        command,
        results_path=results,
        quality_notes="",
        reference=None,
        environment={},
        timeout_seconds=30,
    )

    metrics = record["metrics"]
    # Generous bounds: the point is which side of the marker each phase lands on, not the sleeps.
    assert metrics["load_duration_seconds"] < 0.5
    assert metrics["compile_duration_seconds"] > 0.5


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
            "files": [{"path": "out.png", "sha256": "deadbeef", "bytes": 10}],
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


def test_a_case_is_scored_against_an_unquantized_single_rank_load_of_its_own_model(runner):
    """Judging a quantized or sharded load needs something known-good to judge it against.

    Matched on attributes rather than by name, so a regenerated or renamed matrix cannot leave a
    case quietly scoring against the wrong thing.
    """
    cases = [
        {
            "id": "ref",
            "model": "Z",
            "placement": "eager",
            "quantization": "none",
            "offload": "none",
            "te_fp8": False,
            "world_size": 1,
        },
        {
            "id": "other-model-ref",
            "model": "Y",
            "placement": "eager",
            "quantization": "none",
            "offload": "none",
            "te_fp8": False,
            "world_size": 1,
        },
        {
            "id": "fp8",
            "model": "Z",
            "placement": "fsdp_blockwise",
            "quantization": "fp8",
            "offload": "none",
            "te_fp8": True,
            "world_size": 8,
        },
    ]

    assert runner.reference_case_id(cases[2], cases) == "ref"
    assert runner.reference_case_id(cases[0], cases) is None, (
        "the reference cannot be scored against itself"
    )


def test_a_case_with_no_reference_in_the_matrix_is_left_unscored(runner):
    """Better to report nothing than to invent a baseline from a different model or rank count."""
    cases = [
        {
            "id": "fp8",
            "model": "Z",
            "placement": "fsdp_blockwise",
            "quantization": "fp8",
            "offload": "none",
            "te_fp8": True,
            "world_size": 8,
        }
    ]

    assert runner.reference_case_id(cases[0], cases) is None


def test_references_run_before_the_cases_that_need_them(runner):
    """Otherwise whether a case gets scored depends on the order it was asked for."""
    cases = [
        {
            "id": "fp8",
            "model": "Z",
            "placement": "fsdp_blockwise",
            "quantization": "fp8",
            "offload": "none",
            "te_fp8": True,
            "world_size": 8,
        },
        {
            "id": "ref",
            "model": "Z",
            "placement": "eager",
            "quantization": "none",
            "offload": "none",
            "te_fp8": False,
            "world_size": 1,
        },
    ]

    assert [case["id"] for case in runner.order_references_first(cases)] == ["ref", "fp8"]


def test_scoring_runs_drop_torch_compile(runner):
    """Compile picks kernels by timing, and different fp8 kernels give different images.

    Measured 2 distinct outputs in 3 compiled runs against byte-identical output in 3 uncompiled
    ones, a spread as large as quantization itself, so a compiled comparison mostly reports which
    kernel won.
    """
    command = ["torchrun", "-m", "xfuser.runner", "--use_fp8_gemms", "--use_torch_compile"]

    assert runner.without_compile(command) == [
        "torchrun",
        "-m",
        "xfuser.runner",
        "--use_fp8_gemms",
    ]


def test_a_divergent_image_fails_the_case_rather_than_only_annotating_it(runner):
    """A gate that records a number nobody acts on is not a gate."""
    failed = {"verdict": "fail", "scores": {"comparable": True, "ssim": 0.1}}

    assert runner.quality_status("passed", failed) == "failed_quality"
    assert runner.quality_status("passed", {"verdict": "pass"}) == "passed"
    assert runner.quality_status("passed", None) == "passed"


def test_a_run_that_already_failed_keeps_its_own_reason(runner):
    """The earlier status says what went wrong; a quality verdict would only make it vaguer."""
    failed = {"verdict": "fail", "scores": {"comparable": True, "ssim": 0.1}}

    assert runner.quality_status("failed_inference", failed) == "failed_inference"
    assert runner.quality_status("failed_no_output", failed) == "failed_no_output"


def test_a_missing_artifact_is_reported_as_unscored_not_as_a_pass(runner):
    """Silence here would read as agreement, when nothing was actually compared."""
    result = runner.score_against_reference(
        {"path": None}, {"path": "ref.png"}, reference_id="ref"
    )

    assert result["verdict"] == "unscored"
    assert "no image artifact" in result["reason"]
    assert runner.quality_status("passed", result) == "passed", (
        "an unscored run is not a failing one"
    )


def test_nothing_is_scored_when_the_case_has_no_reference(runner):
    """Scoring stays opt-in, so an ordinary run keeps whatever the operator recorded."""
    assert runner.score_against_reference({"path": "a.png"}, None, reference_id=None) is None


def _reference_record(recorded_at, path, compiled=False, execution="RAN"):
    return {
        "case_id": "ref",
        "execution": execution,
        "recorded_at": recorded_at,
        "command": ["torchrun"] + (["--use_torch_compile"] if compiled else []),
        "output": {"path": path, "sha256": "ab"},
    }


def test_a_reference_already_run_is_reused_instead_of_run_again(runner, tmp_path):
    """Scoring one case is the normal use, and re-running an 8-rank reference for it costs more
    than the candidate does."""
    results = tmp_path / "results.jsonl"
    results.write_text(
        "\n".join(
            json.dumps(record)
            for record in [
                _reference_record("2026-01-01T00:00:00+00:00", "old.png"),
                _reference_record("2026-03-01T00:00:00+00:00", "new.png"),
            ]
        )
        + "\n"
    )

    assert runner.recorded_reference_output(results, "ref")["path"] == "new.png"


def test_a_compiled_reference_is_not_reused(runner, tmp_path):
    """It would put back the kernel-choice spread scoring exists to remove, and do it invisibly."""
    results = tmp_path / "results.jsonl"
    results.write_text(json.dumps(_reference_record("2026-01-01T00:00:00+00:00", "c.png", compiled=True)) + "\n")

    assert runner.recorded_reference_output(results, "ref") is None


def test_a_reference_that_never_ran_is_not_invented(runner, tmp_path):
    """Nothing to compare against has to stay nothing, not a stale or absent artifact."""
    results = tmp_path / "results.jsonl"
    results.write_text(
        json.dumps(_reference_record("2026-01-01T00:00:00+00:00", None))
        + "\n"
        + json.dumps(_reference_record("2026-02-01T00:00:00+00:00", "x.png", execution="NOT RUN"))
        + "\n"
    )

    assert runner.recorded_reference_output(results, "ref") is None
    assert runner.recorded_reference_output(tmp_path / "absent.jsonl", "ref") is None


def test_a_truncated_results_file_does_not_stop_the_lookup(runner, tmp_path):
    """Results are appended per case, so an interrupted run can leave a partial final line."""
    results = tmp_path / "results.jsonl"
    results.write_text(
        json.dumps(_reference_record("2026-01-01T00:00:00+00:00", "good.png"))
        + "\n"
        + '{"case_id": "ref", "output": {"path": "trunc'
    )

    assert runner.recorded_reference_output(results, "ref")["path"] == "good.png"
