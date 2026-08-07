#!/usr/bin/env python3
"""Plan, execute, and record the external xDiT GPU validation matrix."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import datetime as dt
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import re
import signal
import shlex
import shutil
import subprocess
import sys
import threading
import time
from typing import Any
import uuid

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MATRIX = ROOT / "tests/gpu_validation/matrix.json"
# Bumped when a metric's meaning changes, so a reader never puts two definitions in one column.
# 2: GPU memory became the busiest single device's peak; it was previously summed over every device
#    on the node, which grew with the rank count and so hid what sharding did.
# 3: Host memory became the container's anonymous pages. It was summed RSS over the process tree,
#    which re-counted the pages ranks share and so also grew with the rank count.
# 4: The load window now ends at compile warmup instead of running to the end of initialization, so
#    load seconds and the load-phase VRAM peak no longer include tens of seconds of compiling.
GPU_METRICS_VERSION = 4
ID_PATTERN = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
REQUIRED_CASE_FIELDS = {
    "id",
    "tags",
    "model",
    "model_family",
    "hardware",
    "placement",
    "quantization",
    "te_fp8",
    "offload",
    "transformers",
    "checkpoint",
    "world_size",
    "args",
    "expected",
    "quality_notes",
}
BACKENDS = {
    "rdna4_aiter",
    "rocm_torchao",
    "cuda_ada_torchao",
    "cuda_hopper_torchao",
    "cuda_blackwell_torchao",
}
ROCM_ACCELERATORS = {
    "gfx1200_or_gfx1201",
    "non_rdna4_rocm",
    "gfx942_or_gfx950",
    "gfx942",
    "gfx950",
}
# Unknown tokens have to be rejected here: a token nothing recognises matches no device,
# so a typo would silently skip every case that carries it instead of failing.
ACCELERATORS = ROCM_ACCELERATORS | {"sm89", "sm90", "sm100_or_newer"}
# fsdp_eager_fill is the control for fsdp_blockwise: same rank count, same FSDP sharding, same
# final sharded state, but weights are materialised in full before being sharded. Without it a
# memory figure from a blockwise run cannot be attributed to the fill strategy, since spreading a
# model over more ranks changes host and device memory on its own.
PLACEMENTS = {"eager", "replicated", "fsdp_blockwise", "fsdp_eager_fill"}
QUANTIZATION = {"none", "fp8", "fp4", "int8", "hybrid_fp8_fp4"}
OFFLOADS = {"none", "model", "sequential", "group", "group_low_cpu_mem"}
EXPECTED_OUTCOMES = {"inference_success", "preflight_failure"}
GENERATED_ARTIFACT_EXTENSIONS = frozenset(
    {".png", ".jpg", ".jpeg", ".webp", ".gif", ".mp4", ".webm", ".mov"}
)
PACKAGE_NAMES = (
    "torch",
    "torchao",
    "aiter",
    "diffusers",
    "transformers",
    "accelerate",
    "huggingface-hub",
    "xfuser",
)
PROCESS_TERMINATE_GRACE_SECONDS = 5.0


def _positive_seconds(value: Any, *, field: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"{field} must be a finite positive number")
    return value


def resolve_timeout_seconds(
    case: dict[str, Any],
    defaults: dict[str, Any],
    cli_override: float | None,
) -> float:
    value = (
        cli_override
        if cli_override is not None
        else case.get("timeout_seconds", defaults.get("timeout_seconds"))
    )
    if value is None:
        raise ValueError(
            f"{case.get('id', 'case')}: timeout_seconds is required "
            "in the case, matrix defaults, or CLI"
        )
    return _positive_seconds(value, field="timeout_seconds")


def _timeout_argument(value: str) -> float:
    try:
        return _positive_seconds(float(value), field="timeout_seconds")
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def load_matrix(path: Path | str) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        matrix = json.load(handle)
    validate_matrix(matrix)
    return matrix


def validate_matrix(matrix: dict[str, Any]) -> None:
    if matrix.get("schema_version") != 2:
        raise ValueError("matrix schema_version must be 2")
    if matrix.get("validation_status") != "NOT RUN":
        raise ValueError("checked-in matrix validation_status must remain 'NOT RUN'")
    if not isinstance(matrix.get("defaults"), dict):
        raise ValueError("matrix defaults must be an object")
    default_timeout = matrix["defaults"].get("timeout_seconds")
    if default_timeout is not None:
        _positive_seconds(default_timeout, field="matrix defaults timeout_seconds")
    cases = matrix.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("matrix cases must be a non-empty list")

    seen: set[str] = set()
    for index, case in enumerate(cases):
        missing = REQUIRED_CASE_FIELDS - set(case)
        if missing:
            raise ValueError(f"case {index} missing fields: {sorted(missing)}")
        case_id = case["id"]
        if not isinstance(case_id, str) or not ID_PATTERN.fullmatch(case_id):
            raise ValueError(f"invalid case id: {case_id!r}")
        if case_id in seen:
            raise ValueError(f"duplicate case id: {case_id}")
        seen.add(case_id)
        timeout = case.get("timeout_seconds", default_timeout)
        if timeout is None:
            raise ValueError(
                f"{case_id}: timeout_seconds is required in the case or defaults"
            )
        _positive_seconds(timeout, field=f"{case_id}: timeout_seconds")
        if not isinstance(case["tags"], list) or not all(
            isinstance(tag, str) and tag for tag in case["tags"]
        ):
            raise ValueError(f"{case_id}: tags must be non-empty strings")
        hardware = case["hardware"]
        if (
            not isinstance(hardware, dict)
            or hardware.get("backend") not in BACKENDS
            or hardware.get("accelerator") not in ACCELERATORS
        ):
            raise ValueError(f"{case_id}: invalid hardware declaration")
        if case["placement"] not in PLACEMENTS:
            raise ValueError(f"{case_id}: invalid placement")
        if case["quantization"] not in QUANTIZATION:
            raise ValueError(f"{case_id}: invalid quantization")
        if not isinstance(case["te_fp8"], bool):
            raise ValueError(f"{case_id}: te_fp8 must be boolean")
        if case["te_fp8"] and case["quantization"] not in {
            "fp8",
            "hybrid_fp8_fp4",
        }:
            raise ValueError(f"{case_id}: te_fp8 requires FP8 quantization")
        if case["offload"] not in OFFLOADS:
            raise ValueError(f"{case_id}: invalid offload")
        if case["transformers"] not in {"4.x", "5.x"}:
            raise ValueError(f"{case_id}: transformers must be 4.x or 5.x")
        checkpoint = case["checkpoint"]
        if not isinstance(checkpoint, dict) or checkpoint.get("source") not in {
            "hub",
            "local",
        }:
            raise ValueError(f"{case_id}: invalid checkpoint")
        if not checkpoint.get("value"):
            raise ValueError(f"{case_id}: checkpoint value is required")
        if checkpoint["source"] == "local" and not checkpoint.get("env"):
            raise ValueError(f"{case_id}: local checkpoint requires env")
        if not isinstance(case["world_size"], int) or case["world_size"] < 1:
            raise ValueError(f"{case_id}: world_size must be positive")
        if case["placement"] != "eager" and case["world_size"] < 2:
            raise ValueError(
                f"{case_id}: distributed placement requires multiple ranks"
            )
        if not isinstance(case["args"], list) or not all(
            isinstance(arg, str) for arg in case["args"]
        ):
            raise ValueError(f"{case_id}: args must be strings")
        expected = case["expected"]
        if (
            not isinstance(expected, dict)
            or expected.get("outcome") not in EXPECTED_OUTCOMES
        ):
            raise ValueError(f"{case_id}: invalid expected outcome")
        if expected["outcome"] == "preflight_failure" and not expected.get(
            "error_pattern"
        ):
            raise ValueError(f"{case_id}: preflight failure needs error_pattern")
        if expected.get("error_pattern"):
            re.compile(expected["error_pattern"])


def _model_matches(case: dict[str, Any], requested: str) -> bool:
    needle = requested.casefold()
    return needle in {
        case["model"].casefold(),
        case["model_family"].casefold(),
        case["checkpoint"]["value"].casefold(),
    }


def select_cases(
    cases: list[dict[str, Any]],
    *,
    tags: list[str],
    models: list[str],
    backends: list[str],
    case_ids: list[str],
) -> list[dict[str, Any]]:
    return [
        case
        for case in cases
        if (not tags or all(tag in case["tags"] for tag in tags))
        and (not models or any(_model_matches(case, model) for model in models))
        and (not backends or case["hardware"]["backend"] in set(backends))
        and (not case_ids or case["id"] in set(case_ids))
    ]


def build_command(
    case: dict[str, Any],
    defaults: dict[str, Any],
    *,
    run_id: str | None = None,
) -> list[str]:
    command: list[str] = []
    checkpoint = case["checkpoint"]
    if checkpoint["source"] == "local":
        command.extend(
            [
                "env",
                "HF_HUB_OFFLINE=1",
                f"HF_HOME=${{{checkpoint['env']}}}",
            ]
        )

    if case["world_size"] > 1:
        command.extend(
            [
                "torchrun",
                f"--nproc_per_node={case['world_size']}",
                "-m",
                "xfuser.runner",
            ]
        )
    else:
        command.append("xdit")

    output_root = (
        Path(defaults.get("output_root", "gpu-validation-output"))
        .expanduser()
        .resolve()
    )
    if run_id is None:
        timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_id = f"{timestamp}-{uuid.uuid4().hex[:12]}"
    command.extend(
        [
            "--model",
            case["model"],
            "--prompt",
            str(defaults["prompt"]),
            "--seed",
            str(defaults["seed"]),
            "--num_inference_steps",
            str(defaults["num_inference_steps"]),
            "--height",
            str(defaults["height"]),
            "--width",
            str(defaults["width"]),
            "--output_directory",
            str(output_root / case["id"] / run_id),
        ]
    )

    if case["world_size"] > 1:
        # The runner asserts dp*cfg*sp*tp*pp == dit_parallel_size, and
        # --fully_shard_degree does not contribute to that product, so every
        # multi-rank case needs a parallel degree of its own.
        command.extend(["--ulysses_degree", str(case["world_size"])])
    if case["placement"] == "replicated":
        command.append("--memory_efficient_replicated_load")
    elif case["placement"] == "fsdp_blockwise":
        command.extend(
            [
                "--fully_shard_degree",
                str(case["world_size"]),
                "--memory_efficient_sharding",
            ]
        )
    elif case["placement"] == "fsdp_eager_fill":
        command.extend(["--fully_shard_degree", str(case["world_size"])])

    quantization_flags = {
        "none": [],
        "fp8": ["--use_fp8_gemms"],
        "fp4": ["--use_fp4_gemms"],
        "int8": ["--use_int8_gemms"],
        "hybrid_fp8_fp4": [
            "--use_fp8_gemms",
            "--use_fp4_gemms",
            "--use_hybrid_gemm_schedule",
        ],
    }
    command.extend(quantization_flags[case["quantization"]])
    if case["te_fp8"]:
        command.append("--use_fp8_text_encoder")

    offload_flags = {
        "none": [],
        "model": ["--enable_model_cpu_offload"],
        "sequential": ["--enable_sequential_cpu_offload"],
        "group": ["--enable_group_cpu_offload"],
        "group_low_cpu_mem": [
            "--enable_group_cpu_offload",
            "--group_offload_low_cpu_mem",
        ],
    }
    command.extend(offload_flags[case["offload"]])
    command.extend(case["args"])
    return command


def format_command(command: list[str]) -> str:
    """Render a copyable shell command while preserving env placeholders."""
    rendered = []
    placeholder = re.compile(r"^\$\{[A-Za-z_][A-Za-z0-9_]*\}$")
    assignment = re.compile(
        r"^([A-Za-z_][A-Za-z0-9_]*)=(\$\{[A-Za-z_][A-Za-z0-9_]*\})$"
    )
    for argument in command:
        match = assignment.fullmatch(argument)
        if match:
            rendered.append(f'{match.group(1)}="{match.group(2)}"')
        elif placeholder.fullmatch(argument):
            rendered.append(f'"{argument}"')
        else:
            rendered.append(shlex.quote(argument))
    return " ".join(rendered)


# Saved logs carry the recorder's ISO timestamp prefix; in-process
# classification does not. Tolerate both so the same filter can be reapplied to
# an attached validation.log.
LINE_PREFIX = re.compile(r"^(?:\d{4}-\d\d-\d\dT[\d:.+\-]+Z? )?(?:\[rank\d+\]:\s?)?")
TRACEBACK_HEADER = re.compile(r"^Traceback \(most recent call last\):")
# Python renders a terminating exception as a dotted class name followed
# immediately by ": ". Requiring no space before the colon keeps prose log
# lines such as "transformer quantization: requested=fp4" out.
EXCEPTION_LINE = re.compile(r"^[A-Za-z_][\w.]*(?:: \S|$)")
RAISE_LINE = re.compile(r"^\s*raise\b")
ERROR_LEVEL_LINE = re.compile(r"\b(?:ERROR|CRITICAL|FATAL)\b|^E\d{4} ")


def failure_text(log: str) -> str:
    """Return only the lines that report why the process failed.

    An expected rejection must be matched against the reason the process died,
    not against the whole transcript. Routine INFO output can otherwise satisfy
    a rejection pattern: the descriptor line
    "transformer quantization: requested=fp4, backend=aiter,
    storage=aiter_mxfp4_per_1x32" matches "AITER.*FP4" while reporting that FP4
    was accepted, so a case that actually died on a gated-repo 401 was recorded
    as a pass.

    Selecting failure lines rather than excluding known-noisy ones fails closed:
    a rejection that leaves no error trace does not match, instead of matching
    whatever happens to sit nearby in the log.
    """
    selected: list[str] = []
    in_traceback = False
    for raw in log.splitlines():
        line = LINE_PREFIX.sub("", raw).rstrip()
        if not line:
            continue
        if TRACEBACK_HEADER.match(line):
            in_traceback = True
            continue
        if in_traceback:
            # Frames are indented; the first unindented line terminates the
            # traceback and carries the exception type and message.
            if line[:1].isspace():
                if RAISE_LINE.match(line):
                    selected.append(line.strip())
                continue
            in_traceback = False
            selected.append(line)
            continue
        if (
            EXCEPTION_LINE.match(line)
            or RAISE_LINE.match(line)
            or ERROR_LEVEL_LINE.search(line)
        ):
            selected.append(line.strip())
    return "\n".join(selected)


def classify_outcome(
    exit_status: int,
    log: str,
    first_forward: str,
    expected: dict[str, Any],
    output: dict[str, Any],
) -> str:
    if expected["outcome"] == "inference_success":
        if exit_status != 0 or first_forward != "succeeded":
            return "failed_inference"
        files = output.get("files", [])
        if not any(item.get("bytes", 0) > 0 and item.get("sha256") for item in files):
            return "failed_missing_output"
        return "passed"
    if exit_status == 0:
        return "failed_missing_rejection"
    if first_forward != "not_reached":
        return "failed_late_rejection"
    if not re.search(
        expected["error_pattern"], failure_text(log), flags=re.IGNORECASE
    ):
        return "failed_wrong_rejection"
    return "passed_expected_rejection"


def make_result_record(
    *,
    case: dict[str, Any],
    command: list[str],
    environment: dict[str, Any],
    exit_status: int,
    metrics: dict[str, Any],
    output: dict[str, Any],
    log: str,
    quality_notes: str,
    reference: str | None,
) -> dict[str, Any]:
    status = classify_outcome(
        exit_status,
        log,
        metrics["first_forward"],
        case["expected"],
        output,
    )
    return {
        "schema_version": 2,
        "recorded_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "case_id": case["id"],
        "case": case,
        "status": status,
        "execution": "RAN",
        "expected": case["expected"],
        "command": command,
        "environment": environment,
        "exit_status": exit_status,
        "metrics": metrics,
        "output": output,
        "quality": {
            "matrix_notes": case["quality_notes"],
            "operator_notes": quality_notes,
            "reference": reference,
        },
    }


def quality_status(status: str, reference: Any) -> str:
    """Fold a failed comparison into the case status, so a gate that fails is not just a field.

    Only downgrades a case that otherwise passed: a run that already failed to produce an image has
    a more specific status than the score would give it, and overwriting that would lose the reason.
    """
    if not status.startswith("passed"):
        return status
    if isinstance(reference, dict) and reference.get("verdict") == "fail":
        return "failed_quality"
    return status


def append_result(path: Path | str, record: dict[str, Any]) -> None:
    result_path = Path(path)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with result_path.open("a", encoding="utf-8") as handle:
        json.dump(record, handle, sort_keys=True)
        handle.write("\n")


def _run_text(command: list[str]) -> str | None:
    if not shutil.which(command[0]):
        return None
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    text = (completed.stdout + completed.stderr).strip()
    return text or None


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _module_importable(name: str) -> bool:
    """Whether ``name`` can be imported, ignoring distribution metadata.

    AITER is commonly vendored as a source checkout on a PYTHONPATH with no
    dist-info, so importlib.metadata cannot see it even though the runner uses
    it. find_spec locates the module without executing it, which matters because
    importing AITER triggers a JIT build.
    """
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def _dependency_available(name: str, version_getter, module_probe) -> bool:
    return version_getter(name) is not None or module_probe(name)


def probe_environment(
    *,
    command_runner=_run_text,
    version_getter=_package_version,
    module_probe=_module_importable,
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    if environ is None:
        environ = os.environ
    nvidia = command_runner(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,compute_cap",
            "--format=csv,noheader,nounits",
        ]
    )
    rocminfo = command_runner(["rocminfo"])
    rocm_smi = command_runner(["rocm-smi", "--showproductname", "--json"])
    accelerators: list[str] = []
    platform_name = "unknown"
    visibility_variable = None
    visibility_value = None
    if nvidia:
        platform_name = "cuda"
        devices = []
        for fallback_index, line in enumerate(nvidia.splitlines()):
            parts = [part.strip() for part in line.split(",")]
            capability = next(
                (
                    value
                    for value in reversed(parts)
                    if re.fullmatch(r"\d+\.\d+", value)
                ),
                None,
            )
            if capability is None:
                continue
            index = (
                parts[0]
                if len(parts) >= 3 and parts[0].isdigit()
                else str(fallback_index)
            )
            device_uuid = parts[1] if len(parts) >= 3 else None
            major, minor = capability.split(".", 1)
            devices.append(
                {
                    "index": index,
                    "uuid": device_uuid,
                    "accelerator": f"sm{major}{minor}",
                }
            )
        if "CUDA_VISIBLE_DEVICES" in environ:
            visibility_variable = "CUDA_VISIBLE_DEVICES"
            visibility_value = environ["CUDA_VISIBLE_DEVICES"]
            tokens = [
                token.strip()
                for token in visibility_value.split(",")
                if token.strip() and token.strip() != "-1"
            ]
            devices = [
                device
                for token in tokens
                for device in devices
                if token == device["index"]
                or (device["uuid"] is not None and device["uuid"].startswith(token))
            ]
        accelerators.extend(device["accelerator"] for device in devices)
    elif rocminfo or rocm_smi:
        platform_name = "rocm"
        accelerator_text = "\n".join(value for value in (rocminfo, rocm_smi) if value)
        accelerators.extend(
            re.findall(r"(?im)^\s*Name:\s*(gfx\d+)\s*$", accelerator_text)
        )
        for variable in (
            "ROCR_VISIBLE_DEVICES",
            "HIP_VISIBLE_DEVICES",
            "CUDA_VISIBLE_DEVICES",
        ):
            if variable in environ:
                visibility_variable = variable
                visibility_value = environ[variable]
                tokens = [
                    token.strip()
                    for token in visibility_value.split(",")
                    if token.strip() and token.strip() != "-1"
                ]
                accelerators = [
                    accelerators[int(token)]
                    for token in tokens
                    if token.isdigit() and int(token) < len(accelerators)
                ]
                break

    transformers_version = version_getter("transformers")
    transformers_major = None
    if transformers_version:
        match = re.match(r"(\d+)", transformers_version)
        if match:
            transformers_major = int(match.group(1))
    return {
        "platform": platform_name,
        "accelerators": accelerators,
        "device_count": len(accelerators),
        "visibility": {
            "variable": visibility_variable,
            "value": visibility_value,
        },
        "transformers_version": transformers_version,
        "transformers_major": transformers_major,
        "aiter_available": _dependency_available("aiter", version_getter, module_probe),
        "torchao_available": _dependency_available(
            "torchao", version_getter, module_probe
        ),
        "raw": {
            "nvidia_compute_capability": nvidia,
            "rocminfo": rocminfo,
            "rocm_smi": rocm_smi,
        },
    }


_ROCM_RDNA4 = {"gfx1200", "gfx1201"}
_ROCM_MI3XX = {"gfx942", "gfx950"}


def _rocm_accelerator_matches(token: str, accelerator: str) -> bool:
    """Whether one observed gfx name satisfies a case's declared accelerator.

    On ROCm the backend token only separates RDNA4 from everything else, so the arch a
    case was written for lives solely in this field and has to be honoured. FP4 is why:
    AITER builds no FP4 kernels for gfx942 but does for gfx950, so a case pinned to
    gfx942 asserts a rejection that simply does not happen on gfx950.

    gfx942_or_gfx950 is the CDNA3-and-newer datacentre pair that envs._on_mi3xx treats as
    one class. It exists so a case whose behaviour is the same on both is not pinned to
    whichever of them it was first run on, while still excluding older ROCm parts like
    gfx90a that lack the FP8 support such cases assume.
    """
    if token == "non_rdna4_rocm":
        return accelerator not in _ROCM_RDNA4
    if token == "gfx1200_or_gfx1201":
        return accelerator in _ROCM_RDNA4
    if token == "gfx942_or_gfx950":
        return accelerator in _ROCM_MI3XX
    return accelerator == token


def _cuda_capability(accelerator: str) -> tuple[int, int] | None:
    match = re.fullmatch(r"sm(\d+)", accelerator)
    if not match:
        return None
    digits = match.group(1)
    if len(digits) < 2:
        return None
    return int(digits[:-1]), int(digits[-1])


def environment_mismatches(case: dict[str, Any], observed: dict[str, Any]) -> list[str]:
    backend = case["hardware"]["backend"]
    accelerators = observed.get("accelerators", [])
    mismatches = []
    expected_major = int(case["transformers"].split(".", 1)[0])
    if observed.get("transformers_major") != expected_major:
        mismatches.append(
            f"requires Transformers {case['transformers']}; observed "
            f"{observed.get('transformers_version') or 'not installed'}"
        )

    if backend.startswith("cuda_"):
        if observed.get("platform") != "cuda":
            mismatches.append(
                f"requires CUDA; observed {observed.get('platform', 'unknown')}"
            )
        capabilities = [
            capability
            for accelerator in accelerators
            if (capability := _cuda_capability(accelerator)) is not None
        ]
        if backend == "cuda_ada_torchao":
            matching_capabilities = [
                capability for capability in capabilities if capability == (8, 9)
            ]
            requirement = "sm89"
        elif backend == "cuda_hopper_torchao":
            matching_capabilities = [
                capability for capability in capabilities if capability[0] == 9
            ]
            requirement = "sm90-sm99"
        else:
            matching_capabilities = [
                capability for capability in capabilities if capability[0] >= 10
            ]
            requirement = "sm100 or newer"
        valid_accelerator = bool(capabilities) and len(matching_capabilities) == len(
            capabilities
        )
        matching_device_count = len(matching_capabilities)
        if not valid_accelerator:
            mismatches.append(
                f"requires {requirement}; observed {accelerators or ['unknown']}"
            )
        if not observed.get("torchao_available"):
            mismatches.append("requires installed TorchAO")
    else:
        if observed.get("platform") != "rocm":
            mismatches.append(
                f"requires ROCm; observed {observed.get('platform', 'unknown')}"
            )
        token = case["hardware"]["accelerator"]
        matching_device_count = sum(
            _rocm_accelerator_matches(token, accelerator)
            for accelerator in accelerators
        )
        if not accelerators or matching_device_count != len(accelerators):
            mismatches.append(
                f"requires {token}; observed {accelerators or ['unknown']}"
            )
        if backend == "rdna4_aiter":
            if not observed.get("aiter_available"):
                mismatches.append("requires installed AITER")
        else:
            if not observed.get("torchao_available"):
                mismatches.append("requires installed TorchAO")
    if matching_device_count < case["world_size"]:
        mismatches.append(
            f"world_size {case['world_size']} requires "
            f"{case['world_size']} matching devices; observed "
            f"{matching_device_count}"
        )
    return mismatches


def collect_environment(
    validation_probe: dict[str, Any] | None = None,
) -> dict[str, Any]:
    versions = {}
    for name in PACKAGE_NAMES:
        versions[name] = _package_version(name)
    commit = _run_text(["git", "-C", str(ROOT), "rev-parse", "HEAD"])
    dirty = _run_text(["git", "-C", str(ROOT), "status", "--porcelain"])
    device_info = {
        "nvidia_smi": _run_text(
            [
                "nvidia-smi",
                "--query-gpu=name,uuid,driver_version,memory.total,compute_cap",
                "--format=csv,noheader",
            ]
        ),
        "rocm_smi": _run_text(
            [
                "rocm-smi",
                "--showproductname",
                "--showdriverversion",
                "--showmeminfo",
                "vram",
                "--json",
            ]
        ),
    }
    return {
        "commit_sha": commit,
        "git_dirty": bool(dirty),
        "python": sys.version,
        "platform": platform.platform(),
        "packages": versions,
        "devices": device_info,
        "validation_probe": validation_probe or probe_environment(),
    }


def _proc_tree(root_pid: int) -> set[int]:
    parents: dict[int, int] = {}
    for stat_path in Path("/proc").glob("[0-9]*/stat"):
        try:
            fields = stat_path.read_text().split()
            parents[int(fields[0])] = int(fields[3])
        except (OSError, ValueError, IndexError):
            continue
    selected = {root_pid}
    changed = True
    while changed:
        changed = False
        for pid, parent in parents.items():
            if parent in selected and pid not in selected:
                selected.add(pid)
                changed = True
    return selected


def _process_tree_rss(pids: set[int]) -> int:
    """Summed RSS over the tree. Kept for continuity, but not the figure to judge a load by.

    Ranks share the ROCm and torch libraries and mmap the same checkpoint files, and every rank's
    RSS counts those shared pages again, so this reads high in proportion to the rank count. It also
    misses page cache that no process maps, so it can read low as well. _cgroup_memory_breakdown is
    the honest measure: the kernel counts each physical page once for the container.
    """
    total_kib = 0
    for pid in pids:
        try:
            for line in Path(f"/proc/{pid}/status").read_text().splitlines():
                if line.startswith("VmRSS:"):
                    total_kib += int(line.split()[1])
                    break
        except (OSError, ValueError, IndexError):
            continue
    return total_kib * 1024


def _cgroup_memory_breakdown() -> tuple[int | None, int | None]:
    """(anonymous, file-backed) container bytes, each physical page counted once.

    The split is what makes the number interpretable, and mirrors what the load path already logs
    via checkpoint_io.host_mem_gb: anonymous pages are the tensors a load actually allocated and
    cannot be reclaimed under pressure, while file-backed pages are mmap'd checkpoint cache the
    kernel drops when it needs to. Reporting their sum would let page cache masquerade as cost.
    """
    anon = file_backed = None
    try:
        for line in Path("/sys/fs/cgroup/memory.stat").read_text().splitlines():
            key, _, value = line.partition(" ")
            if key == "anon":
                anon = int(value)
            elif key == "file":
                file_backed = int(value)
    except (OSError, ValueError):
        return None, None
    return anon, file_backed


def _cgroup_memory() -> int | None:
    for path in (
        Path("/sys/fs/cgroup/memory.current"),
        Path("/sys/fs/cgroup/memory/memory.usage_in_bytes"),
    ):
        try:
            return int(path.read_text().strip())
        except (OSError, ValueError):
            continue
    return None


def _nvidia_process_memory(pids: set[int]) -> dict[str, int] | None:
    """This run's GPU bytes per device, keyed by device.

    Per device rather than summed: a rank count is not a memory cost, and summing a sharded run
    across eight devices reports a bigger number the more ranks it is spread over, which inverts
    what sharding does.
    """
    output = _run_text(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,gpu_uuid,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ]
    )
    if output is None:
        return None
    by_device: dict[str, int] = {}
    for line in output.splitlines():
        try:
            pid_text, device, memory_text = (part.strip() for part in line.split(",", 2))
            if int(pid_text) in pids:
                # Summed within a device: several ranks can share one GPU.
                by_device[device] = by_device.get(device, 0) + int(memory_text) * 1024**2
        except (ValueError, TypeError):
            continue
    return by_device


def _rocm_global_memory() -> dict[str, int] | None:
    """Whole-device used VRAM per device, keyed by card."""
    output = _run_text(["rocm-smi", "--showmeminfo", "vram", "--json"])
    if output is None:
        return None
    try:
        data = json.loads(output)
    except json.JSONDecodeError:
        return None
    by_device: dict[str, int] = {}

    def visit(value: Any, key: str = "", device: str = "") -> None:
        if isinstance(value, dict):
            for child_key, child in value.items():
                visit(child, child_key, device or key)
        elif isinstance(value, list):
            for child in value:
                visit(child, key, device)
        elif "used" in key.casefold() and "memory" in key.casefold():
            try:
                by_device[device or key] = by_device.get(device or key, 0) + int(value)
            except (TypeError, ValueError):
                pass

    for card, readings in data.items() if isinstance(data, dict) else ():
        visit(readings, device=card)
    return by_device


class ResourceMonitor:
    def __init__(self, root_pid: int, interval: float = 0.25) -> None:
        self.root_pid = root_pid
        self.interval = interval
        self.peak_host_rss = 0
        self.peak_cgroup = 0
        self.peak_host_anon = 0
        self.peak_host_file_cache = 0
        self.peak_gpu: int | None = None
        self.gpu_scope: str | None = None
        # (monotonic time, busiest device's bytes), so a phase's peak can be recovered afterwards
        # from markers the child process reports only once it has finished.
        self.gpu_samples: list[tuple[float, int]] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=max(2.0, self.interval * 4))

    def _sample(self) -> None:
        while not self._stop.is_set():
            pids = _proc_tree(self.root_pid)
            self.peak_host_rss = max(self.peak_host_rss, _process_tree_rss(pids))
            cgroup = _cgroup_memory()
            if cgroup is not None:
                self.peak_cgroup = max(self.peak_cgroup, cgroup)
            anon, file_cache = _cgroup_memory_breakdown()
            if anon is not None:
                self.peak_host_anon = max(self.peak_host_anon, anon)
            if file_cache is not None:
                self.peak_host_file_cache = max(self.peak_host_file_cache, file_cache)
            by_device = _nvidia_process_memory(pids)
            scope = "process_tree"
            if by_device is None:
                by_device = _rocm_global_memory()
                scope = "device_global"
            if by_device:
                busiest = max(by_device.values())
                self.peak_gpu = max(self.peak_gpu or 0, busiest)
                self.gpu_scope = scope
                self.gpu_samples.append((time.monotonic(), busiest))
            self._stop.wait(self.interval)

    def peak_gpu_between(self, start: float, end: float) -> int | None:
        """The busiest device's peak within one phase, or None if no sample landed in it."""
        within = [value for at, value in self.gpu_samples if start <= at <= end]
        return max(within) if within else None


def _placeholder_names(command: list[str]) -> list[str]:
    return sorted(
        {
            name
            for argument in command
            for name in re.findall(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}", argument)
        }
    )


def placeholder_mismatches(
    command: list[str],
    environ: Mapping[str, str] | None = None,
) -> list[str]:
    if environ is None:
        environ = os.environ
    return [
        f"required environment variable {name} is unset"
        for name in _placeholder_names(command)
        if not environ.get(name)
    ]


def expand_command(command: list[str]) -> list[str]:
    missing = placeholder_mismatches(command)
    if missing:
        raise ValueError("; ".join(missing))
    expanded = [os.path.expandvars(arg) for arg in command]
    return expanded


def _redactions(command: list[str]) -> dict[str, str]:
    return {
        value: f"${{{name}}}"
        for name in _placeholder_names(command)
        if (value := os.environ.get(name))
    }


def _redact(text: str, redactions: dict[str, str]) -> str:
    for value in sorted(redactions, key=len, reverse=True):
        text = text.replace(value, redactions[value])
    return text


def reference_case_id(case: dict[str, Any], cases: list[dict[str, Any]]) -> str | None:
    """The case whose output this one should be judged against, or None if it is that case.

    An unquantized eager load at one rank, on the same model: no quantization, no sharding, nothing
    offloaded. Matched on those attributes rather than by name so a renamed or regenerated matrix
    cannot silently leave a case scoring against the wrong thing.
    """
    wanted = {
        "model": case["model"],
        "placement": "eager",
        "quantization": "none",
        "offload": "none",
        "te_fp8": False,
        "world_size": 1,
    }
    if all(case.get(key) == value for key, value in wanted.items()):
        return None
    for candidate in cases:
        if all(candidate.get(key) == value for key, value in wanted.items()):
            return candidate["id"]
    return None


def order_references_first(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Run the cases others are scored against before the cases that need them.

    Without this, scoring a case depends on whether its reference happened to be selected earlier,
    which would make the same command produce scores or not depending on argument order.
    """
    ids = {case["id"] for case in cases}
    referenced = {
        reference
        for case in cases
        if (reference := reference_case_id(case, cases)) is not None and reference in ids
    }
    return [case for case in cases if case["id"] in referenced] + [
        case for case in cases if case["id"] not in referenced
    ]


def without_compile(command: list[str]) -> list[str]:
    """Drop torch.compile from a command, for runs whose output will be compared.

    Compile picks kernels by measured timing, and different fp8 kernels accumulate differently, so
    the same case can render two different images: measured 2 distinct outputs in 3 runs with it on
    and byte-identical output in 3 runs with it off. Since that spread is as large as the difference
    quantization makes, a score taken with compile on would mostly report which kernel won.
    """
    return [argument for argument in command if argument != "--use_torch_compile"]


def score_against_reference(
    output: dict[str, Any],
    reference_output: dict[str, Any] | None,
    *,
    reference_id: str | None,
    thresholds: dict[str, float] | None = None,
) -> dict[str, Any] | None:
    """Compare a run's artifact against its reference's, or explain why it could not be.

    Returns a block rather than a bare score so a recorded verdict stays interpretable: which case
    it was compared against, which artifact, and which thresholds applied.
    """
    if reference_id is None:
        return None
    actual_path = (output or {}).get("path")
    reference_path = (reference_output or {}).get("path")
    if not actual_path or not reference_path:
        missing = "this case" if not actual_path else reference_id
        return {
            "case_id": reference_id,
            "verdict": "unscored",
            "reason": f"no image artifact for {missing}",
        }
    module = _image_quality()
    scores = module.score_images(reference_path, actual_path)
    return {
        "case_id": reference_id,
        "artifact": reference_path,
        "sha256": (reference_output or {}).get("sha256"),
        "scores": scores,
        **module.verdict(scores, thresholds),
    }


def _image_quality():
    """Imported lazily so a dry run needs neither numpy nor pillow."""
    import importlib.util

    path = Path(__file__).resolve().parent / "image_quality.py"
    spec = importlib.util.spec_from_file_location("image_quality", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(spec.name, module)
    spec.loader.exec_module(module)
    return module


def _output_directory(command: list[str]) -> Path:
    index = command.index("--output_directory")
    return Path(command[index + 1])


def reserve_output_directory(output_dir: Path) -> None:
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir()


def hash_outputs(output_dir: Path) -> dict[str, Any]:
    if not output_dir.exists():
        return {
            "path": None,
            "sha256": None,
            "bytes": None,
            "files": [],
        }
    candidates = sorted(
        path
        for path in output_dir.rglob("*")
        if path.is_file()
        and path.stat().st_size > 0
        and path.suffix.casefold() in GENERATED_ARTIFACT_EXTENSIONS
    )
    files = []
    for path in candidates:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        files.append(
            {
                "path": str(path),
                "sha256": digest.hexdigest(),
                "bytes": path.stat().st_size,
            }
        )
    primary = files[-1] if files else {}
    return {
        "path": primary.get("path"),
        "sha256": primary.get("sha256"),
        "bytes": primary.get("bytes"),
        "files": files,
    }


def make_environment_mismatch_record(
    *,
    case: dict[str, Any],
    command: list[str],
    environment: dict[str, Any],
    mismatches: list[str],
) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "recorded_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "case_id": case["id"],
        "case": case,
        "status": "environment_mismatch",
        "execution": "NOT RUN",
        "expected": case["expected"],
        "command": command,
        "environment": environment,
        "environment_mismatches": mismatches,
        "exit_status": None,
        "metrics": {
            "wall_duration_seconds": None,
            "load_duration_seconds": None,
            "first_forward": "not_reached",
            "peak_host_rss_bytes": None,
            "peak_cgroup_memory_bytes": None,
            "peak_gpu_memory_bytes": None,
            "gpu_memory_scope": None,
        },
        "output": {
            "path": None,
            "sha256": None,
            "bytes": None,
            "files": [],
        },
        "quality": {
            "matrix_notes": case["quality_notes"],
            "operator_notes": "",
            "reference": None,
        },
    }


def aggregate_exit_code(statuses: list[str]) -> int:
    if any(
        status
        not in {
            "passed",
            "passed_expected_rejection",
            "environment_mismatch",
        }
        for status in statuses
    ):
        return 1
    if any(status == "environment_mismatch" for status in statuses):
        return 2
    return 0


def _process_group_exists(process_group: int) -> bool:
    try:
        os.killpg(process_group, 0)
    except ProcessLookupError:
        return False
    return True


def _terminate_process_group(
    process: subprocess.Popen[str],
    reader: threading.Thread,
    grace_seconds: float = PROCESS_TERMINATE_GRACE_SECONDS,
) -> int:
    process_group = process.pid
    try:
        os.killpg(process_group, signal.SIGTERM)
    except ProcessLookupError:
        pass

    term_deadline = time.monotonic() + grace_seconds
    while _process_group_exists(process_group):
        process.poll()
        remaining = term_deadline - time.monotonic()
        if remaining <= 0:
            break
        time.sleep(min(0.01, remaining))

    if _process_group_exists(process_group):
        try:
            os.killpg(process_group, signal.SIGKILL)
        except ProcessLookupError:
            pass

    reap_deadline = time.monotonic() + grace_seconds
    try:
        exit_status = process.wait(timeout=max(0.0, reap_deadline - time.monotonic()))
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(
            f"could not reap timed-out root process {process.pid}"
        ) from error
    reader.join(timeout=max(0.0, reap_deadline - time.monotonic()))
    return exit_status


def execute_case(
    case: dict[str, Any],
    command: list[str],
    *,
    results_path: Path,
    quality_notes: str,
    reference: str | None,
    environment: dict[str, Any],
    timeout_seconds: float,
    reference_id: str | None = None,
    reference_output: dict[str, Any] | None = None,
    thresholds: dict[str, float] | None = None,
) -> dict[str, Any]:
    expanded = expand_command(command)
    redactions = _redactions(command)
    output_dir = _output_directory(expanded)
    reserve_output_directory(output_dir)
    log_path = output_dir / "validation.log"
    markers: dict[str, float] = {}
    captured_lines: list[tuple[str, str]] = []
    capture_lock = threading.Lock()

    started = time.monotonic()
    deadline = started + timeout_seconds
    process = subprocess.Popen(
        expanded,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        start_new_session=True,
    )
    monitor = ResourceMonitor(process.pid)
    monitor.start()
    assert process.stdout is not None

    def consume_output() -> None:
        for line in process.stdout:
            now = time.monotonic()
            line = _redact(line, redactions)
            stamped = f"{dt.datetime.now(dt.timezone.utc).isoformat()} {line}"
            print(stamped, end="")
            with capture_lock:
                captured_lines.append((line, stamped))
                if "Initializing model:" in line:
                    markers.setdefault("load_start", now)
                # Compile warmup runs inside model initialization, so it lands between load_start
                # and load_end and would otherwise be charged to the load.
                if "Warming up torch compiler" in line:
                    markers.setdefault("compile_start", now)
                if "Model initialization complete." in line:
                    markers.setdefault("load_end", now)
                if "Running model..." in line:
                    markers.setdefault("forward_start", now)

    timed_out = False
    reader = threading.Thread(target=consume_output, daemon=True)
    reader.start()
    try:
        exit_status = process.wait(timeout=max(0.0, deadline - time.monotonic()))
    except subprocess.TimeoutExpired:
        timed_out = True
        exit_status = _terminate_process_group(
            process, reader, PROCESS_TERMINATE_GRACE_SECONDS
        )
    else:
        reader.join(timeout=max(0.0, deadline - time.monotonic()))
        if reader.is_alive():
            timed_out = True
            exit_status = _terminate_process_group(
                process, reader, PROCESS_TERMINATE_GRACE_SECONDS
            )
    with capture_lock:
        log_lines = [line for line, _ in captured_lines]
        stamped_lines = [stamped for _, stamped in captured_lines]
        markers = dict(markers)
    with log_path.open("w", encoding="utf-8") as log_handle:
        log_handle.writelines(stamped_lines)
    monitor.stop()
    elapsed = time.monotonic() - started

    if "forward_start" not in markers:
        first_forward = "not_reached"
    elif exit_status == 0:
        first_forward = "succeeded"
    else:
        first_forward = "failed"
    # Compile warmup is not loading, and it is a near-constant tens of seconds, so leaving it inside
    # the load window buried what the load itself did: an eight-rank blockwise fill and a full
    # materialization of the same model looked 1.7x apart when the fills are 3.4x apart. It ends the
    # load window rather than being subtracted so the load-phase VRAM peak narrows with it.
    load_end_marker = "compile_start" if "compile_start" in markers else "load_end"
    load_duration = None
    if "load_start" in markers and load_end_marker in markers:
        load_duration = markers[load_end_marker] - markers["load_start"]
    compile_duration = None
    if "compile_start" in markers and "load_end" in markers:
        compile_duration = markers["load_end"] - markers["compile_start"]
    metrics = {
        "wall_duration_seconds": round(elapsed, 3),
        "load_duration_seconds": (
            round(load_duration, 3) if load_duration is not None else None
        ),
        "compile_duration_seconds": (
            round(compile_duration, 3) if compile_duration is not None else None
        ),
        "first_forward": first_forward,
        "peak_host_rss_bytes": monitor.peak_host_rss or None,
        "peak_cgroup_memory_bytes": monitor.peak_cgroup or None,
        "peak_host_anon_bytes": monitor.peak_host_anon or None,
        "peak_host_file_cache_bytes": monitor.peak_host_file_cache or None,
        "metrics_version": GPU_METRICS_VERSION,
        "peak_gpu_memory_bytes": monitor.peak_gpu,
        # The whole-run peak is dominated by inference activations, so on its own it cannot show what
        # a memory-efficient load achieved. This is the peak while the load was in flight.
        "peak_load_gpu_memory_bytes": (
            monitor.peak_gpu_between(markers["load_start"], markers[load_end_marker])
            if "load_start" in markers and load_end_marker in markers
            else None
        ),
        "gpu_memory_scope": monitor.gpu_scope,
    }
    output = hash_outputs(output_dir)
    # A computed comparison supersedes the operator's free-text pointer, since both answer the same
    # question about what this run was judged against and only one of them was measured.
    comparison = score_against_reference(
        output, reference_output, reference_id=reference_id, thresholds=thresholds
    )
    record = make_result_record(
        case=case,
        command=command,
        environment=environment,
        exit_status=exit_status,
        metrics=metrics,
        output=output,
        log="".join(log_lines),
        quality_notes=quality_notes,
        reference=comparison if comparison is not None else reference,
    )
    record["timed_out"] = timed_out
    record["timeout_seconds"] = timeout_seconds
    record["status"] = quality_status(record["status"], comparison)
    if timed_out:
        record["status"] = "timed_out"
    record["log_path"] = str(log_path)
    append_result(results_path, record)
    return record


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="External GPU validation matrix runner (dry-run by default)."
    )
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--case", action="append", default=[], dest="case_ids")
    parser.add_argument("--tag", action="append", default=[])
    parser.add_argument("--model", action="append", default=[])
    parser.add_argument(
        "--backend", action="append", choices=sorted(BACKENDS), default=[]
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--list", action="store_true")
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("gpu-validation-results/results.jsonl"),
    )
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--quality-note", default="")
    parser.add_argument("--reference")
    parser.add_argument(
        "--score-quality",
        action="store_true",
        help=(
            "compare each case's image against its model's eager bf16 case and fail the case if it "
            "diverges; disables torch.compile for every run so the comparison is deterministic"
        ),
    )
    parser.add_argument(
        "--ssim-min",
        type=float,
        help="override the SSIM floor used by --score-quality",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=_timeout_argument,
        help="override the matrix timeout for every selected case",
    )
    parser.add_argument("--continue-on-error", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    matrix = load_matrix(args.matrix)
    defaults = dict(matrix["defaults"])
    if args.output_root is not None:
        defaults["output_root"] = str(args.output_root)
    defaults["output_root"] = str(Path(defaults["output_root"]).expanduser().resolve())
    selected = select_cases(
        matrix["cases"],
        tags=args.tag,
        models=args.model,
        backends=args.backend,
        case_ids=args.case_ids,
    )
    if not selected:
        print("No validation cases matched the requested filters.", file=sys.stderr)
        return 2

    thresholds = {"ssim_min": args.ssim_min} if args.ssim_min is not None else None
    if args.score_quality:
        selected = order_references_first(selected)
    validation_probe = probe_environment() if args.execute else None
    environment = collect_environment(validation_probe) if validation_probe else None
    statuses: list[str] = []
    outputs: dict[str, dict[str, Any]] = {}
    for case in selected:
        command = build_command(case, defaults)
        if args.score_quality:
            command = without_compile(command)
        expected = case["expected"]["outcome"]
        print(f"{case['id']} [{expected}]")
        print(f"  {format_command(command)}")
        if args.list:
            continue
        if args.execute:
            assert validation_probe is not None
            assert environment is not None
            mismatches = environment_mismatches(
                case, validation_probe
            ) + placeholder_mismatches(command)
            if mismatches:
                record = make_environment_mismatch_record(
                    case=case,
                    command=command,
                    environment=environment,
                    mismatches=mismatches,
                )
                append_result(args.results, record)
                statuses.append(record["status"])
                print("  result: environment_mismatch (NOT RUN)")
                for mismatch in mismatches:
                    print(f"    - {mismatch}")
                if not args.continue_on_error:
                    return aggregate_exit_code(statuses)
                continue
            reference_id = (
                reference_case_id(case, selected) if args.score_quality else None
            )
            record = execute_case(
                case,
                command,
                results_path=args.results,
                quality_notes=args.quality_note,
                reference=args.reference,
                environment=environment,
                timeout_seconds=resolve_timeout_seconds(
                    case, defaults, args.timeout_seconds
                ),
                reference_id=reference_id,
                reference_output=outputs.get(reference_id) if reference_id else None,
                thresholds=thresholds,
            )
            outputs[case["id"]] = record.get("output") or {}
            statuses.append(record["status"])
            print(f"  result: {record['status']}")
            comparison = (record.get("quality") or {}).get("reference")
            if isinstance(comparison, dict):
                scores = comparison.get("scores") or {}
                detail = (
                    f"ssim {scores['ssim']:.4f} psnr {scores['psnr']:.1f}"
                    if scores.get("comparable")
                    else comparison.get("reason", "not scored")
                )
                print(f"    quality vs {comparison['case_id']}: {comparison['verdict']} ({detail})")
                for failure in comparison.get("failures") or []:
                    print(f"      - {failure}")
            if not args.continue_on_error and not record["status"].startswith("passed"):
                return 1
    if not args.execute and not args.list:
        print(f"Dry run only: {len(selected)} case(s), GPU validation NOT RUN.")
    return aggregate_exit_code(statuses)


if __name__ == "__main__":
    raise SystemExit(main())
