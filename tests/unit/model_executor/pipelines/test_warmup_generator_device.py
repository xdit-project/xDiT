"""The warmup generator must be built on the accelerator the process runs on.

`prepare_run` seeded the warmup with `torch.Generator(device="cuda")`. A torch
build without CUDA raises `RuntimeError: Cannot get CUDA generator without
ATen_cuda library` for that call, so on an Ascend NPU (or any other non-CUDA
accelerator) those pipelines could not warm up at all, whatever the pipeline's
own device resolution had picked.
"""

import importlib
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from xfuser.config import InputConfig

# Pipelines whose warmup still pinned its generator to CUDA.
PIPELINES = [
    ("xfuser.model_executor.pipelines.pipeline_flux2", "xFuserFlux2Pipeline"),
    ("xfuser.model_executor.pipelines.pipeline_flux_control", "xFuserPipelineFluxControlPipeline"),
    ("xfuser.model_executor.pipelines.pipeline_sana", "xFuserSanaPipeline"),
    ("xfuser.model_executor.pipelines.pipeline_sana_sprint", "xFuserSanaSprintPipeline"),
]


@pytest.mark.parametrize("module_name, class_name", PIPELINES)
def test_warmup_seeds_its_generator_on_the_platform_device(monkeypatch, module_name, class_name):
    module = importlib.import_module(module_name)
    runtime_state = SimpleNamespace(runtime_config=SimpleNamespace(warmup_steps=2))
    monkeypatch.setattr(module, "get_runtime_state", lambda: runtime_state)
    # Stand in for the accelerator this process is on, e.g. an Ascend NPU.
    monkeypatch.setattr(module, "get_device_name", lambda: "npu")

    seeded = []

    def generator(*, device=None):
        if device == "cuda":
            # What a torch without CUDA does; the accelerator the fix must not name.
            raise RuntimeError("Cannot get CUDA generator without ATen_cuda library")
        seeded.append(device)
        return Mock()

    monkeypatch.setattr(torch, "Generator", generator)

    pipeline_cls = getattr(module, class_name)
    pipeline = pipeline_cls.__new__(pipeline_cls)
    pipeline.__call__ = Mock()

    pipeline.prepare_run(InputConfig(height=256, width=256))

    pipeline.__call__.assert_called_once()
    assert seeded == ["npu"]
