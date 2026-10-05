"""Wan runner settings that are decided from the config alone, before any weights load."""

import pytest


@pytest.fixture(autouse=True)
def _single_process(monkeypatch):
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")


def _build(model_name, **config):
    from xfuser import xFuserArgs
    from xfuser.model_executor.models.runner_models import wan  # noqa: F401  registers
    from xfuser.model_executor.models.runner_models.base_model import MODEL_REGISTRY

    return MODEL_REGISTRY[model_name](xFuserArgs(model=model_name, **config))


@pytest.mark.parametrize("name", ["Wan2.1-T2V", "Wan-AI/Wan2.1-T2V-14B-Diffusers", "Wan2.2-T2V"])
def test_wan_t2v_accepts_fp8_text_encoder(name):
    from xfuser.model_executor.models.runner_models.loading.quantization_plan import (
        QuantizationPlan,
    )

    model = _build(name, use_fp8_gemms=True, use_fp8_text_encoder=True)

    assert "text_encoder.encoder.block" in QuantizationPlan(model).module_list("fp8")
