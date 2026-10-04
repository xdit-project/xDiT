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


@pytest.mark.parametrize(
    ("height", "width", "requested", "expected"),
    [
        (720, 1280, None, 5.0),
        (480, 832, None, 3.0),
        (480, 832, 4.0, 4.0),
        (720, 1280, 3.0, 3.0),
    ],
)
def test_wan_vace_flow_shift_follows_resolution_unless_requested(monkeypatch, height, width, requested, expected):
    import dataclasses
    from types import SimpleNamespace

    from diffusers import UniPCMultistepScheduler

    from xfuser.model_executor.models.runner_models.base_model import xFuserModel

    # The base hook materializes loaded weights; there are none here
    monkeypatch.setattr(xFuserModel, "_post_load_and_state_initialization", lambda self, input_args: None)
    model = _build("Wan2.1-VACE-1.3B")
    # The scheduler the Wan2.1 VACE checkpoints ship, with the 480p shift
    scheduler = UniPCMultistepScheduler(
        flow_shift=3.0, prediction_type="flow_prediction", use_flow_sigmas=True, final_sigmas_type="zero"
    )
    model.pipe = SimpleNamespace(scheduler=scheduler)
    input_args = dataclasses.asdict(model.default_input_values)
    input_args.update(height=height, width=width, flow_shift=requested)

    model._post_load_and_state_initialization(input_args)

    assert model.pipe.scheduler.config.flow_shift == expected
