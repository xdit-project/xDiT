"""A per-step hybrid schedule must start at its first step on every pipeline
run, and Krea-2's schedule must span both guidance passes of a step."""

from types import SimpleNamespace

import pytest

from xfuser.core.attention.spec import AttentionBackendType
from xfuser.core.distributed.attention_schedule import AttentionSchedule
from xfuser.core.distributed.runtime_state import DiTRuntimeState
from xfuser.model_executor.models.runner_models import base_model, krea2

HIGH = AttentionBackendType.SDPA
LOW = AttentionBackendType.SDPA_MATH


@pytest.fixture
def state(monkeypatch):
    state = DiTRuntimeState.__new__(DiTRuntimeState)
    state.attention_schedule = None
    state.schedule_total_steps = None
    state.gemm_schedule = None
    state.gemm_schedule_total_steps = None
    state.step_counter = None
    state.attention_backend = HIGH
    state._check_if_backend_compatible_with_current_configuration = lambda backend: None
    monkeypatch.setattr(base_model, "get_runtime_state", lambda: state)
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    return state


def test_run_after_a_shorter_warmup_follows_the_schedule_from_its_first_step(state):
    state.set_attention_schedule(AttentionSchedule([HIGH, LOW, LOW, HIGH]), total_steps=4)
    runner = SimpleNamespace(_vae_manager=SimpleNamespace(prepare_run=lambda *_: None), _decoding_vaes=lambda: [])
    for _ in range(2):  # a warmup run that is shorter than the schedule
        state.increment_step_counter()

    base_model.xFuserModel.prepare_run(runner, {})

    seen = []
    for _ in range(4):
        state.increment_step_counter()
        seen.append(state.attention_backend)
    assert seen == [HIGH, LOW, LOW, HIGH]


@pytest.mark.parametrize(
    ("model_cls", "guidance_scale", "transformer_calls_per_step"),
    [(krea2.xFuserKrea2RawModel, 3.5, 2), (krea2.xFuserKrea2TurboModel, 0.0, 1)],
    ids=["raw-guided", "turbo"],
)
def test_krea2_schedule_spans_every_transformer_call(state, model_cls, guidance_scale, transformer_calls_per_step):
    model = object.__new__(model_cls)
    model.config = SimpleNamespace(
        hybrid_attn_low_precision_backend="sdpa_math", hybrid_attn_high_precision_backend="sdpa"
    )

    model._setup_hybrid_attn_schedule(
        {"num_inference_steps": 8, "guidance_scale": guidance_scale, "num_hybrid_attn_high_precision_steps": 1}
    )

    calls = 8 * transformer_calls_per_step
    assert int(state.schedule_total_steps) == calls
    # One high-precision denoising step at each end covers all of its calls.
    high = [HIGH] * transformer_calls_per_step
    assert state.attention_schedule.backends == high + [LOW] * (calls - 2 * transformer_calls_per_step) + high
