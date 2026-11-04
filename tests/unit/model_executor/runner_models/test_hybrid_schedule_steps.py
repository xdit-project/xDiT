"""A per-step hybrid schedule must start at its first step on every pipeline
run, and Krea-2's schedule must span both guidance passes of a step."""

from types import SimpleNamespace

import pytest

from xfuser import xFuserArgs
from xfuser.core.attention.spec import AttentionBackendType
from xfuser.core.distributed.attention_schedule import AttentionSchedule
from xfuser.core.distributed.runtime_state import DiTRuntimeState, UnetRuntimeState
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


def _krea2(model_cls, name, **hybrid):
    return model_cls(xFuserArgs(model=name, use_hybrid_attn_schedule=True, **hybrid))


@pytest.mark.parametrize(
    ("model_cls", "name", "guidance_scale", "transformer_calls_per_step"),
    [
        (krea2.xFuserKrea2RawModel, "krea/krea-2-raw", 3.5, 2),
        (krea2.xFuserKrea2TurboModel, "krea/krea-2-turbo", 0.0, 1),
    ],
    ids=["raw-guided", "turbo"],
)
def test_krea2_schedule_spans_every_transformer_call(
    state, model_cls, name, guidance_scale, transformer_calls_per_step
):
    model = _krea2(
        model_cls, name, hybrid_attn_low_precision_backend="sdpa_math", hybrid_attn_high_precision_backend="sdpa"
    )

    model._setup_hybrid_attn_schedule(
        {"num_inference_steps": 8, "guidance_scale": guidance_scale, "num_hybrid_attn_high_precision_steps": 1}
    )

    calls = 8 * transformer_calls_per_step
    assert int(state.schedule_total_steps) == calls
    # One high-precision denoising step at each end covers all of its calls.
    high = [HIGH] * transformer_calls_per_step
    assert state.attention_schedule.backends == high + [LOW] * (calls - 2 * transformer_calls_per_step) + high


def test_krea2_raw_explicit_schedule_needs_one_entry_per_transformer_call(state):
    args = {"num_inference_steps": 4, "guidance_scale": 3.5, "num_hybrid_attn_high_precision_steps": 1}
    model = _krea2(krea2.xFuserKrea2RawModel, "krea/krea-2-raw", hybrid_attn_schedule=",".join(["sdpa"] * 4))

    with pytest.raises(ValueError, match="total steps 4 does not match input steps 8"):
        model._setup_hybrid_attn_schedule(args)

    model = _krea2(krea2.xFuserKrea2RawModel, "krea/krea-2-raw", hybrid_attn_schedule=",".join(["sdpa"] * 8))
    model._setup_hybrid_attn_schedule(args)
    assert int(state.schedule_total_steps) == 8


def test_schedule_less_runtime_states_ignore_the_step_reset():
    # prepare_run resets the counter for every runner, whatever runtime state its pipeline gets.
    assert UnetRuntimeState.__new__(UnetRuntimeState).reset_step_counter() is None
