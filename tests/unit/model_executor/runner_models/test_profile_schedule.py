"""xFuserModel.profile returns a trace of the pipeline calls that the schedule marks active."""

import itertools
from types import SimpleNamespace

import pytest
import torch
from torch.profiler import record_function

from xfuser.model_executor.models.runner_models.base_model import DiffusionOutput, xFuserModel


class _Runner(xFuserModel):
    def _load_model(self):
        raise NotImplementedError

    def _run_pipe(self, input_args):
        raise NotImplementedError


def _runner(monkeypatch, *, wait, warmup, active):
    # log() reads the rank from the environment.
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")

    model = object.__new__(_Runner)
    model.config = SimpleNamespace(
        profile_wait=wait,
        profile_warmup=warmup,
        profile_active=active,
        profile_with_stack=False,
        batch_size=None,
    )
    model._validate_args = lambda input_args: None
    model._split_prompts_for_dp = lambda input_args: input_args
    model._gather_dp_outputs = lambda output: output

    call_number = itertools.count(1)

    def _run_timed_pipe(input_args):
        n = next(call_number)
        with record_function(f"pipe_call_{n}"):
            torch.ones(1).add_(1)
        return DiffusionOutput(images=[n], pipe_args=[]), 0.0

    model._run_timed_pipe = _run_timed_pipe
    return model


@pytest.mark.parametrize(
    "wait, warmup, active, expected",
    [
        pytest.param(1, 1, 1, ["pipe_call_3"], id="wait-then-warmup"),
        pytest.param(0, 1, 1, ["pipe_call_2"], id="no-wait"),
        pytest.param(0, 1, 2, ["pipe_call_2", "pipe_call_3"], id="no-wait-two-active"),
    ],
)
def test_profile_traces_only_the_active_calls(monkeypatch, wait, warmup, active, expected):
    model = _runner(monkeypatch, wait=wait, warmup=warmup, active=active)

    _, _, profile = model.profile({"prompt": "p"})

    traced = sorted({event.name for event in profile.events() if event.name.startswith("pipe_call_")})
    assert traced == expected
