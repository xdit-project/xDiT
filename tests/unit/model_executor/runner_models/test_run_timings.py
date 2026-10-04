"""xFuserModel.run reports the timings of the iterations it measured."""

from types import SimpleNamespace

import pytest
import torch

from xfuser.model_executor.models.runner_models import base_model
from xfuser.model_executor.models.runner_models.base_model import DiffusionOutput, xFuserModel


class _Runner(xFuserModel):
    def _load_model(self):
        raise NotImplementedError

    def _run_pipe(self, input_args):
        raise NotImplementedError


class _Event:
    def __init__(self, enable_timing=True):
        pass

    def record(self):
        pass

    def elapsed_time(self, other):
        return 0.0


def _runner(monkeypatch, *, warmup_calls, num_iterations, batch_size=None):
    # log() reads the rank from the environment.
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setattr(torch.cuda, "Event", _Event)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(base_model, "get_world_group", lambda: SimpleNamespace(rank=0))

    model = object.__new__(_Runner)
    model.config = SimpleNamespace(
        num_iterations=num_iterations,
        warmup_calls=warmup_calls,
        determinism_check=0,
        batch_size=batch_size,
        data_parallel_degree=1,
    )
    model.settings = SimpleNamespace(model_output_type="image")
    model._validate_args = lambda input_args: None
    model._split_prompts_for_dp = lambda input_args: input_args
    model._gather_dp_outputs = lambda output: output

    calls = []

    def _run_timed_pipe(input_args):
        calls.append(input_args["prompt"])
        return DiffusionOutput(images=[len(calls)], pipe_args=[]), float(len(calls))

    model._run_timed_pipe = _run_timed_pipe
    return model, calls


@pytest.mark.parametrize("warmup_calls", [1, 2])
def test_warmup_calls_leave_every_measured_iteration_in_the_timings(monkeypatch, warmup_calls):
    model, calls = _runner(monkeypatch, warmup_calls=warmup_calls, num_iterations=3)

    _, timings = model.run({"prompt": "p"})

    assert len(calls) == warmup_calls + 3
    # The warmup calls took the first durations; all three measured ones remain.
    assert timings == [float(warmup_calls + i) for i in (1, 2, 3)]


def test_without_warmup_calls_the_first_iteration_is_left_out(monkeypatch):
    model, _ = _runner(monkeypatch, warmup_calls=0, num_iterations=3)

    _, timings = model.run({"prompt": "p"})

    assert timings == [2.0, 3.0]


def test_without_warmup_calls_a_single_iteration_is_kept(monkeypatch):
    model, _ = _runner(monkeypatch, warmup_calls=0, num_iterations=1)

    _, timings = model.run({"prompt": "p"})

    assert timings == [1.0]


def test_warmup_calls_keep_every_batch_of_a_batched_run(monkeypatch):
    model, calls = _runner(monkeypatch, warmup_calls=1, num_iterations=1, batch_size=1)

    _, timings = model.run({"prompt": ["a", "b"]})

    # One warmup call on the first batch, then one timed call per batch.
    assert calls == [["a"], ["a"], ["b"]]
    assert timings == [2.0, 3.0]
