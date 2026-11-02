"""xFuserModel.run reports the timings of the iterations it measured."""

import itertools
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


def _runner(
    monkeypatch, *, warmup_calls, num_iterations, batch_size=None, use_torch_compile=False, compile_mode="default"
):
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
        use_torch_compile=use_torch_compile,
    )
    model._get_compile_mode = lambda: compile_mode
    model.settings = SimpleNamespace(model_output_type="image")
    model._validate_args = lambda input_args: None
    model._split_prompts_for_dp = lambda input_args: input_args
    model._gather_dp_outputs = lambda output: output

    call_number = itertools.count(1)

    def _run_timed_pipe(input_args):
        n = next(call_number)
        return DiffusionOutput(images=[n], pipe_args=[]), float(n)

    model._run_timed_pipe = _run_timed_pipe
    return model


@pytest.mark.parametrize(
    "warmup_calls, num_iterations, batch_size, prompt, expected",
    [
        pytest.param(1, 3, None, "p", [2.0, 3.0, 4.0], id="warmup-keeps-every-iteration"),
        pytest.param(1, 1, 1, ["a", "b"], [2.0, 3.0], id="warmup-keeps-every-batch"),
        pytest.param(0, 3, None, "p", [2.0, 3.0], id="no-warmup-drops-the-first-iteration"),
        pytest.param(0, 1, None, "p", [1.0], id="no-warmup-keeps-a-single-iteration"),
    ],
)
def test_run_reports_the_measured_timings(monkeypatch, warmup_calls, num_iterations, batch_size, prompt, expected):
    # The stubbed pipe takes 1.0 s on its first call, 2.0 s on its second, and so on.
    model = _runner(monkeypatch, warmup_calls=warmup_calls, num_iterations=num_iterations, batch_size=batch_size)

    _, timings = model.run({"prompt": prompt})

    assert timings == expected


def test_cuda_graph_compile_mode_keeps_every_measured_iteration(monkeypatch):
    """reduce-overhead records one untimed warmup, then reports every measured iteration.

    The stubbed pipe takes 1.0 s, then 2.0 s, and so on. The capture warmup consumes
    the 1.0 s call and is not a timing. All three measured iterations are reported,
    which is only possible once that warmup has absorbed the one-time cost that
    would otherwise make the runner discard the first measurement.
    """
    model = _runner(
        monkeypatch,
        warmup_calls=0,
        num_iterations=3,
        use_torch_compile=True,
        compile_mode="reduce-overhead",
    )

    _, timings = model.run({"prompt": "p"})

    assert timings == [2.0, 3.0, 4.0]


def test_explicit_warmup_calls_are_not_repeated_for_cuda_graphs(monkeypatch):
    """--warmup_calls already runs a full pipe before timings, so reduce-overhead must not add another."""
    model = _runner(
        monkeypatch,
        warmup_calls=1,
        num_iterations=3,
        use_torch_compile=True,
        compile_mode="reduce-overhead",
    )

    _, timings = model.run({"prompt": "p"})

    assert timings == [2.0, 3.0, 4.0]


@pytest.mark.parametrize("compile_mode", ["default", "max-autotune-no-cudagraphs"])
def test_modes_without_cuda_graphs_still_drop_the_first_iteration(monkeypatch, compile_mode):
    model = _runner(
        monkeypatch,
        warmup_calls=0,
        num_iterations=3,
        use_torch_compile=True,
        compile_mode=compile_mode,
    )

    _, timings = model.run({"prompt": "p"})

    assert timings == [2.0, 3.0]
