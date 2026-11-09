from types import SimpleNamespace

from xfuser import runner
from xfuser.model_executor.models.runner_models.base_model import DiffusionOutput


def test_output_owner_is_rank_with_payload(monkeypatch):
    monkeypatch.setattr(runner.dist, "is_available", lambda: True)
    monkeypatch.setattr(runner.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(runner.dist, "get_rank", lambda: 3)
    monkeypatch.setattr(runner.dist, "get_world_size", lambda: 4)
    monkeypatch.setattr(runner.dist, "get_backend", lambda: "gloo")

    def all_reduce(candidate, op):
        assert candidate.item() == -1
        assert op == runner.dist.ReduceOp.MAX
        candidate.fill_(0)

    monkeypatch.setattr(runner.dist, "all_reduce", all_reduce)

    assert runner._select_output_owner(DiffusionOutput()) == 0


def test_runner_saves_from_selected_output_owner(monkeypatch):
    model = SimpleNamespace(
        save_output=lambda output: saved.append(("output", output)),
        save_timings=lambda timings: saved.append(("timings", timings)),
    )
    model_runner = object.__new__(runner.xFuserModelRunner)
    model_runner.model = model
    output = DiffusionOutput(images=[object()])
    saved = []

    monkeypatch.setattr(runner.dist, "is_available", lambda: True)
    monkeypatch.setattr(runner.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(runner.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(runner, "_select_output_owner", lambda _output: 0)

    model_runner.save(output=output, timings=[1.0])

    assert saved == [("output", output), ("timings", [1.0])]
