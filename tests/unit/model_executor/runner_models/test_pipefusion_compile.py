from types import SimpleNamespace

import pytest
import torch

import xfuser.model_executor.models.runner_models.base_model as base_model_module
from xfuser.model_executor.models.runner_models.base_model import xFuserModel


class _Runner(xFuserModel):
    def _load_model(self):
        raise NotImplementedError

    def _run_pipe(self, input_args):
        raise NotImplementedError


class _Stage(torch.nn.Module):
    def __init__(self, events):
        super().__init__()
        self.events = events
        self.transformer_blocks = torch.nn.ModuleList([torch.nn.Identity()])

    def forward(self, hidden_states):
        self.events.append("stage")
        for block in self.transformer_blocks:
            hidden_states = block(hidden_states)
        return hidden_states


def test_pipefusion_compiles_the_bound_stage_forward(monkeypatch):
    transformer = torch.nn.Linear(4, 4)

    def compiled_forward(value):
        return value

    model = object.__new__(_Runner)
    model.config = SimpleNamespace(
        pipefusion_parallel_degree=2,
        fully_shard_degree=1,
        fully_shard_components=None,
        cache_method=None,
    )
    model.pipe = SimpleNamespace(transformer=transformer)
    model.settings = SimpleNamespace(fsdp_strategy={})
    model._enable_compute_comm_overlap = lambda: None
    model._get_compile_mode = lambda: "default"
    model._get_compile_dynamic = lambda _input_args=None: False
    model._get_compiled_pipe_components = lambda: ["transformer"]
    model._get_compile_warmup_steps = lambda _input_args: None
    model._run_compile_warmup = lambda _input_args: None
    monkeypatch.setattr(torch, "compile", lambda _candidate, **_kwargs: compiled_forward)

    model._compile_model({"num_inference_steps": 4})

    assert model.pipe.transformer is transformer
    assert transformer.forward is compiled_forward


def test_pipefusion_compiles_stage_local_blocks_when_declared(monkeypatch):
    transformer = torch.nn.Module()
    transformer.transformer_blocks = torch.nn.ModuleList([torch.nn.Linear(4, 4), torch.nn.Linear(4, 4)])
    original_forward = transformer.forward
    original_blocks = list(transformer.transformer_blocks)
    model = object.__new__(_Runner)
    model.config = SimpleNamespace(
        pipefusion_parallel_degree=2,
        fully_shard_degree=1,
        fully_shard_components=None,
        cache_method=None,
    )
    model.pipe = SimpleNamespace(transformer=transformer)
    model.settings = SimpleNamespace(fsdp_strategy={"transformer": {"wrap_attrs": ["transformer_blocks"]}})
    model._enable_compute_comm_overlap = lambda: None
    model._get_compile_mode = lambda: "default"
    model._get_compile_dynamic = lambda _input_args=None: False
    model._get_compiled_pipe_components = lambda: ["transformer"]
    model._get_compile_warmup_steps = lambda _input_args: None
    model._run_compile_warmup = lambda _input_args: None

    compiled_blocks = []

    def compile_block(candidate, **_kwargs):
        compiled_blocks.append(candidate)
        return candidate

    monkeypatch.setattr(torch, "compile", compile_block)

    model._compile_model({"num_inference_steps": 4})

    assert compiled_blocks == original_blocks
    assert transformer.forward == original_forward


def test_pipefusion_captures_before_wrap_and_replays_before_validation(monkeypatch):
    events = []
    state = SimpleNamespace(
        patch_mode=False,
        pipeline_patch_idx=0,
        pp_patches_height=[1],
        pp_patches_start_idx_local=[0, 1],
        pp_patches_start_end_idx_global=[[0, 1]],
        pp_patches_token_num=[1],
        pp_patches_token_start_idx_local=[0, 1],
        pp_patches_token_start_end_idx_global=[[0, 1]],
    )
    transformer = _Stage(events)
    model = object.__new__(_Runner)
    model.config = SimpleNamespace(
        pipefusion_parallel_degree=2,
        fully_shard_degree=1,
        fully_shard_components=None,
        cache_method=None,
    )
    model.pipe = SimpleNamespace(transformer=transformer)
    model.settings = SimpleNamespace(fsdp_strategy={"transformer": {"wrap_attrs": ["transformer_blocks"]}})
    model.engine_config = SimpleNamespace(runtime_config=SimpleNamespace(warmup_steps=1))
    model._enable_compute_comm_overlap = lambda: None
    model._get_compile_mode = lambda: "default"
    model._get_compile_dynamic = lambda _input_args=None: False
    model._get_compiled_pipe_components = lambda: ["transformer"]
    model._get_compile_warmup_steps = lambda _input_args: 2
    model._local_onload_device = lambda: torch.device("cpu")
    model._reset_pipefusion_compile_state = lambda _components: events.append("reset")

    def eager_capture(_input_args):
        events.append("capture")
        state.patch_mode = False
        state.pipeline_patch_idx = 0
        transformer(torch.ones(1, 2))
        state.patch_mode = True
        transformer(torch.ones(1, 1))

    model._run_timed_pipe = eager_capture
    model._run_compile_warmup = lambda _input_args: events.append("validation")

    class _Replica:
        def barrier(self):
            events.append("barrier")

        def all_gather_object(self, value):
            events.append("status")
            return [value]

    monkeypatch.setattr(base_model_module, "runtime_state_is_initialized", lambda: True)
    monkeypatch.setattr(base_model_module, "get_runtime_state", lambda: state)
    monkeypatch.setattr(base_model_module, "get_model_replica_group", lambda: _Replica())
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: events.append("synchronize"))
    monkeypatch.setattr(
        torch,
        "compile",
        lambda candidate, **_kwargs: events.append("compile") or candidate,
    )

    model._compile_model({"num_inference_steps": 4})

    assert events.index("capture") < events.index("compile")
    assert events.index("compile") < events.index("synchronize")
    assert events.index("synchronize") < events.index("validation")
    assert events.count("stage") == 4  # two eager captures and two local replays


def test_pipefusion_compile_replay_failure_is_raised_collectively(monkeypatch):
    events = []
    state = SimpleNamespace(
        patch_mode=False,
        pipeline_patch_idx=0,
        pp_patches_height=[1],
        pp_patches_start_idx_local=[0, 1],
        pp_patches_start_end_idx_global=[[0, 1]],
        pp_patches_token_num=[1],
        pp_patches_token_start_idx_local=[0, 1],
        pp_patches_token_start_end_idx_global=[[0, 1]],
    )
    transformer = _Stage(events)
    model = object.__new__(_Runner)
    model.config = SimpleNamespace(
        pipefusion_parallel_degree=2,
        fully_shard_degree=1,
        fully_shard_components=None,
        cache_method=None,
    )
    model.pipe = SimpleNamespace(transformer=transformer)
    model.settings = SimpleNamespace(fsdp_strategy={"transformer": {"wrap_attrs": ["transformer_blocks"]}})
    model.engine_config = SimpleNamespace(runtime_config=SimpleNamespace(warmup_steps=1))
    model._enable_compute_comm_overlap = lambda: None
    model._get_compile_mode = lambda: "default"
    model._get_compile_dynamic = lambda _input_args=None: False
    model._get_compiled_pipe_components = lambda: ["transformer"]
    model._get_compile_warmup_steps = lambda _input_args: 2
    model._local_onload_device = lambda: torch.device("cpu")
    model._reset_pipefusion_compile_state = lambda _components: events.append("reset")
    model._run_compile_warmup = lambda _input_args: events.append("validation")

    def eager_capture(_input_args):
        transformer(torch.ones(1, 2))

    model._run_timed_pipe = eager_capture

    class _Replica:
        def barrier(self):
            pass

        def all_gather_object(self, value):
            assert value is None
            return ["RuntimeError: compile failed", value]

    monkeypatch.setattr(base_model_module, "runtime_state_is_initialized", lambda: True)
    monkeypatch.setattr(base_model_module, "get_runtime_state", lambda: state)
    monkeypatch.setattr(base_model_module, "get_model_replica_group", lambda: _Replica())
    monkeypatch.setattr(
        torch,
        "compile",
        lambda candidate, **_kwargs: candidate,
    )
    monkeypatch.setattr(
        base_model_module.PipeFusionCompileCapture,
        "replay",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)

    with pytest.raises(RuntimeError, match="rank 0: RuntimeError: compile failed"):
        model._compile_model({"num_inference_steps": 4})

    assert events == ["stage", "reset", "reset"]


def test_pipefusion_raises_recompile_limit_for_patch_specializations(monkeypatch):
    model = object.__new__(_Runner)
    model.config = SimpleNamespace(num_pipeline_patch=8)
    monkeypatch.setattr(base_model_module, "get_pipeline_parallel_world_size", lambda: 8)
    monkeypatch.setattr(torch._dynamo.config, "recompile_limit", 8)

    model._enable_compute_comm_overlap()

    assert torch._dynamo.config.recompile_limit == 20
