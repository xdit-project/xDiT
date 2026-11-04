from types import SimpleNamespace

import pytest
import torch

import xfuser.model_executor.models.runner_models.base_model as base_model_module
import xfuser.model_executor.pipefusion.compile as pipefusion_compile
from xfuser.core import distributed
from xfuser.model_executor.cache import (
    CacheDitAdapterConfig,
    DBCachePreset,
    DBCacheSettings,
)
from xfuser.model_executor.models.runner_models.base_model import xFuserModel


@pytest.fixture(autouse=True)
def _restore_recompile_limit():
    # Assigned directly: monkeypatch.setattr deletes this config key on teardown in torch 2.15.
    original = torch._dynamo.config.recompile_limit
    yield
    torch._dynamo.config.recompile_limit = original


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


def test_pipefusion_cache_uses_stage_local_residual_engine(monkeypatch):
    applied = []
    settings = DBCacheSettings(
        adapter=CacheDitAdapterConfig(blocks=(("transformer_blocks", "Pattern_1"),)),
        preset=DBCachePreset(),
    )
    model = object.__new__(_Runner)
    model.config = SimpleNamespace(
        cache_method="pipefusion",
        cache_config=None,
        num_inference_steps=25,
    )
    model.settings = SimpleNamespace(
        step_cache_config={"dbcache": settings},
        transformer_attr_names=["transformer"],
    )
    model.pipe = SimpleNamespace(transformer=object())
    monkeypatch.setattr(base_model_module, "get_pipeline_parallel_world_size", lambda: 4)
    monkeypatch.setattr(base_model_module, "get_data_parallel_world_size", lambda: 1)
    monkeypatch.setattr(distributed, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(
        "xfuser.model_executor.cache.adapters.apply_cache",
        lambda **kwargs: applied.append(kwargs),
    )
    monkeypatch.setattr(base_model_module, "log", lambda _message: None)

    model._apply_step_cache()

    assert len(applied) == 1
    assert applied[0]["cache_method"] == "dbcache"
    assert applied[0]["preset_kwargs"] is not settings.preset
    assert applied[0]["preset_kwargs"].share_pipefusion_decisions is True
    assert settings.preset.share_pipefusion_decisions is False
    assert applied[0]["adapter_config"] is settings.adapter


def test_pipefusion_cache_rejects_multi_transformer_adapter():
    settings = DBCacheSettings(
        adapter=[
            CacheDitAdapterConfig(blocks=(("transformer_blocks", "Pattern_1"),)),
            CacheDitAdapterConfig(blocks=(("transformer_blocks", "Pattern_1"),)),
        ],
        preset=DBCachePreset(),
    )
    model = object.__new__(_Runner)
    model.config = SimpleNamespace(
        cache_method="pipefusion",
        cache_config=None,
        num_inference_steps=25,
    )
    model.settings = SimpleNamespace(
        step_cache_config={"dbcache": settings},
        transformer_attr_names=["transformer", "transformer_2"],
    )

    with pytest.raises(ValueError, match="does not yet support multi-transformer"):
        model._apply_step_cache()


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


def test_step_cache_only_forces_target_transformer_blockwise(monkeypatch):
    transformer = torch.nn.Module()
    transformer.transformer_blocks = torch.nn.ModuleList([torch.nn.Linear(4, 4)])
    text_encoder = torch.nn.Linear(4, 4)
    block = transformer.transformer_blocks[0]
    model = object.__new__(_Runner)
    model.config = SimpleNamespace(
        pipefusion_parallel_degree=1,
        fully_shard_degree=1,
        fully_shard_components=None,
        cache_method="dbcache",
    )
    model.pipe = SimpleNamespace(
        transformer=transformer,
        text_encoder=text_encoder,
    )
    model.settings = SimpleNamespace(
        transformer_attr_names=["transformer"],
        fsdp_strategy={
            "transformer": {"wrap_attrs": ["transformer_blocks"]},
        },
    )
    model._enable_compute_comm_overlap = lambda: None
    model._get_compile_mode = lambda: "default"
    model._get_compile_dynamic = lambda: False
    model._get_compiled_pipe_components = lambda: [
        "transformer",
        "text_encoder",
    ]
    model._get_compile_warmup_steps = lambda _input_args: None
    model._prefer_blockwise_compile = lambda: False
    model._run_compile_warmup = lambda _input_args: None
    compiled = []

    def compile_candidate(candidate, **_kwargs):
        compiled.append(candidate)
        return candidate

    monkeypatch.setattr(torch, "compile", compile_candidate)

    model._compile_model({"num_inference_steps": 4})

    assert compiled == [block, text_encoder]


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
        runtime_config=SimpleNamespace(warmup_steps=1),
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
        runtime_config=SimpleNamespace(warmup_steps=1),
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
        pipefusion_compile.PipeFusionCompileCapture,
        "replay",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)

    with pytest.raises(RuntimeError, match="rank 0: RuntimeError: compile failed"):
        model._compile_model({"num_inference_steps": 4})

    assert events == ["stage", "reset", "reset"]


def _blocks_only_runner(pp_degree, num_patches, num_blocks):
    model = object.__new__(_Runner)
    model.config = SimpleNamespace(
        pipefusion_parallel_degree=pp_degree,
        num_pipeline_patch=num_patches,
        fully_shard_degree=1,
        fully_shard_components=None,
        cache_method=None,
    )
    model.pipe = SimpleNamespace(transformer=SimpleNamespace(transformer_blocks=[object() for _ in range(num_blocks)]))
    model.settings = SimpleNamespace(fsdp_strategy={"transformer": {"wrap_attrs": ["transformer_blocks"]}})
    model._enable_compute_comm_overlap = lambda: None
    model._get_compile_mode = lambda: "default"
    model._get_compile_dynamic = lambda _input_args=None: False
    model._get_compiled_pipe_components = lambda: ["transformer"]
    model._get_compile_warmup_steps = lambda _input_args: None
    model._run_compile_warmup = lambda _input_args: None
    return model


def test_pipefusion_compile_leaves_recompile_limit_raised_for_every_local_block(monkeypatch):
    model = _blocks_only_runner(pp_degree=4, num_patches=4, num_blocks=15)
    monkeypatch.setattr(base_model_module, "runtime_state_is_initialized", lambda: False)
    monkeypatch.setattr(torch, "compile", lambda candidate, **_kwargs: candidate)
    torch._dynamo.config.recompile_limit = 8

    model._compile_model({})

    # Step-cache adapters are installed after compile and trace their graphs during run().
    assert torch._dynamo.config.recompile_limit > 8


def test_pipefusion_compile_keeps_a_higher_recompile_limit(monkeypatch):
    model = _blocks_only_runner(pp_degree=4, num_patches=4, num_blocks=15)
    monkeypatch.setattr(base_model_module, "runtime_state_is_initialized", lambda: False)
    monkeypatch.setattr(torch, "compile", lambda candidate, **_kwargs: candidate)
    torch._dynamo.config.recompile_limit = 1000

    model._compile_model({})

    assert torch._dynamo.config.recompile_limit == 1000


def test_compile_without_pipefusion_leaves_recompile_limit_alone(monkeypatch):
    model = _blocks_only_runner(pp_degree=1, num_patches=1, num_blocks=15)
    monkeypatch.setattr(torch, "compile", lambda candidate, **_kwargs: candidate)
    torch._dynamo.config.recompile_limit = 8

    model._compile_model({})

    assert torch._dynamo.config.recompile_limit == 8
