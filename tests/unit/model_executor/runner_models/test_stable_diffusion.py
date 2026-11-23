import pytest
import copy
from types import SimpleNamespace

torch = pytest.importorskip("torch")

from xfuser import xFuserArgs
from xfuser.model_executor.models.runner_models.stable_diffusion import (
    xFuserStableDiffusionModel,
)
from xfuser.model_executor.models.runner_models import stable_diffusion


@pytest.fixture(autouse=True)
def single_process_env(monkeypatch):
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")


@pytest.mark.parametrize("world_size", [1, 8])
def test_sd35_non_pipefusion_flag_keeps_single_rank_eager(tmp_path, world_size):
    from xfuser.model_executor.models.runner_models.loading.contracts import (
        MaterializationMode,
        UnsupportedLoadContract,
    )

    model = xFuserStableDiffusionModel(
        xFuserArgs(model="SD3.5", memory_efficient_replicated_load=True, output_directory=str(tmp_path))
    )
    if world_size == 1:
        model.loader.preflight(world_size=world_size)
        assert model.loader.load_contract.materialization_mode is MaterializationMode.EAGER
    else:
        with pytest.raises(UnsupportedLoadContract, match="does not support replicated_meta"):
            model.loader.preflight(world_size=world_size)


def test_sd35_pipefusion_accepts_replicated_load():
    model = object.__new__(xFuserStableDiffusionModel)
    model._validate_config(
        xFuserArgs(
            model="stabilityai/stable-diffusion-3.5-large",
            pipefusion_parallel_degree=4,
            use_cfg_parallel=True,
            memory_efficient_replicated_load=True,
        )
    )


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"fully_shard_degree": 2, "memory_efficient_sharding": True}, "memory_efficient_sharding"),
        (
            {
                "pipefusion_parallel_degree": 4,
                "memory_efficient_replicated_load": True,
                "fully_shard_degree": 2,
                "fully_shard_components": ["text_encoder_3"],
            },
            "FSDP",
        ),
        (
            {"pipefusion_parallel_degree": 4, "memory_efficient_replicated_load": True, "use_fp8_gemms": True},
            "quantized GEMMs",
        ),
    ],
)
def test_sd35_refuses_unverified_loading(kwargs, message):
    model = object.__new__(xFuserStableDiffusionModel)
    with pytest.raises(ValueError, match=message):
        model._validate_config(xFuserArgs(model="stabilityai/stable-diffusion-3.5-large", **kwargs))


@pytest.mark.parametrize("flag, pp_degree", [(False, 1), (False, 4), (True, 1), (True, 4)])
def test_sd35_pipeline_loading_preserves_eager_and_marks_meta_stage(monkeypatch, flag, pp_degree):
    from diffusers import SD3Transformer2DModel

    runner = object.__new__(xFuserStableDiffusionModel)
    runner.config = SimpleNamespace(
        pipefusion_parallel_degree=pp_degree, memory_efficient_replicated_load=flag, use_fp8_gemms=False
    )
    runner.settings = copy.deepcopy(xFuserStableDiffusionModel.settings)
    runner.engine_config = object()
    source = object()
    stage = object()
    pipe = SimpleNamespace(transformer=stage)
    events = []

    def plan(transformer_cls, **kwargs):
        events.append(("plan", transformer_cls, kwargs))
        return source, {"transformer": source}

    runner.loader = SimpleNamespace(
        plan_pipefusion_components=plan,
        mark_pipeline_stage_blockwise=lambda stage, source: events.append(("mark", stage, source)),
    )
    monkeypatch.setattr(
        stable_diffusion.xFuserStableDiffusion3Pipeline,
        "from_pretrained",
        lambda **kwargs: events.append(("load", kwargs)) or pipe,
    )

    assert runner._load_model() is pipe
    dtype = torch.float16 if pp_degree > 1 else torch.bfloat16
    expected_load = {
        "pretrained_model_name_or_path": runner.settings.model_name,
        "engine_config": runner.engine_config,
        "torch_dtype": dtype,
    }
    if flag and pp_degree > 1:
        expected_load["transformer"] = source
        assert events == [
            ("plan", SD3Transformer2DModel, {"torch_dtype": dtype}),
            ("load", expected_load),
            ("mark", stage, source),
        ]
    else:
        assert events == [("load", expected_load)]


def test_sd35_single_rank_flag_keeps_quantized_eager_load(tmp_path):
    xFuserStableDiffusionModel(
        xFuserArgs(
            model="SD3.5", memory_efficient_replicated_load=True, use_fp8_gemms=True, output_directory=str(tmp_path)
        )
    )
