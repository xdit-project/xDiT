import sys
from types import ModuleType
from unittest import mock

import pytest


COMMUNITY = "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers"


@pytest.fixture
def build(monkeypatch):
    from xfuser import xFuserArgs
    from xfuser.model_executor.models.runner_models import hunyuan  # noqa: F401  registers
    from xfuser.model_executor.models.runner_models.base_model import MODEL_REGISTRY

    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")

    def _build(name, task, **kwargs):
        return MODEL_REGISTRY[name](xFuserArgs(model=name, task=task, **kwargs))

    return _build


@pytest.mark.parametrize(
    ("name", "task", "checkpoint", "default_size"),
    [
        (f"{COMMUNITY}-480p_i2v", "i2v", f"{COMMUNITY}-480p_i2v", (480, 848)),
        (f"{COMMUNITY}-480p_t2v", "t2v", f"{COMMUNITY}-480p_t2v", (480, 848)),
        (f"{COMMUNITY}-720p_i2v", "i2v", f"{COMMUNITY}-720p_i2v", (720, 1280)),
        (f"{COMMUNITY}-720p_t2v", "t2v", f"{COMMUNITY}-720p_t2v", (720, 1280)),
        ("Hunyuanvideo-1.5", "i2v", f"{COMMUNITY}-720p_i2v", (720, 1280)),
        ("Hunyuanvideo-1.5", "t2v", f"{COMMUNITY}-720p_t2v", (720, 1280)),
        ("tencent/HunyuanVideo-1.5", "i2v", f"{COMMUNITY}-720p_i2v", (720, 1280)),
        ("tencent/HunyuanVideo-1.5", "t2v", f"{COMMUNITY}-720p_t2v", (720, 1280)),
    ],
)
def test_hunyuanvideo15_name_selects_its_checkpoint(build, name, task, checkpoint, default_size):
    model = build(name, task)

    assert model.loader.checkpoint_request().model_name_or_path == checkpoint
    assert (model.default_input_values.height, model.default_input_values.width) == default_size


@pytest.mark.parametrize(
    ("name", "height", "width", "checkpoint"),
    [
        (f"{COMMUNITY}-480p_t2v", 720, 1280, f"{COMMUNITY}-480p_t2v"),
        (f"{COMMUNITY}-720p_t2v", 480, 848, f"{COMMUNITY}-720p_t2v"),
        ("Hunyuanvideo-1.5", 480, 848, f"{COMMUNITY}-720p_t2v"),
        ("tencent/HunyuanVideo-1.5", 480, 848, f"{COMMUNITY}-720p_t2v"),
    ],
)
def test_hunyuanvideo15_output_size_does_not_change_checkpoint(build, name, height, width, checkpoint):
    model = build(name, "t2v", height=height, width=width)
    input_args = model.preprocess_args(
        {"prompt": "A dog runs through a meadow", "dataset_path": None, "height": height, "width": width}
    )

    assert model.loader.checkpoint_request().model_name_or_path == checkpoint
    assert (input_args["height"], input_args["width"]) == (height, width)


@pytest.mark.parametrize(
    ("name", "task"),
    [(f"{COMMUNITY}-480p_i2v", "t2v"), (f"{COMMUNITY}-720p_t2v", "i2v")],
)
def test_hunyuanvideo15_checkpoint_refuses_the_other_task(build, name, task):
    with pytest.raises(ValueError, match=f"cannot run --task {task}"):
        build(name, task)


def test_hunyuanvideo15_distilled_keeps_its_own_checkpoint(build):
    name = f"{COMMUNITY}-720p_i2v_distilled"

    assert build(name, "i2v").settings.model_name == name


@pytest.mark.parametrize(
    "requested, world_size, use_fp8_gemms",
    [(False, 1, False), (True, 1, False), (True, 8, False), (False, 1, True), (True, 1, True)],
)
@pytest.mark.parametrize("task", ["t2v", "i2v"])
def test_hunyuanvideo15_load_preserves_task_and_transformer_identity(
    build, monkeypatch, requested, world_size, use_fp8_gemms, task
):
    import diffusers
    import torch

    from types import SimpleNamespace
    from xfuser.model_executor.models.runner_models.loading import meta_load

    model = build("Hunyuanvideo-1.5", task, memory_efficient_replicated_load=requested, use_fp8_gemms=use_fp8_gemms)
    monkeypatch.setattr(meta_load, "get_world_group", lambda: SimpleNamespace(world_size=world_size))
    transformer = object()
    wrapper = mock.Mock()
    wrapper.from_pretrained.return_value = transformer
    wrapper_module = ModuleType("xfuser.model_executor.models.transformers.transformer_hunyuan_video15")
    wrapper_module.xFuserHunyuanVideo15Transformer3DWrapper = wrapper
    monkeypatch.setitem(sys.modules, wrapper_module.__name__, wrapper_module)
    pipeline = mock.Mock()
    other_pipeline = mock.Mock()
    name = "HunyuanVideo15Pipeline" if task == "t2v" else "HunyuanVideo15ImageToVideoPipeline"
    other_name = "HunyuanVideo15ImageToVideoPipeline" if task == "t2v" else "HunyuanVideo15Pipeline"
    monkeypatch.setattr(diffusers, name, pipeline, raising=False)
    monkeypatch.setattr(diffusers, other_name, other_pipeline, raising=False)
    monkeypatch.setattr(model.loader, "load_transformer", mock.Mock(return_value=transformer))

    result = model._load_model()

    assert result is pipeline.from_pretrained.return_value
    pipeline.from_pretrained.assert_called_once_with(
        pretrained_model_name_or_path=f"{COMMUNITY}-720p_{task}",
        transformer=transformer,
        torch_dtype=torch.bfloat16,
    )
    other_pipeline.from_pretrained.assert_not_called()
    if requested and world_size > 1:
        model.loader.load_transformer.assert_called_once_with(wrapper, subfolder="transformer")
        wrapper.from_pretrained.assert_not_called()
    else:
        model.loader.load_transformer.assert_not_called()
        wrapper.from_pretrained.assert_called_once_with(
            f"{COMMUNITY}-720p_{task}",
            torch_dtype=torch.bfloat16,
            **model.loader.checkpoint_request("transformer").from_pretrained_kwargs(),
        )


def test_hunyuanvideo15_replicated_preflight_accepts_unquantized_sequence_parallel(build, monkeypatch):
    from xfuser.model_executor.models.runner_models.loading import meta_load
    from xfuser.model_executor.models.runner_models.loading.contracts import MaterializationMode

    monkeypatch.setattr(meta_load, "_is_cuda", lambda: False)
    model = build("Hunyuanvideo-1.5", "t2v", ulysses_degree=8, memory_efficient_replicated_load=True)

    model.loader.preflight(world_size=8)

    assert model.loader.load_contract.materialization_mode is MaterializationMode.REPLICATED_META


@pytest.mark.parametrize(
    "name, attention_backend",
    [("Hunyuanvideo-1.5-Distilled", "SDPA"), ("Hunyuanvideo-1.5-Sparse", "FLEX_BLOCK_ATTN")],
)
def test_hunyuanvideo15_specialized_weights_refuse_replicated_preflight(build, name, attention_backend, monkeypatch):
    from xfuser.model_executor.models.runner_models.loading import meta_load
    from xfuser.model_executor.models.runner_models.loading.contracts import UnsupportedLoadContract

    monkeypatch.setattr(meta_load, "_is_cuda", lambda: False)
    model = build(name, "i2v", memory_efficient_replicated_load=True, attention_backend=attention_backend)

    with pytest.raises(UnsupportedLoadContract, match="does not support replicated_meta"):
        model.loader.preflight(world_size=8)


def test_hunyuanvideo15_replicated_load_refuses_fp8_before_allocation(build, monkeypatch):
    from types import SimpleNamespace
    from xfuser.model_executor.models.runner_models.loading import meta_load

    model = build("Hunyuanvideo-1.5", "t2v", memory_efficient_replicated_load=True, use_fp8_gemms=True)
    monkeypatch.setattr(meta_load, "get_world_group", lambda: SimpleNamespace(world_size=8))
    with pytest.raises(ValueError, match="requires unquantized GEMMs"):
        model._load_model()
