"""Each Qwen-Image-Edit checkpoint must load through the pipeline diffusers uses for it.

Qwen-Image-Edit-2509 and -2511 are Edit Plus checkpoints: diffusers loads them
with QwenImageEditPlusPipeline, which builds its prompt from a list of reference
images. The runner loaded them through QwenImageEditPipeline instead.
"""

import copy
from types import SimpleNamespace

import diffusers
import pytest
import torch
from diffusers import DiffusionPipeline, QwenImageEditPipeline
from xfuser.model_executor.models.runner_models.qwen import (
    xFuserQwenImageEditModel,
    xFuserQwenImageModel,
)

# Diffusers added the Edit Plus pipeline after Qwen-Image-Edit itself.
QwenImageEditPlusPipeline = getattr(diffusers, "QwenImageEditPlusPipeline", None)
requires_edit_plus = pytest.mark.skipif(
    QwenImageEditPlusPipeline is None, reason="this diffusers release has no QwenImageEditPlusPipeline"
)


@pytest.fixture
def build(monkeypatch):
    from xfuser import xFuserArgs
    from xfuser.model_executor.models.runner_models import qwen  # noqa: F401  registers
    from xfuser.model_executor.models.runner_models.base_model import MODEL_REGISTRY

    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")

    def _build(name):
        return MODEL_REGISTRY[name](xFuserArgs(model=name))

    return _build


@pytest.fixture
def loaded_classes(monkeypatch):
    loaded = []

    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        loaded.append((cls, pretrained_model_name_or_path))
        return SimpleNamespace()

    monkeypatch.setattr(DiffusionPipeline, "from_pretrained", classmethod(from_pretrained))
    return loaded


def _load(model):
    model.loader = SimpleNamespace(
        load_transformer=lambda wrapper_cls: object(),
        plan_text_encoders=lambda: ({}, None),
    )
    model._load_model()


@requires_edit_plus
@pytest.mark.parametrize(
    ("name", "checkpoint"),
    [
        ("Qwen/Qwen-Image-Edit-2509", "Qwen/Qwen-Image-Edit-2509"),
        ("Qwen-Image-Edit-2509", "Qwen/Qwen-Image-Edit-2509"),
        ("Qwen/Qwen-Image-Edit-2511", "Qwen/Qwen-Image-Edit-2511"),
        ("Qwen-Image-Edit-2511", "Qwen/Qwen-Image-Edit-2511"),
    ],
)
def test_edit_plus_checkpoints_load_the_edit_plus_pipeline(build, loaded_classes, name, checkpoint):
    _load(build(name))

    [(pipeline_cls, loaded_checkpoint)] = loaded_classes
    assert loaded_checkpoint == checkpoint
    assert issubclass(pipeline_cls, QwenImageEditPlusPipeline)


@pytest.mark.parametrize("name", ["Qwen/Qwen-Image-Edit", "Qwen-Image-Edit"])
def test_original_edit_checkpoint_keeps_the_edit_pipeline(build, loaded_classes, name):
    _load(build(name))

    [(pipeline_cls, loaded_checkpoint)] = loaded_classes
    assert loaded_checkpoint == "Qwen/Qwen-Image-Edit"
    assert issubclass(pipeline_cls, QwenImageEditPipeline)
    assert QwenImageEditPlusPipeline is None or not issubclass(pipeline_cls, QwenImageEditPlusPipeline)


def test_edit_plus_passes_every_reference_image(build):
    model = build("Qwen/Qwen-Image-Edit-2509")
    calls = []

    def pipe(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(images=["edited"])

    model.pipe = pipe
    model._make_generator = lambda seed: None
    references = ["first.png", "second.png"]
    input_args = {
        "input_images": references,
        "prompt": "put the cat next to the dog",
        "negative_prompt": " ",
        "num_inference_steps": 2,
        "guidance_scale": 4.0,
        "seed": 0,
    }

    model._run_pipe(input_args)

    assert calls[0]["image"] == references


def _customize(runner_cls, *, model, pp, cfg=False, split=None, ulysses=1, ring=1):
    runner = object.__new__(runner_cls)
    runner.settings = copy.deepcopy(runner_cls.settings)
    config = SimpleNamespace(
        model=model,
        pipefusion_parallel_degree=pp,
        use_cfg_parallel=cfg,
        attn_layer_num_for_pp=split,
        ulysses_degree=ulysses,
        ring_degree=ring,
    )
    runner._customize_settings(config)
    return runner, config


@pytest.mark.parametrize(
    ("name", "pp", "expected"),
    [
        ("Qwen/Qwen-Image", 2, [30, 30]),
        ("Qwen-Image", 3, [20, 20, 20]),
        ("Qwen/Qwen-Image-2512", 4, [15, 15, 15, 15]),
        ("Qwen-Image-2512", 8, [8, 8, 8, 8, 7, 7, 7, 7]),
    ],
)
def test_qwen_image_aliases_enable_balanced_pipefusion(name, pp, expected):
    _, config = _customize(xFuserQwenImageModel, model=name, pp=pp)

    assert config.attn_layer_num_for_pp == expected


@pytest.mark.parametrize(
    ("cfg", "pp", "expected"),
    [
        (False, 4, [15, 15, 15, 15]),
        (True, 2, [30, 30]),
        (True, 3, [20, 20, 20]),
        (True, 8, [8, 8, 8, 8, 7, 7, 7, 7]),
    ],
)
def test_base_edit_supports_general_pipefusion_topologies(cfg, pp, expected):
    _, config = _customize(
        xFuserQwenImageEditModel,
        model="Qwen/Qwen-Image-Edit",
        pp=pp,
        cfg=cfg,
    )

    assert config.attn_layer_num_for_pp == expected


@pytest.mark.parametrize("name", ["Qwen/Qwen-Image-Edit-2509", "Qwen-Image-Edit-2511"])
def test_edit_plus_rejects_pipefusion(name):
    with pytest.raises(ValueError, match="Edit Plus.*does not support PipeFusion"):
        _customize(xFuserQwenImageEditModel, model=name, pp=4)


def test_qwen_image_accepts_user_tuned_stage_split():
    _, config = _customize(
        xFuserQwenImageModel,
        model="Qwen/Qwen-Image",
        pp=4,
        split=[14, 15, 15, 16],
    )

    assert config.attn_layer_num_for_pp == [14, 15, 15, 16]


def test_qwen_image_rejects_incomplete_stage_split():
    with pytest.raises(ValueError, match="must assign all 60"):
        _customize(
            xFuserQwenImageModel,
            model="Qwen/Qwen-Image",
            pp=4,
            split=[14, 15, 15, 15],
        )


@pytest.mark.parametrize(("parallelism", "degree"), [("ulysses", 2), ("ring", 2)])
def test_qwen_pipefusion_rejects_sequence_parallel_composition(parallelism, degree):
    with pytest.raises(ValueError, match="does not yet compose with sequence parallelism"):
        _customize(
            xFuserQwenImageModel,
            model="Qwen/Qwen-Image",
            pp=4,
            **{parallelism: degree},
        )


@pytest.mark.parametrize(
    ("runner_cls", "extra_args"),
    [
        (xFuserQwenImageModel, {}),
        (xFuserQwenImageEditModel, {"input_images": ["cat.png"]}),
    ],
)
def test_qwen_runner_accepts_no_output_on_non_owner_rank(runner_cls, extra_args):
    runner = object.__new__(runner_cls)
    runner.pipe = lambda **_kwargs: None
    runner._make_generator = lambda _seed: None
    runner._is_edit_plus = False
    input_args = {
        "prompt": "cat",
        "negative_prompt": " ",
        "height": 512,
        "width": 512,
        "num_inference_steps": 2,
        "guidance_scale": 4.0,
        "seed": 0,
        **extra_args,
    }

    assert runner._run_pipe(input_args).images == []


@pytest.mark.parametrize(
    ("runner_cls", "pipeline_path"),
    [
        (
            xFuserQwenImageModel,
            "xfuser.model_executor.pipelines.pipeline_qwen_image.xFuserQwenImagePipeFusionPipeline",
        ),
        (
            xFuserQwenImageEditModel,
            "xfuser.model_executor.pipelines.pipeline_qwen_image_edit.xFuserQwenImageEditPipeFusionPipeline",
        ),
    ],
)
def test_pipefusion_load_uses_stage_local_component_plan(monkeypatch, runner_cls, pipeline_path):
    runner = object.__new__(runner_cls)
    runner.config = SimpleNamespace(pipefusion_parallel_degree=4)
    runner.settings = copy.deepcopy(runner_cls.settings)
    runner.engine_config = SimpleNamespace(runtime_config=SimpleNamespace(dtype=None))
    source_transformer = object()
    stage_transformer = object()
    stage_pipe = SimpleNamespace(transformer=stage_transformer)
    events = []
    runner.loader = SimpleNamespace(
        plan_pipefusion_components=lambda transformer_cls, **kwargs: (
            events.append(("plan", transformer_cls, kwargs)) or source_transformer,
            {"text_encoder": object()},
        ),
        mark_pipeline_stage_blockwise=lambda stage, source: events.append(("mark", stage, source)),
    )
    pipeline_cls = __import__(pipeline_path.rsplit(".", 1)[0], fromlist=[pipeline_path.rsplit(".", 1)[1]])
    pipeline_cls = getattr(pipeline_cls, pipeline_path.rsplit(".", 1)[1])
    monkeypatch.setattr(pipeline_cls, "from_pretrained", lambda **kwargs: events.append(("load", kwargs)) or stage_pipe)

    result = runner._load_model()

    assert result is stage_pipe
    assert events[0][0] == "plan"
    assert events[1][0] == "load"
    assert events[2] == ("mark", stage_transformer, source_transformer)
    assert runner.engine_config.runtime_config.dtype is torch.bfloat16
