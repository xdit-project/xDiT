from types import SimpleNamespace
from unittest import mock

import pytest
from diffusers.guiders import ClassifierFreeGuidance


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


def _hv15_args(model, **overrides):
    # Keys the CLI always passes; unset flags are None.
    args = {"prompt": "a cat", "dataset_path": None, "seed": 0, "num_frames": 5, "height": 64, "width": 64}
    args |= {"negative_prompt": None, "guidance_scale": None, "num_inference_steps": 2}
    return model.preprocess_args({**args, **overrides})


def test_hunyuanvideo15_forwards_negative_prompt_and_guidance_for_one_call(build):
    model = build(f"{COMMUNITY}-720p_t2v", "t2v")
    guider = ClassifierFreeGuidance(guidance_scale=6.0)
    scales = []

    def run(**_):
        scales.append(model.pipe.guider.guidance_scale)
        return SimpleNamespace(frames=["video"])

    model.pipe = mock.MagicMock(side_effect=run)
    model.pipe.guider = guider
    model.pipe._execution_device = "cpu"

    model._run_pipe(_hv15_args(model, negative_prompt="blurry", guidance_scale=2.5))
    model._run_pipe(_hv15_args(model))

    assert [call.kwargs["negative_prompt"] for call in model.pipe.call_args_list] == ["blurry", None]
    assert scales == [2.5, 6.0]
    assert model.pipe.guider is guider


@pytest.mark.parametrize(
    ("name", "config"),
    [("Hunyuanvideo-1.5-Distilled", {}), ("Hunyuanvideo-1.5-Sparse", {"attention_backend": "FLEX_BLOCK_ATTN"})],
)
@pytest.mark.parametrize("flag", [{"negative_prompt": "blurry"}, {"guidance_scale": 6.0}])
def test_hunyuanvideo15_guidance_free_checkpoints_refuse_guidance_flags(build, name, config, flag):
    model = build(name, "i2v", **config)

    with pytest.raises(ValueError, match="drop --negative_prompt and --guidance_scale"):
        _hv15_args(model, input_images=["image.png"], **flag)


def test_hunyuanvideo15_refuses_negative_prompt_when_guidance_scale_is_one(build):
    model = build(f"{COMMUNITY}-720p_t2v", "t2v")

    with pytest.raises(ValueError, match="--guidance_scale 1 turns guidance off"):
        _hv15_args(model, negative_prompt="blurry", guidance_scale=1)
    _hv15_args(model, guidance_scale=1)
