from types import SimpleNamespace

import pytest
from PIL import Image

pytest.importorskip(
    "diffusers.pipelines.qwenimage21.pipeline_qwenimage21",
    reason="installed diffusers does not include Qwen-Image-2.1",
)

from xfuser.model_executor.models.runner_models.qwen import xFuserQwenImage21Model

_qwen_image21_output_size = xFuserQwenImage21Model._qwen_image21_output_size


def test_explicit_size_wins_over_images():
    assert _qwen_image21_output_size([Image.new("RGB", (1600, 900))], 512, 768) == (512, 768)


def test_text_only_defaults_to_square():
    assert _qwen_image21_output_size([], None, None) == (2048, 2048)


def test_landscape_image_sets_landscape_output():
    assert _qwen_image21_output_size([Image.new("RGB", (1600, 900))], None, None) == (1536, 2720)


def test_last_image_decides_the_aspect_ratio():
    images = [Image.new("RGB", (1600, 900)), Image.new("RGB", (900, 1600))]
    assert _qwen_image21_output_size(images, None, None) == (2720, 1536)


def test_only_one_dimension_is_rejected():
    with pytest.raises(ValueError, match="both --height and --width"):
        _qwen_image21_output_size([], 1024, None)


@pytest.fixture
def build_qwen_image21(monkeypatch):
    from xfuser import xFuserArgs
    from xfuser.runner import xFuserModelRunner

    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    runner = object.__new__(xFuserModelRunner)

    def _build(name):
        return runner._select_model(name, xFuserArgs(model=name))

    return _build


@pytest.mark.parametrize(
    ("name", "checkpoint"),
    [
        ("Qwen-Image-2.1", "Qwen/Qwen-Image-2.1"),
        ("Qwen-Image-2.1-Turbo", "Qwen/Qwen-Image-2.1-Turbo"),
        ("Qwen/Qwen-Image-2.1-Turbo", "Qwen/Qwen-Image-2.1-Turbo"),
    ],
)
def test_name_selects_its_checkpoint(build_qwen_image21, name, checkpoint):
    model = build_qwen_image21(name)

    assert model.loader.checkpoint_request().model_name_or_path == checkpoint


def _turbo_args(model, **overrides):
    args = {"prompt": "a cat", "dataset_path": None, "height": 1024, "width": 1024}
    args["num_inference_steps"] = model.default_input_values.num_inference_steps
    return {**args, **overrides}


def test_turbo_rejects_steps_the_saved_schedule_would_ignore(build_qwen_image21):
    model = build_qwen_image21("Qwen-Image-2.1-Turbo")

    with pytest.raises(ValueError, match="num_inference_steps must be 8, got 40"):
        model._validate_args(_turbo_args(model, num_inference_steps=40))


def test_turbo_refuses_a_diffusers_that_dropped_the_saved_schedule(build_qwen_image21):
    model = build_qwen_image21("Qwen-Image-2.1-Turbo")
    model.pipe = SimpleNamespace(config=SimpleNamespace())

    with pytest.raises(RuntimeError, match="dropped the checkpoint's sample_sigmas"):
        model._post_load_and_state_initialization(_turbo_args(model))
