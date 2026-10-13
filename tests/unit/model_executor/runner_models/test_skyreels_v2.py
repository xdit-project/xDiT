"""SkyReels-V2 runner contract: argument validation and handling. No downloads."""

from types import SimpleNamespace

import pytest
from PIL import Image

pytest.importorskip(
    "diffusers.pipelines.skyreels_v2",
    reason="installed diffusers does not include SkyReels-V2",
)

from diffusers import UniPCMultistepScheduler  # noqa: E402

from xfuser import xFuserArgs  # noqa: E402
from xfuser.model_executor.models.runner_models.base_model import (  # noqa: E402
    MODEL_REGISTRY,
    xFuserModel,
)


@pytest.fixture(autouse=True)
def _single_process_env(monkeypatch):
    # runner_utils.log reads RANK/WORLD_SIZE straight from the environment.
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")


def _model(name, **config):
    return MODEL_REGISTRY[name](xFuserArgs(model=name, **config))


@pytest.mark.parametrize(
    "name, ulysses_degree",
    [("SkyReels-V2-I2V-1.3B", 8), ("SkyReels-V2-I2V-1.3B", 5), ("SkyReels-V2-T2V-14B", 3), ("SkyReels-V2-I2V-14B", 16)],
)
def test_ulysses_degree_must_divide_the_attention_heads(name, ulysses_degree):
    with pytest.raises(ValueError, match="must divide"):
        _model(name, ulysses_degree=ulysses_degree)


@pytest.mark.parametrize(
    "name, ulysses_degree",
    [("SkyReels-V2-I2V-1.3B", 6), ("SkyReels-V2-T2V-14B", 8), ("SkyReels-V2-I2V-14B", 4)],
)
def test_ulysses_degree_dividing_the_heads_is_accepted(name, ulysses_degree):
    _model(name, ulysses_degree=ulysses_degree)


def _args(**overrides):
    args = {
        "prompt": "a cat",
        "dataset_path": None,
        "height": 544,
        "width": 960,
        "input_images": [],
    }
    args.update(overrides)
    return args


def test_text_to_video_refuses_an_input_image():
    with pytest.raises(ValueError, match="text-to-video"):
        _model("SkyReels-V2-T2V-14B")._validate_args(_args(input_images=["image.png"]))


def test_image_to_video_needs_exactly_one_image():
    model = _model("SkyReels-V2-I2V-1.3B")
    for images in ([], ["a.png", "b.png"]):
        with pytest.raises(ValueError, match="exactly one input image"):
            model._validate_args(_args(input_images=images))


def test_image_to_video_sizes_the_video_from_the_input_image(tmp_path):
    path = tmp_path / "portrait.png"
    Image.new("RGB", (544, 960)).save(path)
    model = _model("SkyReels-V2-I2V-1.3B")

    args = model.preprocess_args({"prompt": "a cat", "dataset_path": None, "input_images": [str(path)]})

    # A portrait image keeps its aspect ratio within the default 544x960 area.
    assert (args["height"], args["width"]) == (960, 544)
    assert args["image"].size == (544, 960)


@pytest.mark.parametrize("name", ["SkyReels-V2-T2V-14B", "SkyReels-V2-I2V-1.3B"])
def test_flow_shift_reaches_the_scheduler(monkeypatch, name):
    """The checkpoints ship flow_shift=1.0, so the requested shift has to replace it."""
    monkeypatch.setattr(xFuserModel, "_post_load_and_state_initialization", lambda self, input_args: None)
    model = _model(name)
    model.pipe = SimpleNamespace(scheduler=UniPCMultistepScheduler(use_flow_sigmas=True, flow_shift=1.0))

    model._post_load_and_state_initialization({"flow_shift": 7.0})

    assert model.pipe.scheduler.config.flow_shift == 7.0
    assert model.pipe.scheduler.config.use_flow_sigmas
