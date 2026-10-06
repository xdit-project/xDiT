"""--resize_input_images must resize to the requested height and width, not their transpose.

Each runner is built without __init__ (which wants a fully parsed CLI config and would
set up loading), so only its own settings and the task it reads from the config are in
place. Nothing here loads weights or touches the network.
"""

import copy
from types import SimpleNamespace

import pytest
from PIL import Image

from xfuser.model_executor.models.runner_models.cosmos3 import (
    xFuserCosmos3NanoModel,
    xFuserCosmos3SuperModel,
)
from xfuser.model_executor.models.runner_models.flux import (
    xFuserFlux2Klein4BModel,
    xFuserFlux2Klein9BModel,
    xFuserFlux2Model,
    xFuserFluxKontextModel,
)
from xfuser.model_executor.models.runner_models.hunyuan import (
    xFuserHunyuanvideo15Model,
)
from xfuser.model_executor.models.runner_models.wan import (
    xFuserWan21I2VModel,
    xFuserWan22I2VModel,
    xFuserWan22TI2VModel,
)

# Landscape target, divisible by every model's alignment (16 or 32).
HEIGHT, WIDTH = 64, 128


def _runner(cls, task):
    runner = object.__new__(cls)
    runner.settings = copy.deepcopy(cls.settings)
    runner.config = SimpleNamespace(task=task)
    return runner


def _conditioning_image(input_args):
    if "images" in input_args:
        (image,) = input_args["images"]
        return image
    return input_args["image"]


@pytest.mark.parametrize(
    ("cls", "task"),
    [
        (xFuserWan21I2VModel, None),
        (xFuserWan22I2VModel, None),
        (xFuserWan22TI2VModel, "i2v"),
        (xFuserFluxKontextModel, None),
        (xFuserFlux2Model, None),
        (xFuserFlux2Klein9BModel, None),
        (xFuserFlux2Klein4BModel, None),
        (xFuserHunyuanvideo15Model, "i2v"),
        (xFuserCosmos3SuperModel, None),
        (xFuserCosmos3NanoModel, None),
    ],
    ids=lambda value: getattr(value, "__name__", str(value)),
)
def test_resize_input_images_keeps_requested_orientation(monkeypatch, cls, task):
    # The resize helpers log from the last rank, which they read from the launcher env.
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    # Portrait source, so a transposed target cannot pass by matching the input.
    source = Image.new("RGB", (40, 90), color=(10, 20, 30))
    input_args = {
        "prompt": "a prompt",
        "dataset_path": None,
        "input_images": [source],
        "resize_input_images": True,
        "height": HEIGHT,
        "width": WIDTH,
    }

    input_args = _runner(cls, task)._preprocess_args_images(input_args)

    assert _conditioning_image(input_args).size == (WIDTH, HEIGHT)
    assert (input_args["height"], input_args["width"]) == (HEIGHT, WIDTH)
