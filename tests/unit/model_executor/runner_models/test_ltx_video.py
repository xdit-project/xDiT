"""Runner contract for LTX-Video 0.9.7: what it accepts, refuses, and hands the pipeline.

CPU only; the pipeline and transformer loads are replaced, nothing is downloaded.
"""

from types import SimpleNamespace
from unittest import mock

import pytest

from xfuser.config import xFuserArgs
from xfuser.model_executor.models.runner_models.base_model import MODEL_REGISTRY
from xfuser.model_executor.models.runner_models.ltx_video import (
    ltx_video_token_count,
    xFuserLTXVideoModel,
)


@pytest.fixture(autouse=True)
def _single_process(monkeypatch):
    """The runner's logging reads the launcher's rank variables, as xfuser/runner.py sets them."""
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")


def _model(**config):
    config.setdefault("task", "t2v")
    return xFuserLTXVideoModel(xFuserArgs(model="LTX-Video-0.9.7-dev", **config))


def _args(model, **overrides):
    args = {"prompt": "a red fox in the snow", "dataset_path": None, "input_images": [], "seed": 0}
    args.update(overrides)
    return model.preprocess_args(args)


@pytest.mark.parametrize("name", ["LTX-Video-0.9.7-dev", "Lightricks/LTX-Video-0.9.7-dev"])
def test_both_names_select_the_runner(name):
    assert MODEL_REGISTRY[name] is xFuserLTXVideoModel


@pytest.mark.parametrize(
    "parallel",
    [
        {"ulysses_degree": 8},
        {"ring_degree": 8},
        {"ulysses_degree": 4, "ring_degree": 2},
        {"use_cfg_parallel": True, "ulysses_degree": 4},
    ],
)
def test_defaults_are_a_valid_request_for_every_supported_layout(parallel):
    model = _model(**parallel)

    args = _args(model)

    assert args["negative_prompt"]
    assert args["guidance_scale"] > 1
    assert ltx_video_token_count(args["height"], args["width"], args["num_frames"]) % 8 == 0


@pytest.mark.parametrize("ulysses_degree", [3, 64])
def test_ulysses_degree_must_divide_the_attention_heads(ulysses_degree):
    with pytest.raises(ValueError, match="attention heads"):
        _model(ulysses_degree=ulysses_degree)


def test_a_task_is_required():
    with pytest.raises(ValueError, match="requires a task"):
        xFuserLTXVideoModel(xFuserArgs(model="LTX-Video-0.9.7-dev"))


def test_num_frames_must_be_eight_k_plus_one():
    with pytest.raises(ValueError, match="8k\\+1"):
        _args(_model(), num_frames=120)


def test_ring_needs_a_divisible_token_count_but_ulysses_pads():
    # 480x704x41 -> 6 x 15 x 22 = 1980 tokens, not divisible by 8.
    shape = {"height": 480, "width": 704, "num_frames": 41}
    assert ltx_video_token_count(**shape) % 8 != 0

    _args(_model(ulysses_degree=8), **shape)
    with pytest.raises(ValueError, match="ring_degree"):
        _args(_model(ulysses_degree=4, ring_degree=2), **shape)


def test_cfg_parallel_needs_guidance():
    with pytest.raises(ValueError, match="guidance_scale > 1"):
        _args(_model(use_cfg_parallel=True), guidance_scale=1.0)


def test_i2v_needs_exactly_one_image_and_t2v_none():
    with pytest.raises(ValueError, match="exactly one input image"):
        _args(_model(task="i2v"))
    with pytest.raises(ValueError, match="does not take input images"):
        _model(task="t2v")._validate_args({**_args(_model()), "input_images": ["image.png"]})


@pytest.mark.parametrize("task, pipeline_name", [("t2v", "LTXPipeline"), ("i2v", "LTXImageToVideoPipeline")])
def test_task_selects_the_pipeline_and_its_inputs(task, pipeline_name, tmp_path):
    import diffusers
    from PIL import Image

    model = _model(task=task)
    pipeline = mock.MagicMock()
    pipeline.return_value = SimpleNamespace(frames=["video"])
    pipeline._execution_device = "cpu"
    model.loader = mock.MagicMock()

    with mock.patch.object(
        getattr(diffusers, pipeline_name), "from_pretrained", return_value=pipeline
    ) as from_pretrained:
        model.pipe = model._load_model()

    from_pretrained.assert_called_once()
    assert from_pretrained.call_args.kwargs["transformer"] is model.loader.load_transformer.return_value

    images = []
    if task == "i2v":
        images = [str(tmp_path / "first_frame.png")]
        Image.new("RGB", (64, 32)).save(images[0])
    args = _args(model, input_images=images)
    model._run_pipe(args)

    kwargs = pipeline.call_args.kwargs
    assert kwargs["frame_rate"] == model.settings.fps
    assert kwargs["output_type"] == "np"
    assert ("image" in kwargs) == (task == "i2v")
    if task == "i2v":
        assert kwargs["image"].size == (64, 32)
