"""Runner contract for Prism: which layouts and requests it accepts and refuses.

Ulysses splits the video tower's 40 heads, so a degree that does not divide them must
be refused at config time, before 130 GB of weights are read. Ring attention would need
the key trimming per ring step, and the pipeline has no offload or batching, so those
are refused up front too.
"""

import pytest

pytest.importorskip("diffusers")

from xfuser import xFuserArgs  # noqa: E402
from xfuser.model_executor.models.runner_models.prism import (  # noqa: E402
    snap_num_frames,
    xFuserPrismModel,
)


@pytest.fixture(autouse=True)
def _single_process(monkeypatch):
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")


def _model(**config):
    return xFuserPrismModel(xFuserArgs(model="Prism", **config))


@pytest.mark.parametrize("ulysses_degree", [1, 2, 4, 5, 8])
def test_ulysses_degrees_that_divide_the_video_heads_are_accepted(ulysses_degree):
    _model(ulysses_degree=ulysses_degree)


@pytest.mark.parametrize("ulysses_degree", [3, 6, 16])
def test_ulysses_degree_must_divide_the_40_video_heads(ulysses_degree):
    with pytest.raises(ValueError, match="40 attention heads.*--ulysses_degree must divide 40"):
        _model(ulysses_degree=ulysses_degree)


@pytest.mark.parametrize(
    "config",
    [
        {"ring_degree": 2},
        {"use_cfg_parallel": True},
        {"data_parallel_degree": 2},
        {"pipefusion_parallel_degree": 2},
    ],
)
def test_unsupported_parallelism_is_refused(config):
    with pytest.raises(ValueError, match="does not support"):
        _model(**config)


@pytest.mark.parametrize(
    "config",
    [
        {"enable_model_cpu_offload": True},
        {"enable_sequential_cpu_offload": True},
        {"batch_size": 2},
    ],
)
def test_offload_and_batching_are_refused(config):
    with pytest.raises(ValueError, match="Prism does not support CPU offloading|one clip per request"):
        _model(**config)


@pytest.mark.parametrize("input_images", [[], ["first.png", "second.png"]])
def test_exactly_one_reference_image_is_required(input_images):
    model = _model()
    with pytest.raises(ValueError, match="exactly one --input_images"):
        model._validate_args(
            {"prompt": "a", "dataset_path": None, "height": 480, "width": 848, "input_images": input_images}
        )


@pytest.mark.parametrize(
    ("requested", "snapped"),
    [(205, 205), (206, 205), (208, 205), (209, 209), (2, 5)],
)
def test_frame_counts_snap_down_to_what_the_video_vae_accepts(requested, snapped):
    assert snap_num_frames(requested) == snapped
