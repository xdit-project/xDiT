import copy
from types import SimpleNamespace

import pytest

from xfuser.model_executor.models.runner_models.z_image import xFuserZImageModel


def _customize(*, pp, split=None, ulysses=1, ring=1):
    runner = object.__new__(xFuserZImageModel)
    runner.settings = copy.deepcopy(xFuserZImageModel.settings)
    config = SimpleNamespace(
        pipefusion_parallel_degree=pp,
        attn_layer_num_for_pp=split,
        ulysses_degree=ulysses,
        ring_degree=ring,
        use_int8_gemms=False,
    )
    runner._customize_settings(config)
    return config


@pytest.mark.parametrize(
    ("pp", "expected"),
    [
        (2, [14, 16]),
        (3, [9, 11, 10]),
        (4, [6, 8, 8, 8]),
        (8, [2, 4, 4, 4, 4, 4, 4, 4]),
    ],
)
def test_z_image_pipefusion_balances_first_stage_refiner_work(pp, expected):
    assert _customize(pp=pp).attn_layer_num_for_pp == expected


def test_z_image_pipefusion_accepts_user_stage_split():
    split = [13, 17]
    assert _customize(pp=2, split=split).attn_layer_num_for_pp == split


def test_z_image_pipefusion_non_output_rank_returns_empty_images():
    runner = object.__new__(xFuserZImageModel)
    runner.pipe = lambda **_kwargs: None
    runner._make_generator = lambda _seed: None
    args = {
        "height": 256,
        "width": 256,
        "prompt": "test",
        "num_inference_steps": 2,
        "guidance_scale": 4.0,
        "seed": 42,
    }

    assert runner._run_pipe(args).images == []


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"ulysses": 2}, "does not yet compose with sequence parallelism"),
        ({"ring": 2}, "does not yet compose with sequence parallelism"),
        ({"split": [15, 14]}, "assign all 30"),
        ({"split": [30, 0]}, "positive layer count"),
        ({"split": [10, 10, 10]}, "positive layer count"),
    ],
)
def test_z_image_pipefusion_rejects_invalid_topologies(kwargs, message):
    with pytest.raises(ValueError, match=message):
        _customize(pp=2, **kwargs)
