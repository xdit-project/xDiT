import copy
from types import SimpleNamespace

import pytest

from xfuser.model_executor.models.runner_models import wan
from xfuser.model_executor.models.runner_models.wan import xFuserWan22TI2VModel


def _customize(*, pp, task="i2v", split=None, ulysses=1, ring=1):
    runner = object.__new__(xFuserWan22TI2VModel)
    runner.settings = copy.deepcopy(xFuserWan22TI2VModel.settings)
    config = SimpleNamespace(
        pipefusion_parallel_degree=pp,
        task=task,
        attn_layer_num_for_pp=split,
        ulysses_degree=ulysses,
        ring_degree=ring,
    )
    runner._customize_settings(config)
    return config


@pytest.mark.parametrize("pp", range(2, 31))
def test_wan_ti2v_supports_balanced_pipefusion_topologies(pp):
    split = _customize(pp=pp).attn_layer_num_for_pp

    assert len(split) == pp
    assert sum(split) == 30
    assert min(split) > 0
    assert max(split) - min(split) <= 1


def test_wan_ti2v_accepts_user_tuned_pipefusion_split():
    split = [7, 8, 7, 8]
    assert _customize(pp=4, split=split).attn_layer_num_for_pp == split


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"task": "t2v"}, "only the i2v task"),
        ({"ulysses": 2}, "does not yet compose with sequence parallelism"),
        ({"ring": 2}, "does not yet compose with sequence parallelism"),
        ({"split": [8, 8, 7, 6]}, "must assign all 30"),
        ({"split": [15, 15, 0, 0]}, "positive block count"),
        ({"split": [10, 10, 10]}, "positive block count"),
    ],
)
def test_wan_ti2v_rejects_unsupported_pipefusion_configurations(kwargs, message):
    with pytest.raises(ValueError, match=message):
        _customize(pp=4, **kwargs)


def test_wan_ti2v_rejects_more_stages_than_blocks():
    with pytest.raises(ValueError, match="cannot split 30 blocks"):
        _customize(pp=31)


def test_wan_ti2v_runner_accepts_no_output_on_non_owner_rank():
    runner = object.__new__(xFuserWan22TI2VModel)
    runner.config = SimpleNamespace(task="i2v")
    runner.pipe = lambda **_kwargs: None
    runner._make_generator = lambda _seed: None
    output = runner._run_pipe(
        {
            "height": 128,
            "width": 128,
            "prompt": ["cat"],
            "negative_prompt": "bad",
            "num_inference_steps": 2,
            "num_frames": 5,
            "guidance_scale": 5.0,
            "guidance_scale_2": None,
            "seed": 0,
            "image": object(),
        }
    )

    assert output.videos == []


@pytest.mark.parametrize(
    ("attention_schedule", "gemm_schedule", "expected"),
    [(False, False, 2), (True, False, None), (False, True, None)],
)
def test_wan_ti2v_pipefusion_compile_capture_respects_backend_schedules(
    monkeypatch,
    attention_schedule,
    gemm_schedule,
    expected,
):
    runner = object.__new__(xFuserWan22TI2VModel)
    runner.config = SimpleNamespace(pipefusion_parallel_degree=4)
    state = SimpleNamespace(
        has_attention_schedule=lambda: attention_schedule,
        has_gemm_schedule=lambda: gemm_schedule,
    )
    monkeypatch.setattr(wan, "get_runtime_state", lambda: state)

    assert runner._get_compile_warmup_steps({}) == expected
