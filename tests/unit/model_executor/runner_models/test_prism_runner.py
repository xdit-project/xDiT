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


def test_block_sparse_attention_runs_beside_a_dense_cross_attention_backend():
    _model(attention_backend="TRITON_BSA", cross_attention_backend="AITER", ulysses_degree=8)


@pytest.mark.parametrize("config", [{"bsa_sparsity": 1.0}, {"bsa_sparsity": -0.1}, {"bsa_cdf_threshold": 1.5}])
def test_block_sparse_settings_must_lie_in_the_unit_interval(config):
    with pytest.raises(ValueError, match=r"must lie in \[0, 1\)"):
        _model(**config)


@pytest.mark.parametrize("input_images", [[], ["first.png", "second.png"]])
def test_exactly_one_reference_image_is_required(input_images):
    model = _model()
    with pytest.raises(ValueError, match="exactly one --input_images"):
        model._validate_args(
            {"prompt": "a", "dataset_path": None, "height": 480, "width": 848, "input_images": input_images}
        )


@pytest.mark.parametrize("flow_shift", [7.0, 9.0, 13.0, 17.0])
def test_compile_warmup_reaches_the_low_noise_expert(flow_shift):
    """Each video expert is its own compiled graph; a warmup that never switches
    experts leaves the second graph to compile inside the first timed run."""
    from types import SimpleNamespace

    from xfuser.model_executor.models.customized.prism.scheduler import FlowMatchPairScheduler

    model = _model()
    # MOVA-360p's scheduler config.
    scheduler = FlowMatchPairScheduler(shift=5, sigma_min=0.0, extra_one_step=True)
    model.pipe = SimpleNamespace(transformer=SimpleNamespace(boundary_ratio=0.9), scheduler=scheduler)

    steps = model._get_compile_warmup_steps({"num_inference_steps": 50, "flow_shift": flow_shift})

    video_timesteps = scheduler.set_pair_timesteps(steps, flow_shift, 7.0)[:, 0]
    assert steps < 50
    assert (video_timesteps < 0.9 * 1000).any()


@pytest.mark.parametrize(
    ("requested", "snapped"),
    [(205, 205), (206, 205), (208, 205), (209, 209), (2, 5)],
)
def test_frame_counts_snap_down_to_what_the_video_vae_accepts(requested, snapped):
    assert snap_num_frames(requested) == snapped
