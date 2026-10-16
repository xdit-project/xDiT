import pytest

from xfuser import xFuserArgs
from xfuser.model_executor.models.runner_models.causal_wan import xFuserCausalWanModel


@pytest.fixture(autouse=True)
def _single_process(monkeypatch):
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")


def _preprocess(num_frames):
    model = xFuserCausalWanModel(xFuserArgs(model="CausalWan", task="t2v"))
    args = {"prompt": "a fox", "dataset_path": None, "input_images": [], "seed": 0, "num_frames": num_frames}
    model.preprocess_args(args)


@pytest.mark.parametrize("num_frames", [81, 83])  # 83 is rounded down to 81 by the pipeline
def test_num_frames_that_fit_the_kv_cache_are_accepted(num_frames):
    _preprocess(num_frames)


@pytest.mark.parametrize("num_frames", [84, 93])  # 84 is rounded up to 85 = 22 latent frames
def test_num_frames_beyond_the_kv_cache_are_rejected_with_the_largest_value(num_frames):
    with pytest.raises(ValueError, match=r"--num_frames 81 or fewer"):
        _preprocess(num_frames)
