import pytest

from xfuser import xFuserArgs
from xfuser.model_executor.models.runner_models.z_image import xFuserZImageModel


@pytest.fixture(autouse=True)
def _single_process(monkeypatch):
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")


def _args(model, guidance_scale):
    args = {"prompt": "a fox", "dataset_path": None, "input_images": [], "seed": 0, "guidance_scale": guidance_scale}
    return model.preprocess_args(args)


def test_cfg_parallel_needs_guidance():
    model = xFuserZImageModel(xFuserArgs(model="Z-Image", use_cfg_parallel=True))

    _args(model, 4.0)
    with pytest.raises(ValueError, match="--guidance_scale > 0 with --use_cfg_parallel"):
        _args(model, 0.0)
