import pytest

from xfuser import xFuserArgs
from xfuser.model_executor.models.runner_models.stable_diffusion import (
    xFuserStableDiffusionModel,
)


def test_sd35_rejects_replicated_meta_load_with_pipefusion():
    model = object.__new__(xFuserStableDiffusionModel)
    config = xFuserArgs(
        model="stabilityai/stable-diffusion-3.5-large",
        pipefusion_parallel_degree=2,
        memory_efficient_replicated_load=True,
    )

    with pytest.raises(ValueError, match="cannot construct a stage-local transformer"):
        model._validate_config(config)
