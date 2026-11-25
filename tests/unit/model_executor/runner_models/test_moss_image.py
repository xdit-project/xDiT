"""Runner contract for MOSS-Image: recipe defaults and Ulysses head divisibility."""

import pytest

pytest.importorskip("diffusers")

from xfuser import xFuserArgs  # noqa: E402
from xfuser.model_executor.models.runner_models.base_model import MODEL_REGISTRY  # noqa: E402
from xfuser.model_executor.models.runner_models.moss_image import xFuserMOSSImageModel  # noqa: E402


@pytest.fixture(autouse=True)
def _single_process(monkeypatch):
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")


def _model(**config):
    return xFuserMOSSImageModel(xFuserArgs(model="MOSS-Image", prompt="a cat", **config))


def test_registry_names_the_base_checkpoint():
    assert MODEL_REGISTRY["OpenMOSS-Team/MOSS-Image-Base"] is xFuserMOSSImageModel
    assert MODEL_REGISTRY["MOSS-Image"] is xFuserMOSSImageModel
    assert MODEL_REGISTRY["MOSS-Image-Base"] is xFuserMOSSImageModel


@pytest.mark.parametrize("ulysses_degree", [1, 2, 3, 4, 6, 8, 12, 24])
def test_ulysses_degrees_that_divide_the_24_heads_are_accepted(ulysses_degree):
    _model(ulysses_degree=ulysses_degree)


@pytest.mark.parametrize("ulysses_degree", [5, 16])
def test_ulysses_degree_must_divide_the_24_heads(ulysses_degree):
    with pytest.raises(ValueError, match="24 attention heads"):
        _model(ulysses_degree=ulysses_degree)


def test_pipefusion_and_tensor_parallel_are_refused():
    with pytest.raises(ValueError, match="does not support pipefusion_parallel_degree"):
        _model(pipefusion_parallel_degree=2)
    with pytest.raises(ValueError, match="does not support tensor_parallel_degree"):
        _model(tensor_parallel_degree=2)


def test_omitted_sampling_args_use_the_model_defaults():
    args = _model().preprocess_args(
        {
            "prompt": "a cat",
            "num_inference_steps": None,
            "guidance_scale": None,
            "max_sequence_length": None,
            "flow_shift": None,
            "height": None,
            "width": None,
            "negative_prompt": "",
        }
    )
    assert args["num_inference_steps"] == 50
    assert args["guidance_scale"] == 3.0
    assert args["max_sequence_length"] == 4096
    assert args["flow_shift"] == 5.0
    assert args["height"] == 1024
    assert args["width"] == 1024


def test_explicit_cli_defaults_are_kept():
    args = _model().preprocess_args(
        {
            "prompt": "a cat",
            "num_inference_steps": 20,
            "guidance_scale": 3.5,
            "max_sequence_length": 256,
            "flow_shift": None,
            "height": None,
            "width": None,
            "negative_prompt": "",
        }
    )
    assert args["num_inference_steps"] == 20
    assert args["guidance_scale"] == 3.5
    assert args["max_sequence_length"] == 256
    assert args["flow_shift"] == 5.0


def test_explicit_sampling_choices_are_kept():
    args = _model().preprocess_args(
        {
            "prompt": "a cat",
            "num_inference_steps": 8,
            "guidance_scale": 1.0,
            "max_sequence_length": 128,
            "flow_shift": 2.0,
            "height": 512,
            "width": 768,
            "negative_prompt": "blur",
        }
    )
    assert args["num_inference_steps"] == 8
    assert args["guidance_scale"] == 1.0
    assert args["max_sequence_length"] == 128
    assert args["flow_shift"] == 2.0


def test_resolution_must_be_divisible_by_16():
    with pytest.raises(ValueError, match="divisible by 16"):
        _model().preprocess_args({"prompt": "a cat", "height": 24, "width": 32})


def test_compile_mode_records_cuda_graphs_except_on_rdna4(monkeypatch):
    from xfuser.model_executor.models.runner_models import moss_image

    model = _model()
    monkeypatch.setattr(moss_image.PACKAGES_CHECKER, "_on_rdna4", lambda: False)
    assert model._get_compile_mode() == "reduce-overhead"
    assert model._get_compiled_pipe_components() == ["transformer", "text_encoder"]
    monkeypatch.setattr(moss_image.PACKAGES_CHECKER, "_on_rdna4", lambda: True)
    assert model._get_compile_mode() == "default"


@pytest.mark.parametrize("cfg_parallel, fused", [(False, ("q", "k", "v")), (True, ("q", "k"))])
def test_projection_fusion_follows_the_cfg_layout(monkeypatch, cfg_parallel, fused):
    import torch

    from xfuser.model_executor.pipelines import pipeline_moss_image

    seen = {}
    monkeypatch.setattr(
        pipeline_moss_image.xFuserMOSSImagePipeline, "from_pretrained", lambda **kwargs: seen.update(kwargs)
    )
    model = _model()
    monkeypatch.setattr(model.config, "use_cfg_parallel", cfg_parallel)
    model._load_model()
    assert seen["fuse_projections"] == fused
    assert seen["vae_dtype"] is torch.bfloat16
