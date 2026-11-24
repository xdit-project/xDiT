"""LTX-2.5 loading boundaries; all checkpoint and pipeline work stays offline."""

import sys
from types import SimpleNamespace
from unittest import mock

import pytest

from xfuser.config import xFuserArgs
from xfuser.model_executor.models.runner_models.loading.contracts import (
    MaterializationMode,
    UnsupportedLoadContract,
)
from xfuser.model_executor.models.runner_models.ltx import (
    xFuserLTX25DistilledVideoModel,
    xFuserLTX25FullVideoModel,
)


@pytest.fixture(autouse=True)
def _single_process(monkeypatch):
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")


def _model(tmp_path, cls=xFuserLTX25DistilledVideoModel, **config):
    config.setdefault("task", "t2v")
    return cls(xFuserArgs(model="LTX-2.5", output_directory=str(tmp_path), **config))


@pytest.mark.parametrize(
    "requested, world_size, gemm_quantization",
    [
        (False, 1, "none"),
        (True, 1, "none"),
        (True, 8, "none"),
        (False, 1, "fp8"),
        (True, 1, "fp8"),
        (False, 1, "fp4"),
        (True, 1, "fp4"),
    ],
)
@pytest.mark.parametrize("task", ["t2v", "i2v"])
def test_only_transformer_uses_replicated_loader(monkeypatch, tmp_path, requested, world_size, gemm_quantization, task):
    import diffusers

    from xfuser.model_executor.models.runner_models.loading import meta_load

    model = _model(tmp_path, task=task, memory_efficient_replicated_load=requested, gemm_quantization=gemm_quantization)
    monkeypatch.setattr(meta_load, "get_world_group", lambda: SimpleNamespace(world_size=world_size))
    transformer_cls = mock.Mock()
    decoder_cls = mock.Mock()
    upsampler_cls = mock.Mock()
    pipeline_cls = mock.Mock()
    image_pipeline_cls = mock.Mock()
    upsample_pipeline_cls = mock.Mock()
    decode_pipeline_cls = mock.Mock()
    for name, value in (
        ("LTX2Pipeline", pipeline_cls),
        ("LTX2ImageToVideoPipeline", image_pipeline_cls),
        ("LTX2LatentUpsamplePipeline", upsample_pipeline_cls),
    ):
        monkeypatch.setattr(diffusers, name, value, raising=False)
    for name, exports in (
        (
            "xfuser.model_executor.models.transformers.transformer_ltx2",
            {"xFuserLTX2VideoTransformer3DWrapper": transformer_cls},
        ),
        (
            "xfuser.model_executor.layers.ltx2.diffusion_decoder",
            {"xFuserLTX2VideoDiffusionDecoderWrapper": decoder_cls},
        ),
        (
            "diffusers.pipelines.ltx2.latent_upsampler",
            {"LTX2LatentUpsamplerModel": upsampler_cls},
        ),
        (
            "diffusers.pipelines.ltx2.pipeline_ltx2_diffusion_decode",
            {"LTX2VideoDiffusionDecodePipeline": decode_pipeline_cls},
        ),
        (
            "diffusers.models.autoencoders.ltx2_diffusion_decoder",
            {"LTX2VideoVaeNeighborhoodNattenProcessor": mock.Mock()},
        ),
    ):
        monkeypatch.setitem(sys.modules, name, SimpleNamespace(**exports))
    loader = model.loader
    with mock.patch.object(loader, "load_transformer") as load_transformer:
        pipe = model._load_model()

    selected = image_pipeline_cls if task == "i2v" else pipeline_cls
    other = pipeline_cls if task == "i2v" else image_pipeline_cls
    assert pipe is selected.from_pretrained.return_value
    other.from_pretrained.assert_not_called()
    kwargs = selected.from_pretrained.call_args.kwargs
    if requested and world_size > 1:
        load_transformer.assert_called_once_with(transformer_cls, subfolder="transformer")
        transformer_cls.from_pretrained.assert_not_called()
        assert kwargs["transformer"] is load_transformer.return_value
    else:
        load_transformer.assert_not_called()
        transformer_cls.from_pretrained.assert_called_once()
        assert transformer_cls.from_pretrained.call_args.kwargs["subfolder"] == "transformer"
        assert kwargs["transformer"] is transformer_cls.from_pretrained.return_value
    # No text-encoder override: pipeline owns its eager load in either branch.
    assert "text_encoder" not in kwargs
    upsampler_cls.from_pretrained.assert_called_once()
    decoder_cls.from_pretrained.assert_called_once()
    assert upsampler_cls.from_pretrained.call_args.kwargs["subfolder"] == "latent_upsampler"
    assert decoder_cls.from_pretrained.call_args.kwargs["subfolder"] == "diffusion_decoder"
    assert model.upsample_pipe is upsample_pipeline_cls.return_value
    assert model.decode_pipe is decode_pipeline_cls.return_value


@pytest.mark.parametrize("world_size, mode", [(1, MaterializationMode.EAGER), (8, MaterializationMode.REPLICATED_META)])
def test_distilled_preflight_accepts_unquantized_replicated_load(tmp_path, world_size, mode):
    model = _model(tmp_path, memory_efficient_replicated_load=True)
    model.loader.preflight(world_size=world_size)
    assert model.loader.load_contract.materialization_mode is mode


def test_full_preflight_refuses_replicated_load(tmp_path):
    model = _model(tmp_path, cls=xFuserLTX25FullVideoModel, memory_efficient_replicated_load=True)
    with pytest.raises(UnsupportedLoadContract, match="does not support replicated_meta"):
        model.loader.preflight(world_size=8)


@pytest.mark.parametrize("gemm_quantization", ["fp8", "fp4", "low=fp4,high=fp8"])
def test_replicated_load_refuses_quantized_gemms_before_loading(tmp_path, monkeypatch, gemm_quantization):
    from xfuser.model_executor.models.runner_models.loading import meta_load

    model = _model(tmp_path, memory_efficient_replicated_load=True, gemm_quantization=gemm_quantization)
    monkeypatch.setattr(meta_load, "get_world_group", lambda: SimpleNamespace(world_size=8))
    with pytest.raises(ValueError, match="requires unquantized GEMMs"):
        model._load_model()
