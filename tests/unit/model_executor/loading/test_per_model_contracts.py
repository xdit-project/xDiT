"""Behavioral load contracts for models with specialized loading paths."""

import inspect
import json
import types

import pytest


def test_tokenizer_reload_reads_the_tokenizer_directory_not_the_repo_root(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    if not transformers.__version__.startswith("5"):
        pytest.skip("the reload only runs on transformers v5")

    from xfuser.core.utils import runner_utils

    repo = tmp_path / "HunyuanVideo"
    component = repo / "tokenizer"
    component.mkdir(parents=True)
    (repo / "config.json").write_text('{\n  "Name": [\n    "HunyuanVideo"\n  ],\n}')
    tokenizers = pytest.importorskip("tokenizers")
    backend = tokenizers.Tokenizer(tokenizers.models.WordLevel({"<unk>": 0, "hello": 1}, unk_token="<unk>"))
    backend.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    backend.save(str(component / "tokenizer.json"))
    (component / "tokenizer_config.json").write_text(json.dumps({"tokenizer_class": "LlamaTokenizerFast"}))

    class FakeLlamaTokenizerFast:
        pass

    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    pipeline = types.SimpleNamespace(components={"tokenizer": FakeLlamaTokenizerFast()}, tokenizer=None)

    assert runner_utils._tokenizer_directory(str(repo), "tokenizer", {}) == str(component)
    runner_utils.fix_llama_tokenizer_pretokenizer(pipeline, str(repo))
    assert pipeline.tokenizer.tokenize("hello") == ["hello"]

    (component / "tokenizer.json").unlink()
    assert runner_utils._tokenizer_directory(str(repo), "tokenizer", {}) is None


def test_krea2_text_encoder_shard_path_exists_on_the_encoder_it_loads():
    pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    accelerate = pytest.importorskip("accelerate")
    from xfuser.core.distributed.sharding import rgetattr
    from xfuser.model_executor.models.runner_models.krea2 import (
        xFuserKrea2RawModel,
        xFuserKrea2TurboModel,
    )

    config = transformers.Qwen3VLConfig()
    config.text_config.num_hidden_layers = 2
    config.vision_config.depth = 2
    with accelerate.init_empty_weights():
        encoder = transformers.Qwen3VLModel(config)

    for model in (xFuserKrea2RawModel, xFuserKrea2TurboModel):
        for attr in model.settings.fsdp_strategy["text_encoder"]["wrap_attrs"]:
            layers = rgetattr(encoder, attr)
            assert len(layers) == 2, f"{attr} does not reach the decoder layers"


def test_undeclared_fsdp_strategy_text_encoders_are_not_auto_enrolled():
    from xfuser.model_executor.models.runner_models.loading.contracts import (
        LoadDeclaration,
        LoadSupport,
        STANDARD_LOAD_ROUTES,
    )

    capability = LoadDeclaration.for_runner(
        type(
            "Capabilities",
            (),
            {
                "use_fp8_gemms": False,
                "use_fp4_gemms": False,
                "use_int8_gemms": False,
                "fully_shard_degree": True,
            },
        )(),
        load_support=LoadSupport(
            meta_transformers=("transformer",),
            meta_text_encoders=(),
            replicated_meta=True,
            routes=STANDARD_LOAD_ROUTES,
        ),
        fsdp_strategy={
            "transformer": {"wrap_attrs": ["blocks"]},
            "text_encoder": {"wrap_attrs": ["layers"]},
        },
    )

    assert capability.meta_text_encoders == ()


def test_hunyuan_wrapper_keeps_the_parent_config_signature():
    pytest.importorskip("diffusers")
    wrapper_module = pytest.importorskip("xfuser.model_executor.models.transformers.transformer_hunyuan_video")
    from diffusers.models.transformers.transformer_hunyuan_video import (
        HunyuanVideoTransformer3DModel,
    )

    assert inspect.signature(wrapper_module.xFuserHunyuanVideoTransformer3DWrapper.__init__) == inspect.signature(
        HunyuanVideoTransformer3DModel.__init__
    )
    assert "from_config" in wrapper_module.xFuserHunyuanVideoTransformer3DWrapper.__dict__


def test_ltx_wrapper_keeps_diffusers_config_api():
    pytest.importorskip("diffusers")
    wrapper_module = pytest.importorskip("xfuser.model_executor.models.transformers.transformer_ltx2")
    wrapper = wrapper_module.xFuserLTX2VideoTransformer3DWrapper

    assert callable(wrapper.load_config)
    assert callable(wrapper.from_config)
