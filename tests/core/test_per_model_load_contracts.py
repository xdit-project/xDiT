"""Load contracts for the individual models that needed one of their own.

Each test here is about a named model whose checkpoint, encoder or wrapper made it
the exception: HunyuanVideo's pinned revision and Llama encoder, LTX's withheld
declaration, Krea-2's excluded text encoder, the runners on dedicated adapters.

The AST checks stay dependency-light. Model API checks are guarded so a core
test environment without Diffusers can still enforce the declarations.
"""

import ast
import importlib.util
import inspect
import json
from pathlib import Path
import types

import pytest

ROOT = Path(__file__).resolve().parents[2]
RUNNERS = ROOT / "xfuser/model_executor/models/runner_models"
CONTRACTS = RUNNERS / "loading/contracts.py"


def _classes(filename):
    return {
        node.name: node
        for node in ast.parse((RUNNERS / filename).read_text()).body
        if isinstance(node, ast.ClassDef)
    }


def _load_declaration(filename, class_name):
    return next(
        decorator
        for decorator in _classes(filename)[class_name].decorator_list
        if isinstance(decorator, ast.Call)
        and isinstance(decorator.func, ast.Attribute)
        and decorator.func.attr == "declare"
    )


def _literal(node):
    """A literal-ish view of an AST node.

    A declaration is written to be read, so a test that asks what it says should not also
    depend on how it is spaced or which line it wrapped on. Containers recurse so that one
    value ast cannot evaluate -- torch.bfloat16, a named constant, an enum member -- costs
    only itself and not the structure around it; those come back as their source text,
    which is the thing worth asserting on anyway.
    """
    if isinstance(node, ast.Dict):
        return {_literal(k): _literal(v) for k, v in zip(node.keys, node.values)}
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        return [_literal(element) for element in node.elts]
    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError, SyntaxError):
        return ast.unparse(node)


def _keyword(call, name, default=None):
    """One keyword argument of a call, as a literal-ish value."""
    node = next((kw.value for kw in call.keywords if kw.arg == name), None)
    return default if node is None else _literal(node)


def _class_attribute(filename, class_name, attribute):
    """A class-body assignment's value node, e.g. the ModelSettings(...) call behind `settings`."""
    for statement in _classes(filename)[class_name].body:
        targets = getattr(statement, "targets", [])
        if any(isinstance(t, ast.Name) and t.id == attribute for t in targets):
            return statement.value
    return None


def _class_attribute_value(filename, class_name, attribute):
    node = _class_attribute(filename, class_name, attribute)
    return None if node is None else _literal(node)


def _self_calls(filename, class_name):
    """Every method the class calls on self, however the call is formatted."""
    return {
        node.func.attr
        for node in ast.walk(_classes(filename)[class_name])
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "self"
    }


def _calls_on(filename, class_name, receiver):
    """Every attribute called on a named receiver, e.g. SomeWrapper.from_pretrained."""
    return {
        node.func.attr
        for node in ast.walk(_classes(filename)[class_name])
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == receiver
    }


def _call_keywords(filename, class_name, attribute):
    """The keyword names passed to one call anywhere in a class body, plus its **kwargs names."""
    found = set()
    for node in ast.walk(_classes(filename)[class_name]):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
        if name != attribute:
            continue
        for keyword in node.keywords:
            found.add(keyword.arg if keyword.arg is not None else ast.unparse(keyword.value))
    return found


def _load_contracts():
    spec = importlib.util.spec_from_file_location(
        "per_model_loading_contracts", CONTRACTS
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_hunyuan_pinned_revision_uses_one_checkpoint_request():
    declaration = _load_declaration("hunyuan.py", "xFuserHunyuanvideoModel")
    settings = _class_attribute("hunyuan.py", "xFuserHunyuanvideoModel", "settings")
    capabilities = _class_attribute("hunyuan.py", "xFuserHunyuanvideoModel", "capabilities")

    assert [ast.literal_eval(arg) for arg in declaration.args] == ["transformer"]
    assert _keyword(declaration, "replicated") is True
    assert _keyword(capabilities, "fully_shard_degree") is True
    assert _keyword(settings, "fsdp_strategy")["transformer"]["wrap_attrs"] == [
        "transformer_blocks",
        "single_transformer_blocks",
    ]
    assert (
        _class_attribute_value(
            "hunyuan.py", "xFuserHunyuanvideoModel", "_DIFFUSERS_FORMAT_REVISION"
        )
        == "refs/pr/18"
    )
    assert "checkpoint_request" in _call_keywords(
        "hunyuan.py", "xFuserHunyuanvideoModel", "_build_transformer"
    )
    assert _keyword(
        next(
            node
            for node in ast.walk(_classes("hunyuan.py")["xFuserHunyuanvideoModel"])
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "from_pretrained_kwargs"
        ),
        "include_subfolder",
    ) is False
    # The pin belongs to the runner's checkpoint identity, not to the one call site that used to
    # pass it: the text encoder's meta construction and its checkpoint mapping ask the runner.
    # test_hunyuanvideo_pins_its_revision_for_every_component_it_resolves calls it for real.
    assert "_checkpoint_request" in {
        node.name
        for node in _classes("hunyuan.py")["xFuserHunyuanvideoModel"].body
        if isinstance(node, ast.FunctionDef)
    }


def test_hunyuanvideo_offers_its_llama_encoder_to_the_memory_efficient_load():
    """The encoder every rank loaded whole is what kept this model's host peak where it was.

    Fourteen gigabytes of Llama per rank, against a transformer that fills a block at a time, so
    the transformer path working was not visible in the host figure at all. Both meta paths pick
    their components out of fsdp_strategy, and the pipeline only skips loading a component it was
    handed, so declaring the encoder and passing the kwargs are one change.
    """
    settings = _class_attribute("hunyuan.py", "xFuserHunyuanvideoModel", "settings")
    strategy = _keyword(settings, "fsdp_strategy")
    pipeline_kwargs = _call_keywords(
        "hunyuan.py", "xFuserHunyuanvideoModel", "from_pretrained"
    )

    assert strategy["text_encoder"]["wrap_attrs"] == ["layers"]
    assert "_meta_te_kwargs" in _self_calls("hunyuan.py", "xFuserHunyuanvideoModel")
    assert "quantization_config" in pipeline_kwargs
    assert "te_kwargs" in pipeline_kwargs
    # The 0.2G CLIP stays out: a collective per prompt to save a fraction of a gigabyte.
    assert "text_encoder_2" not in strategy


def test_hunyuanvideo_encoder_shard_path_exists_on_the_encoder_it_loads():
    """A wrap path is a claim about another library's module layout, so check it against that library.

    LlamaModel keeps its decoder layers at the top level. "model.layers", which is where the
    causal-LM wrapper and several other encoders keep theirs, would raise an AttributeError in
    every sharded case and nowhere else.
    """
    pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    accelerate = pytest.importorskip("accelerate")
    from xfuser.core.distributed.sharding import rgetattr
    from xfuser.model_executor.models.runner_models.hunyuan import (
        xFuserHunyuanvideoModel,
    )

    config = transformers.LlamaConfig(
        num_hidden_layers=2, hidden_size=8, intermediate_size=16, num_attention_heads=2
    )
    with accelerate.init_empty_weights():
        encoder = transformers.LlamaModel(config)

    strategy = xFuserHunyuanvideoModel.settings.fsdp_strategy["text_encoder"]
    for attr in strategy["wrap_attrs"]:
        assert len(rgetattr(encoder, attr)) == 2, f"{attr} does not reach the decoder layers"


def test_hunyuanvideo_pins_its_revision_for_every_component_it_resolves():
    """Whoever asks for the checkpoint identity gets the pinned revision, subfolder or not.

    The text encoder is built from its config and then mapped onto its own checkpoint, and both
    steps ask the runner rather than being handed a request. On the default revision this repo has
    no diffusers-format subfolders to find, so an unpinned request cannot resolve them.
    """
    pytest.importorskip("torch")
    pytest.importorskip("diffusers")
    from xfuser.model_executor.models.runner_models.hunyuan import (
        xFuserHunyuanvideoModel,
    )

    # No __init__: the request only reads the class's settings, and a runner needs a config and a
    # device this test has neither of.
    runner = object.__new__(xFuserHunyuanvideoModel)

    for subfolder in (None, "transformer", "text_encoder"):
        request = runner._checkpoint_request(subfolder)
        assert request.revision == "refs/pr/18"
        assert request.subfolder == subfolder
    # A caller that has already decided still wins, as on the base runner.
    assert runner._checkpoint_request(revision="main").revision == "main"


@pytest.mark.parametrize(
    "class_name",
    ["xFuserLTX2VideoModel", "xFuserLTX23VideoModel"],
)
def test_ltx_direct_transformers_use_standard_construction_seam(class_name):
    declaration = _load_declaration("ltx.py", class_name)
    reason = ast.literal_eval(
        next(
            keyword.value
            for keyword in declaration.keywords
            if keyword.arg == "unsupported_reason"
        )
    )

    capabilities = _class_attribute("ltx.py", class_name, "capabilities")
    settings = _class_attribute("ltx.py", class_name, "settings")

    assert declaration.args == []
    assert not any(keyword.arg == "replicated" for keyword in declaration.keywords)
    assert _keyword(capabilities, "fully_shard_degree") is True
    assert _keyword(settings, "fsdp_strategy")["transformer"]["wrap_attrs"] == [
        "transformer_blocks"
    ]
    assert "_build_transformer" in _self_calls("ltx.py", class_name)
    assert "from_pretrained" not in _calls_on(
        "ltx.py", class_name, "xFuserLTX2VideoTransformer3DWrapper"
    )
    assert "stage 2 distilled LoRA" in reason

    contracts = _load_contracts()
    capability = contracts.LoadDeclaration.for_runner(
        type(
            "LTXCapabilities",
            (),
            {
                "use_fp8_gemms": class_name == "xFuserLTX2VideoModel",
                "use_fp4_gemms": False,
                "use_int8_gemms": False,
                "fully_shard_degree": True,
            },
        )(),
        fsdp_strategy={"transformer": {"wrap_attrs": ["transformer_blocks"]}},
        unsupported_reason=reason,
    )
    for mode in (
        contracts.MaterializationMode.FSDP_META,
        contracts.MaterializationMode.REPLICATED_META,
    ):
        with pytest.raises(
            contracts.UnsupportedLoadContract,
            match="stage 2 distilled LoRA",
        ):
            contracts.validate_materialization_contract(
                capability,
                mode,
                {"transformer": {"wrap_attrs": ["transformer_blocks"]}},
                runner_name=class_name,
            )


def test_named_custom_adapters_reject_standard_collective_modes():
    contracts = _load_contracts()
    custom = (
        contracts.LoaderAdapter.DISTILLED_WAN,
        contracts.LoaderAdapter.SD35_COMPOSITION,
        contracts.LoaderAdapter.CAUSAL_WAN,
        contracts.LoaderAdapter.HUNYUAN15_VARIANTS,
    )

    for adapter in custom:
        capability = contracts.LoadDeclaration.for_runner(
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
            meta_transformers=("transformer",),
            replicated=True,
            fsdp_strategy={"transformer": {"wrap_attrs": ["blocks"]}},
            loader_adapter=adapter,
            unsupported_reason="custom checkpoint semantics",
        )

        assert capability.meta_transformers == ()
        assert capability.construction_seam is None
        malformed = contracts.LoadDeclaration(
            fsdp_meta_transformers=("transformer",),
            materialization_modes=frozenset(
                {
                    contracts.MaterializationMode.EAGER,
                    contracts.MaterializationMode.FSDP_META,
                }
            ),
            construction_seam=contracts.ConstructionSeam.BUILD_TRANSFORMER,
            loader_adapter=adapter,
            unsupported_reason="custom checkpoint semantics",
        )
        with pytest.raises(
            contracts.UnsupportedLoadContract,
            match="custom checkpoint semantics",
        ):
            contracts.validate_materialization_contract(
                malformed,
                contracts.MaterializationMode.FSDP_META,
                {"transformer": {"wrap_attrs": ["blocks"]}},
                runner_name="CustomRunner",
            )


def test_distilled_wan_declares_local_meta_without_collective_support():
    contracts = _load_contracts()
    adapter = contracts.LoaderAdapter.DISTILLED_WAN
    capability = contracts.LoadDeclaration.for_runner(
        type(
            "Capabilities",
            (),
            {
                "use_fp8_gemms": True,
                "use_fp4_gemms": True,
                "use_int8_gemms": False,
                "fully_shard_degree": True,
            },
        )(),
        meta_transformers=("transformer", "transformer_2"),
        replicated=True,
        fsdp_strategy={
            "transformer": {"wrap_attrs": ["blocks"]},
            "transformer_2": {"wrap_attrs": ["blocks"]},
        },
        loader_adapter=adapter,
    )

    assert adapter.supports_local_blockwise is True
    assert adapter.supports_standard_collectives is False
    assert capability.local_meta_transformers == (
        "transformer",
        "transformer_2",
    )
    assert capability.meta_transformers == ()
    assert capability.materialization_modes == {contracts.MaterializationMode.EAGER}


@pytest.mark.parametrize(
    ("filename", "class_name", "adapter"),
    [
        ("wan.py", "xFuserWan22DistilledI2VModel", "DISTILLED_WAN"),
        ("stable_diffusion.py", "xFuserStableDiffusionModel", "SD35_COMPOSITION"),
        ("causal_wan.py", "xFuserCausalWanModel", "CAUSAL_WAN"),
        ("hunyuan.py", "xFuserHunyuanvideo15Model", "HUNYUAN15_VARIANTS"),
        (
            "hunyuan.py",
            "xFuserHunyuanvideo15DistilledModel",
            "HUNYUAN15_VARIANTS",
        ),
        (
            "hunyuan.py",
            "xFuserHunyuanvideo15SparseModel",
            "HUNYUAN15_VARIANTS",
        ),
    ],
)
def test_custom_runners_declare_their_dedicated_adapter(filename, class_name, adapter):
    declaration = _load_declaration(filename, class_name)

    assert _keyword(declaration, "loader_adapter") == f"LoaderAdapter.{adapter}"
    assert _keyword(declaration, "unsupported_reason")


def test_krea2_text_encoder_is_declaratively_excluded():
    """The exclusion carries the reason, so reading the declaration explains the omission.

    Asserting on the class source instead would accept the words appearing in a comment,
    which is where "ROCm" appears in this file anyway.
    """
    declaration = _load_declaration("krea2.py", "_Krea2BaseModel")

    assert _keyword(declaration, "component_exclusions") == ["KREA2_TEXT_ENCODER_EXCLUSION"]

    exclusion = _load_contracts().KREA2_TEXT_ENCODER_EXCLUSION

    assert exclusion.component == "text_encoder"
    assert "Qwen3VL" in exclusion.reason
    assert "ROCm" in exclusion.reason


def test_tokenizer_reload_reads_the_tokenizer_directory_not_the_repo_root(
    tmp_path, monkeypatch
):
    """HunyuanVideo's repo root config.json is not valid JSON, and it is not ours to fix.

    Transformers v5 parses that file for an unrelated Mistral regex fix, for every repo once
    HF_HUB_OFFLINE is set, so reloading the tokenizer by repo id raised JSONDecodeError and the model
    could not load at all. The tokenizer's own directory has no config.json to trip over.
    """
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
    backend = tokenizers.Tokenizer(
        tokenizers.models.WordLevel({"<unk>": 0, "hello": 1}, unk_token="<unk>")
    )
    backend.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    backend.save(str(component / "tokenizer.json"))
    (component / "tokenizer_config.json").write_text(
        json.dumps({"tokenizer_class": "LlamaTokenizerFast"})
    )

    class FakeLlamaTokenizerFast:
        pass

    # The helper logs through the rank-aware logger, which reads the launcher's environment.
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    pipeline = types.SimpleNamespace(
        components={"tokenizer": FakeLlamaTokenizerFast()}, tokenizer=None
    )

    assert runner_utils._tokenizer_directory(str(repo), "tokenizer", {}) == str(
        component
    )
    runner_utils.fix_llama_tokenizer_pretokenizer(pipeline, str(repo))
    assert pipeline.tokenizer.tokenize("hello") == ["hello"]

    # No fast tokenizer file beside it: the directory cannot build one, so keep the name given.
    (component / "tokenizer.json").unlink()
    assert runner_utils._tokenizer_directory(str(repo), "tokenizer", {}) is None


def test_krea2_text_encoder_shard_path_exists_on_the_encoder_it_loads():
    """A wrap path is a claim about another library's module layout, so check it against that library.

    Krea-2 named its decoder layers "model.layers", which is where a text-only encoder keeps them.
    Its encoder is a Qwen3VLModel, which holds them under language_model beside the vision tower, so
    every sharded case died with an AttributeError before reaching a load. Nothing else noticed:
    eager and replicated cases do not walk this path.
    """
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


def test_component_exclusions_are_queryable_by_loading_adapters():
    contracts = _load_contracts()
    capability = contracts.LoadDeclaration(
        component_exclusions=(contracts.KREA2_TEXT_ENCODER_EXCLUSION,)
    )

    exclusion = capability.exclusion_for("text_encoder")
    assert exclusion is contracts.KREA2_TEXT_ENCODER_EXCLUSION
    assert capability.exclusion_for("transformer") is None


def test_hunyuan_wrapper_keeps_the_parent_config_signature():
    pytest.importorskip("diffusers")
    wrapper_module = pytest.importorskip(
        "xfuser.model_executor.models.transformers.transformer_hunyuan_video"
    )
    from diffusers.models.transformers.transformer_hunyuan_video import (
        HunyuanVideoTransformer3DModel,
    )

    assert inspect.signature(
        wrapper_module.xFuserHunyuanVideoTransformer3DWrapper.__init__
    ) == inspect.signature(HunyuanVideoTransformer3DModel.__init__)
    assert (
        "from_config" in wrapper_module.xFuserHunyuanVideoTransformer3DWrapper.__dict__
    )


def test_ltx_wrapper_keeps_diffusers_config_api():
    pytest.importorskip("diffusers")
    wrapper_module = pytest.importorskip(
        "xfuser.model_executor.models.transformers.transformer_ltx2"
    )
    wrapper = wrapper_module.xFuserLTX2VideoTransformer3DWrapper

    assert callable(wrapper.load_config)
    assert callable(wrapper.from_config)
