"""Task-6 contracts for incremental model loading coverage.

The AST checks stay dependency-light. Model API checks are guarded so a core
test environment without Diffusers can still enforce the declarations.
"""

import ast
import importlib.util
import inspect
from pathlib import Path

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


def _source(filename, class_name):
    text = (RUNNERS / filename).read_text()
    node = _classes(filename)[class_name]
    start = min(
        [node.lineno, *(decorator.lineno for decorator in node.decorator_list)]
    )
    return "\n".join(text.splitlines()[start - 1 : node.end_lineno])


def _load_declaration(filename, class_name):
    return next(
        decorator
        for decorator in _classes(filename)[class_name].decorator_list
        if isinstance(decorator, ast.Call)
        and isinstance(decorator.func, ast.Attribute)
        and decorator.func.attr == "declare"
    )


def _load_contracts():
    spec = importlib.util.spec_from_file_location(
        "roadmap6_loading_contracts", CONTRACTS
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_hunyuan_pinned_revision_uses_one_checkpoint_request():
    source = _source("hunyuan.py", "xFuserHunyuanvideoModel")

    assert '@LoadCapability.declare("transformer", replicated=True)' in source
    assert "fully_shard_degree=True" in source
    assert '"transformer_blocks", "single_transformer_blocks"' in source
    assert 'revision="refs/pr/18"' in source
    assert "self._build_transformer(" in source
    assert "checkpoint_request=transformer_request" in source
    assert "request.from_pretrained_kwargs(include_subfolder=False)" in source


@pytest.mark.parametrize(
    "class_name",
    ["xFuserLTX2VideoModel", "xFuserLTX23VideoModel"],
)
def test_ltx_direct_transformers_use_standard_construction_seam(class_name):
    source = _source("ltx.py", class_name)
    declaration = _load_declaration("ltx.py", class_name)
    reason = ast.literal_eval(
        next(
            keyword.value
            for keyword in declaration.keywords
            if keyword.arg == "unsupported_reason"
        )
    )

    assert declaration.args == []
    assert not any(
        keyword.arg == "replicated"
        for keyword in declaration.keywords
    )
    assert "fully_shard_degree=True" in source
    assert '"wrap_attrs": ["transformer_blocks"]' in source
    assert "self._build_transformer(" in source
    assert "xFuserLTX2VideoTransformer3DWrapper.from_pretrained(" not in source
    assert "stage 2 distilled LoRA" in reason

    contracts = _load_contracts()
    capability = contracts.LoadCapability.for_runner(
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
        fsdp_strategy={
            "transformer": {"wrap_attrs": ["transformer_blocks"]}
        },
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
        capability = contracts.LoadCapability.for_runner(
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
        malformed = contracts.LoadCapability(
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
def test_custom_runners_declare_their_dedicated_adapter(
    filename, class_name, adapter
):
    source = _source(filename, class_name)

    assert f"loader_adapter=LoaderAdapter.{adapter}" in source
    assert "unsupported_reason=" in source


def test_krea2_text_encoder_is_declaratively_excluded():
    source = _source("krea2.py", "_Krea2BaseModel")

    assert "KREA2_TEXT_ENCODER_EXCLUSION" in source
    assert "component_exclusions=" in source
    assert "Qwen3VL" in source
    assert "ROCm" in source


def test_component_exclusions_are_queryable_by_loading_adapters():
    contracts = _load_contracts()
    capability = contracts.LoadCapability(
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
    ) == inspect.signature(
        HunyuanVideoTransformer3DModel.__init__
    )
    assert (
        "from_config"
        in wrapper_module.xFuserHunyuanVideoTransformer3DWrapper.__dict__
    )


def test_ltx_wrapper_keeps_diffusers_config_api():
    pytest.importorskip("diffusers")
    wrapper_module = pytest.importorskip(
        "xfuser.model_executor.models.transformers.transformer_ltx2"
    )
    wrapper = wrapper_module.xFuserLTX2VideoTransformer3DWrapper

    assert callable(wrapper.load_config)
    assert callable(wrapper.from_config)
