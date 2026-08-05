"""Dependency-free structural checks for explicit runner meta-load declarations."""

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUNNERS = ROOT / "xfuser/model_executor/models/runner_models"


def _classes(path):
    return {
        node.name: node
        for node in ast.parse(path.read_text()).body
        if isinstance(node, ast.ClassDef)
    }


def _assigned_names(class_node):
    return {
        target.id
        for node in class_node.body
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        for target in (
            node.targets if isinstance(node, ast.Assign) else [node.target]
        )
        if isinstance(target, ast.Name)
    }


def _declares_load_capability(class_node):
    if "load_capability" in _assigned_names(class_node):
        return True
    return any(
        isinstance(decorator, ast.Call)
        and isinstance(decorator.func, ast.Attribute)
        and decorator.func.attr == "declare"
        for decorator in class_node.decorator_list
    )


def test_base_runner_defaults_to_an_explicit_unsupported_declaration():
    classes = _classes(RUNNERS / "base_model.py")
    base = classes["xFuserModel"]

    assert "load_capability" in _assigned_names(base)
    methods = {
        node.name for node in base.body if isinstance(node, ast.FunctionDef)
    }
    assert "_supports_replicated_meta_load" not in methods


def test_supported_runner_families_declare_the_meta_construction_seam():
    expected = {
        "flux.py": {
            "xFuserFluxModel",
            "xFuserFluxKontextModel",
            "xFuserFlux2Model",
            "xFuserFlux2Klein9BModel",
        },
        "krea2.py": {"_Krea2BaseModel"},
        "z_image.py": {"xFuserZImageModel", "xFuserZImageTurboModel"},
        "cosmos3.py": {"xFuserCosmos3SuperModel"},
        "wan.py": {
            "xFuserWan21I2VModel",
            "xFuserWan22I2VModel",
            "xFuserWan21T2VModel",
            "xFuserWan22T2VModel",
            "xFuserWan22TI2VModel",
            "xFuserWan21VACEModel",
        },
        "qwen.py": {"xFuserQwenImageEditModel", "xFuserQwenImageModel"},
    }

    missing = []
    for filename, names in expected.items():
        classes = _classes(RUNNERS / filename)
        for name in names:
            if not _declares_load_capability(classes[name]):
                missing.append(f"{filename}:{name}")

    assert not missing, "missing load_capability declaration: " + ", ".join(missing)


def test_custom_runner_exclusions_remain_explicit():
    expected = {
        "ltx.py": {"xFuserLTX23VideoModel", "xFuserLTX2VideoModel"},
        "stable_diffusion.py": {"xFuserStableDiffusionModel"},
        "hunyuan.py": {"xFuserHunyuanvideoModel", "xFuserHunyuanvideo15Model"},
        "wan.py": {"xFuserWan22DistilledI2VModel"},
        "causal_wan.py": {"xFuserCausalWanModel"},
    }

    missing = []
    for filename, names in expected.items():
        classes = _classes(RUNNERS / filename)
        for name in names:
            if not _declares_load_capability(classes[name]):
                missing.append(f"{filename}:{name}")

    assert not missing, "missing explicit exclusion: " + ", ".join(missing)
