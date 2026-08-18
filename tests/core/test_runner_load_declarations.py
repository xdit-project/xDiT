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
        for target in (node.targets if isinstance(node, ast.Assign) else [node.target])
        if isinstance(target, ast.Name)
    }


def _has_own_load_support(class_node):
    return "load_support" in _assigned_names(class_node)


def test_supported_runner_families_declare_load_support():
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
        "hunyuan.py": {"xFuserHunyuanvideoModel"},
    }

    missing = []
    for filename, names in expected.items():
        classes = _classes(RUNNERS / filename)
        for name in names:
            if not _has_own_load_support(classes[name]):
                missing.append(f"{filename}:{name}")

    assert not missing, "missing load declaration: " + ", ".join(missing)


def test_unsupported_runner_capabilities_remain_explicit():
    expected = {
        "stable_diffusion.py": {"xFuserStableDiffusionModel"},
        "hunyuan.py": {
            "xFuserHunyuanvideo15Model",
            "xFuserHunyuanvideo15DistilledModel",
            "xFuserHunyuanvideo15SparseModel",
        },
        "wan.py": {"xFuserWan22DistilledI2VModel"},
        "causal_wan.py": {"xFuserCausalWanModel"},
        "ltx.py": {"xFuserLTX23VideoModel", "xFuserLTX2VideoModel"},
    }

    missing = []
    for filename, names in expected.items():
        classes = _classes(RUNNERS / filename)
        for name in names:
            if not _has_own_load_support(classes[name]):
                missing.append(f"{filename}:{name}")

    assert not missing, "missing explicit unsupported declaration: " + ", ".join(missing)


def test_every_registered_runner_has_its_own_load_support():
    missing = []
    for path in sorted(RUNNERS.glob("*.py")):
        for name, class_node in _classes(path).items():
            registered = any(
                isinstance(decorator, ast.Call)
                and isinstance(decorator.func, ast.Name)
                and decorator.func.id == "register_model"
                for decorator in class_node.decorator_list
            )
            if registered and not _has_own_load_support(class_node):
                missing.append(f"{path.name}:{name}")

    assert (
        not missing
    ), "registered runner inherits an implicit declaration: " + ", ".join(missing)
