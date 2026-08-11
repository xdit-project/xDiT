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


def _strategy_dict(node, constants):
    if isinstance(node, ast.Name):
        node = constants.get(node.id)
    if not isinstance(node, ast.Dict):
        return {}
    strategy = {}
    for key, value in zip(node.keys, node.values):
        if not (
            isinstance(key, ast.Constant)
            and isinstance(key.value, str)
            and isinstance(value, ast.Dict)
        ):
            continue
        fields = {
            field_key.value: field_value
            for field_key, field_value in zip(value.keys, value.values)
            if isinstance(field_key, ast.Constant) and isinstance(field_key.value, str)
        }
        wraps = fields.get("wrap_attrs")
        if isinstance(wraps, (ast.List, ast.Tuple)):
            strategy[key.value] = tuple(
                item.value
                for item in wraps.elts
                if isinstance(item, ast.Constant) and isinstance(item.value, str)
            )
    return strategy


def _module_strategy_constants(path):
    constants = {}
    for node in ast.parse(path.read_text()).body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and isinstance(node.value, ast.Dict)
        ):
            constants[node.targets[0].id] = node.value
    return constants


def _class_strategy(path, class_node):
    constants = _module_strategy_constants(path)
    strategy = {}
    for statement in class_node.body:
        if isinstance(statement, ast.Assign):
            if (
                len(statement.targets) == 1
                and isinstance(statement.targets[0], ast.Name)
                and statement.targets[0].id == "settings"
                and isinstance(statement.value, ast.Call)
            ):
                for keyword in statement.value.keywords:
                    if keyword.arg == "fsdp_strategy":
                        strategy.update(_strategy_dict(keyword.value, constants))
        if not isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for node in ast.walk(statement):
            if not (
                isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Subscript)
            ):
                continue
            target = node.targets[0]
            owner = target.value
            if not (
                isinstance(owner, ast.Attribute)
                and owner.attr == "fsdp_strategy"
                and isinstance(owner.value, ast.Attribute)
                and owner.value.attr == "settings"
                and isinstance(owner.value.value, ast.Name)
                and owner.value.value.id == "self"
                and isinstance(target.slice, ast.Constant)
                and isinstance(target.slice.value, str)
            ):
                continue
            parsed = _strategy_dict(
                ast.Dict(
                    keys=[ast.Constant(target.slice.value)],
                    values=[node.value],
                ),
                constants,
            )
            strategy.update(parsed)
    return strategy


def _assigned_names(class_node):
    return {
        target.id
        for node in class_node.body
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        for target in (node.targets if isinstance(node, ast.Assign) else [node.target])
        if isinstance(target, ast.Name)
    }


def _has_own_load_declaration(class_node):
    if "load_declaration" in _assigned_names(class_node):
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

    assert "load_declaration" in _assigned_names(base)
    methods = {node.name for node in base.body if isinstance(node, ast.FunctionDef)}
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
        "hunyuan.py": {"xFuserHunyuanvideoModel"},
    }

    missing = []
    for filename, names in expected.items():
        classes = _classes(RUNNERS / filename)
        for name in names:
            if not _has_own_load_declaration(classes[name]):
                missing.append(f"{filename}:{name}")

    assert not missing, "missing load declaration: " + ", ".join(missing)


def test_custom_runner_exclusions_remain_explicit():
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
            if not _has_own_load_declaration(classes[name]):
                missing.append(f"{filename}:{name}")

    assert not missing, "missing explicit exclusion: " + ", ".join(missing)


def test_every_registered_runner_has_its_own_load_declaration():
    missing = []
    for path in sorted(RUNNERS.glob("*.py")):
        for name, class_node in _classes(path).items():
            registered = any(
                isinstance(decorator, ast.Call)
                and isinstance(decorator.func, ast.Name)
                and decorator.func.id == "register_model"
                for decorator in class_node.decorator_list
            )
            if registered and not _has_own_load_declaration(class_node):
                missing.append(f"{path.name}:{name}")

    assert (
        not missing
    ), "registered runner inherits an implicit declaration: " + ", ".join(missing)


def test_strategy_extraction_uses_assignments_not_unrelated_strings(tmp_path):
    path = tmp_path / "runner.py"
    path.write_text("""
COMMON = {
    "transformer": {"wrap_attrs": ["blocks"], "dtype": object()},
}

class Example:
    settings = ModelSettings(fsdp_strategy=COMMON)
    unrelated = "transformer_2 wrap_attrs"

    def customize(self):
        self.settings.fsdp_strategy["transformer_2"] = {
            "wrap_attrs": ["layers"],
        }
""")
    example = _classes(path)["Example"]

    assert _class_strategy(path, example) == {
        "transformer": ("blocks",),
        "transformer_2": ("layers",),
    }


def test_every_meta_declaration_matches_a_build_seam_and_strategy():
    index = {}
    for path in sorted(RUNNERS.glob("*.py")):
        for name, class_node in _classes(path).items():
            index[name] = (path, class_node)

    def hierarchy(class_node):
        for base in class_node.bases:
            if isinstance(base, ast.Name) and base.id in index:
                yield from hierarchy(index[base.id][1])
        yield class_node

    mismatches = []
    for name, (path, class_node) in index.items():
        registered = any(
            isinstance(decorator, ast.Call)
            and isinstance(decorator.func, ast.Name)
            and decorator.func.id == "register_model"
            for decorator in class_node.decorator_list
        )
        if not registered:
            continue
        declaration = next(
            (
                decorator
                for decorator in class_node.decorator_list
                if isinstance(decorator, ast.Call)
                and isinstance(decorator.func, ast.Attribute)
                and decorator.func.attr == "declare"
            ),
            None,
        )
        if declaration is None:
            continue
        components = tuple(
            arg.value
            for arg in declaration.args
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str)
        )
        if not components:
            continue

        nodes = tuple(hierarchy(class_node))
        uses_build_seam = any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "_build_transformer"
            for owner in nodes
            for node in ast.walk(owner)
        )
        strategy = {}
        for owner in nodes:
            owner_path = index[owner.name][0]
            strategy.update(_class_strategy(owner_path, owner))
        missing_strategy = [
            component for component in components if not strategy.get(component)
        ]
        if not uses_build_seam or missing_strategy:
            mismatches.append(
                f"{path.name}:{name} components={components} "
                f"build={uses_build_seam} missing_strategy={missing_strategy}"
            )

    assert not mismatches, "meta declaration does not match construction: " + ", ".join(
        mismatches
    )
