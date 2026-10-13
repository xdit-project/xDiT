"""Dependency-free structural checks on the load path.

The rest of the original file went with the test reorganisation in #798; these
are the checks this branch added, kept because each one pins a regression it
fixed rather than restating what the code says.
"""

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]


def _class(tree, name):
    return next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name)


def test_no_profile_gate_names_a_format_outside_the_registry():
    """Which formats go together, and which a machine has, is asked in one place.

    The argument validator keeps only the questions about the *run* -- a flag
    that needs a tiered profile, a schedule naming formats the profile does
    not. Whether a format can be stored here, and whether two can be named
    together, is the registry's answer at load time.
    """
    args_path = ROOT / "xfuser/config/args.py"
    args_source = args_path.read_text()
    args_tree = ast.parse(args_source)
    args_class = _class(args_tree, "xFuserArgs")
    validate = next(
        node
        for node in args_class.body
        if isinstance(node, ast.FunctionDef) and node.name == "_validate_gemm_quantization_flags"
    )
    source = ast.get_source_segment(args_source, validate)
    for refusal in ("INT8 cannot be ", "cannot be\n                tiered"):
        assert refusal not in source

    base_path = ROOT / "xfuser/model_executor/models/runner_models/base_model.py"
    base_source = base_path.read_text()
    base_tree = ast.parse(base_source)
    base_class = _class(base_tree, "xFuserModel")
    base_validate = next(
        node for node in base_class.body if isinstance(node, ast.FunctionDef) and node.name == "_validate_config"
    )
    base = ast.get_source_segment(base_source, base_validate)
    assert "Cannot use int8 gemms with fp8 or fp4 gemms." not in base
    # Nor a second copy of the hardware rules the probes already measure.
    for gate in ("CUDA capability >= 10.0", "require AITER", "on ROCm are not"):
        assert gate not in base


def _text_encoder_formats():
    """The formats whose adapters declare `places_text_encoder`, read out of
    the source so this stays a structural check."""
    path = ROOT / "xfuser/model_executor/models/runner_models/loading/backends.py"
    found = set()
    for node in ast.parse(path.read_text()).body:
        if not isinstance(node, ast.ClassDef):
            continue
        declares = any(
            isinstance(stmt, ast.Assign)
            and getattr(stmt.targets[0], "id", "") == "places_text_encoder"
            and getattr(stmt.value, "value", False) is True
            for stmt in node.body
        )
        if not declares:
            continue
        for decorator in node.decorator_list:
            if (
                isinstance(decorator, ast.Call)
                and getattr(decorator.func, "id", "") == "stores"
                and isinstance(decorator.args[0], ast.Constant)
            ):
                found.add(decorator.args[0].value)
    assert found, "no adapter declares places_text_encoder any more"
    return found


def test_the_text_encoder_format_is_resolved_only_when_an_encoder_needs_it():
    """Asking for it up front asks this machine to store a format the run may
    never have named.

    It used to be resolved before the loop, which was harmless only while
    capability probing answered for every format on every run. Once probing was
    narrowed to the formats a run actually names, that lookup had no truthful
    answer and refused the load -- on a plain bf16 run of any model that merely
    declares encoder targets.

    So the lookup must sit inside the loop, after the guard that skips an
    encoder with no targets.
    """
    path = ROOT / "xfuser/model_executor/models/runner_models/loading/text_encoder_plan.py"
    source = path.read_text()
    plan = next(
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.FunctionDef) and node.name == "plan_text_encoders"
    )

    loop = next(node for node in plan.body if isinstance(node, ast.For))
    lookups = [
        node
        for node in ast.walk(plan)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "adapter_for"
    ]
    assert lookups, "nothing resolves the text encoder's adapter any more"

    guard = min(node.lineno for node in ast.walk(loop) if isinstance(node, ast.Continue))
    for call in lookups:
        assert loop.lineno < call.lineno, (
            "the text encoder's adapter is resolved before the loop, so a run "
            "that quantizes nothing still has to be able to store its format"
        )
        assert call.lineno > guard, (
            "the adapter is resolved before the no-targets guard, so an "
            "encoder with nothing to place still forces the lookup"
        )


def test_the_meta_text_encoder_layout_matches_the_declared_format():
    """`places_text_encoder` claims this pair can be placed in an encoder.

    The meta and broadcast paths get there through `_swap_meta_te_to_fp8`,
    which builds one specific layer class rather than consulting anything. So
    declaring a second format would leave that swap behind while the plan, the
    adapter lookup and the ledger all moved; this fails until it moves too.
    """
    declared = _text_encoder_formats()

    path = ROOT / "xfuser/model_executor/models/runner_models/loading/meta_load.py"
    source = path.read_text()
    swap = next(
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.FunctionDef) and node.name == "_swap_meta_te_to_fp8"
    )
    body = ast.get_source_segment(source, swap)

    layer_for = {"fp8": "xFuserFP8BlockScaleLinear"}
    assert declared <= set(layer_for), (
        f"the meta swap builds {sorted(layer_for)}, but {sorted(declared)} "
        "declare they can be placed in an encoder; the swap has to learn the "
        "new layout before the declaration is true"
    )
    for format_name in declared:
        assert layer_for[format_name] in body
