"""Dependency-free structural checks for roadmap task 2."""

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _class(tree, name):
    return next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name)


def test_meta_load_defines_collective_layout_and_persistent_buffer_helpers():
    path = ROOT / "xfuser/model_executor/models/runner_models/loading/meta_load.py"
    tree = ast.parse(path.read_text())
    functions = {
        node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    assert {
        "_tensor_layout",
        "_collective_assert_same_layout",
        "_collective_build_call",
        "_collective_reconcile_tensor_specs",
        "_collective_source_call",
        "_persistent_named_buffers",
    } <= functions


def test_args_and_base_both_validate_gemm_flag_ownership():
    args_path = ROOT / "xfuser/config/args.py"
    args_tree = ast.parse(args_path.read_text())
    args_class = _class(args_tree, "xFuserArgs")
    methods = {
        node.name for node in args_class.body if isinstance(node, ast.FunctionDef)
    }
    assert "_validate_gemm_quantization_flags" in methods

    base_path = ROOT / "xfuser/model_executor/models/runner_models/base_model.py"
    base_tree = ast.parse(base_path.read_text())
    base_class = _class(base_tree, "xFuserModel")
    validate = next(
        node for node in base_class.body
        if isinstance(node, ast.FunctionDef) and node.name == "_validate_config"
    )
    calls = {
        node.func.attr
        for node in ast.walk(validate)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert "_validate_gemm_quantization_flags" in calls


def test_central_gemm_validation_owns_int8_conflicts():
    args_path = ROOT / "xfuser/config/args.py"
    args_source = args_path.read_text()
    args_tree = ast.parse(args_source)
    args_class = _class(args_tree, "xFuserArgs")
    validate = next(
        node for node in args_class.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_validate_gemm_quantization_flags"
    )
    attributes = {
        node.attr for node in ast.walk(validate) if isinstance(node, ast.Attribute)
    }
    assert {"use_int8_gemms", "use_fp8_gemms", "use_fp4_gemms"} <= attributes
    assert "--use_int8_gemms cannot be combined" in ast.get_source_segment(
        args_source, validate
    )

    base_path = ROOT / "xfuser/model_executor/models/runner_models/base_model.py"
    base_source = base_path.read_text()
    base_tree = ast.parse(base_source)
    base_class = _class(base_tree, "xFuserModel")
    base_validate = next(
        node for node in base_class.body
        if isinstance(node, ast.FunctionDef) and node.name == "_validate_config"
    )
    assert "Cannot use int8 gemms with fp8 or fp4 gemms." not in ast.get_source_segment(
        base_source, base_validate
    )


def test_fp4_owner_prevents_the_later_generic_fp8_walk():
    path = ROOT / "xfuser/model_executor/models/runner_models/base_model.py"
    source = path.read_text()
    tree = ast.parse(source)
    base_class = _class(tree, "xFuserModel")
    post_load = next(
        node
        for node in base_class.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_post_load_and_state_initialization"
    )
    post_load_source = ast.get_source_segment(source, post_load)

    assert "if self.config.use_fp4_gemms:" in post_load_source
    assert "and not self.config.use_fp4_gemms" in post_load_source
    assert "_setup_fp8_only_gemm_modules" in source


def test_mxfp4_quantized_weight_is_registered_as_parameter():
    path = ROOT / "xfuser/model_executor/layers/mxfp4_linear.py"
    tree = ast.parse(path.read_text())
    cls = _class(tree, "xFuserMXFP4Linear")
    quantize = next(
        node for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == "_quantize_weights"
    )
    source = ast.get_source_segment(path.read_text(), quantize)

    assert "nn.Parameter(weight_shuffle, requires_grad=False)" in source
    assert "register_buffer('weight_scale', weight_scale, persistent=True)" in source
    assert any(
        isinstance(node, ast.FunctionDef) and node.name == "_load_from_state_dict"
        for node in cls.body
    )
    load_state = next(
        node for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == "_load_from_state_dict"
    )
    assert "destination_device" in ast.get_source_segment(path.read_text(), load_state)
    assert "packed state cannot be loaded after FSDP" in ast.get_source_segment(
        path.read_text(), load_state
    )
    load_source = ast.get_source_segment(path.read_text(), load_state)
    assert "full-precision state cannot replace an FSDP-managed packed parameter" in load_source
    assert "incoming_device" in load_source
    assert any(
        isinstance(node, ast.FunctionDef)
        and node.name == "_is_fsdp_managed_parameter"
        for node in cls.body
    )


def test_disk_fill_preserves_nonpersistent_buffers_and_reports_source_errors():
    sharding_path = ROOT / "xfuser/core/distributed/sharding.py"
    sharding_source = sharding_path.read_text()
    sharding_tree = ast.parse(sharding_source)
    functions = {
        node.name for node in sharding_tree.body if isinstance(node, ast.FunctionDef)
    }
    assert {"_save_nonpersistent_buffers", "_restore_nonpersistent_buffers"} <= functions
    shard_component = next(
        node for node in sharding_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "shard_component"
    )
    shard_source = ast.get_source_segment(sharding_source, shard_component)
    assert shard_source.index("_restore_nonpersistent_buffers") < shard_source.index(
        "load_block_fn(block, i)"
    )

    meta_path = ROOT / "xfuser/model_executor/models/runner_models/loading/meta_load.py"
    meta_source = meta_path.read_text()
    meta_tree = ast.parse(meta_source)
    filler = _class(meta_tree, "_TransformerDiskFiller")
    init_source = ast.get_source_segment(
        meta_source,
        next(
            node for node in filler.body
            if isinstance(node, ast.FunctionDef) and node.name == "__init__"
        ),
    )
    fill_source = ast.get_source_segment(
        meta_source,
        next(
            node for node in filler.body
            if isinstance(node, ast.FunctionDef) and node.name == "fill_block"
        ),
    )
    assert "_collective_source_call" in init_source
    assert "_collective_source_call" in fill_source
    assert "_collective_assert_same_layout" in fill_source
    assert "_collective_reconcile_tensor_specs" in fill_source

    loader = _class(meta_tree, "MemoryEfficientLoader")
    replicated_fill = ast.get_source_segment(
        meta_source,
        next(
            node for node in loader.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_fill_transformer_replicated"
        ),
    )
    assert replicated_fill.index(
        "_save_nonpersistent_buffers(block, device)"
    ) < replicated_fill.index("block.to_empty")
    assert replicated_fill.index("block.to_empty") < replicated_fill.index(
        "_restore_nonpersistent_buffers(nonpersistent_buffers)"
    )

    te_load = ast.get_source_segment(
        meta_source,
        next(
            node for node in loader.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_broadcast_load_component"
        ),
    )
    assert "_collective_source_call" in te_load
    assert te_load.index("_collective_source_call") < te_load.index(
        "set_model_state_dict(block"
    )
    assert "finally:" in te_load
    assert "_release_rank0_source" in te_load

    for method_name in (
        "build_meta_transformer",
        "meta_te_kwargs",
        "meta_te_kwargs_replicated",
    ):
        method_source = ast.get_source_segment(
            meta_source,
            next(
                node for node in loader.body
                if isinstance(node, ast.FunctionDef) and node.name == method_name
            ),
        )
        assert "_collective_build_call" in method_source

    assert "_collective_build_call" in replicated_fill
    assert "quantizing replicated transformer block" in replicated_fill

    quant_helper = next(
        node for node in sharding_tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_collective_quantize_call"
    )
    assert quant_helper is not None
    shard_source = ast.get_source_segment(sharding_source, shard_component)
    assert shard_source.index("_collective_quantize_call") < shard_source.index(
        "fully_shard(block, mesh="
    )
