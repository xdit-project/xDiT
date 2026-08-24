"""Dependency-free structural checks on the load path: who owns a quantization flag,
which helpers the collective fill requires, and what survives a disk fill."""

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _class(tree, name):
    return next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == name
    )


def test_meta_load_defines_collective_layout_and_persistent_buffer_helpers():
    path = ROOT / "xfuser/model_executor/models/runner_models/loading/meta_load.py"
    tree = ast.parse(path.read_text())
    functions = {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
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
        node
        for node in base_class.body
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
        node
        for node in args_class.body
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
        node
        for node in base_class.body
        if isinstance(node, ast.FunctionDef) and node.name == "_validate_config"
    )
    assert "Cannot use int8 gemms with fp8 or fp4 gemms." not in ast.get_source_segment(
        base_source, base_validate
    )


def test_mxfp4_quantized_weight_is_registered_as_parameter():
    path = ROOT / "xfuser/model_executor/layers/mxfp4_linear.py"
    tree = ast.parse(path.read_text())
    cls = _class(tree, "xFuserMXFP4Linear")
    quantize = next(
        node
        for node in cls.body
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
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == "_load_from_state_dict"
    )
    assert "destination_device" in ast.get_source_segment(path.read_text(), load_state)
    assert "packed state cannot be loaded after FSDP" in ast.get_source_segment(
        path.read_text(), load_state
    )
    load_source = ast.get_source_segment(path.read_text(), load_state)
    assert (
        "full-precision state cannot replace an FSDP-managed packed parameter"
        in load_source
    )
    assert "incoming_device" in load_source
    assert any(
        isinstance(node, ast.FunctionDef) and node.name == "_is_fsdp_managed_parameter"
        for node in cls.body
    )


def test_disk_fill_preserves_nonpersistent_buffers_and_reports_source_errors():
    sharding_path = ROOT / "xfuser/core/distributed/sharding.py"
    sharding_source = sharding_path.read_text()
    sharding_tree = ast.parse(sharding_source)
    functions = {
        node.name for node in sharding_tree.body if isinstance(node, ast.FunctionDef)
    }
    assert {
        "_save_nonpersistent_buffers",
        "_restore_nonpersistent_buffers",
    } <= functions
    shard_component = next(
        node
        for node in sharding_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "shard_component"
    )
    shard_source = ast.get_source_segment(sharding_source, shard_component)
    assert shard_source.index("_restore_nonpersistent_buffers") < shard_source.index(
        "load_block_fn(block, i)"
    )

    meta_path = ROOT / "xfuser/model_executor/models/runner_models/loading/meta_load.py"
    meta_source = meta_path.read_text()
    meta_tree = ast.parse(meta_source)
    filler = _class(meta_tree, "_BlockwiseDiskFiller")
    init_source = ast.get_source_segment(
        meta_source,
        next(
            node
            for node in filler.body
            if isinstance(node, ast.FunctionDef) and node.name == "__init__"
        ),
    )
    fill_source = ast.get_source_segment(
        meta_source,
        next(
            node
            for node in filler.body
            if isinstance(node, ast.FunctionDef) and node.name == "fill_block"
        ),
    )
    read_source = ast.get_source_segment(
        meta_source,
        next(
            node
            for node in filler.body
            if isinstance(node, ast.FunctionDef) and node.name == "_read_tensors"
        ),
    )
    share_source = ast.get_source_segment(
        meta_source,
        next(
            node
            for node in filler.body
            if isinstance(node, ast.FunctionDef) and node.name == "_share_from_source"
        ),
    )
    # Source reads may call _collective_source_call directly or go through the
    # _source_call wrapper; both names end in _source_call. The wrapper is only
    # equivalent if it still delegates, which is asserted below.
    # __init__ resolves the checkpoint map through _share_from_source, which broadcasts the result so
    # every rank can read a block, and which is only guarded if it still delegates to _source_call.
    assert "_share_from_source" in init_source
    assert "_source_call" in share_source
    # fill_block batches a block's reads into _read_tensors so they share one status exchange
    # instead of paying a pickled collective each, so the guard lives one level down.
    assert "_read_tensors" in fill_source
    assert "_source_call" in read_source
    assert "_assert_same_layout" in fill_source
    assert "_reconcile_tensor_specs" in fill_source

    source_call = ast.get_source_segment(
        meta_source,
        next(
            node
            for node in filler.body
            if isinstance(node, ast.FunctionDef) and node.name == "_source_call"
        ),
    )
    assert "_collective_source_call" in source_call

    loader = _class(meta_tree, "ModelLoader")

    def _loader_method(method_name):
        return ast.get_source_segment(
            meta_source,
            next(
                node
                for node in loader.body
                if isinstance(node, ast.FunctionDef) and node.name == method_name
            ),
        )

    # The replicated and local paths share one block loop, so the buffer
    # save/restore ordering is asserted there rather than once per caller.
    for entry_point in ("_fill_transformer_replicated", "fill_transformer_local"):
        assert "self._fill_blocks(" in _loader_method(entry_point)

    block_fill = _loader_method("_fill_blocks")
    assert block_fill.index("_save_nonpersistent_buffers(block, device)") < block_fill.index(
        "block.to_empty"
    )
    assert block_fill.index("block.to_empty") < block_fill.index(
        "_restore_nonpersistent_buffers(nonpersistent_buffers)"
    )
    assert block_fill.index(
        "_restore_nonpersistent_buffers(nonpersistent_buffers)"
    ) < block_fill.index("fill_block(block, i)")

    te_load = ast.get_source_segment(
        meta_source,
        next(
            node
            for node in loader.body
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

    # Every meta construction must be collectively agreed, so a rank that fails to build fails the
    # run rather than diverging into a different collective. te_blockwise_routes is where the FSDP
    # path builds its text encoders, since the route decision and the construction are one step.
    for method_name in (
        "build_meta_transformer",
        "te_blockwise_routes",
        "meta_te_kwargs_replicated",
    ):
        assert "_collective_build_call" in _loader_method(method_name)

    assert "_collective_build_call" in block_fill
    assert "quantizing replicated transformer block" in block_fill

    quant_helper = next(
        node
        for node in sharding_tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_collective_quantize_call"
    )
    assert quant_helper is not None

    # A block has to be quantized before it is sharded: afterwards its weight is a DTensor holding
    # one rank's slice, which no quantizer can scale. Compared by position in the tree rather than in
    # the text so reformatting the call cannot quietly retire the contract.
    def _call_line(predicate):
        return min(
            node.lineno
            for node in ast.walk(shard_component)
            if isinstance(node, ast.Call) and predicate(node)
        )

    def _is_block_shard(node):
        return (
            isinstance(node.func, ast.Name)
            and node.func.id == "fully_shard"
            and node.args
            and isinstance(node.args[0], ast.Name)
            and node.args[0].id == "block"
        )

    quantize_line = _call_line(
        lambda node: isinstance(node.func, ast.Name)
        and node.func.id == "_collective_quantize_call"
    )
    assert quantize_line < _call_line(_is_block_shard)
