"""Dependency-light tests for load/quantization contract selection."""

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
CONTRACTS_PATH = (
    ROOT / "xfuser/model_executor/models/runner_models/loading/contracts.py"
)


def _load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def contracts():
    return _load_module(CONTRACTS_PATH, "loading_contracts_under_test")


def test_declaration_defaults_are_explicitly_unsupported(contracts):
    declaration = contracts.LoadDeclaration()

    assert declaration.meta_transformers == ()
    assert declaration.meta_text_encoders == ()
    assert declaration.materialization_modes == frozenset(
        {contracts.MaterializationMode.EAGER}
    )
    assert declaration.quantization_backends == frozenset(
        {contracts.QuantizationBackend.NONE}
    )


def test_load_support_is_frozen_static_intent(contracts):
    spec = contracts.LoadSupport(replicated_meta=True)

    with pytest.raises(AttributeError):
        spec.replicated_meta = False


def test_declared_meta_mode_requires_every_transformer_in_strategy(contracts):
    declaration = contracts.LoadDeclaration.meta(
        "transformer", "transformer_2", replicated=True
    )

    with pytest.raises(
        contracts.UnsupportedLoadContract,
        match=r"transformer_2.*fsdp_strategy",
    ):
        contracts.validate_materialization_contract(
            declaration,
            contracts.MaterializationMode.FSDP_META,
            {"transformer": {"wrap_attrs": ["blocks"]}},
            runner_name="ExampleRunner",
        )


def test_declared_meta_mode_requires_a_construction_seam(contracts):
    declaration = contracts.LoadDeclaration(
        fsdp_meta_transformers=("transformer",),
        materialization_modes=frozenset(
            {
                contracts.MaterializationMode.EAGER,
                contracts.MaterializationMode.FSDP_META,
            }
        ),
    )

    with pytest.raises(
        contracts.UnsupportedLoadContract,
        match=r"ExampleRunner.*construction seam",
    ):
        contracts.validate_materialization_contract(
            declaration,
            contracts.MaterializationMode.FSDP_META,
            {"transformer": {"wrap_attrs": ["blocks"]}},
            runner_name="ExampleRunner",
        )


def test_contract_selection_accepts_only_declared_backend_and_mode(contracts):
    declaration = contracts.LoadDeclaration.meta(
        "transformer",
        replicated=True,
        quantization_backends={
            contracts.QuantizationBackend.NONE,
            contracts.QuantizationBackend.AITER,
        },
        quantization_formats={
            contracts.QuantizationFormat.NONE,
            contracts.QuantizationFormat.FP8,
        },
    )

    selected = contracts.select_load_contract(
        requested_format=contracts.QuantizationFormat.FP8,
        selected_backend=contracts.QuantizationBackend.AITER,
        materialization_mode=contracts.MaterializationMode.REPLICATED_META,
        declaration=declaration,
        fsdp_strategy={"transformer": {"wrap_attrs": ["blocks"]}},
        runner_name="ExampleRunner",
    )

    assert selected.requested_format is contracts.QuantizationFormat.FP8
    assert selected.selected_backend is contracts.QuantizationBackend.AITER
    assert (
        selected.materialization_mode is contracts.MaterializationMode.REPLICATED_META
    )


def test_contract_selection_rejects_unsupported_pair_before_runtime(contracts):
    declaration = contracts.LoadDeclaration.meta("transformer")

    with pytest.raises(
        contracts.UnsupportedLoadContract,
        match=r"TORCHAO.*FP8.*ExampleRunner",
    ):
        contracts.select_load_contract(
            requested_format=contracts.QuantizationFormat.FP8,
            selected_backend=contracts.QuantizationBackend.TORCHAO,
            materialization_mode=contracts.MaterializationMode.FSDP_META,
            declaration=declaration,
            fsdp_strategy={"transformer": {"wrap_attrs": ["blocks"]}},
            runner_name="ExampleRunner",
        )


def test_runner_declaration_derives_quantization_contracts(contracts):
    model_capabilities = type(
        "ModelCapabilities",
        (),
        {
            "gemm_formats": frozenset({"fp8", "int8"}),
        },
    )()

    declaration = contracts.LoadDeclaration.for_runner(
        model_capabilities,
        load_support=contracts.LoadSupport(
            meta_transformers=("transformer",),
            meta_text_encoders=("text_encoder",),
            replicated_meta=True,
            routes=contracts.STANDARD_LOAD_ROUTES,
        ),
        fsdp_strategy={"transformer": {"wrap_attrs": ["blocks"]}},
    )

    assert declaration.quantization_contracts == frozenset(
        {
            (
                contracts.QuantizationFormat.NONE,
                contracts.QuantizationBackend.NONE,
            ),
            (
                contracts.QuantizationFormat.FP8,
                contracts.QuantizationBackend.AITER,
            ),
            (
                contracts.QuantizationFormat.FP8,
                contracts.QuantizationBackend.TORCHAO,
            ),
            (
                contracts.QuantizationFormat.INT8,
                contracts.QuantizationBackend.TORCHAO,
            ),
        }
    )
    assert declaration.meta_text_encoders == ("text_encoder",)


def test_runner_declaration_does_not_allow_cross_product_backend_pairs(contracts):
    model_capabilities = type(
        "ModelCapabilities",
        (),
        {
            "gemm_formats": frozenset({"int8"}),
        },
    )()
    declaration = contracts.LoadDeclaration.for_runner(model_capabilities)

    with pytest.raises(
        contracts.UnsupportedLoadContract,
        match=r"AITER.*INT8.*ExampleRunner",
    ):
        contracts.select_load_contract(
            requested_format=contracts.QuantizationFormat.INT8,
            selected_backend=contracts.QuantizationBackend.AITER,
            materialization_mode=contracts.MaterializationMode.EAGER,
            declaration=declaration,
            fsdp_strategy={},
            runner_name="ExampleRunner",
        )


def test_fp8_fp4_hybrid_is_an_explicit_valid_contract(contracts):
    model_capabilities = type(
        "ModelCapabilities",
        (),
        {
            "gemm_formats": frozenset({"fp8", "fp4"}),
            "fully_shard_degree": False,
        },
    )()
    declaration = contracts.LoadDeclaration.for_runner(model_capabilities)

    # FP8 and FP4 are separate contracts: a tiered run takes the FP4 one and
    # places its FP8 tier through the blockwise converter.
    assert (
        contracts.QuantizationFormat.FP4,
        contracts.QuantizationBackend.AITER,
    ) in declaration.quantization_contracts
    assert (
        contracts.QuantizationFormat.FP8,
        contracts.QuantizationBackend.TORCHAO,
    ) in declaration.quantization_contracts


def test_int8_cannot_be_tiered_with_another_format():
    """`select_runtime_quantization` no longer sees impossible combinations:
    a spec that names them is refused when the run's options are validated."""
    from xfuser.config.args import xFuserArgs

    args = xFuserArgs(model="m", gemm_quantization="low=int8,high=fp8")
    with pytest.raises(ValueError, match="INT8 cannot be tiered"):
        args._validate_gemm_quantization_flags()


def test_fsdp_and_replicated_meta_support_are_derived_separately(contracts):
    capable = type(
        "ModelCapabilities",
        (),
        {
            "gemm_formats": frozenset({}),
            "fully_shard_degree": True,
        },
    )()
    replicated_only = type(
        "ModelCapabilities",
        (),
        {
            "gemm_formats": frozenset({}),
            "fully_shard_degree": False,
        },
    )()
    strategy = {"transformer": {"wrap_attrs": ["blocks"]}}

    both = contracts.LoadDeclaration.for_runner(
        capable,
        load_support=contracts.LoadSupport(
            meta_transformers=("transformer",),
            replicated_meta=True,
            routes=contracts.STANDARD_LOAD_ROUTES,
        ),
        fsdp_strategy=strategy,
    )
    only_replicated = contracts.LoadDeclaration.for_runner(
        replicated_only,
        load_support=contracts.LoadSupport(
            meta_transformers=("transformer",),
            replicated_meta=True,
            routes=contracts.STANDARD_LOAD_ROUTES,
        ),
        fsdp_strategy=strategy,
    )

    assert both.fsdp_meta_transformers == ("transformer",)
    assert both.replicated_meta_transformers == ("transformer",)
    assert only_replicated.fsdp_meta_transformers == ()
    assert only_replicated.replicated_meta_transformers == ("transformer",)
    assert contracts.MaterializationMode.FSDP_META not in (
        only_replicated.materialization_modes
    )


@pytest.mark.parametrize(
    ("world_size", "degrees", "expected"),
    [
        (1, {}, "EAGER"),
        (2, {}, "REPLICATED_META"),
        (2, {"fully_shard_degree": 2}, "EAGER"),
        (2, {"pipefusion_parallel_degree": 2}, "EAGER"),
        (2, {"tensor_parallel_degree": 2}, "EAGER"),
    ],
)
def test_effective_replicated_mode_applies_runtime_exclusions(
    contracts, world_size, degrees, expected
):
    config = type(
        "Config",
        (),
        {
            "memory_efficient_sharding": False,
            "memory_efficient_replicated_load": True,
            "fully_shard_degree": 1,
            "pipefusion_parallel_degree": 1,
            "tensor_parallel_degree": 1,
            **degrees,
        },
    )()

    mode = contracts.select_effective_materialization_mode(
        config, world_size=world_size
    )

    assert mode.name == expected


@pytest.mark.parametrize(
    ("degrees", "world_size", "expected_in_reason"),
    [
        ({"fully_shard_degree": 8}, 8, "--fully_shard_degree"),
        ({"tensor_parallel_degree": 2}, 8, "--tensor_parallel_degree"),
        ({"pipefusion_parallel_degree": 2}, 8, "--pipefusion_parallel_degree"),
    ],
)
def test_a_replicated_request_that_would_be_dropped_is_refused(
    contracts, degrees, world_size, expected_in_reason
):
    """Silently returning an eager load reads as the feature being on and doing nothing."""
    config = type(
        "Config",
        (),
        {
            "memory_efficient_sharding": False,
            "memory_efficient_replicated_load": True,
            "fully_shard_degree": 1,
            "pipefusion_parallel_degree": 1,
            "tensor_parallel_degree": 1,
            **degrees,
        },
    )()

    with pytest.raises(contracts.UnsupportedLoadContract) as refusal:
        contracts.assert_requested_materialization_is_honoured(
            config, world_size=world_size
        )

    assert expected_in_reason in str(refusal.value)


@pytest.mark.parametrize("world_size", [1, 8])
def test_a_request_nothing_contradicts_is_allowed_through(contracts, world_size):
    """A single rank degrades to eager rather than failing, so the same command line still runs."""
    config = type(
        "Config",
        (),
        {
            "memory_efficient_sharding": False,
            "memory_efficient_replicated_load": True,
            "fully_shard_degree": 1,
            "pipefusion_parallel_degree": 1,
            "tensor_parallel_degree": 1,
        },
    )()

    contracts.assert_requested_materialization_is_honoured(
        config, world_size=world_size
    )


def _offload_config(**flags):
    defaults = {
        "enable_group_cpu_offload": False,
        "enable_sequential_cpu_offload": False,
        "enable_model_cpu_offload": False,
        "group_offload_low_cpu_mem": False,
        "fully_shard_degree": 1,
    }
    return type("Config", (), {**defaults, **flags})()


# Which offload a single format's storage survives moved onto the adapter:
# `group_offload_refusal` carries the measured reason, and the loop that reads
# it lives in test_fp8_blockwise_hybrid_routing.py, where an adapter exists.
# What is left here is the one claim about a *compound* contract.


@pytest.mark.parametrize(
    "flag",
    [
        "enable_model_cpu_offload",
        "enable_sequential_cpu_offload",
        "enable_group_cpu_offload",
    ],
)
def test_every_offload_mode_is_refused_for_mixed_fp4_fp6(contracts, flag):
    config = _offload_config(**{flag: True})

    with pytest.raises(contracts.UnsupportedLoadContract, match="MXFP4 packing"):
        contracts.assert_offload_is_compatible_with_format(
            config,
            requested_format=contracts.QuantizationFormat.FP4_FP6,
            selected_backend=contracts.QuantizationBackend.AITER,
        )


@pytest.mark.parametrize(
    ("format_name", "backend_name", "offload"),
    [
        ("FP4", "AITER", True),
        ("FP8", "AITER", True),
        ("FP4", "TORCHAO", True),
        ("FP4_FP6", "TORCHAO", True),
    ],
)
def test_a_single_format_contract_is_left_to_its_adapter(
    contracts, format_name, backend_name, offload
):
    """Only the compound claim lives here; the rest is the adapter's answer."""
    config = _offload_config(enable_group_cpu_offload=offload)

    contracts.assert_offload_is_compatible_with_format(
        config,
        requested_format=getattr(contracts.QuantizationFormat, format_name),
        selected_backend=getattr(contracts.QuantizationBackend, backend_name),
    )


@pytest.mark.parametrize(
    ("flag", "expected_in_reason"),
    [
        ("enable_group_cpu_offload", "is_pinned"),
        ("enable_sequential_cpu_offload", "DTensor spec"),
    ],
)
def test_offload_that_reaches_inside_a_sharded_parameter_is_refused(
    contracts, flag, expected_in_reason
):
    """Both fail deep in the hook mid-denoise, long after the load being watched."""
    config = _offload_config(fully_shard_degree=4, **{flag: True})

    with pytest.raises(contracts.UnsupportedLoadContract) as refusal:
        contracts.assert_offload_is_compatible_with_sharding(config)

    assert expected_in_reason in str(refusal.value)


@pytest.mark.parametrize(
    ("flags", "shard_degree"),
    [
        ({"enable_model_cpu_offload": True}, 4),
        ({"enable_group_cpu_offload": True}, 1),
        ({"enable_sequential_cpu_offload": True}, 1),
    ],
)
def test_offload_that_was_measured_working_is_allowed(contracts, flags, shard_degree):
    """Whole-model offload moves components rather than parameters, and it ran sharded."""
    config = _offload_config(fully_shard_degree=shard_degree, **flags)

    contracts.assert_offload_is_compatible_with_sharding(config)


@pytest.mark.parametrize(
    ("raw", "aiter_fp8", "cuda", "expected"),
    [
        ("none", False, False, ("NONE", "NONE")),
        ("fp8", True, False, ("FP8", "AITER")),
        ("fp8", False, True, ("FP8", "TORCHAO")),
        ("fp4", False, False, ("FP4", "AITER")),
        ("fp4", False, True, ("FP4", "TORCHAO")),
        # A tiered fp4/fp8 run is an FP4 contract; the FP8 tier is placed by
        # the blockwise converter, not by a contract of its own.
        ("low=fp4,high=fp8", True, False, ("FP4", "AITER")),
        ("int8", False, True, ("INT8", "TORCHAO")),
    ],
)
def test_runtime_quantization_selection(contracts, raw, aiter_fp8, cuda, expected):
    from xfuser.config.gemm import GemmQuantizationSpec

    requested, backend = contracts.select_runtime_quantization(
        GemmQuantizationSpec.parse(raw),
        aiter_fp8_active=aiter_fp8,
        cuda_active=cuda,
    )

    assert (requested.name, backend.name) == expected
