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
    return _load_module(CONTRACTS_PATH, "roadmap3_loading_contracts")


def test_declaration_defaults_are_explicitly_unsupported(contracts):
    declaration = contracts.LoadDeclaration()

    assert declaration.meta_transformers == ()
    assert declaration.materialization_modes == frozenset(
        {contracts.MaterializationMode.EAGER}
    )
    assert declaration.quantization_backends == frozenset(
        {contracts.QuantizationBackend.NONE}
    )


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
            "use_fp8_gemms": True,
            "use_fp4_gemms": False,
            "use_int8_gemms": True,
        },
    )()

    declaration = contracts.LoadDeclaration.for_runner(
        model_capabilities,
        meta_transformers=("transformer",),
        replicated=True,
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


def test_runner_declaration_does_not_allow_cross_product_backend_pairs(contracts):
    model_capabilities = type(
        "ModelCapabilities",
        (),
        {
            "use_fp8_gemms": False,
            "use_fp4_gemms": False,
            "use_int8_gemms": True,
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
            "use_fp8_gemms": True,
            "use_fp4_gemms": True,
            "use_int8_gemms": False,
            "fully_shard_degree": False,
        },
    )()
    declaration = contracts.LoadDeclaration.for_runner(model_capabilities)

    assert (
        contracts.QuantizationFormat.FP8_FP4,
        contracts.QuantizationBackend.AITER,
    ) in declaration.quantization_contracts
    assert (
        contracts.QuantizationFormat.FP8_FP4,
        contracts.QuantizationBackend.TORCHAO,
    ) in declaration.quantization_contracts


def test_int8_still_conflicts_with_hybrid_quantization(contracts):
    config = type(
        "Config",
        (),
        {
            "use_fp8_gemms": True,
            "use_fp4_gemms": True,
            "use_int8_gemms": True,
        },
    )()

    with pytest.raises(
        contracts.UnsupportedLoadContract,
        match=r"INT8.*FP8.*FP4",
    ):
        contracts.select_runtime_quantization(
            config, aiter_fp8_active=True, cuda_active=False
        )


def test_fsdp_and_replicated_meta_support_are_derived_separately(contracts):
    capable = type(
        "ModelCapabilities",
        (),
        {
            "use_fp8_gemms": False,
            "use_fp4_gemms": False,
            "use_int8_gemms": False,
            "fully_shard_degree": True,
        },
    )()
    replicated_only = type(
        "ModelCapabilities",
        (),
        {
            "use_fp8_gemms": False,
            "use_fp4_gemms": False,
            "use_int8_gemms": False,
            "fully_shard_degree": False,
        },
    )()
    strategy = {"transformer": {"wrap_attrs": ["blocks"]}}

    both = contracts.LoadDeclaration.for_runner(
        capable,
        meta_transformers=("transformer",),
        replicated=True,
        fsdp_strategy=strategy,
    )
    only_replicated = contracts.LoadDeclaration.for_runner(
        replicated_only,
        meta_transformers=("transformer",),
        replicated=True,
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


@pytest.mark.parametrize(
    ("flags", "aiter_fp8", "cuda", "expected"),
    [
        (
            {},
            False,
            False,
            ("NONE", "NONE"),
        ),
        (
            {"use_fp8_gemms": True},
            True,
            False,
            ("FP8", "AITER"),
        ),
        (
            {"use_fp8_gemms": True},
            False,
            True,
            ("FP8", "TORCHAO"),
        ),
        (
            {"use_fp4_gemms": True},
            False,
            False,
            ("FP4", "AITER"),
        ),
        (
            {"use_fp4_gemms": True},
            False,
            True,
            ("FP4", "TORCHAO"),
        ),
        (
            {"use_fp8_gemms": True, "use_fp4_gemms": True},
            True,
            False,
            ("FP8_FP4", "AITER"),
        ),
        (
            {"use_int8_gemms": True},
            False,
            True,
            ("INT8", "TORCHAO"),
        ),
    ],
)
def test_runtime_quantization_selection(contracts, flags, aiter_fp8, cuda, expected):
    config = type(
        "Config",
        (),
        {
            "use_fp8_gemms": False,
            "use_fp4_gemms": False,
            "use_int8_gemms": False,
            **flags,
        },
    )()

    requested, backend = contracts.select_runtime_quantization(
        config, aiter_fp8_active=aiter_fp8, cuda_active=cuda
    )

    assert (requested.name, backend.name) == expected
