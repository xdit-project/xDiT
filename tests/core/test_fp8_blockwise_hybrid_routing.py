"""Routing tests for FP8-only blocks inside FP4/hybrid materialization."""

from types import SimpleNamespace

import pytest

from xfuser.model_executor.models.runner_models import base_model
from xfuser.model_executor.models.runner_models.base_model import xFuserModel
from xfuser.model_executor.models.runner_models.loading import shard
from xfuser.model_executor.models.runner_models.loading import fp8_backends
from xfuser.model_executor.models.runner_models.loading.contracts import (
    MaterializationMode,
    QuantizationBackend,
    QuantizationFormat,
)
from xfuser.model_executor.models.runner_models.loading.fp8_backends import (
    Fp8BackendCapabilities,
    select_blockwise_fp8_backend,
)


class RecordingAdapter:
    def __init__(self):
        self.calls = []

    def convert_block(self, block, *, device):
        self.calls.append((block, device))


def _hybrid_model(*, adapter, overrides=(), suffixes=None):
    return SimpleNamespace(
        config=SimpleNamespace(
            use_fp4_gemms=True,
            use_fp8_gemms=False,
            use_int8_gemms=False,
            use_hybrid_gemm_schedule=False,
        ),
        settings=SimpleNamespace(
            fp4_gemm_module_list=["transformer.blocks"],
            fp8_precision_overrides=overrides,
            fp8_precision_override_suffixes=suffixes,
            int8_gemm_module_list=None,
        ),
        fp8=SimpleNamespace(
            module_list=lambda: [
                "transformer.blocks",
                "transformer_2.blocks",
            ]
        ),
        blockwise_fp8_backend=adapter,
    )


def test_wan_fp8_only_second_transformer_uses_blockwise_backend(monkeypatch):
    adapter = RecordingAdapter()
    model = _hybrid_model(adapter=adapter)
    fp4_calls = []
    monkeypatch.setattr(
        shard,
        "quantize_linear_layers_to_fp4",
        lambda *args, **kwargs: fp4_calls.append((args, kwargs)),
    )
    monkeypatch.setattr(
        shard,
        "quantize_linear_layers_to_nvfp4",
        lambda *args, **kwargs: fp4_calls.append((args, kwargs)),
    )

    quantize = shard.build_block_quantize_fn(
        model, "transformer_2", ["blocks"], local_rank=3
    )
    block = object()
    quantize(block, 0)

    assert adapter.calls == [(block, "cuda:3")]
    assert fp4_calls == []


def test_fp4_target_keeps_precision_overrides_in_fp4_owner(monkeypatch):
    adapter = RecordingAdapter()
    model = _hybrid_model(
        adapter=adapter,
        overrides=("3.attn.proj", "8.mlp"),
        suffixes=(".net.0.proj",),
    )
    calls = []
    monkeypatch.setattr(shard, "_is_cuda", lambda: True)
    monkeypatch.setattr(
        shard,
        "quantize_linear_layers_to_nvfp4",
        lambda block, **kwargs: calls.append((block, kwargs)),
    )

    quantize = shard.build_block_quantize_fn(
        model, "transformer", ["blocks"], local_rank=1
    )
    block = object()
    quantize(block, 3)

    assert calls == [
        (
            block,
            {
                "fp8_layers": ("attn.proj",),
                "fp8_suffix_layers": (".net.0.proj",),
                "device": "cuda:1",
            },
        )
    ]
    assert adapter.calls == []


def test_pure_fp4_wan_targets_require_backend_preflight():
    model = _hybrid_model(adapter=RecordingAdapter())

    assert xFuserModel._requires_blockwise_fp8_backend(model)


def test_eager_fp4_with_fp8_only_target_preflights_component_backend():
    model = _hybrid_model(adapter=RecordingAdapter())
    model.load_contract = SimpleNamespace(
        requested_format=QuantizationFormat.FP4,
        materialization_mode=MaterializationMode.EAGER,
    )
    model._requires_blockwise_fp8_backend = (
        lambda: xFuserModel._requires_blockwise_fp8_backend(model)
    )

    assert xFuserModel._uses_blockwise_fp8_backend(model)


def _fsdp_patch_model(
    *,
    strategy,
    fp4_targets=("transformer.blocks",),
    fp8_targets=("transformer.blocks", "transformer_2.blocks"),
    prefixes=(),
    suffixes=(),
    hybrid=False,
    fully_shard_degree=2,
):
    model = SimpleNamespace(
        config=SimpleNamespace(
            fully_shard_degree=fully_shard_degree,
            use_fp4_gemms=True,
            use_hybrid_gemm_schedule=hybrid,
        ),
        settings=SimpleNamespace(
            fsdp_strategy=strategy,
            fp4_gemm_module_list=list(fp4_targets),
            fp8_precision_overrides=prefixes,
            fp8_precision_override_suffixes=suffixes,
        ),
        fp8=SimpleNamespace(module_list=lambda: list(fp8_targets)),
    )
    model._places_torchao_tensor_subclass_under_fsdp2 = (
        lambda adapter, **kwargs: (
            xFuserModel._places_torchao_tensor_subclass_under_fsdp2(
                model, adapter, **kwargs
            )
        )
    )
    return model


def test_fp8_only_target_outside_fsdp_strategy_needs_no_torchao_patches():
    model = _fsdp_patch_model(
        strategy={"transformer": {"wrap_attrs": ["blocks"]}},
    )
    adapter = SimpleNamespace(backend=QuantizationBackend.TORCHAO)

    assert not xFuserModel._places_torchao_tensor_subclass_under_fsdp2(
        model, adapter
    )


def test_fsdp_sharded_fp8_only_torchao_target_needs_patches():
    model = _fsdp_patch_model(
        strategy={"transformer_2": {"wrap_attrs": ["blocks"]}},
    )
    adapter = SimpleNamespace(backend=QuantizationBackend.TORCHAO)

    assert xFuserModel._places_torchao_tensor_subclass_under_fsdp2(
        model, adapter
    )


def test_fsdp_sharded_fp8_only_aiter_target_needs_no_torchao_patches():
    model = _fsdp_patch_model(
        strategy={"transformer_2": {"wrap_attrs": ["blocks"]}},
    )
    adapter = SimpleNamespace(backend=QuantizationBackend.AITER)

    assert not xFuserModel._places_torchao_tensor_subclass_under_fsdp2(
        model, adapter
    )


@pytest.mark.parametrize(
    ("prefixes", "suffixes", "hybrid"),
    [
        (("0.",), (), False),
        ((), (".net.0.proj",), False),
        ((), (), True),
    ],
)
def test_fsdp_sharded_fp4_torchao_fp8_paths_need_patches(
    prefixes,
    suffixes,
    hybrid,
):
    model = _fsdp_patch_model(
        strategy={"transformer": {"wrap_attrs": ["blocks"]}},
        prefixes=prefixes,
        suffixes=suffixes,
        hybrid=hybrid,
    )
    adapter = SimpleNamespace(backend=QuantizationBackend.AITER)

    assert xFuserModel._places_torchao_tensor_subclass_under_fsdp2(
        model, adapter
    )


def test_fp4_torchao_fp8_paths_outside_fsdp_strategy_need_no_patches():
    model = _fsdp_patch_model(
        strategy={"transformer_2": {"wrap_attrs": ["blocks"]}},
        prefixes=("0.",),
    )
    adapter = SimpleNamespace(backend=QuantizationBackend.AITER)

    assert not xFuserModel._places_torchao_tensor_subclass_under_fsdp2(
        model, adapter
    )


def test_fsdp_fp4_override_triggers_startup_backend_preflight():
    model = _fsdp_patch_model(
        strategy={"transformer": {"wrap_attrs": ["blocks"]}},
        fp8_targets=("transformer.blocks",),
        prefixes=("0.",),
    )
    model.load_contract = SimpleNamespace(
        requested_format=QuantizationFormat.FP4,
        materialization_mode=MaterializationMode.FSDP_META,
    )
    model._requires_blockwise_fp8_backend = (
        lambda: xFuserModel._requires_blockwise_fp8_backend(model)
    )

    assert xFuserModel._uses_blockwise_fp8_backend(model)


def test_eager_fsdp_fp8_triggers_startup_backend_preflight():
    model = _fsdp_patch_model(
        strategy={"transformer": {"wrap_attrs": ["blocks"]}},
        fp4_targets=(),
        fp8_targets=("transformer.blocks",),
    )
    model.config.use_fp4_gemms = False
    model.load_contract = SimpleNamespace(
        requested_format=QuantizationFormat.FP8,
        materialization_mode=MaterializationMode.EAGER,
    )
    model._requires_blockwise_fp8_backend = (
        lambda: xFuserModel._requires_blockwise_fp8_backend(model)
    )

    assert xFuserModel._uses_blockwise_fp8_backend(model)


def test_fp4_override_outside_strategy_skips_startup_backend_preflight():
    model = _fsdp_patch_model(
        strategy={"transformer_2": {"wrap_attrs": ["blocks"]}},
        fp8_targets=("transformer.blocks",),
        prefixes=("0.",),
    )
    model.load_contract = SimpleNamespace(
        requested_format=QuantizationFormat.FP4,
        materialization_mode=MaterializationMode.FSDP_META,
    )
    model._requires_blockwise_fp8_backend = (
        lambda: xFuserModel._requires_blockwise_fp8_backend(model)
    )

    assert not xFuserModel._uses_blockwise_fp8_backend(model)


def test_backend_preflight_uses_component_target_requirement(monkeypatch):
    observed = []
    adapter = SimpleNamespace(backend=QuantizationBackend.TORCHAO)
    monkeypatch.setattr(
        fp8_backends,
        "probe_fp8_backend_capabilities",
        lambda: Fp8BackendCapabilities(),
    )

    def select_backend(contract, *, capabilities):
        return adapter

    monkeypatch.setattr(
        fp8_backends,
        "select_blockwise_fp8_backend",
        select_backend,
    )
    monkeypatch.setattr(
        fp8_backends,
        "validate_torchao_fsdp2_patches",
        lambda contract, *, capabilities, required: observed.append(required),
    )
    model = _fsdp_patch_model(
        strategy={"transformer_2": {"wrap_attrs": ["blocks"]}},
    )
    model.load_contract = SimpleNamespace(
        requested_format=QuantizationFormat.FP4,
        selected_backend=QuantizationBackend.TORCHAO,
        materialization_mode=MaterializationMode.FSDP_META,
    )

    selected = xFuserModel.blockwise_fp8_backend.func(model)

    assert selected is adapter
    assert observed == [True]


def test_component_outside_fsdp_strategy_does_not_block_startup(monkeypatch):
    monkeypatch.setattr(
        fp8_backends,
        "probe_fp8_backend_capabilities",
        lambda: Fp8BackendCapabilities(
            torchao_fp8=True,
            torchao_fsdp_patches=False,
            torchao_fsdp_reason="patches unavailable",
        ),
    )
    model = _fsdp_patch_model(
        strategy={"transformer": {"wrap_attrs": ["blocks"]}},
    )
    model.load_contract = SimpleNamespace(
        requested_format=QuantizationFormat.FP4,
        selected_backend=QuantizationBackend.TORCHAO,
    )

    adapter = xFuserModel.blockwise_fp8_backend.func(model)

    assert adapter.backend is QuantizationBackend.TORCHAO


def test_fsdp_boundary_has_no_global_fp8_patch_assertion(monkeypatch):
    from xfuser.core.utils import runner_utils

    monkeypatch.setattr(shard, "_is_cuda", lambda: True)
    monkeypatch.setattr(runner_utils, "_TORCHAO_FLOAT8_FSDP2_PATCHES", [])
    monkeypatch.setattr(
        shard,
        "get_world_group",
        lambda: SimpleNamespace(local_rank=0),
    )
    monkeypatch.setattr(
        shard,
        "get_fs_group",
        lambda: SimpleNamespace(local_rank=0, device_group=object()),
    )
    model = SimpleNamespace(
        config=SimpleNamespace(use_fp8_gemms=True),
        settings=SimpleNamespace(fsdp_strategy={}),
        pipe=SimpleNamespace(components={}),
        _loader=object(),
    )

    shard.shard_pipeline_components(model)


def test_fp4_override_under_fsdp_requires_patches_with_aiter_fp8_backend(
    monkeypatch,
):
    monkeypatch.setattr(
        fp8_backends,
        "probe_fp8_backend_capabilities",
        lambda: Fp8BackendCapabilities(
            aiter_block_scale=True,
            torchao_fp8=True,
            torchao_fsdp_patches=False,
            torchao_fsdp_reason="missing fsdp_post_all_gather",
        ),
    )
    model = _fsdp_patch_model(
        strategy={"transformer": {"wrap_attrs": ["blocks"]}},
        prefixes=("0.",),
    )
    model.load_contract = SimpleNamespace(
        requested_format=QuantizationFormat.FP4,
        selected_backend=QuantizationBackend.AITER,
    )

    with pytest.raises(ValueError, match=r"FSDP.*fsdp_post_all_gather"):
        xFuserModel.blockwise_fp8_backend.func(model)


@pytest.mark.parametrize(
    (
        "platform",
        "fp4_backend",
        "aiter_fp8",
        "setup_name",
        "expected_fp8_backend",
    ),
    [
        (
            "cuda",
            QuantizationBackend.TORCHAO,
            False,
            "_setup_nvfp4_gemms",
            QuantizationBackend.TORCHAO,
        ),
        (
            "rdna4_rocm",
            QuantizationBackend.AITER,
            True,
            "_setup_mxfp4_gemms",
            QuantizationBackend.AITER,
        ),
        (
            "other_rocm",
            QuantizationBackend.AITER,
            False,
            "_setup_mxfp4_gemms",
            QuantizationBackend.TORCHAO,
        ),
    ],
)
def test_eager_fp4_routes_fp8_only_module_by_hardware(
    monkeypatch,
    platform,
    fp4_backend,
    aiter_fp8,
    setup_name,
    expected_fp8_backend,
):
    contract = SimpleNamespace(
        requested_format=QuantizationFormat.FP4,
        selected_backend=fp4_backend,
    )
    adapter = select_blockwise_fp8_backend(
        contract,
        capabilities=Fp8BackendCapabilities(
            aiter_block_scale=aiter_fp8,
            torchao_fp8=True,
            torchao_fsdp_patches=True,
        ),
    )
    fp4_module = object()
    fp8_module = object()
    fp4_calls = []
    fp8_calls = []
    adapter.convert_module = lambda module, **kwargs: fp8_calls.append(
        (module, kwargs)
    )
    monkeypatch.setattr(
        base_model,
        "quantize_linear_layers_to_fp4",
        lambda module, **kwargs: fp4_calls.append((module, kwargs)),
    )
    monkeypatch.setattr(
        base_model,
        "quantize_linear_layers_to_nvfp4",
        lambda module, **kwargs: fp4_calls.append((module, kwargs)),
    )
    monkeypatch.setattr(base_model, "log", lambda *args, **kwargs: None)
    model = SimpleNamespace(
        settings=SimpleNamespace(
            fp4_gemm_module_list=["transformer.blocks"],
            fp8_precision_overrides=None,
            fp8_precision_override_suffixes=None,
        ),
        config=SimpleNamespace(use_hybrid_gemm_schedule=False),
        fp8=SimpleNamespace(
            module_list=lambda: [
                "transformer.blocks",
                "transformer_2.blocks",
            ]
        ),
        pipe=SimpleNamespace(
            transformer=SimpleNamespace(blocks=fp4_module),
            transformer_2=SimpleNamespace(blocks=fp8_module),
        ),
        blockwise_fp8_backend=adapter,
    )
    model._setup_fp8_only_gemm_modules = (
        lambda rank: xFuserModel._setup_fp8_only_gemm_modules(model, rank)
    )

    getattr(xFuserModel, setup_name)(model, local_rank=2)

    assert adapter.backend is expected_fp8_backend, platform
    assert [call[0] for call in fp4_calls] == [fp4_module]
    assert fp8_calls == [(fp8_module, {"device": "cuda:2"})]
