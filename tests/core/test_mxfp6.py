import copy
from types import SimpleNamespace

import pytest
import torch


@pytest.mark.parametrize("raw", ["fp6", "low=fp4,high=fp6"])
def test_a_tiered_fp6_run_takes_its_low_format_contract(raw):
    """No compound contract: the FP6 tier is resolved by name where it lands."""
    from xfuser.model_executor.models.runner_models.loading.contracts import (
        QuantizationBackend,
        QuantizationFormat,
        select_runtime_quantization,
    )

    from xfuser.config.gemm import GemmQuantizationSpec

    format_, backend = select_runtime_quantization(
        GemmQuantizationSpec.parse(raw),
        impl_for=lambda _format: "aiter",
    )

    assert backend is QuantizationBackend.AITER
    assert format_ is (
        QuantizationFormat.FP6 if raw == "fp6" else QuantizationFormat.FP4
    )


def test_wan22_reuses_existing_fp4_and_quality_targets_for_mxfp6():
    from xfuser.config.gemm import GemmQuantizationSpec
    from xfuser.model_executor.models.runner_models.loading.quantization_plan import (
        QuantizationPlan,
    )
    from xfuser.model_executor.models.runner_models.wan import xFuserWan22T2VModel

    model = object.__new__(xFuserWan22T2VModel)
    model.settings = copy.deepcopy(xFuserWan22T2VModel.settings)
    model._customize_settings(SimpleNamespace())
    model.settings.gemm_targets = model.settings.gemm_targets
    # An MXFP6 run always carries a spec: args refuses --use_fp6_gemms on its
    # own, so the mixed mode is only reachable as low=fp4,high=fp6.
    model.config = SimpleNamespace(
        quantize_text_encoder=False,
        gemm_formats=frozenset({"fp6"}),
        gemm_quantization_spec=GemmQuantizationSpec("fp4", "fp6"),
        ulysses_degree=1,
        ring_degree=1,
    )
    plan = QuantizationPlan(model)

    gemm_plan = plan.gemm_plan
    # The primary format owns the high-noise pass; the refiner is held back.
    assert list(gemm_plan.roots(gemm_plan.low)) == ["transformer.blocks"]
    assert list(gemm_plan.roots(gemm_plan.high)) == ["transformer_2.blocks"]
    assert gemm_plan.declared_roots() == (
        "transformer.blocks",
        "transformer_2.blocks",
    )


def test_wan22_logs_an_explicit_fp4_and_fp6_mapping(monkeypatch):
    from xfuser.config.gemm import GemmQuantizationSpec
    from xfuser.model_executor.models.runner_models.loading import quantization_plan
    from xfuser.model_executor.quant.targets import GemmTargets, Select

    messages = []
    model = SimpleNamespace(
        config=SimpleNamespace(
            gemm_quantization_spec=GemmQuantizationSpec("fp4", "fp6"),
            gemm_high_precision_targets="model",
            use_hybrid_gemm_schedule=False,
        ),
        settings=SimpleNamespace(
            gemm_targets=GemmTargets(
                transformer=Select(
                    modules=("transformer.blocks", "transformer_2.blocks")
                ),
                keep_high=Select(modules=("transformer_2.blocks",)),
            ),
        ),
    )
    model.config.quantize_text_encoder = False
    model.config.ulysses_degree = 1
    model.config.ring_degree = 1
    monkeypatch.setattr(quantization_plan, "log", messages.append)

    quantization_plan.QuantizationPlan(model).log_gemm_plan()

    assert messages[-2:] == [
        "GEMM quantization: transformer.blocks -> FP4",
        "GEMM quantization: transformer_2.blocks -> FP6",
    ]


def test_wan22_mixed_mode_routes_primary_to_fp4_and_second_transformer_to_fp6(
    monkeypatch,
):
    """The low-noise refiner is held at MXFP6 while the rest takes MXFP4."""
    from xfuser.config.gemm import GemmQuantizationSpec
    from xfuser.model_executor.models.runner_models.loading import placement
    from xfuser.model_executor.models.runner_models.loading.quantization_ledger import (
        QuantizationLedger,
    )
    from xfuser.model_executor.models.runner_models.loading.quantization_plan import (
        QuantizationPlan,
    )
    from xfuser.model_executor.quant.targets import GemmTargets, Select

    calls = []
    monkeypatch.setattr(placement, "log", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        placement,
        "descriptor_for",
        lambda *a, **k: SimpleNamespace(log_message=lambda: ""),
    )
    fp4_blocks, fp6_blocks = object(), object()
    model = SimpleNamespace(
        config=SimpleNamespace(
            gemm_quantization_spec=GemmQuantizationSpec("fp4", "fp6"),
            quantize_text_encoder=False,
            use_hybrid_gemm_schedule=False,
            ulysses_degree=1,
            ring_degree=1,
        ),
        settings=SimpleNamespace(
            gemm_targets=GemmTargets(
                transformer=Select(
                    modules=("transformer.blocks", "transformer_2.blocks")
                ),
                keep_high=Select(modules=("transformer_2.blocks",)),
            ),
        ),
        pipe=SimpleNamespace(
            transformer=SimpleNamespace(blocks=fp4_blocks),
            transformer_2=SimpleNamespace(blocks=fp6_blocks),
        ),
    )
    backends = SimpleNamespace(
        format=SimpleNamespace(
            convert_module=lambda module, **kwargs: calls.append(("fp4", module))
        ),
        fp6=SimpleNamespace(
            convert_module=lambda module, **kwargs: calls.append(("fp6", module))
        ),
        fp8=None,
        blockwise_fp8=None,
    )
    backends.adapter_for = lambda format_name: (
        backends.fp6 if format_name == "fp6" else backends.format
    )
    loader = SimpleNamespace(
        model=model,
        backends=backends,
        quantization_plan=QuantizationPlan(model),
        quantization_ledger=QuantizationLedger(),
    )

    for before in (True, False):
        placement.setup_gemm_quantization(
            loader, local_rank=0, offload_requested=False, before_device_move=before
        )

    assert calls == [("fp4", fp4_blocks), ("fp6", fp6_blocks)]


def test_only_supported_wan_runners_enable_mxfp6():
    from xfuser.model_executor.models.runner_models.wan import (
        xFuserWan21I2VModel,
        xFuserWan21T2VModel,
        xFuserWan21VACEModel,
        xFuserWan22DistilledI2VModel,
        xFuserWan22TI2VModel,
    )

    assert "fp6" in xFuserWan21I2VModel.capabilities.supported_gemm_formats()
    assert "fp6" in xFuserWan21T2VModel.capabilities.supported_gemm_formats()
    assert "fp6" in xFuserWan22DistilledI2VModel.capabilities.supported_gemm_formats()
    assert "fp6" in xFuserWan22TI2VModel.capabilities.supported_gemm_formats()
    assert "fp6" not in xFuserWan21VACEModel.capabilities.supported_gemm_formats()


def test_wan_mxfp6_declaration_keeps_existing_load_modes():
    from xfuser.model_executor.models.runner_models.loading.contracts import (
        LoadDeclaration,
        MaterializationMode,
        QuantizationBackend,
        QuantizationFormat,
    )
    from xfuser.model_executor.models.runner_models.wan import xFuserWan21T2VModel

    declaration = LoadDeclaration.for_runner(
        xFuserWan21T2VModel.capabilities,
        load_support=xFuserWan21T2VModel.load_support,
        fsdp_strategy=xFuserWan21T2VModel.settings.fsdp_strategy,
    )

    assert (
        QuantizationFormat.FP6,
        QuantizationBackend.AITER,
    ) in declaration.quantization_contracts
    assert (
        QuantizationFormat.FP4,
        QuantizationBackend.AITER,
    ) in declaration.quantization_contracts
    assert MaterializationMode.FSDP_META in declaration.materialization_modes
    assert MaterializationMode.REPLICATED_META in declaration.materialization_modes


def test_aiter_mxfp6_probe_requires_gfx950(monkeypatch):
    from xfuser.model_executor.models.runner_models.loading import backends

    api = SimpleNamespace(
        quant_mxfp6_gemm=lambda value: value,
        gemm_a6w6=lambda *args: None,
        mxfp6_gemm_pack_size=lambda rows, features: (rows, features),
    )
    real_import = backends.import_module
    monkeypatch.setattr(
        backends,
        "import_module",
        lambda name: api if name == "aiter" else real_import(name),
    )

    assert backends._probe_aiter_mxfp6_apis(lambda: "gfx950") == (
        True,
        None,
    )
    available, reason = backends._probe_aiter_mxfp6_apis(lambda: "gfx942")
    assert not available
    assert "gfx950" in reason


def test_fp4_hybrid_builds_whatever_companion_it_is_given(monkeypatch):
    """The FP4 converter no longer picks the high branch; the caller does.

    It used to choose between MXFP6 and torchao FP8 from a flag of its own,
    which is the last cross-format knowledge that lived inside a converter.
    """
    from xfuser.core.utils import runner_utils
    from xfuser.model_executor.layers import mxfp4_linear, mxfp6_linear
    from xfuser.model_executor.layers.hybrid_linear import xFuserHybridLinear

    class StubLinear(torch.nn.Module):
        def __init__(self, in_features, out_features, **kwargs):
            super().__init__()

        def load_and_quantize_weights(self, weight, bias=None, **kwargs):
            pass

    class StubFP4(StubLinear):
        pass

    class StubFP6(StubLinear):
        pass

    monkeypatch.setattr(mxfp4_linear, "xFuserMXFP4Linear", StubFP4)
    monkeypatch.setattr(mxfp6_linear, "xFuserMXFP6Linear", StubFP6)
    model = torch.nn.Sequential(torch.nn.Linear(4, 3, bias=False, dtype=torch.bfloat16))

    runner_utils.quantize_linear_layers_to_fp4(
        model,
        device="cpu",
        companion=runner_utils.packed_layer_factory(StubFP6, "cpu"),
    )

    assert isinstance(model[0], xFuserHybridLinear)
    assert isinstance(model[0].low_precision_linear, StubFP4)
    assert isinstance(model[0].high_precision_linear, StubFP6)


def test_mxfp6_linear_uses_aiter_pack_and_gemm(monkeypatch):
    from xfuser.model_executor.layers import mxfp6_linear

    calls = []

    def pack(value):
        calls.append(("pack", tuple(value.shape)))
        return (
            torch.ones(8, dtype=torch.uint8),
            torch.ones(2, dtype=torch.uint8),
        )

    def gemm(a, b, a_scale, b_scale, rows, out_features, in_features, **kwargs):
        calls.append(("gemm", rows, out_features, in_features))
        return torch.full((rows, out_features), 2.0, dtype=torch.bfloat16)

    monkeypatch.setattr(mxfp6_linear, "quant_mxfp6_gemm", pack)
    monkeypatch.setattr(mxfp6_linear, "gemm_a6w6", gemm)

    layer = mxfp6_linear.xFuserMXFP6Linear(4, 3, bias=False)
    layer.load_and_quantize_weights(torch.zeros(3, 4, dtype=torch.bfloat16))
    output = layer(torch.zeros(2, 4, dtype=torch.bfloat16))

    assert output.shape == (2, 3)
    assert torch.all(output == 2)
    assert calls == [
        ("pack", (3, 4)),
        ("pack", (2, 4)),
        ("gemm", 2, 3, 4),
    ]

    state = layer.state_dict()
    monkeypatch.setattr(mxfp6_linear, "mxfp6_gemm_pack_size", lambda *args: (8, 2))
    restored = mxfp6_linear.xFuserMXFP6Linear(4, 3, bias=False)
    restored.load_state_dict(state)
    assert restored.weight is None
    assert isinstance(restored.weight_packed, torch.nn.Parameter)

    meta_restored = mxfp6_linear.xFuserMXFP6Linear(4, 3, bias=False, device="meta")
    meta_restored.load_state_dict(state)
    assert meta_restored.weight_packed.device.type == "cpu"
