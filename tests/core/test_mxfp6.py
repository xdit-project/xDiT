import copy
from types import SimpleNamespace

import pytest
import torch


@pytest.mark.parametrize(
    ("fp4", "hybrid", "expected"),
    [
        (False, False, "fp6"),
        (True, False, "fp4_fp6"),
        (True, True, "fp4_fp6"),
    ],
)
def test_runtime_selects_aiter_mxfp6_formats(fp4, hybrid, expected):
    from xfuser.model_executor.models.runner_models.loading.contracts import (
        QuantizationBackend,
        select_runtime_quantization,
    )

    config = SimpleNamespace(
        use_fp8_gemms=False,
        use_fp4_gemms=fp4,
        use_fp6_gemms=True,
        use_int8_gemms=False,
        use_hybrid_gemm_schedule=hybrid,
    )

    format_, backend = select_runtime_quantization(
        config,
        aiter_fp8_active=False,
        cuda_active=False,
    )

    assert format_.value == expected
    assert backend is QuantizationBackend.AITER


def test_wan22_reuses_existing_fp4_and_quality_targets_for_mxfp6():
    from xfuser.model_executor.models.runner_models.loading.quantization_plan import (
        QuantizationPlan,
    )
    from xfuser.model_executor.models.runner_models.wan import xFuserWan22T2VModel

    model = object.__new__(xFuserWan22T2VModel)
    model.settings = copy.deepcopy(xFuserWan22T2VModel.settings)
    model._customize_settings(SimpleNamespace())
    model.config = SimpleNamespace(use_fp8_text_encoder=False, use_fp6_gemms=True)
    plan = QuantizationPlan(model)

    assert plan.module_list("fp4") == ["transformer.blocks"]
    assert plan.module_list("fp6") == [
        "transformer.blocks",
        "transformer_2.blocks",
    ]
    assert set(plan.module_list("fp8")) - set(plan.module_list("fp4")) == {
        "transformer_2.blocks"
    }


def test_wan22_logs_explicit_fp4_fp6_mapping(monkeypatch):
    from xfuser.config.gemm import GemmQuantizationSpec
    from xfuser.model_executor.models.runner_models.loading import quantization_plan

    messages = []
    model = SimpleNamespace(
        config=SimpleNamespace(
            gemm_quantization_spec=GemmQuantizationSpec("fp4", "fp6"),
            gemm_high_precision_targets="model",
            use_hybrid_gemm_schedule=False,
        ),
        settings=SimpleNamespace(
            fp4_gemm_module_list=["transformer.blocks"],
            fp8_gemm_module_list=[
                "transformer.blocks",
                "transformer_2.blocks",
            ],
            fp8_precision_overrides=None,
            fp8_precision_override_suffixes=None,
        ),
    )
    monkeypatch.setattr(quantization_plan, "log", messages.append)

    quantization_plan.QuantizationPlan(model).log_gemm_plan()

    assert messages[-2:] == [
        "GEMM quantization: transformer.blocks -> FP4",
        "GEMM quantization: transformer_2.blocks -> FP6",
    ]


def test_wan22_mixed_mode_routes_primary_to_fp4_and_second_transformer_to_fp6(
    monkeypatch,
):
    from xfuser.model_executor.models.runner_models.loading import placement

    calls = []
    monkeypatch.setattr(placement, "log", lambda *args, **kwargs: None)
    fp4_blocks, fp6_blocks = object(), object()
    model = SimpleNamespace(
        config=SimpleNamespace(
            use_fp4_gemms=True,
            use_fp6_gemms=True,
            use_hybrid_gemm_schedule=False,
            enable_model_cpu_offload=False,
            enable_sequential_cpu_offload=False,
            enable_group_cpu_offload=False,
        ),
        settings=SimpleNamespace(
            fp8_precision_overrides=None,
            fp8_precision_override_suffixes=None,
            fp8_gemm_include_suffixes=None,
        ),
        pipe=SimpleNamespace(
            transformer=SimpleNamespace(blocks=fp4_blocks),
            transformer_2=SimpleNamespace(blocks=fp6_blocks),
        ),
    )
    loader = SimpleNamespace(
        model=model,
        backends=SimpleNamespace(
            format_entries=lambda: ("transformer.blocks",),
            format=SimpleNamespace(
                convert_module=lambda module, **kwargs: calls.append(("fp4", module))
            ),
            fp6=SimpleNamespace(
                convert_module=lambda module, **kwargs: calls.append(("fp6", module))
            ),
        ),
        quantization_plan=SimpleNamespace(
            module_list=lambda format_name="fp8": (
                ["transformer.blocks"]
                if format_name == "fp4"
                else ["transformer.blocks", "transformer_2.blocks"]
            )
        ),
        quantization_ledger=SimpleNamespace(
            streaming_targets=set(),
            claim_description=lambda component: None,
            already_quantized=lambda **kwargs: set(),
        ),
    )

    placement.setup_mxfp4_gemms(loader, local_rank=0)

    assert calls == [("fp4", fp4_blocks), ("fp6", fp6_blocks)]


def test_only_supported_wan_runners_enable_mxfp6():
    from xfuser.model_executor.models.runner_models.wan import (
        xFuserWan21I2VModel,
        xFuserWan21T2VModel,
        xFuserWan21VACEModel,
        xFuserWan22DistilledI2VModel,
        xFuserWan22TI2VModel,
    )

    assert xFuserWan21I2VModel.capabilities.use_fp6_gemms
    assert xFuserWan21T2VModel.capabilities.use_fp6_gemms
    assert xFuserWan22DistilledI2VModel.capabilities.use_fp6_gemms
    assert xFuserWan22TI2VModel.capabilities.use_fp6_gemms
    assert not xFuserWan21VACEModel.capabilities.use_fp6_gemms


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
        QuantizationFormat.FP4_FP6,
        QuantizationBackend.AITER,
    ) in declaration.quantization_contracts
    assert MaterializationMode.FSDP_META in declaration.materialization_modes
    assert MaterializationMode.REPLICATED_META in declaration.materialization_modes


def test_aiter_mxfp6_probe_requires_gfx950(monkeypatch):
    from xfuser.model_executor.models.runner_models.loading import format_backends

    api = SimpleNamespace(
        quant_mxfp6_gemm=lambda value: value,
        gemm_a6w6=lambda *args: None,
        mxfp6_gemm_pack_size=lambda rows, features: (rows, features),
    )
    real_import = format_backends.import_module
    monkeypatch.setattr(
        format_backends,
        "import_module",
        lambda name: api if name == "aiter.ops.gemm_op_a6w6" else real_import(name),
    )
    monkeypatch.delenv("AITER_TRITON_ONLY", raising=False)

    assert format_backends._probe_aiter_mxfp6_apis(lambda: "gfx950") == (
        True,
        None,
    )
    available, reason = format_backends._probe_aiter_mxfp6_apis(lambda: "gfx942")
    assert not available
    assert "gfx950" in reason


def test_fp4_hybrid_builds_an_mxfp6_high_branch(monkeypatch):
    from xfuser.core.utils import runner_utils
    from xfuser.model_executor.layers import mxfp4_linear, mxfp6_linear

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
        use_hybrid_schedule=True,
        use_fp6_for_overrides=True,
        device="cpu",
    )

    assert isinstance(model[0], mxfp4_linear.xFuserHybridMXFP4Linear)
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
