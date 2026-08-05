"""Regression tests for mutually-owned generic FP8/FP4 GEMM modes."""

from types import SimpleNamespace

import pytest


@pytest.fixture(scope="module")
def runtime():
    pytest.importorskip("torch", reason="PyTorch is required for runner validation")
    from xfuser.config.args import xFuserArgs
    from xfuser.model_executor.models.runner_models import base_model

    return SimpleNamespace(
        args_cls=xFuserArgs,
        base=base_model,
        capabilities_cls=base_model.ModelCapabilities,
        model_cls=base_model.xFuserModel,
    )


def _args(runtime, **overrides):
    values = {
        "model": "test/model",
        "use_fp8_gemms": False,
        "use_fp4_gemms": False,
        "use_hybrid_gemm_schedule": False,
    }
    values.update(overrides)
    return runtime.args_cls(**values)


def test_args_reject_generic_fp8_and_fp4_without_hybrid_owner(runtime):
    config = _args(runtime, use_fp8_gemms=True, use_fp4_gemms=True)

    with pytest.raises(ValueError, match="cannot both be enabled"):
        config._validate_gemm_quantization_flags()


@pytest.mark.parametrize(
    "flags",
    [
        {"use_int8_gemms": True, "use_fp8_gemms": True},
        {"use_int8_gemms": True, "use_fp4_gemms": True},
        {
            "use_int8_gemms": True,
            "use_fp8_gemms": True,
            "use_fp4_gemms": True,
            "use_hybrid_gemm_schedule": True,
        },
    ],
)
def test_args_reject_int8_combined_with_fp8_or_fp4(runtime, flags):
    config = _args(runtime, **flags)

    with pytest.raises(ValueError, match="--use_int8_gemms cannot be combined"):
        config._validate_gemm_quantization_flags()


@pytest.mark.parametrize(
    "use_fp8_gemms",
    [False, True],
)
def test_args_allow_explicit_hybrid_schedule_to_own_fp8_inside_fp4(
    runtime, use_fp8_gemms
):
    config = _args(
        runtime,
        use_fp8_gemms=use_fp8_gemms,
        use_fp4_gemms=True,
        use_hybrid_gemm_schedule=True,
    )

    config._validate_gemm_quantization_flags()


def test_base_model_validation_rejects_generic_fp8_and_fp4_early(runtime):
    model = object.__new__(runtime.model_cls)
    model.settings = SimpleNamespace(model_name="test/model", valid_tasks=[])
    model.capabilities = runtime.capabilities_cls(
        use_fp8_gemms=True,
        use_fp4_gemms=True,
        use_hybrid_gemm_schedule=True,
    )
    config = _args(runtime, use_fp8_gemms=True, use_fp4_gemms=True)

    with pytest.raises(ValueError, match="cannot both be enabled"):
        model._validate_config(config)


def test_base_model_uses_central_int8_conflict_validation(runtime):
    model = object.__new__(runtime.model_cls)
    model.settings = SimpleNamespace(model_name="test/model", valid_tasks=[])
    model.capabilities = runtime.capabilities_cls(
        use_int8_gemms=True,
        use_fp8_gemms=True,
    )
    config = _args(runtime, use_int8_gemms=True, use_fp8_gemms=True)

    with pytest.raises(ValueError, match="--use_int8_gemms cannot be combined"):
        model._validate_config(config)


def test_explicit_hybrid_fp4_owns_conversion_without_generic_fp8_walk(
    runtime, monkeypatch
):
    calls = []
    model = object.__new__(runtime.model_cls)
    model.config = SimpleNamespace(
        fully_shard_degree=1,
        enable_model_cpu_offload=False,
        enable_sequential_cpu_offload=False,
        enable_group_cpu_offload=False,
        use_fp4_gemms=True,
        use_fp8_gemms=True,
        use_int8_gemms=False,
        use_hybrid_attn_schedule=False,
        use_hybrid_gemm_schedule=True,
        use_vae_channels_last_format=False,
        use_torch_compile=False,
    )
    model.settings = SimpleNamespace(
        fp8_precision_overrides=None,
        fp8_precision_override_suffixes=None,
        int8_gemm_module_list=None,
    )
    model.fp8 = SimpleNamespace(aiter_active=False, module_list=lambda: ["transformer"])
    model.pipe = SimpleNamespace(to=lambda device: model.pipe)
    model._replicated_broadcast_load = lambda: False
    model._setup_mxfp4_gemms = lambda local_rank: calls.append(("fp4", local_rank))
    model._setup_hybrid_gemm_schedule = lambda input_args: calls.append(
        ("schedule", input_args)
    )

    monkeypatch.setattr(
        runtime.base, "get_world_group", lambda: SimpleNamespace(local_rank=0)
    )
    monkeypatch.setattr(runtime.base, "_is_cuda", lambda: False)
    monkeypatch.setattr(runtime.base, "_use_aiter_fp8_rdna4", lambda: False)
    monkeypatch.setattr(
        runtime.base,
        "quantize_linear_layers_to_fp8",
        lambda module, device: calls.append(("generic-fp8", device)),
    )

    model._post_load_and_state_initialization({"num_inference_steps": 4})

    assert calls == [
        ("fp4", 0),
        ("schedule", {"num_inference_steps": 4}),
    ]
