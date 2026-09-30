"""Regression tests for mutually-owned generic FP8/FP4 GEMM modes."""

from types import MethodType, SimpleNamespace

import pytest

from xfuser.config.gemm import GemmQuantizationSpec


@pytest.fixture(scope="module")
def runtime():
    pytest.importorskip("torch", reason="PyTorch is required for runner validation")
    from xfuser.config.args import xFuserArgs
    from xfuser.model_executor.models.runner_models import base_model
    from xfuser.model_executor.models.runner_models.loading import placement
    from xfuser.model_executor.models.runner_models.loading.meta_load import ModelLoader
    from xfuser.model_executor.models.runner_models.loading.quantization_plan import (
        QuantizationPlan,
    )

    class _StubModel(base_model.xFuserModel):
        """Concrete runner used to exercise base-class flag validation.

        xFuserModel is an ABC, so object.__new__ on it raises rather than
        producing the uninitialized instance these tests poke at.
        """

        def _load_model(self):
            raise NotImplementedError

        def _run_pipe(self, input_args):
            raise NotImplementedError

    return SimpleNamespace(
        args_cls=xFuserArgs,
        base=base_model,
        capabilities_cls=base_model.ModelCapabilities,
        model_cls=_StubModel,
        placement=placement,
        loader_cls=ModelLoader,
        plan_cls=QuantizationPlan,
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


def test_unsupported_runner_rejects_fp8_text_encoder_via_capability_validation(
    runtime,
):
    model = object.__new__(runtime.model_cls)
    model.settings = SimpleNamespace(model_name="test/model", valid_tasks=[])
    model.capabilities = runtime.capabilities_cls(use_fp8_gemms=True)
    config = _args(
        runtime,
        use_fp8_gemms=True,
        quantize_text_encoder=True,
    )

    with pytest.raises(
        ValueError,
        match="does not support quantize_text_encoder",
    ):
        model._validate_config(config)


def test_supported_runner_logs_when_text_encoder_targets_remain_bf16(
    runtime, monkeypatch
):
    messages = []
    model = object.__new__(runtime.model_cls)
    from xfuser.model_executor.quant.targets import GemmTargets, Select

    model.settings = SimpleNamespace(
        gemm_targets=GemmTargets(
            transformer=Select(modules=("transformer.blocks",)),
            text_encoder=Select(modules=("text_encoder.layers",)),
        ),
    )
    config = _args(runtime, use_fp8_gemms=True)
    monkeypatch.setattr(runtime.base, "log", messages.append)

    model._update_model_settings(config)

    assert len(messages) == 1
    assert "text-encoder target(s) stay bf16" in messages[0]


# There is one walk now, so no generic FP8 pass can follow the FP4 one
# and re-quantize inside its wrappers; the premise this guarded is gone.


@pytest.mark.parametrize(
    ("value", "expected_flags"),
    [
        ("fp8", (True, False, False, False)),
        ("fp6", (False, False, True, False)),
        ("int8", (False, False, False, True)),
        ("low=fp4,high=fp8", (False, True, False, False)),
        ("low=fp4,high=fp6", (False, True, True, False)),
    ],
)
def test_explicit_gemm_profiles_map_to_existing_flags(
    runtime, value, expected_flags
):
    config = _args(runtime, gemm_quantization=value)

    assert str(config.gemm_quantization_spec) == value
    assert (
        config.use_fp8_gemms,
        config.use_fp4_gemms,
        config.use_fp6_gemms,
        config.use_int8_gemms,
    ) == expected_flags


def test_explicit_profile_wins_over_deprecated_format_flag(runtime):
    with pytest.warns(FutureWarning, match="ignored"):
        config = _args(
            runtime,
            gemm_quantization="fp6",
            use_fp4_gemms=True,
        )

    assert config.gemm_quantization_spec == GemmQuantizationSpec("fp6")
    assert config.use_fp6_gemms is True
    assert config.use_fp4_gemms is False


def test_runner_parser_accepts_explicit_gemm_profile(runtime):
    from xfuser.config.args import FlexibleArgumentParser

    parser = runtime.args_cls.add_runner_args(FlexibleArgumentParser())
    assert "--use_fp6_gemms" not in parser._option_string_actions
    parsed = parser.parse_args(
        ["--model", "test/model", "--gemm-quantization", "low=fp4,high=fp6"]
    )
    config = runtime.args_cls.from_runner_args(vars(parsed))

    assert config.use_fp4_gemms is True
    assert config.use_fp6_gemms is True


def test_any_profile_can_include_the_text_encoder(runtime):
    """The flag says to include the encoder, not which format to give it."""
    for profile in ("fp8", "fp4", "low=fp4,high=fp8", "low=fp4,high=fp6"):
        _args(
            runtime,
            gemm_quantization=profile,
            quantize_text_encoder=True,
        )._validate_gemm_quantization_flags()


def test_including_the_text_encoder_needs_a_profile(runtime):
    with pytest.raises(ValueError, match="needs a gemm_quantization profile"):
        _args(
            runtime,
            gemm_quantization="none",
            quantize_text_encoder=True,
        )._validate_gemm_quantization_flags()


# The advanced YAML no longer rewrites per-format lists; it overrides
# `keep_high` directly, which test_flux2_gemm_equivalence covers under
# "a config file can hold extra modules high".


# A pure low-format run quantizing every declared target is now the
# resolver's own rule, checked for every model by
# tests/quant/test_migrated_models.py against the recorded lists.


def test_yaml_schedule_conflicts_with_simple_schedule_flags(runtime, tmp_path):
    path = tmp_path / "gemm.yaml"
    path.write_text("hybrid_gemm_schedule: [fp8, fp4]\n")

    with pytest.raises(ValueError, match="cannot be combined"):
        _args(
            runtime,
            gemm_quantization="low=fp4,high=fp8",
            gemm_config=str(path),
            use_hybrid_gemm_schedule=True,
        )


def test_yaml_schedule_expands_over_wan_cfg_calls(runtime, monkeypatch):
    captured = {}
    state = SimpleNamespace(
        set_gemm_schedule=lambda schedule, total_steps: captured.update(
            schedule=schedule.use_high_precision_schedule,
            total_steps=total_steps,
        )
    )
    model = object.__new__(runtime.model_cls)
    model.config = SimpleNamespace(
        hybrid_gemm_schedule="fp6,fp4",
        gemm_quantization_spec=GemmQuantizationSpec("fp4", "fp6"),
    )
    model._calculate_hybrid_attention_step_multiplier = lambda input_args: 2
    monkeypatch.setattr(runtime.base, "get_runtime_state", lambda: state)
    monkeypatch.setattr(runtime.base, "log", lambda *args, **kwargs: None)

    model._setup_hybrid_gemm_schedule(
        {
            "num_inference_steps": 2,
            "num_hybrid_gemm_high_precision_steps": None,
        }
    )

    assert captured == {
        "schedule": [True, True, False, False],
        "total_steps": 4,
    }


def test_fp6_profile_supports_simple_hybrid_schedule(runtime):
    config = _args(
        runtime,
        gemm_quantization="low=fp4,high=fp6",
        use_hybrid_gemm_schedule=True,
        num_hybrid_gemm_high_precision_steps=1,
    )

    config._validate_gemm_quantization_flags()
