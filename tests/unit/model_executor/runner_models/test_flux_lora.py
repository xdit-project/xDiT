import pytest


def test_lora_cli_and_config_validation():
    from xfuser.config import FlexibleArgumentParser, xFuserArgs

    parser = xFuserArgs.add_runner_args(FlexibleArgumentParser())
    args = parser.parse_args(
        [
            "--model",
            "black-forest-labs/FLUX.1-dev",
            "--lora-path",
            "adapter",
            "--lora-weight-name",
            "weights.safetensors",
            "--lora-scale",
            "0.5",
        ]
    )
    config = xFuserArgs.from_runner_args(vars(args))
    assert (config.lora_path, config.lora_weight_name, config.lora_scale) == (
        "adapter",
        "weights.safetensors",
        0.5,
    )
    with pytest.raises(ValueError, match="require lora_path"):
        xFuserArgs(lora_weight_name="weights.safetensors")
    with pytest.raises(ValueError, match="finite"):
        xFuserArgs(lora_path="adapter", lora_scale=float("nan"))


@pytest.mark.parametrize(
    "options",
    [
        {"pipefusion_parallel_degree": 2},
        {"fully_shard_degree": 2},
        {"memory_efficient_sharding": True},
        {"memory_efficient_replicated_load": True},
        {"gemm_quantization": "fp8"},
    ],
)
def test_lora_refuses_incompatible_loading_before_weights_are_loaded(options):
    from xfuser.config import xFuserArgs
    from xfuser.model_executor.models.runner_models.flux import xFuserFluxModel

    with pytest.raises(ValueError, match="eager, unquantized"):
        xFuserFluxModel(xFuserArgs(lora_path="adapter", **options))


def test_lora_is_rejected_on_runners_without_support():
    from xfuser.config import xFuserArgs
    from xfuser.model_executor.models.runner_models.wan import xFuserWan21T2VModel

    with pytest.raises(ValueError, match="does not support startup LoRA"):
        xFuserWan21T2VModel(xFuserArgs(lora_path="adapter"))
