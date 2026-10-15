import pytest

from xfuser.config.args import FlexibleArgumentParser, xFuserArgs


def test_pipefusion_rejects_full_transformer_fsdp():
    with pytest.raises(ValueError, match="Full-transformer FSDP"):
        xFuserArgs(pipefusion_parallel_degree=2, fully_shard_degree=2)


def test_pipefusion_allows_targeted_fsdp_for_replicated_components():
    args = xFuserArgs(
        pipefusion_parallel_degree=2,
        fully_shard_degree=2,
        fully_shard_components=["text_encoder"],
    )

    assert args.fully_shard_components == ["text_encoder"]


def test_pipefusion_rejects_non_native_step_cache_methods():
    with pytest.raises(ValueError, match="requires --cache_method pipefusion"):
        xFuserArgs(pipefusion_parallel_degree=2, cache_method="fbcache")


def test_pipefusion_accepts_its_patch_aware_cache_method():
    args = xFuserArgs(
        pipefusion_parallel_degree=2,
        cache_method="pipefusion",
    )

    assert args.cache_method == "pipefusion"


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"num_pipeline_patch": 0}, "num_pipeline_patch"),
        ({"warmup_steps": 0}, "at least 1"),
        ({"attn_layer_num_for_pp": [4]}, "one entry per"),
        ({"attn_layer_num_for_pp": [4, 0]}, "greater than 0"),
    ],
)
def test_pipefusion_schedule_configuration_rejects_invalid_boundaries(
    kwargs,
    message,
):
    with pytest.raises(ValueError, match=message):
        xFuserArgs(pipefusion_parallel_degree=2, **kwargs)


def test_runner_accepts_pipefusion_schedule_configuration():
    parser = xFuserArgs.add_runner_args(FlexibleArgumentParser())

    args = parser.parse_args(
        [
            "--model",
            "black-forest-labs/FLUX.1-dev",
            "--pipefusion_parallel_degree",
            "2",
            "--num_pipeline_patch",
            "4",
            "--attn_layer_num_for_pp",
            "29",
            "28",
            "--warmup_steps",
            "2",
        ]
    )

    assert args.num_pipeline_patch == 4
    assert args.attn_layer_num_for_pp == [29, 28]
    assert args.warmup_steps == 2
