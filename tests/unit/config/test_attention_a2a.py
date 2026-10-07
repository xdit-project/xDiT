import pytest

from xfuser.config.args import xFuserArgs
from xfuser.config.attention_a2a import AttentionA2AConfig
from xfuser.core.attention import registry
from xfuser.core.attention.backends.aiter_mha_v4.spec import (
    DENSE_BACKENDS,
    Fmt,
)
from xfuser.core.attention.spec import AttentionBackendType


@pytest.mark.parametrize(
    (
        "profile",
        "backend",
        "codecs",
        "consumer_codecs",
        "hadamard",
        "multiple",
    ),
    (
        (
            "e4m3-e4m3",
            "aiter_fp8",
            ("e4m3", "e4m3", "e4m3"),
            ("e4m3", "e4m3", "e4m3"),
            "preprocess",
            32,
        ),
        (
            "int8-e4m3",
            "aiter_i8fp8",
            ("int8", "int8", "e4m3"),
            ("int8", "int8", "e4m3"),
            "none",
            32,
        ),
        (
            "mxfp8-e4m3",
            "aiter_mxfp8",
            ("mxfp8", "mxfp8", "e4m3"),
            ("e4m3", "e4m3", "e4m3"),
            "preprocess",
            32,
        ),
        (
            "e4m3-mxfp6",
            "aiter_f8f6",
            ("e4m3", "e4m3", "mxfp6_p"),
            ("e4m3", "e4m3", "mxfp6"),
            "preprocess",
            64,
        ),
        (
            "mxfp6-e4m3",
            "aiter_mxfp6",
            ("mxfp6", "mxfp6", "e4m3_pc"),
            ("mxfp6", "mxfp6", "e4m3"),
            "preprocess",
            32,
        ),
        (
            "mxfp6-mxfp6",
            "aiter_f6f6",
            ("mxfp6", "mxfp6", "mxfp6_p"),
            ("mxfp6", "mxfp6", "mxfp6"),
            "preprocess",
            64,
        ),
        (
            "mxfp4-mxfp4",
            "aiter_mxfp4",
            ("mxfp4", "mxfp4", "mxfp4"),
            ("mxfp4", "mxfp4", "mxfp4"),
            "preprocess",
            64,
        ),
        (
            "mxfp6-mxfp4",
            "aiter_f6f4",
            ("mxfp6", "mxfp6", "mxfp4"),
            ("mxfp6", "mxfp6", "mxfp4"),
            "preprocess",
            64,
        ),
    ),
)
def test_profile_contract(
    profile,
    backend,
    codecs,
    consumer_codecs,
    hadamard,
    multiple,
):
    config = AttentionA2AConfig(profile=profile)

    assert config.attention_backend == backend
    assert config.codecs == codecs
    assert config.consumer_codecs == consumer_codecs
    assert config.hadamard_placement == hadamard
    assert config.local_sequence_multiple == multiple


def test_auto_infers_dense_mha_v4_profiles():
    policy = AttentionA2AConfig(profile="auto")

    assert policy.resolve_for_backend("aiter_mxfp6").profile == "mxfp6-e4m3"
    assert policy.resolve_for_backend("aiter_f6f6").profile == "mxfp6-mxfp6"
    assert policy.resolve_for_backend("aiter_f6f4").profile == "mxfp6-mxfp4"
    assert policy.resolve_for_backend("aiter_mxfp4").profile == "mxfp4-mxfp4"
    assert policy.resolve_for_backend("aiter_f4f4").profile == "mxfp4-mxfp4"


def test_profiles_use_validated_wan_launch_geometry():
    assert AttentionA2AConfig(profile="mxfp6-mxfp4").block_num == 512
    assert AttentionA2AConfig(profile="e4m3-e4m3").block_num == 512


@pytest.mark.parametrize(
    ("profile", "scale_modes", "v_pack"),
    (
        ("e4m3-e4m3", ("f32_per_tensor",) * 3, "default"),
        ("int8-e4m3", ("f32_per_tensor",) * 3, "default"),
        (
            "mxfp8-e4m3",
            ("e8m0_per_1x32", "e8m0_per_1x32", "f32_per_tensor"),
            "default",
        ),
        (
            "e4m3-mxfp6",
            ("f32_per_tensor", "f32_per_tensor", "e8m0_per_1x32"),
            "fp6_p",
        ),
        (
            "mxfp6-e4m3",
            ("e8m0_per_1x32", "e8m0_per_1x32", "f32_per_channel"),
            "default",
        ),
        ("mxfp6-mxfp6", ("e8m0_per_1x32",) * 3, "fp6_p"),
        ("mxfp6-mxfp4", ("e8m0_per_1x32",) * 3, "fp6_p"),
        ("mxfp4-mxfp4", ("e8m0_per_1x32",) * 3, "fp6_p"),
    ),
)
def test_profile_scale_and_pack_contract(profile, scale_modes, v_pack):
    config = AttentionA2AConfig(profile=profile)

    assert config.scale_modes == scale_modes
    assert config.v_pack == v_pack


def test_explicit_profile_requires_matching_backend():
    with pytest.raises(ValueError, match="requires an explicit"):
        xFuserArgs(
            model="test",
            ulysses_degree=8,
            attention_a2a="mxfp8-e4m3",
        )

    args = xFuserArgs(
        model="test",
        ulysses_degree=8,
        attention_backend="aiter_mxfp8",
        attention_a2a="mxfp8-e4m3",
    )
    assert args.attention_a2a == "mxfp8-e4m3"

    with pytest.raises(ValueError, match="requires backend"):
        xFuserArgs(
            model="test",
            ulysses_degree=8,
            attention_backend="aiter_fp8",
            attention_a2a="mxfp8-e4m3",
        )


def test_hybrid_schedule_requires_auto_profile():
    args = xFuserArgs(
        model="test",
        ulysses_degree=8,
        use_hybrid_attn_schedule=True,
        hybrid_attn_low_precision_backend="aiter_f4f4",
        hybrid_attn_high_precision_backend="aiter_fp8",
        attention_a2a="auto",
    )
    assert args.attention_a2a == "auto"

    with pytest.raises(ValueError, match="hybrid attention requires"):
        xFuserArgs(
            model="test",
            ulysses_degree=8,
            use_hybrid_attn_schedule=True,
            hybrid_attn_low_precision_backend="aiter_f4f4",
            hybrid_attn_high_precision_backend="aiter_fp8",
            attention_a2a="mxfp4-mxfp4",
        )


def test_attention_a2a_rejects_incompatible_parallel_modes():
    with pytest.raises(ValueError, match="ring parallelism"):
        xFuserArgs(
            model="test",
            ulysses_degree=4,
            ring_degree=2,
            attention_backend="aiter_fp8",
            attention_a2a="auto",
        )
    with pytest.raises(ValueError, match="batch size 1"):
        xFuserArgs(
            model="test",
            ulysses_degree=8,
            batch_size=2,
            attention_backend="aiter_fp8",
            attention_a2a="auto",
        )


def test_f6f6_is_a_dense_gfx950_mha_v4_backend():
    backend = AttentionBackendType.AITER_F6F6
    spec = registry.get(backend)

    assert backend in DENSE_BACKENDS
    assert spec.impl.target == "kernel:mha_v4_dense"
    assert (spec.impl.bound["fmt"].qk, spec.impl.bound["fmt"].v) == (
        Fmt.MXFP6,
        Fmt.MXFP6,
    )
    assert spec.impl.bound["fmt"].dense_on.names == ("gfx950",)
