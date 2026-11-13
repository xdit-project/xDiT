"""Which models may select an AITER Sol-Attn backend, and what they must configure alongside it.

Config validation only -- no kernel, no device. Whether the machine actually carries the manifest
row for a recipe is a separate check made against the device.
"""

import pytest

SOL_BACKENDS = ["aiter_fp8_sol", "aiter_bf16_sol", "aiter_f6f4_sol"]

# The registered name and any extra args its own validation demands.
MODELS = [
    ("Wan2.1-T2V", {}),
    ("Wan2.2-T2V", {}),
    ("Wan2.1-I2V", {}),
    ("Wan2.2-I2V", {}),
    ("Wan2.2-TI2V", {"task": "t2v"}),
    ("Wan2.2-Distilled-I2V", {}),
]


@pytest.fixture(autouse=True)
def _single_process(monkeypatch):
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")


def _build(model_name, **config):
    from xfuser import xFuserArgs
    from xfuser.model_executor.models.runner_models import wan  # noqa: F401  registers
    from xfuser.model_executor.models.runner_models.base_model import MODEL_REGISTRY

    return MODEL_REGISTRY[model_name](xFuserArgs(model=model_name, **config))


@pytest.mark.parametrize("model,extra", MODELS)
@pytest.mark.parametrize("backend", SOL_BACKENDS)
def test_sol_backends_are_accepted_with_a_dense_cross_attention_backend(model, extra, backend):
    _build(model, attention_backend=backend, cross_attention_backend="aiter_bf16", **extra)


@pytest.mark.parametrize("model,extra", MODELS)
def test_sol_needs_a_dense_cross_attention_backend(model, extra):
    """Unset, cross-attention would fall back to Sol on Wan's 512 text tokens: four KV blocks, too
    few for the routing threshold to mean anything. It would not fail, just answer worse."""
    with pytest.raises(ValueError, match="needs --cross_attention_backend set to a dense backend"):
        _build(model, attention_backend="aiter_fp8_sol", **extra)
    with pytest.raises(ValueError, match="cannot be AITER_FP8_SOL"):
        _build(model, attention_backend="aiter_fp8_sol", cross_attention_backend="aiter_fp8_sol", **extra)


@pytest.mark.parametrize("model,extra", MODELS)
def test_sol_refuses_ring_parallelism_at_config_time(model, extra):
    """Sol-Attn cannot be a ring rank, and finding that out mid-denoise is too late."""
    with pytest.raises(ValueError, match="does not support ring parallelism"):
        _build(
            model,
            attention_backend="aiter_fp8_sol",
            cross_attention_backend="aiter_bf16",
            ring_degree=2,
            **extra,
        )


@pytest.mark.parametrize("model,extra", MODELS)
def test_ulysses_stays_available(model, extra):
    """Ulysses is the sequence parallelism Sol-Attn does support, so it must survive the ring check."""
    _build(
        model,
        attention_backend="aiter_fp8_sol",
        cross_attention_backend="aiter_bf16",
        ulysses_degree=2,
        **extra,
    )


def test_a_hybrid_schedule_on_sol_is_gated_like_an_explicit_backend():
    with pytest.raises(ValueError, match="does not support ring parallelism"):
        _build(
            "Wan2.2-T2V",
            use_hybrid_attn_schedule=True,
            hybrid_attn_low_precision_backend="aiter_fp8_sol",
            hybrid_attn_high_precision_backend="aiter_bf16",
            cross_attention_backend="aiter_bf16",
            ring_degree=2,
        )


def test_a_model_without_the_capability_refuses_sol():
    """The gate is opt-in, so a Wan runner that has not been checked for Sol-Attn keeps saying no."""
    with pytest.raises(ValueError, match="does not support Sol-Attn"):
        _build("Wan2.1-VACE-14B", attention_backend="aiter_fp8_sol", cross_attention_backend="aiter_bf16")
