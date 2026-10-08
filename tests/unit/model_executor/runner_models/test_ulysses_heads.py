"""Registered runners validate Ulysses before loading weights, including GQA."""

import pytest

from xfuser import xFuserArgs
from xfuser.model_executor.models.runner_models.base_model import MODEL_REGISTRY
from xfuser.model_executor.models.runner_models.z_image import xFuserZImageModel, xFuserZImageTurboModel


@pytest.fixture(autouse=True)
def _single_process(monkeypatch):
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")


def _build(name, degree):
    model_class = MODEL_REGISTRY[name]
    tasks = model_class.settings.valid_tasks
    checkpoint_task = name.rsplit("_", 1)[-1]
    task = checkpoint_task if checkpoint_task in tasks else (tasks[0] if tasks else None)
    return model_class(
        xFuserArgs(
            model=name,
            ulysses_degree=degree,
            attention_backend="FLEX_BLOCK_ATTN"
            if model_class.capabilities.supports_sparse_attention_backends
            else "SDPA",
            task=task,
        )
    )


@pytest.mark.parametrize(
    "name",
    sorted(name for name, cls in MODEL_REGISTRY.items() if cls not in (xFuserZImageModel, xFuserZImageTurboModel)),
)
def test_registered_model_refuses_invalid_ulysses_before_loading(name):
    model_class = MODEL_REGISTRY[name]
    message = (
        "--ulysses_degree must divide" if model_class.capabilities.ulysses_degree else "does not support ulysses_degree"
    )
    with pytest.raises(ValueError, match=message):
        _build(name, 97)


@pytest.mark.parametrize("name", sorted(MODEL_REGISTRY))
def test_registered_model_accepts_supported_ulysses_before_loading(name):
    degree = 2 if MODEL_REGISTRY[name].capabilities.ulysses_degree else 1
    model = _build(name, degree)
    assert model.pipe is None


@pytest.mark.parametrize(
    ("name", "heads", "accepted", "refused"),
    [
        ("FLUX.1-dev", 24, 3, 5),
        ("FLUX.1-Kontext-dev", 24, 3, 5),
        ("FLUX.2-dev", 48, 16, 32),
        ("FLUX.2-klein-9B", 32, 16, 3),
        ("FLUX.2-klein-4B", 24, 3, 16),
        ("SD3.5", 38, 19, 3),
        ("HunyuanVideo", 24, 3, 16),
        ("Hunyuanvideo-1.5", 16, 16, 3),
        ("Hunyuanvideo-1.5-Distilled", 16, 16, 3),
        ("Hunyuanvideo-1.5-Sparse", 16, 16, 3),
        ("LTX-2", 32, 16, 3),
        ("LTX-2.3", 32, 16, 3),
        ("LTX-2.5", 32, 16, 3),
        ("LTX-2.5-full", 32, 16, 3),
        ("Krea-2-Raw", 48, 16, 32),
        ("Krea-2-Turbo", 48, 16, 32),
        ("LingBot-Video-MoE", 16, 16, 3),
        ("LingBot-Video-Dense", 16, 16, 3),
        ("MiniMax-H3", 56, 8, 3),
        ("MiniMax-H3-Ref2VA", 56, 8, 3),
        ("FastH3", 56, 8, 3),
        ("FastH3-Dense", 56, 8, 3),
        ("Qwen-Image", 24, 3, 16),
        ("Qwen-Image-2512", 24, 3, 16),
        ("Qwen-Image-Edit", 24, 3, 16),
        ("Qwen-Image-Edit-2509", 24, 3, 16),
        ("Qwen-Image-Edit-2511", 24, 3, 16),
    ],
)
def test_ulysses_degrees_follow_each_checkpoint_head_layout(name, heads, accepted, refused):
    assert _build(name, accepted).pipe is None
    with pytest.raises(ValueError, match=rf"has {heads} attention heads.*got {refused}\."):
        _build(name, refused)


@pytest.mark.parametrize("name", ["Lumina2", "Cosmos3-Super", "Cosmos3-Nano"])
def test_compact_kv_heads_also_limit_ulysses(name):
    # 3 divides Lumina2's 24 Q heads; 16 divides both Cosmos3 Q layouts.
    # Neither degree divides their eight compact KV heads.
    degree = 3 if name == "Lumina2" else 16
    with pytest.raises(ValueError, match=rf"and 8 KV heads.*\(1, 2, 4, 8\); got {degree}\."):
        _build(name, degree)
    assert _build(name, 8).pipe is None


@pytest.mark.parametrize("name", ["Z-Image", "Z-Image-Turbo", "Tongyi-MAI/Z-Image", "Tongyi-MAI/Z-Image-Turbo"])
def test_head_padding_preserves_non_divisible_ulysses(name):
    assert _build(name, 8).pipe is None


@pytest.mark.parametrize("name", ["Lumina2", "Krea-2-Raw", "MiniMax-H3"])
def test_invalid_ulysses_does_not_recommend_unsupported_ring(name):
    with pytest.raises(ValueError, match="--ulysses_degree must divide") as exc:
        _build(name, 5)
    assert "--ring_degree" not in str(exc.value)
