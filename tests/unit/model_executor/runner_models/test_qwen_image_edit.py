"""Each Qwen-Image-Edit checkpoint must load through the pipeline diffusers uses for it.

Qwen-Image-Edit-2509 and -2511 are Edit Plus checkpoints: diffusers loads them
with QwenImageEditPlusPipeline, which builds its prompt from a list of reference
images. The runner loaded them through QwenImageEditPipeline instead.
"""

from types import SimpleNamespace

import diffusers
import pytest
from diffusers import DiffusionPipeline, QwenImageEditPipeline

# Diffusers added the Edit Plus pipeline after Qwen-Image-Edit itself.
QwenImageEditPlusPipeline = getattr(diffusers, "QwenImageEditPlusPipeline", None)
requires_edit_plus = pytest.mark.skipif(
    QwenImageEditPlusPipeline is None, reason="this diffusers release has no QwenImageEditPlusPipeline"
)


@pytest.fixture
def build(monkeypatch):
    from xfuser import xFuserArgs
    from xfuser.model_executor.models.runner_models import qwen  # noqa: F401  registers
    from xfuser.model_executor.models.runner_models.base_model import MODEL_REGISTRY

    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")

    def _build(name):
        return MODEL_REGISTRY[name](xFuserArgs(model=name))

    return _build


@pytest.fixture
def loaded_classes(monkeypatch):
    loaded = []

    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        loaded.append((cls, pretrained_model_name_or_path))
        return SimpleNamespace()

    monkeypatch.setattr(DiffusionPipeline, "from_pretrained", classmethod(from_pretrained))
    return loaded


def _load(model):
    model.loader = SimpleNamespace(
        load_transformer=lambda wrapper_cls: object(),
        plan_text_encoders=lambda: ({}, None),
    )
    model._load_model()


@requires_edit_plus
@pytest.mark.parametrize(
    ("name", "checkpoint"),
    [
        ("Qwen/Qwen-Image-Edit-2509", "Qwen/Qwen-Image-Edit-2509"),
        ("Qwen-Image-Edit-2509", "Qwen/Qwen-Image-Edit-2509"),
        ("Qwen/Qwen-Image-Edit-2511", "Qwen/Qwen-Image-Edit-2511"),
        ("Qwen-Image-Edit-2511", "Qwen/Qwen-Image-Edit-2511"),
    ],
)
def test_edit_plus_checkpoints_load_the_edit_plus_pipeline(build, loaded_classes, name, checkpoint):
    _load(build(name))

    [(pipeline_cls, loaded_checkpoint)] = loaded_classes
    assert loaded_checkpoint == checkpoint
    assert issubclass(pipeline_cls, QwenImageEditPlusPipeline)


@pytest.mark.parametrize("name", ["Qwen/Qwen-Image-Edit", "Qwen-Image-Edit"])
def test_original_edit_checkpoint_keeps_the_edit_pipeline(build, loaded_classes, name):
    _load(build(name))

    [(pipeline_cls, loaded_checkpoint)] = loaded_classes
    assert loaded_checkpoint == "Qwen/Qwen-Image-Edit"
    assert issubclass(pipeline_cls, QwenImageEditPipeline)
    assert QwenImageEditPlusPipeline is None or not issubclass(pipeline_cls, QwenImageEditPlusPipeline)


def test_edit_plus_passes_every_reference_image(build):
    model = build("Qwen/Qwen-Image-Edit-2509")
    calls = []

    def pipe(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(images=["edited"])

    model.pipe = pipe
    model._make_generator = lambda seed: None
    references = ["first.png", "second.png"]
    input_args = {
        "input_images": references,
        "prompt": "put the cat next to the dog",
        "negative_prompt": " ",
        "num_inference_steps": 2,
        "guidance_scale": 4.0,
        "seed": 0,
    }

    model._run_pipe(input_args)

    assert calls[0]["image"] == references
