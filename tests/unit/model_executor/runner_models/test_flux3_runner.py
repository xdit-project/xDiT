import inspect
from types import SimpleNamespace

import pytest
import torch
from PIL import Image


@pytest.fixture
def build_flux3(monkeypatch):
    from xfuser import xFuserArgs
    from xfuser.runner import xFuserModelRunner

    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    runner = object.__new__(xFuserModelRunner)

    def _build(name="FLUX.3-image-klein", **config):
        return runner._select_model(name, xFuserArgs(model=name, **config))

    return _build


def _args(**overrides):
    args = {
        "prompt": ["a bison"],
        "dataset_path": None,
        "height": None,
        "width": None,
        "num_inference_steps": 16,
        "seed": 0,
        "input_images": [],
    }
    return {**args, **overrides}


@pytest.mark.parametrize(
    ("name", "checkpoint"),
    [
        ("FLUX.3-image-klein", "black-forest-labs/flux-3-image-klein"),
        ("flux-3-image-klein", "black-forest-labs/flux-3-image-klein"),
        ("black-forest-labs/flux-3-image-klein", "black-forest-labs/flux-3-image-klein"),
        ("FLUX.3-image-klein-base", "black-forest-labs/flux-3-image-klein-base"),
        ("flux-3-image-klein-base", "black-forest-labs/flux-3-image-klein-base"),
        ("black-forest-labs/flux-3-image-klein-base", "black-forest-labs/flux-3-image-klein-base"),
    ],
)
def test_name_selects_its_checkpoint(build_flux3, name, checkpoint):
    model = build_flux3(name)

    assert model.loader.checkpoint_request().model_name_or_path == checkpoint


def test_batching_more_than_one_prompt_per_call_is_refused(build_flux3):
    with pytest.raises(ValueError, match="--batch_size must be 1"):
        build_flux3(batch_size=2)


def test_the_distilled_model_refuses_cfg_parallel(build_flux3):
    with pytest.raises(ValueError, match="does not support use_cfg_parallel"):
        build_flux3(use_cfg_parallel=True)


def test_the_base_model_validates_with_cfg_parallel(build_flux3):
    # Construction validates the config and raises for a flag the model does not support.
    build_flux3("FLUX.3-image-klein-base", use_cfg_parallel=True)


def test_cfg_parallel_without_guidance_is_refused(build_flux3):
    model = build_flux3("FLUX.3-image-klein-base", use_cfg_parallel=True)

    with pytest.raises(ValueError, match="--guidance_scale 1 disables it"):
        model._validate_args(_args(guidance_scale=1.0))


def test_negative_prompt_is_refused(build_flux3):
    model = build_flux3()

    with pytest.raises(ValueError, match="does not accept a negative prompt"):
        model._validate_args(_args(negative_prompt="blurry"))


def test_size_off_the_token_grid_is_refused(build_flux3):
    model = build_flux3()

    with pytest.raises(ValueError, match="--width divisible by 16, got 1000"):
        model._validate_args(_args(height=1024, width=1000))


def test_text_to_image_defaults_to_1024_square(build_flux3):
    model = build_flux3()

    args = model._preprocess_args_images(_args())

    assert (args["height"], args["width"]) == (1024, 1024)


def test_edit_leaves_the_size_to_the_reference(build_flux3, monkeypatch):
    model = build_flux3()
    monkeypatch.setattr(
        "xfuser.model_executor.models.runner_models.base_model.load_image",
        lambda path: Image.new("RGB", (1600, 900)),
    )

    args = model._preprocess_args_images(_args(input_images=["ref.png"]))

    assert (args["height"], args["width"]) == (None, None)


def test_resizing_references_needs_a_target_size(build_flux3, monkeypatch):
    model = build_flux3()
    monkeypatch.setattr(
        "xfuser.model_executor.models.runner_models.base_model.load_image",
        lambda path: Image.new("RGB", (1600, 900)),
    )

    with pytest.raises(ValueError, match="--resize_input_images needs --height and --width"):
        model._preprocess_args_images(_args(input_images=["ref.png"], resize_input_images=True))


class _RecordingPipe:
    def __init__(self, size):
        self.size = size
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return [Image.new("RGB", self.size)]


def test_each_prompt_is_its_own_pipeline_call(build_flux3):
    model = build_flux3()
    model.pipe = _RecordingPipe((1360, 768))
    reference = Image.new("RGB", (1600, 900))

    output = model._run_pipe(_args(prompt=["a bison", "a heron"], input_images=[reference], max_sequence_length=512))

    assert [call["prompt"] for call in model.pipe.calls] == ["a bison", "a heron"]
    assert all(call["image"] == [reference] for call in model.pipe.calls)
    assert all(call["max_sequence_length"] == 512 for call in model.pipe.calls)
    # Edits are sized by the pipeline; the saved name follows the image it produced.
    assert [(a["prompt"], a["height"], a["width"]) for a in output.pipe_args] == [
        ("a bison", 768, 1360),
        ("a heron", 768, 1360),
    ]


def test_a_seed_draws_the_same_noise_as_a_cpu_generator(build_flux3):
    # The model card seeds a CPU generator; the runner has to match it to reproduce its images.
    model = build_flux3()
    model.pipe = _RecordingPipe((1024, 1024))

    model._run_pipe(_args(seed=7))

    generator = model.pipe.calls[0]["generator"]
    assert generator.device.type == "cpu"
    assert torch.equal(torch.randn(4, generator=generator), torch.randn(4, generator=torch.Generator().manual_seed(7)))


def test_the_base_model_runs_the_requested_guidance_and_negative_prompt(build_flux3):
    model = build_flux3("FLUX.3-image-klein-base")
    pipe = _RecordingPipe((1024, 1024))
    pipe.guider = SimpleNamespace(guidance_scale=3.5)
    model.pipe = pipe

    model._run_pipe(_args(guidance_scale=5.0, negative_prompt="blurry", num_inference_steps=50))

    assert pipe.guider.guidance_scale == 5.0
    assert pipe.calls[0]["negative_prompt"] == "blurry"
    assert pipe.calls[0]["num_inference_steps"] == 50


class _Transformer(torch.nn.Module):
    def forward(self, hidden_states, encoder_hidden_states, timestep, img_ids, txt_ids):
        return hidden_states + 1


class _OpaqueWrapper(torch.nn.Module):
    """Stands in for FSDP1: the inner module behind a forward(*args, **kwargs)."""

    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


def test_a_wrapped_transformer_still_shows_its_forward_parameters(build_flux3):
    model = build_flux3()
    model.pipe = SimpleNamespace(transformer=_OpaqueWrapper(_Transformer()))

    model._expose_transformer_signature()

    transformer = model.pipe.transformer
    assert {"img_ids", "txt_ids"} <= set(inspect.signature(transformer.forward).parameters)
    out = transformer(torch.zeros(1), encoder_hidden_states=None, timestep=None, img_ids=None, txt_ids=None)
    assert torch.equal(out, torch.ones(1))


def test_an_unwrapped_transformer_is_left_alone(build_flux3):
    model = build_flux3()
    transformer = _Transformer()
    model.pipe = SimpleNamespace(transformer=transformer)

    model._expose_transformer_signature()

    assert "forward" not in vars(transformer)
