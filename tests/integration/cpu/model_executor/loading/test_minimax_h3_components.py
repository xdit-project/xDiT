"""Exercise H3's component loading with tiny local files and no Hub access."""

import json

import pytest
import torch

pytest.importorskip("diffusers.modular_pipelines.minimax_h3")

from diffusers import ConfigMixin, ModelMixin, ModularPipeline
from diffusers.configuration_utils import register_to_config
from diffusers.modular_pipelines.modular_pipeline import ModularPipelineBlocks
from diffusers.modular_pipelines.modular_pipeline_utils import ComponentSpec
from transformers import ProcessorMixin

from xfuser.model_executor.models.runner_models.minimax_h3 import _load_minimax_h3_components


class TinyComponent(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(self, width=512):
        super().__init__()
        self.first = torch.nn.Parameter(torch.ones(width, device="cpu"))
        self.second = torch.nn.Parameter(torch.zeros(width, device="cpu"))


class TinyProcessor(ProcessorMixin):
    """Exercise the shared tokenizer loader without image/video dependencies."""

    def __init__(self, tokenizer, chat_template=None):
        super().__init__(tokenizer, chat_template=chat_template)


def _pipeline(*specs):
    class Blocks(ModularPipelineBlocks):
        @property
        def expected_components(self):
            return list(specs)

    return ModularPipeline(blocks=Blocks())


@pytest.fixture(autouse=True)
def offline(tmp_path, monkeypatch):
    cache = tmp_path / "hub"
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("HF_HUB_CACHE", str(cache))
    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_OFFLINE", True)
    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_CACHE", str(cache))
    metadata_requests = []

    def unexpected_metadata(*args, **kwargs):
        metadata_requests.append((args, kwargs))
        raise AssertionError("Component loading attempted a Hub metadata request")

    monkeypatch.setattr("diffusers.utils.hub_utils.model_info", unexpected_metadata)
    yield cache
    assert not metadata_requests


def test_processor_loads_h3_special_tokens_without_model_config(tmp_path, monkeypatch):
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from transformers import AutoConfig, Qwen2TokenizerFast

    processor_dir = tmp_path / "processor"
    tokenizer = Qwen2TokenizerFast(
        tokenizer_object=Tokenizer(BPE({"<|endoftext|>": 0, "x": 1}, [])),
        additional_special_tokens=["<d>"],
    )
    tokenizer.save_pretrained(processor_dir)
    template = "{{ messages }}"
    (processor_dir / "chat_template.json").write_text(json.dumps({"chat_template": template}))

    def unexpected_config(*args, **kwargs):
        raise AssertionError("H3's processor has no model config.json")

    monkeypatch.setattr(AutoConfig, "from_pretrained", unexpected_config)
    pipe = _pipeline(
        ComponentSpec(
            name="processor",
            type_hint=TinyProcessor,
            pretrained_model_name_or_path=str(tmp_path),
            subfolder="processor",
        )
    )

    _load_minimax_h3_components(pipe)

    assert pipe.processor.chat_template == template
    assert pipe.processor.tokenizer.encode("<d>", add_special_tokens=False) == [tokenizer.convert_tokens_to_ids("<d>")]
    assert "<d>" in pipe.processor.tokenizer.all_special_tokens
    assert not (processor_dir / "config.json").exists()


def test_sharded_components_use_their_own_cached_repository(offline):
    specs = []
    for name, value, revision in [("vae", 1, "a" * 40), ("audio_vae", 2, "b" * 40)]:
        root = offline / f"models--example--{name}" / "snapshots" / revision
        component = TinyComponent()
        with torch.no_grad():
            component.first.fill_(value)
        component.save_pretrained(root / name, max_shard_size="1KB")
        assert (root / name / "diffusion_pytorch_model.safetensors.index.json").exists()
        repo = f"example/{name}"
        specs.append(
            ComponentSpec(
                name=name,
                type_hint=TinyComponent,
                pretrained_model_name_or_path=repo,
                subfolder=name,
                revision=revision,
            )
        )

    pipe = _pipeline(*specs)

    _load_minimax_h3_components(pipe)

    torch.testing.assert_close(pipe.vae.first, torch.ones(512, dtype=torch.bfloat16, device="cpu"))
    torch.testing.assert_close(pipe.audio_vae.first, torch.full((512,), 2, dtype=torch.bfloat16, device="cpu"))


def test_missing_component_stops_loading_before_inference(tmp_path):
    pipe = _pipeline(
        ComponentSpec(
            name="vae",
            type_hint=TinyComponent,
            pretrained_model_name_or_path=str(tmp_path),
            subfolder="vae",
        )
    )

    with pytest.raises(RuntimeError, match="failed to load required components: vae"):
        _load_minimax_h3_components(pipe)


def test_missing_offline_snapshot_stops_loading():
    pipe = _pipeline(
        ComponentSpec(
            name="vae",
            type_hint=TinyComponent,
            pretrained_model_name_or_path="example/uncached",
            subfolder="vae",
        )
    )

    with pytest.raises(RuntimeError, match="failed to load required components: vae"):
        _load_minimax_h3_components(pipe)
