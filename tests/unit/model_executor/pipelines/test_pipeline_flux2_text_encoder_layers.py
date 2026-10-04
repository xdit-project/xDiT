"""The PipeFusion FLUX.2 pipelines must condition on the prompt as diffusers does.

dev and klein stack different hidden-state layers of their text encoders. The PipeFusion
wrappers share one ``__call__``, which must still hand each variant's denoising loop the
prompt embedding that variant's own diffusers pipeline produces.
"""

from types import SimpleNamespace

import pytest
import torch

pipeline_flux2 = pytest.importorskip("xfuser.model_executor.pipelines.pipeline_flux2")

from diffusers import (  # noqa: E402
    FlowMatchEulerDiscreteScheduler,
    Flux2KleinPipeline,
    Flux2Transformer2DModel,
)
from transformers import Qwen3Config, Qwen3ForCausalLM  # noqa: E402

PROMPT = "a lighthouse at dusk"
MAX_SEQUENCE_LENGTH = 6
TEXT_HIDDEN_SIZE = 8
# Deep enough for both the dev and klein layer choices, so a wrong choice still runs.
TEXT_LAYERS = 31


class _Tokenizer:
    """Stands in for klein's Qwen tokenizer: any text becomes the same fixed ids."""

    def apply_chat_template(self, messages, **kwargs):
        return messages[-1]["content"]

    def __call__(self, text, max_length, **kwargs):
        input_ids = torch.arange(1, max_length + 1).unsqueeze(0)
        return {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}


def _tiny_klein_pipeline():
    torch.manual_seed(0)
    text_encoder = Qwen3ForCausalLM(
        Qwen3Config(
            vocab_size=16,
            hidden_size=TEXT_HIDDEN_SIZE,
            intermediate_size=16,
            num_hidden_layers=TEXT_LAYERS,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=4,
        )
    ).eval()
    transformer = Flux2Transformer2DModel(
        in_channels=8,
        num_layers=1,
        num_single_layers=1,
        attention_head_dim=16,
        num_attention_heads=2,
        joint_attention_dim=3 * TEXT_HIDDEN_SIZE,
        timestep_guidance_channels=32,
        axes_dims_rope=(4, 4, 4, 4),
        guidance_embeds=False,
    ).eval()
    return Flux2KleinPipeline(
        scheduler=FlowMatchEulerDiscreteScheduler(),
        vae=None,
        text_encoder=text_encoder,
        tokenizer=_Tokenizer(),
        transformer=transformer,
        is_distilled=True,
    )


class _DenoisingStarted(Exception):
    def __init__(self, prompt_embeds):
        super().__init__()
        self.prompt_embeds = prompt_embeds


@pytest.fixture
def single_stage_pipefusion(monkeypatch):
    """Route ``__call__`` into xDiT's own loop on one process, without a process group."""
    runtime_state = SimpleNamespace(
        set_input_parameters=lambda **kwargs: None,
        runtime_config=SimpleNamespace(warmup_steps=0),
    )
    monkeypatch.setattr(pipeline_flux2, "get_runtime_state", lambda: runtime_state)
    monkeypatch.setattr(pipeline_flux2, "get_pipeline_parallel_world_size", lambda: 1)
    monkeypatch.setattr(pipeline_flux2.xFuserFlux2PipelineBase, "use_naive_forward", lambda self: False)


def _xfuser_klein(pipeline, monkeypatch):
    wrapper = object.__new__(pipeline_flux2.xFuserFlux2KleinPipeline)
    wrapper.module = pipeline

    def stop_before_denoising(*, prompt_embeds, **kwargs):
        raise _DenoisingStarted(prompt_embeds)

    monkeypatch.setattr(wrapper, "_sync_pipeline", stop_before_denoising)
    return wrapper


@torch.no_grad()
def test_klein_pipefusion_denoises_with_diffusers_klein_prompt_embedding(single_stage_pipefusion, monkeypatch):
    pipeline = _tiny_klein_pipeline()
    expected, _ = pipeline.encode_prompt(PROMPT, device="cpu", max_sequence_length=MAX_SEQUENCE_LENGTH)

    with pytest.raises(_DenoisingStarted) as started:
        _xfuser_klein(pipeline, monkeypatch)(
            prompt=PROMPT,
            height=64,
            width=64,
            num_inference_steps=1,
            guidance_scale=1.0,
            max_sequence_length=MAX_SEQUENCE_LENGTH,
            output_type="latent",
        )

    torch.testing.assert_close(started.value.prompt_embeds, expected, rtol=0, atol=0)
