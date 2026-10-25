"""Trimming FLUX.3's Qwen3-VL text encoder must leave every hidden state it reads unchanged."""

from types import SimpleNamespace

import pytest
import torch

transformers = pytest.importorskip("transformers")
if not hasattr(transformers, "Qwen3VLForConditionalGeneration"):
    pytest.skip("installed transformers has no Qwen3-VL", allow_module_level=True)

from xfuser.model_executor.models.runner_models.flux3 import (  # noqa: E402
    _deepest_text_encoder_layer,
    trim_text_encoder,
)

LAYERS = 6
READ = (2, 4)


def _tiny_qwen3_vl():
    config = transformers.Qwen3VLConfig(
        text_config={
            "vocab_size": 64,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": LAYERS,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "rope_scaling": {"rope_type": "default", "mrope_section": [1, 1, 2]},
        },
        vision_config={"depth": 1, "hidden_size": 16, "intermediate_size": 32, "num_heads": 2, "out_hidden_size": 32},
    )
    torch.manual_seed(0)
    return transformers.Qwen3VLForConditionalGeneration(config).eval()


def _hidden_states(model, input_ids):
    with torch.no_grad():
        return model(input_ids=input_ids, output_hidden_states=True, use_cache=False).hidden_states


def test_the_hidden_states_read_are_unchanged():
    model = _tiny_qwen3_vl()
    input_ids = torch.randint(0, 64, (1, 9), generator=torch.Generator().manual_seed(1))
    stock = _hidden_states(model, input_ids)

    trim_text_encoder(model, max(READ))

    trimmed = _hidden_states(model, input_ids)
    assert len(model.model.language_model.layers) == max(READ)
    for index in (0, *READ):
        assert torch.equal(trimmed[index], stock[index]), index


def test_an_encoder_no_deeper_than_needed_is_left_alone():
    model = _tiny_qwen3_vl()
    norm, lm_head = model.model.language_model.norm, model.lm_head

    trim_text_encoder(model, LAYERS)

    assert model.model.language_model.norm is norm and model.lm_head is lm_head


def test_the_deepest_layer_comes_from_the_encoder_step():
    step = SimpleNamespace(inputs=[SimpleNamespace(name="prompt", default=None)])
    step.inputs.append(SimpleNamespace(name="text_encoder_out_layers", default=(4, 8, 32, 12)))
    blocks = SimpleNamespace(sub_blocks={"text_encoder": step})

    assert _deepest_text_encoder_layer(blocks) == 32
