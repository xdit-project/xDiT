"""The xDiT Cosmos3 transformer answers the calls the diffusers pipeline makes.

diffusers 0.40 added ``return_dict`` to Cosmos3OmniTransformer.forward, and its
pipeline passes ``return_dict=False`` on every denoising step. The wrapper has
to accept it and return what the stock transformer returns.
"""

import pytest
import torch

cosmos3 = pytest.importorskip("diffusers.models.transformers.transformer_cosmos3")
if not hasattr(cosmos3, "Cosmos3OmniTransformerOutput"):
    pytest.skip("this diffusers' Cosmos3 transformer has no return_dict", allow_module_level=True)

from xfuser.model_executor.models.transformers.transformer_cosmos3 import (  # noqa: E402
    get_cosmos3_transformer_wrapper_class,
)

# The tiny configuration diffusers tests its own Cosmos3 transformer with.
CONFIG = {
    "head_dim": 6,
    "hidden_act": "relu2",
    "hidden_size": 12,
    "intermediate_size": 24,
    "latent_channel": 2,
    "latent_patch_size": 1,
    "num_attention_heads": 2,
    "num_hidden_layers": 2,
    "num_key_value_heads": 1,
    "patch_latent_dim": 2,
    "qk_norm_for_text": False,
    "rms_norm_eps": 1e-5,
    "rope_axes_dim": [1, 1, 1],
    "rope_theta": 1e8,
    "vocab_size": 32,
}


def _inputs(height=2, width=2):
    num_vision_tokens = height * width
    sequence_length = 2 + num_vision_tokens
    vision_indexes = torch.arange(2, sequence_length)
    generator = torch.Generator().manual_seed(0)
    return {
        "input_ids": torch.tensor([1, 2]),
        "text_indexes": torch.tensor([0, 1]),
        "position_ids": torch.zeros((3, sequence_length), dtype=torch.long),
        "und_len": 2,
        "sequence_length": sequence_length,
        "vision_tokens": [torch.randn((1, 2, 1, height, width), generator=generator)],
        "vision_token_shapes": [(1, height, width)],
        "vision_sequence_indexes": vision_indexes,
        "vision_mse_loss_indexes": vision_indexes,
        "vision_timesteps": torch.ones(num_vision_tokens),
        "vision_noisy_frame_indexes": [torch.tensor([0])],
    }


@pytest.fixture
def models():
    torch.manual_seed(0)
    stock = cosmos3.Cosmos3OmniTransformer(**CONFIG).eval()
    wrapper = get_cosmos3_transformer_wrapper_class()(**CONFIG).eval()
    wrapper.load_state_dict(stock.state_dict())
    wrapper._install_xfuser_processors()
    return stock, wrapper


@torch.no_grad()
def test_return_dict_false_returns_the_stock_tuple(models):
    stock, wrapper = models
    expected = stock(**_inputs(), return_dict=False)

    out = wrapper(**_inputs(), return_dict=False)

    assert isinstance(out, tuple) and len(out) == 3
    torch.testing.assert_close(out[0][0], expected[0][0])
    assert out[1] is None and out[2] is None


@torch.no_grad()
def test_return_dict_defaults_to_the_stock_output_class(models):
    stock, wrapper = models
    expected = stock(**_inputs())

    out = wrapper(**_inputs())

    assert isinstance(out, cosmos3.Cosmos3OmniTransformerOutput)
    torch.testing.assert_close(out.sample[0], expected.sample[0])
    assert out.sound is None and out.action is None
