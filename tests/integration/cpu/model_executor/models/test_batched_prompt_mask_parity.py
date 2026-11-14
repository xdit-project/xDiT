"""Z-Image, Qwen-Image and HunyuanVideo-1.5 match their stock transformers for prompts of different lengths.

A batch pads its shorter prompts, and the stock models mask that padding out of
the attention keys. The xDiT wrappers must give the same output as diffusers on
one rank, shorter prompt included. Runs tiny random fp32 configs on CPU against
the `single_rank` Gloo group (conftest.py); nothing is downloaded.
"""

import pytest
import torch

pytest.importorskip("diffusers.models.transformers.transformer_hunyuan_video15")
pytest.importorskip("diffusers.models.transformers.transformer_z_image")

from diffusers import QwenImageTransformer2DModel, ZImageTransformer2DModel  # noqa: E402
from diffusers.models.transformers.transformer_hunyuan_video15 import HunyuanVideo15Transformer3DModel  # noqa: E402

from xfuser.core.distributed import runtime_state  # noqa: E402
from xfuser.model_executor.models.transformers.transformer_hunyuan_video15 import (  # noqa: E402
    xFuserHunyuanVideo15Transformer3DWrapper,
)
from xfuser.model_executor.models.transformers.transformer_qwen import xFuserQwenImageTransformerWrapper  # noqa: E402
from xfuser.model_executor.models.transformers.transformer_z_image import xFuserZImageTransformer2DWrapper  # noqa: E402

_HEADS, _HEAD_DIM = 4, 16


class _Pipeline:
    def __init__(self, transformer):
        self.transformer = transformer


def _compare(engine_config, reference, wrapper, inputs):
    wrapper.load_state_dict(reference.state_dict())
    runtime_state.initialize_runtime_state(pipeline=_Pipeline(wrapper), engine_config=engine_config)
    with torch.no_grad():
        expected = reference(**inputs, return_dict=False)[0]
        actual = wrapper(**inputs, return_dict=False)[0]
    return expected, actual


def _z_image_models():
    config = dict(
        all_patch_size=(2,),
        all_f_patch_size=(1,),
        in_channels=4,
        dim=_HEADS * _HEAD_DIM,
        n_layers=2,
        n_refiner_layers=1,
        n_heads=_HEADS,
        n_kv_heads=_HEADS,
        cap_feat_dim=16,
        axes_dims=[4, 6, 6],
        axes_lens=[128, 64, 64],
    )
    return ZImageTransformer2DModel(**config).eval(), xFuserZImageTransformer2DWrapper(**config).eval()


def _qwen_models():
    config = dict(
        patch_size=2,
        in_channels=16,
        out_channels=4,
        num_layers=2,
        attention_head_dim=_HEAD_DIM,
        num_attention_heads=_HEADS,
        joint_attention_dim=16,
        axes_dims_rope=(4, 6, 6),
    )
    return QwenImageTransformer2DModel(**config).eval(), xFuserQwenImageTransformerWrapper(**config).eval()


def _hunyuan_models(task_type):
    config = dict(
        in_channels=4,
        out_channels=4,
        num_attention_heads=_HEADS,
        attention_head_dim=_HEAD_DIM,
        num_layers=2,
        num_refiner_layers=1,
        mlp_ratio=2.0,
        patch_size=1,
        patch_size_t=1,
        qk_norm="rms_norm",
        text_embed_dim=16,
        text_embed_2_dim=8,
        image_embed_dim=8,
        rope_axes_dim=(4, 6, 6),
        task_type=task_type,
    )
    return HunyuanVideo15Transformer3DModel(**config).eval(), xFuserHunyuanVideo15Transformer3DWrapper(**config).eval()


def _prompt_mask(lengths, size):
    return (torch.arange(size) < torch.tensor(lengths)[:, None]).float()


@pytest.mark.parametrize(
    "captions", [(10, 40), (40, 10), (10,), (40, 40)], ids=["short-first", "short-last", "one", "equal"]
)
def test_z_image_matches_diffusers_for_captions_of_different_lengths(single_rank, captions):
    torch.manual_seed(0)
    generator = torch.Generator().manual_seed(1)
    inputs = dict(
        x=[torch.randn(4, 1, 8, 8, generator=generator) for _ in captions],
        t=torch.full((len(captions),), 0.5),
        cap_feats=[torch.randn(length, 16, generator=generator) for length in captions],
    )

    expected, actual = _compare(single_rank, *_z_image_models(), inputs)

    for stock, ours in zip(expected, actual):
        torch.testing.assert_close(ours, stock, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("prompts", [(8, 5), (5, 8), (7,), (6, 6)], ids=["short-last", "short-first", "one", "equal"])
def test_qwen_image_matches_diffusers_for_prompts_of_different_lengths(single_rank, prompts):
    torch.manual_seed(0)
    generator = torch.Generator().manual_seed(1)
    batch, text_tokens = len(prompts), max(prompts)
    inputs = dict(
        hidden_states=torch.randn(batch, 24, 16, generator=generator),
        encoder_hidden_states=torch.randn(batch, text_tokens, 16, generator=generator),
        encoder_hidden_states_mask=_prompt_mask(prompts, text_tokens),
        timestep=torch.full((batch,), 0.5),
        img_shapes=[[(1, 4, 6)]] * batch,
    )

    expected, actual = _compare(single_rank, *_qwen_models(), inputs)

    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("task_type", ["t2v", "i2v"])
@pytest.mark.parametrize("prompts", [(7, 4), (4, 7), (5,), (6, 6)], ids=["short-last", "short-first", "one", "equal"])
def test_hunyuan_video15_matches_diffusers_for_prompts_of_different_lengths(single_rank, task_type, prompts):
    torch.manual_seed(0)
    generator = torch.Generator().manual_seed(1)
    batch, text_tokens, text_2_tokens = len(prompts), max(prompts) + 2, 3
    # The byT5 prompts differ in length too, unless the prompts are equal.
    text_2_lengths = [text_2_tokens] * (batch - 1) + [text_2_tokens if len(set(prompts)) == 1 else 1]
    # Text-to-video passes all-zero image embeddings, which the model masks out.
    image_embeds = torch.randn(batch, 2, 8, generator=generator)
    if task_type == "t2v":
        image_embeds = torch.zeros_like(image_embeds)
    inputs = dict(
        hidden_states=torch.randn(batch, 4, 2, 3, 4, generator=generator),
        timestep=torch.full((batch,), 500.0),
        encoder_hidden_states=torch.randn(batch, text_tokens, 16, generator=generator),
        encoder_attention_mask=_prompt_mask(prompts, text_tokens),
        encoder_hidden_states_2=torch.randn(batch, text_2_tokens, 8, generator=generator),
        encoder_attention_mask_2=_prompt_mask(text_2_lengths, text_2_tokens),
        image_embeds=image_embeds,
    )

    expected, actual = _compare(single_rank, *_hunyuan_models(task_type), inputs)

    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
