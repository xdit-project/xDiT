"""The LTX-Video wrapper on one rank must compute what diffusers' model computes.

A tiny random LTXVideoTransformer3DModel is run on CPU next to the xDiT wrapper
holding the same weights. Prompts of different lengths make the text mask
differ per sample, which the wrapper serves by gathering valid keys instead of
diffusers' additive bias. Nothing is downloaded.
"""

from types import SimpleNamespace
from unittest import mock

import pytest
import torch

pytest.importorskip("diffusers")

from diffusers.models.transformers.transformer_ltx import LTXVideoTransformer3DModel

from xfuser.core.attention.spec import AttentionBackendType
from xfuser.model_executor.models.transformers import transformer_ltx_video as ltx_video
from xfuser.model_executor.models.transformers.transformer_ltx_video import (
    xFuserLTXVideoTransformer3DWrapper,
)

_CONFIG = dict(
    in_channels=8,
    out_channels=8,
    num_attention_heads=2,
    attention_head_dim=16,
    cross_attention_dim=32,
    num_layers=2,
    caption_channels=24,
)
_FRAMES, _HEIGHT, _WIDTH = 2, 3, 4
_TOKENS = _FRAMES * _HEIGHT * _WIDTH
_TEXT_LEN = 7


@pytest.fixture(autouse=True)
def _sdpa_runtime():
    """attention() reads its backend from the runtime state, which needs a launched run."""
    runtime = SimpleNamespace(attention_backend=AttentionBackendType.SDPA)
    with mock.patch("xfuser.model_executor.layers.usp.get_runtime_state", return_value=runtime):
        yield


def _models():
    torch.manual_seed(0)
    reference = LTXVideoTransformer3DModel(**_CONFIG).eval()
    wrapper = xFuserLTXVideoTransformer3DWrapper.from_config(reference.config).eval()
    wrapper.load_state_dict(reference.state_dict())
    return reference, wrapper


def _inputs(text_lengths, timestep):
    generator = torch.Generator().manual_seed(1)
    batch = len(text_lengths)
    mask = torch.zeros(batch, _TEXT_LEN, dtype=torch.int64)
    for row, length in enumerate(text_lengths):
        mask[row, :length] = 1
    return dict(
        hidden_states=torch.randn(batch, _TOKENS, _CONFIG["in_channels"], generator=generator),
        encoder_hidden_states=torch.randn(batch, _TEXT_LEN, _CONFIG["caption_channels"], generator=generator),
        timestep=timestep,
        encoder_attention_mask=mask,
        num_frames=_FRAMES,
        height=_HEIGHT,
        width=_WIDTH,
        rope_interpolation_scale=(8 / 25, 32, 32),
        return_dict=False,
    )


@pytest.mark.parametrize(
    "text_lengths",
    [(3, 7), (4, 4), (7, 7)],
    ids=["per-sample-text-mask", "shared-text-mask", "unpadded-text"],
)
def test_wrapper_matches_diffusers_with_padded_text(text_lengths):
    reference, wrapper = _models()
    inputs = _inputs(text_lengths, timestep=torch.tensor([700.0, 700.0]))

    with torch.no_grad():
        expected = reference(**inputs)[0]
        actual = wrapper(**inputs)[0]

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


def test_wrapper_matches_diffusers_with_per_token_timestep():
    """Image-to-video passes [batch, tokens] timesteps with the first frame at zero."""
    reference, wrapper = _models()
    timestep = torch.full((2, _TOKENS), 600.0)
    timestep[:, : _HEIGHT * _WIDTH] = 0.0
    inputs = _inputs((5, 2), timestep=timestep)

    with torch.no_grad():
        expected = reference(**inputs)[0]
        actual = wrapper(**inputs)[0]

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


def test_wrapper_matches_diffusers_with_explicit_video_coords():
    """LTXConditionPipeline passes precomputed coordinates instead of a grid size."""
    reference, wrapper = _models()
    inputs = _inputs((6, 3), timestep=torch.tensor([300.0, 300.0]))
    grid = torch.stack(
        torch.meshgrid(torch.arange(_FRAMES), torch.arange(_HEIGHT), torch.arange(_WIDTH), indexing="ij")
    ).flatten(1)
    inputs["video_coords"] = (grid.float() * torch.tensor([[0.32], [32.0], [32.0]])).expand(2, -1, -1)

    with torch.no_grad():
        expected = reference(**inputs)[0]
        actual = wrapper(**inputs)[0]

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


def test_sample_without_valid_text_attends_to_every_key_like_diffusers():
    """An all-zero mask row is an all-equal bias in diffusers: plain attention over every key."""
    reference, wrapper = _models()
    inputs = _inputs((0, 4), timestep=torch.tensor([500.0, 500.0]))

    with torch.no_grad():
        expected = reference(**inputs)[0]
        actual = wrapper(**inputs)[0]

    # Adding -10000 to every fp32 score rounds them to ~1e-3, hence the looser bound.
    torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-4)


def test_sequence_parallelism_without_ulysses_or_ring_is_rejected():
    """Without long-context attention the SP group reports Ulysses and ring degree 1.

    Sharding the sequence anyway would make each rank attend only to its own shard.
    """
    _, wrapper = _models()
    inputs = _inputs((3, 7), timestep=torch.tensor([700.0, 700.0]))
    layout = ltx_video._ParallelLayout(sp_world_size=2, sp_rank=0, ulysses_world_size=1, ring_world_size=1)

    with (
        mock.patch.object(ltx_video, "_parallel_layout", return_value=layout),
        torch.no_grad(),
        pytest.raises(RuntimeError, match="unavailable on this host"),
    ):
        wrapper(**inputs)
