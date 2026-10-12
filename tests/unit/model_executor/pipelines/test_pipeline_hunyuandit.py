"""HunyuanDiT's image RoPE builds on every diffusers release xDiT supports.

get_2d_rotary_pos_embed defaults to output_type="np", and diffusers raises on
that default from 0.33.0 on, so the pipeline failed before its first step.
"""

from types import SimpleNamespace

import pytest
import torch
from diffusers.models import embeddings

from xfuser.model_executor.pipelines.pipeline_hunyuandit import _image_rotary_emb

# HunyuanDiT-v1.2: 1408 wide, 16 heads of 88, patch size 2.
TRANSFORMER = SimpleNamespace(config=SimpleNamespace(patch_size=2), inner_dim=1408, num_heads=16)


@pytest.mark.parametrize("height, width", [(1024, 1024), (768, 1280)])
def test_image_rotary_emb_is_a_tensor_pair_over_the_image_tokens(height, width):
    cos, sin = _image_rotary_emb(TRANSFORMER, height, width, torch.device("cpu"))

    tokens = (height // 16) * (width // 16)
    for part in (cos, sin):
        assert isinstance(part, torch.Tensor)
        assert part.shape == (tokens, 88)
        assert part.device.type == "cpu"


@pytest.mark.parametrize("height, width", [(512, 512), (1024, 1024)])
def test_image_rotary_emb_matches_the_numpy_embedding_it_replaces(height, width):
    """The values the pipeline got from output_type="np" before diffusers removed it.

    Square sizes only: for a crop that does not start at zero, diffusers'
    tensor path spaces the grid differently from its numpy one. The pipeline
    follows the stock diffusers pipeline, which uses the tensor path.
    """
    legacy = getattr(embeddings, "_get_2d_rotary_pos_embed_np", None)
    if legacy is None:
        pytest.skip("this diffusers no longer ships the numpy implementation")
    from diffusers.pipelines.hunyuandit.pipeline_hunyuandit import get_resize_crop_region_for_grid

    grid = (height // 16, width // 16)
    expected = legacy(88, get_resize_crop_region_for_grid(grid, 512 // 16), grid)

    actual = _image_rotary_emb(TRANSFORMER, height, width, torch.device("cpu"))

    for got, want in zip(actual, expected):
        torch.testing.assert_close(got, torch.as_tensor(want, dtype=got.dtype), rtol=0, atol=1e-5)
