"""FBCache on FLUX.2 must reproduce the uncached model with the installed diffusers.

The adapter takes over the model's block loops and calls the single-stream blocks
itself, so it has to use the keywords the installed blocks accept. diffusers 0.37
renamed them, and FLUX.2-dev supports 0.36, so run this against both.
"""

import copy

import pytest
import torch

pytest.importorskip("diffusers.models.transformers.transformer_flux2")

from diffusers.models.transformers.transformer_flux2 import Flux2Transformer2DModel  # noqa: E402

from xfuser.model_executor.cache import utils as cache_utils  # noqa: E402
from xfuser.model_executor.cache.adapters.flux2 import apply_fbcache  # noqa: E402


def _tiny_model():
    torch.manual_seed(0)
    return Flux2Transformer2DModel(
        in_channels=8,
        num_layers=2,
        num_single_layers=2,
        attention_head_dim=16,
        num_attention_heads=2,
        joint_attention_dim=12,
        timestep_guidance_channels=32,
        axes_dims_rope=(4, 4, 4, 4),
    ).eval()


def _inputs(image_tokens=6, text_tokens=3):
    generator = torch.Generator().manual_seed(0)
    img_ids = torch.zeros(image_tokens, 4)
    img_ids[:, 1] = torch.arange(image_tokens)
    txt_ids = torch.zeros(text_tokens, 4)
    txt_ids[:, 3] = torch.arange(text_tokens)
    return {
        "hidden_states": torch.randn(1, image_tokens, 8, generator=generator),
        "encoder_hidden_states": torch.randn(1, text_tokens, 12, generator=generator),
        "timestep": torch.tensor([0.4]),
        "guidance": torch.tensor([3.5]),
        "img_ids": img_ids,
        "txt_ids": txt_ids,
    }


def test_fbcache_matches_uncached_flux2_forward(monkeypatch):
    # Keep the cache's bookkeeping on the CPU, next to the model, whether or not torch
    # was built for an accelerator. There is no sequence-parallel group either, so the
    # cache compares residuals locally.
    monkeypatch.setattr(cache_utils, "get_device", lambda local_rank: torch.device("cpu"))
    monkeypatch.setattr(cache_utils, "get_sequence_parallel_world_size", lambda: 1)
    model = _tiny_model()
    cached = apply_fbcache(copy.deepcopy(model), rel_l1_thresh=0.12, num_steps=2)
    cached_blocks = cached.transformer_blocks[0]
    inputs = _inputs()

    with torch.no_grad():
        expected = model(**inputs, return_dict=False)[0]
        # The first step has nothing cached, so it runs every block.
        computed = cached(**inputs, return_dict=False)[0]
        assert not cached_blocks.use_cache
        # The same input again leaves the first block's residual unchanged, so the
        # second step reuses the residuals cached by the first.
        reused = cached(**inputs, return_dict=False)[0]
        assert cached_blocks.use_cache

    torch.testing.assert_close(computed, expected)
    torch.testing.assert_close(reused, expected)
