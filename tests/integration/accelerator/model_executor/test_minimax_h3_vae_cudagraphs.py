"""MiniMax-H3's compiled VAE decoder must decode tile after tile like the eager decoder.

The runner may replay the decoder blocks as CUDA graphs. Every tile reuses the same
graphs, so an earlier tile's output must survive the later replays. The model is a
tiny random VAE and nothing is downloaded.
"""

import copy

import pytest
import torch

minimax_h3_vae = pytest.importorskip("diffusers.models.autoencoders.autoencoder_kl_minimax_h3")

from xfuser.model_executor.models.runner_models import minimax_h3  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires an accelerator")

LATENT_CHANNELS = 4
TILES = 4


def _tiny_vae():
    torch.manual_seed(0)
    vae = minimax_h3_vae.AutoencoderKLMiniMaxH3(
        block_out_channels=(32,),
        layers_per_block=1,
        spatial_downsample_factors=(2,),
        temporal_downsample_factors=(1,),
        norm_num_groups=8,
        latent_channels=LATENT_CHANNELS,
        decoder_num_layers=3,
        decoder_num_attention_heads=2,
        decoder_attention_head_dim=32,
        latents_mean=(0.0,) * LATENT_CHANNELS,
        latents_std=(1.0,) * LATENT_CHANNELS,
    )
    # The residual scales start at zero, which would make every block an identity.
    with torch.no_grad():
        for block in vae.decoder.transformer_blocks:
            block.scale1.normal_(std=0.5)
            block.scale2.normal_(std=0.5)
    return vae.to("cuda").eval()


def _decode_tiles(vae, tiles):
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        return [vae.decoder(vae.post_quant_conv(tile)) for tile in tiles]


def test_compiled_vae_decoder_keeps_every_tile_output(monkeypatch):
    monkeypatch.setattr(minimax_h3, "log", lambda message: None)
    eager = _tiny_vae()
    compiled = copy.deepcopy(eager)
    model = object.__new__(minimax_h3.xFuserMiniMaxH3Model)
    model._compile_vae_decoder_blocks(compiled)
    generator = torch.Generator(device="cuda").manual_seed(1)
    tiles = [torch.randn(1, LATENT_CHANNELS, 2, 4, 4, device="cuda", generator=generator) for _ in range(TILES)]

    try:
        compiled_outputs = _decode_tiles(compiled, tiles)
        expected = _decode_tiles(eager, tiles)
        for actual, reference in zip(compiled_outputs, expected):
            torch.testing.assert_close(actual.float(), reference.float(), rtol=2e-2, atol=2e-2)
    finally:
        torch._dynamo.reset()
