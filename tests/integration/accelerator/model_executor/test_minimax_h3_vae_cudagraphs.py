"""MiniMax-H3's compiled VAE decoders must decode like the eager decoders.

The runner may replay the video decoder blocks and the audio decoder as CUDA graphs.
Every replay reuses the graph's buffers, so an earlier tile or waveform must survive
the later replays. The models are tiny random VAEs and nothing is downloaded.
"""

import copy

import pytest
import torch

minimax_h3_vae = pytest.importorskip("diffusers.models.autoencoders.autoencoder_kl_minimax_h3")
minimax_h3_audio_vae = pytest.importorskip("diffusers.models.autoencoders.autoencoder_kl_minimax_h3_audio")

from xfuser.model_executor.models.runner_models import minimax_h3  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires an accelerator")

LATENT_CHANNELS = 4
AUDIO_LATENT_CHANNELS = 8
TILES = 4


@pytest.fixture
def model(monkeypatch):
    monkeypatch.setattr(minimax_h3, "log", lambda message: None)
    yield object.__new__(minimax_h3.xFuserMiniMaxH3Model)
    torch._dynamo.reset()


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


def _tiny_audio_vae():
    torch.manual_seed(0)
    return (
        minimax_h3_audio_vae.AutoencoderKLMiniMaxH3Audio(
            encoder_dim=4,
            encoder_rates=(2, 2),
            latent_dim=32,
            latent_channels=AUDIO_LATENT_CHANNELS,
            num_attention_heads=2,
            decoder_dim=32,
            decoder_rates=(2, 2),
            decoder_kernel_sizes=(4, 4),
            resblock_kernel_sizes=(3,),
            resblock_dilation_sizes=((1,),),
        )
        .to("cuda")
        .eval()
    )


def _decode_tiles(vae, tiles):
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        return [vae.decoder(vae.post_quant_conv(tile)) for tile in tiles]


def _decode_audio(audio_vae, latents):
    with torch.no_grad():
        return [audio_vae.decode(latent, return_dict=False)[0] for latent in latents]


def test_compiled_vae_decoder_keeps_every_tile_output(model):
    eager = _tiny_vae()
    compiled = copy.deepcopy(eager)
    model._compile_vae_decoder_blocks(compiled)
    generator = torch.Generator(device="cuda").manual_seed(1)
    tiles = [torch.randn(1, LATENT_CHANNELS, 2, 4, 4, device="cuda", generator=generator) for _ in range(TILES)]

    compiled_outputs = _decode_tiles(compiled, tiles)
    expected = _decode_tiles(eager, tiles)

    for actual, reference in zip(compiled_outputs, expected):
        torch.testing.assert_close(actual.float(), reference.float(), rtol=2e-2, atol=2e-2)


def test_compiled_audio_vae_decoder_keeps_every_waveform(model):
    eager = _tiny_audio_vae()
    compiled = copy.deepcopy(eager)
    model._compile_audio_vae_decoder(compiled)
    generator = torch.Generator(device="cuda").manual_seed(1)
    # Stereo is decoded as a batch of two, one request at a time.
    latents = [torch.randn(2, AUDIO_LATENT_CHANNELS, 6, device="cuda", generator=generator) for _ in range(3)]

    compiled_outputs = _decode_audio(compiled, latents)
    expected = _decode_audio(eager, latents)

    for actual, reference in zip(compiled_outputs, expected):
        torch.testing.assert_close(actual, reference, rtol=1e-3, atol=1e-3)
