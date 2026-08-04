"""Which VAE classes DistVAE can shard, checked against real decoders rather than a list.

The adapters assert their block types from inside a half-built replacement decoder, so a VAE they
cannot take has to be recognised before wrapping. These build each VAE class a runner model loads
and demand the answer, so a model declaring use_parallel_vae cannot quietly become unshardable
when diffusers reworks a decoder.
"""

import unittest

import diffusers
import torch.nn as nn

from xfuser.core.utils import vae_parallel

from tests.core import test_vae_tiling

# The same tiny builds the tiling tests decode through, taken from there rather than repeated, so
# one VAE class is described in one place. Imported as a module, since binding its TestCase here
# would have unittest collect and run those decodes a second time. Only the config is needed:
# picking an adapter reads the decoder's block types and never runs it.
CONFIGS = {
    name: config
    for name, (config, _, _) in test_vae_tiling.TestEveryVAEARunnerLoads.VAES.items()
}

# The adapter each class needs, or None where DistVAE has nothing for its decoder. A None here is
# what --use_parallel_vae refuses, and what a new DistVAE adapter would change.
EXPECTED = {
    "AutoencoderKL": vae_parallel.TWO_D,
    "AutoencoderKLFlux2": vae_parallel.TWO_D,
    "AutoencoderKLWan": vae_parallel.WAN,
    "AutoencoderKLQwenImage": None,
    "AutoencoderKLHunyuanVideo": None,
    "AutoencoderKLHunyuanVideo15": None,
    "AutoencoderKLLTX2Video": None,
}


class TestDecoderAdapterChoice(unittest.TestCase):

    def test_every_vae_class_gets_the_adapter_it_needs(self):
        for name, expected in EXPECTED.items():
            with self.subTest(vae=name):
                vae = getattr(diffusers, name)(**CONFIGS[name])
                self.assertEqual(vae_parallel.decoder_adapter_name(vae), expected)

    def test_a_vae_with_no_decoder_is_not_shardable(self):
        class Bare:
            pass

        self.assertIsNone(vae_parallel.decoder_adapter_name(Bare()))

    def test_a_two_d_decoder_without_group_norm_is_not_shardable(self):
        # DecoderAdapter replaces conv_norm_out with a sharded GroupNorm and asserts it found one,
        # so a decoder normalising some other way is out even with the right up blocks.
        vae = diffusers.AutoencoderKL(**CONFIGS["AutoencoderKL"])
        vae.decoder.conv_norm_out = nn.Identity()
        self.assertIsNone(vae_parallel.decoder_adapter_name(vae))


class TestEncoderScaleFactor(unittest.TestCase):
    """The number the encoder adapter shards by, which used to be derived per model"""

    def test_a_vae_that_does_not_patch_uses_its_spatial_ratio(self):
        vae = diffusers.AutoencoderKLWan(**CONFIGS["AutoencoderKLWan"])
        self.assertEqual(vae_parallel.encoder_scale_factor(vae), 8)

    def test_patching_is_divided_out(self):
        # Cosmos 3's 16 is 8 from the encoder's convolutions and 2 from patching, and the adapter
        # shards the convolutions.
        vae = diffusers.AutoencoderKLWan(**CONFIGS["AutoencoderKLWan"])
        vae.register_to_config(scale_factor_spatial=16, patch_size=2)
        self.assertEqual(vae_parallel.encoder_scale_factor(vae), 8)

    def test_a_vae_with_no_spatial_ratio_falls_back(self):
        vae = diffusers.AutoencoderKLWan(**CONFIGS["AutoencoderKLWan"])
        vae.register_to_config(scale_factor_spatial=None)
        self.assertEqual(vae_parallel.encoder_scale_factor(vae), 8)


class TestUnshardableIsRefused(unittest.TestCase):

    def test_it_names_the_flag_and_points_at_the_alternative(self):
        vae = diffusers.AutoencoderKLQwenImage(**CONFIGS["AutoencoderKLQwenImage"])
        decoder = vae.decoder
        with self.assertRaises(ValueError) as caught:
            vae_parallel.parallelize_decoder(vae, vae_group=None)
        message = str(caught.exception)
        self.assertIn("--use_parallel_vae", message)
        self.assertIn("--vae_tile_size", message)
        # Refused before touching anything, so the caller is left a working decode to fall back on.
        self.assertIs(vae.decoder, decoder)

    def test_encoding_is_refused_for_a_vae_the_one_encoder_adapter_does_not_fit(self):
        # DistVAE has only WanEncoderAdapter, so a 2D VAE it can decode with it cannot encode with.
        vae = diffusers.AutoencoderKL(**CONFIGS["AutoencoderKL"])
        encoder = vae.encoder
        with self.assertRaises(ValueError):
            vae_parallel.parallelize_encoder(vae, vae_group=None)
        self.assertIs(vae.encoder, encoder)


if __name__ == "__main__":
    unittest.main()
