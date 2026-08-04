"""Which VAE classes DistVAE can shard, checked against real VAEs rather than a list.

The adapters assert their block types from inside a half-built replacement, so a VAE they cannot
take has to be recognised before wrapping. These build each VAE class a runner model loads and
demand the answer for both halves of it, so a model declaring use_parallel_vae or
use_parallel_vae_encoder cannot quietly become unshardable when diffusers reworks a block.
"""

import os
import unittest
from types import SimpleNamespace
from unittest import mock

import diffusers
import torch.distributed as dist
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
# what --use_parallel_vae refuses. Every class a runner model loads is shardable as of DistVAE's
# QwenImage, HunyuanVideo and LTX-2 adapters, so a None appearing here again would mean a newly
# supported model arrived ahead of the adapter for its VAE.
EXPECTED = {
    "AutoencoderKL": vae_parallel.TWO_D,
    "AutoencoderKLFlux2": vae_parallel.TWO_D,
    "AutoencoderKLWan": vae_parallel.WAN,
    "AutoencoderKLQwenImage": vae_parallel.QWEN_IMAGE,
    "AutoencoderKLHunyuanVideo": vae_parallel.HUNYUAN_VIDEO,
    "AutoencoderKLHunyuanVideo15": vae_parallel.HUNYUAN_VIDEO_15,
    "AutoencoderKLLTX2Video": vae_parallel.LTX2_VIDEO,
}

# The encoder adapter each class needs. DistVAE reached full encoder coverage alongside its
# decoders, so a None here would mean an encoder adapter was lost rather than never written.
EXPECTED_ENCODERS = {
    "AutoencoderKL": vae_parallel.TWO_D_ENCODER,
    "AutoencoderKLFlux2": vae_parallel.TWO_D_ENCODER,
    "AutoencoderKLWan": "WanEncoderAdapter",
    "AutoencoderKLQwenImage": "QwenImageEncoderAdapter",
    "AutoencoderKLHunyuanVideo": "HunyuanVideoEncoderAdapter",
    "AutoencoderKLHunyuanVideo15": "HunyuanVideo15EncoderAdapter",
    "AutoencoderKLLTX2Video": "LTX2VideoEncoderAdapter",
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

    def test_an_ltx2_decoder_that_injects_noise_is_not_shardable(self):
        # Every rank would draw noise for its own rows, and together they would not reconstruct
        # what one rank draws, so the decode could not match an unsharded one. No released LTX-2
        # checkpoint enables this, which is why the shardable config above is the shipped shape.
        vae = diffusers.AutoencoderKLLTX2Video(
            **CONFIGS["AutoencoderKLLTX2Video"], decoder_inject_noise=True
        )
        self.assertIsNone(vae_parallel.decoder_adapter_name(vae))

    def test_a_two_d_decoder_without_group_norm_is_not_shardable(self):
        # DecoderAdapter replaces conv_norm_out with a sharded GroupNorm and asserts it found one,
        # so a decoder normalising some other way is out even with the right up blocks.
        vae = diffusers.AutoencoderKL(**CONFIGS["AutoencoderKL"])
        vae.decoder.conv_norm_out = nn.Identity()
        self.assertIsNone(vae_parallel.decoder_adapter_name(vae))


class TestEncoderAdapterChoice(unittest.TestCase):
    """Which adapter shards each class's encoder, read off the family its decoder names"""

    def test_every_vae_class_gets_the_encoder_adapter_it_needs(self):
        for name, expected in EXPECTED_ENCODERS.items():
            with self.subTest(vae=name):
                vae = getattr(diffusers, name)(**CONFIGS[name])
                self.assertEqual(vae_parallel.encoder_adapter_name(vae), expected)

    def test_an_encoder_with_no_down_blocks_is_not_shardable(self):
        class Bare:
            pass

        self.assertIsNone(vae_parallel.encoder_adapter_name(Bare()))

    def test_the_two_halves_are_recognised_independently(self):
        # Sharding either half replaces its blocks with adapters, so an encoder read off the
        # decoder would come back unrecognised once the decoder had been done first, which is the
        # order the runner models shard them in.
        vae = diffusers.AutoencoderKLWan(**CONFIGS["AutoencoderKLWan"])
        expected = vae_parallel.encoder_adapter_name(vae)
        vae.decoder.conv_norm_out = nn.Identity()
        vae.decoder.up_blocks = nn.ModuleList()
        self.assertIsNone(vae_parallel.decoder_adapter_name(vae))
        self.assertEqual(vae_parallel.encoder_adapter_name(vae), expected)


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

    def test_flux2s_pair_of_patch_sizes_is_not_mistaken_for_a_ratio(self):
        # Flux.2 names patch_size for how its latents are packed for the transformer, which its
        # convolutions know nothing about, and names it as a pair. Dividing by that both divides
        # by the wrong thing and cannot be compared against a number in the first place.
        vae = diffusers.AutoencoderKLFlux2(**CONFIGS["AutoencoderKLFlux2"])
        self.assertEqual(vae.config.patch_size, (2, 2))
        self.assertEqual(vae_parallel.encoder_scale_factor(vae), 8)

    def test_a_two_d_encoder_is_counted_off_its_stages(self):
        # These VAEs record no ratio, so the four-stage 8 has to be counted rather than assumed.
        # A three-stage one narrows by 4, and sizing its bands by 8 would leave its last stage
        # halving a band into a row belonging to the next rank.
        config = dict(CONFIGS["AutoencoderKL"])
        config["block_out_channels"] = [8, 8, 16]
        config["down_block_types"] = ["DownEncoderBlock2D"] * 3
        config["up_block_types"] = ["UpDecoderBlock2D"] * 3
        self.assertEqual(vae_parallel.encoder_scale_factor(diffusers.AutoencoderKL(**config)), 4)


class _StubAdapter(nn.Module):
    """Stands in for a DistVAE adapter, which needs a process group to build"""

    def __init__(self, decoder, vae_group=None, **kwargs):
        super().__init__()
        self.wrapped = decoder
        self.kwargs = kwargs
        # WanDecoderAdapter crops by the factor it upsamples; the 2D one has no such step.
        self.patchify = SimpleNamespace(scale_factor=1)


class TestWrappingReadsEveryVAEConfig(unittest.TestCase):
    """Wrapping reads the VAE's config, so it has to survive how each class spells it"""

    def _parallelize(self, vae):
        with mock.patch.object(vae_parallel, "_adapter", return_value=_StubAdapter):
            vae_parallel.parallelize_decoder(vae, vae_group=None)
        return vae.decoder

    def test_every_shardable_vae_class_can_be_wrapped(self):
        for name, adapter in EXPECTED.items():
            if adapter is None:
                continue
            with self.subTest(vae=name):
                vae = getattr(diffusers, name)(**CONFIGS[name])
                self.assertIsInstance(self._parallelize(vae), _StubAdapter)

    def test_a_patching_vae_tells_the_adapter_its_factor(self):
        vae = diffusers.AutoencoderKLWan(**CONFIGS["AutoencoderKLWan"])
        vae.register_to_config(patch_size=2)
        self.assertEqual(self._parallelize(vae).patchify.scale_factor, 2)

    def test_flux_2s_patching_is_not_read_as_a_factor(self):
        # Flux 2 declares patch_size (2, 2) for the pixel unshuffle at its boundary, which is not
        # the single factor an adapter's patchify takes.
        vae = diffusers.AutoencoderKLFlux2(**CONFIGS["AutoencoderKLFlux2"])
        self.assertEqual(self._parallelize(vae).patchify.scale_factor, 1)


class TestUnshardableIsRefused(unittest.TestCase):

    def test_it_names_the_flag_and_points_at_the_alternative(self):
        # DistVAE now fits every VAE class a runner model loads, so the refusal is provoked with
        # a decoder taken out of the shape its adapter needs rather than with a real VAE.
        vae = diffusers.AutoencoderKL(**CONFIGS["AutoencoderKL"])
        vae.decoder.conv_norm_out = nn.Identity()
        decoder = vae.decoder
        with self.assertRaises(ValueError) as caught:
            vae_parallel.parallelize_decoder(vae, vae_group=None)
        message = str(caught.exception)
        self.assertIn("--use_parallel_vae", message)
        self.assertIn("--vae_tile_size", message)
        # Refused before touching anything, so the caller is left a working decode to fall back on.
        self.assertIs(vae.decoder, decoder)

    def test_encoding_is_refused_for_a_vae_no_encoder_adapter_fits(self):
        # Provoked with an encoder taken out of the shape its adapter needs, since DistVAE fits
        # every VAE class a runner model loads.
        vae = diffusers.AutoencoderKL(**CONFIGS["AutoencoderKL"])
        vae.encoder.down_blocks = nn.ModuleList()
        encoder = vae.encoder
        with self.assertRaises(ValueError) as caught:
            vae_parallel.parallelize_encoder(vae, vae_group=None)
        self.assertIn("Parallel VAE encoding is not available", str(caught.exception))
        # Refused before touching anything, so the caller is left a working encode.
        self.assertIs(vae.encoder, encoder)


class TestBothHalvesShardTogether(unittest.TestCase):
    """Every VAE class a runner model loads has both halves replaced, in the runner's order

    Naming an adapter and installing it are different things: the adapters rebuild a half in
    place, so the half done first no longer answers to the blocks it was recognised by. Choosing
    both names off intact blocks and then wrapping is what these check, over a one-rank gloo
    group, since a name that resolves is no use if the wrapping it is chosen for cannot run.
    """

    @classmethod
    def setUpClass(cls):
        cls.owns_group = not dist.is_initialized()
        if cls.owns_group:
            os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
            os.environ.setdefault("MASTER_PORT", "24118")
            os.environ.setdefault("RANK", "0")
            os.environ.setdefault("WORLD_SIZE", "1")
            dist.init_process_group(backend="gloo", init_method="env://")

    @classmethod
    def tearDownClass(cls):
        if cls.owns_group:
            dist.destroy_process_group()

    def test_every_vae_class_shards_both_halves(self):
        for name, config in CONFIGS.items():
            with self.subTest(vae=name):
                vae = getattr(diffusers, name)(**config).eval()
                expected = vae_parallel.encoder_adapter_name(vae)
                encoder, decoder = vae.encoder, vae.decoder
                # The runner models shard the decoder first, which is what makes the order matter.
                self.assertEqual(
                    vae_parallel.parallelize_decoder(vae, vae_group=None), EXPECTED[name]
                )
                self.assertEqual(vae_parallel.encoder_adapter_name(vae), expected)
                self.assertEqual(
                    vae_parallel.parallelize_encoder(vae, vae_group=None),
                    EXPECTED_ENCODERS[name],
                )
                self.assertIsNot(vae.decoder, decoder)
                self.assertIsNot(vae.encoder, encoder)


if __name__ == "__main__":
    unittest.main()
