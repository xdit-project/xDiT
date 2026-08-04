import unittest

from xfuser.core.utils import vae_tiling


class StubVAE:
    """Stands in for a diffusers VAE, carrying only the tiling attributes one would set"""

    def __init__(self, **attrs):
        for name, value in attrs.items():
            setattr(self, name, value)


def legacy_pair_vae():
    """AutoencoderKL and friends: a pixel window, a latent window, an overlap fraction"""
    return StubVAE(
        tile_sample_min_size=256, tile_latent_min_size=32, tile_overlap_factor=0.25
    )


def stride_vae():
    """Wan, Qwen-Image, the video VAEs: a pixel window and an explicit pixel stride"""
    return StubVAE(
        tile_sample_min_height=256,
        tile_sample_min_width=256,
        tile_sample_stride_height=192,
        tile_sample_stride_width=192,
        spatial_compression_ratio=8,
    )


def overlap_hw_vae():
    """CogVideoX-style: pixel and latent windows keyed by height and width, plus fractions"""
    return StubVAE(
        tile_sample_min_height=256,
        tile_sample_min_width=256,
        tile_latent_min_height=32,
        tile_latent_min_width=32,
        tile_overlap_factor_height=0.25,
        tile_overlap_factor_width=0.25,
    )


def asymmetric_vae():
    """CogVideoX-style: a window taller than it is wide, which one edge cannot describe"""
    return StubVAE(
        tile_sample_min_height=240,
        tile_sample_min_width=360,
        tile_latent_min_height=30,
        tile_latent_min_width=45,
        tile_overlap_factor_height=1 / 6,
        tile_overlap_factor_width=0.2,
    )


def per_axis_overlap_vae():
    """A square window whose two axes carry their own overlap fractions"""
    return StubVAE(
        tile_sample_min_height=256,
        tile_sample_min_width=256,
        tile_latent_min_height=32,
        tile_latent_min_width=40,
        tile_overlap_factor_height=0.25,
        tile_overlap_factor_width=0.2,
    )


class TestSupportProbe(unittest.TestCase):

    def test_the_method_alone_does_not_count_as_support(self):
        # Diffusers hands out enable_tiling from a mixin whether or not the class implements it,
        # so a VAE can carry the method and still raise NotImplementedError when called.
        unsupported = StubVAE(enable_tiling=lambda: None)
        with self.assertRaises(ValueError):
            vae_tiling.require_vae_support(unsupported, "tiling", "--enable_tiling")

    def test_the_state_flag_counts_as_support(self):
        vae_tiling.require_vae_support(
            StubVAE(use_tiling=False), "tiling", "--enable_tiling"
        )
        vae_tiling.require_vae_support(
            StubVAE(use_slicing=False), "slicing", "--enable_slicing"
        )


class TestTilePaddingError(unittest.TestCase):
    """The padding failure a too-thin tile causes, told apart from failures with other causes"""

    # Verbatim from AutoencoderKLLTX2Video decoding a 16x16 latent at a 128px window.
    REAL = (
        "Argument #4: Padding size should be less than the corresponding input dimension, "
        "but got: padding (1, 1) at dimension 4 of input [1, 8, 3, 4, 1]"
    )

    def test_the_padding_failure_is_recognised(self):
        self.assertTrue(vae_tiling.is_tile_padding_error(RuntimeError(self.REAL)))

    def test_other_decode_failures_are_not(self):
        for message in (
            "expected scalar type BFloat16 but found Float",
            "Expected all tensors to be on the same device",
            "shape '[1, 8, 16, 16]' is invalid for input of size 1024",
            "CUDA error: an illegal memory access was encountered",
        ):
            with self.subTest(error=message):
                self.assertFalse(
                    vae_tiling.is_tile_padding_error(RuntimeError(message))
                )


class TestTileWindow(unittest.TestCase):

    def test_reads_the_pixel_window_of_each_family(self):
        self.assertEqual(vae_tiling.tile_window(legacy_pair_vae()), 256)
        self.assertEqual(vae_tiling.tile_window(stride_vae()), 256)
        self.assertEqual(vae_tiling.tile_window(overlap_hw_vae()), 256)

    def test_a_vae_without_a_window_reports_none(self):
        self.assertIsNone(vae_tiling.tile_window(StubVAE(tile_overlap_h=0.25)))
        self.assertIsNone(vae_tiling.tile_plan(StubVAE(tile_overlap_h=0.25), 128))

    def test_a_window_that_is_not_square_reports_none(self):
        # One edge cannot set a 240x360 window: moving both to one number would leave the latent
        # window on one axis describing a different region than the pixel window above it.
        self.assertIsNone(vae_tiling.tile_window(asymmetric_vae()))
        self.assertIsNone(vae_tiling.tile_plan(asymmetric_vae(), 240))

    def test_spatial_ratio_falls_back_from_config_to_the_module(self):
        self.assertEqual(vae_tiling.spatial_ratio(stride_vae()), 8)
        self.assertIsNone(vae_tiling.spatial_ratio(legacy_pair_vae()))


class TestTilePlan(unittest.TestCase):

    def test_every_attribute_is_rescaled_by_the_same_factor(self):
        plan = vae_tiling.tile_plan(stride_vae(), 128)
        self.assertEqual(
            plan,
            {
                "tile_sample_min_height": 128,
                "tile_sample_min_width": 128,
                "tile_sample_stride_height": 96,
                "tile_sample_stride_width": 96,
            },
        )

    def test_a_window_that_does_not_divide_whole_is_refused(self):
        # 100px would put the latent window at 12.5, which no VAE can hold.
        self.assertIsNone(vae_tiling.tile_plan(legacy_pair_vae(), 100))

    def test_an_overlap_that_does_not_land_whole_is_refused(self):
        # 200px gives a latent window of 25, and 25 x 0.75 truncates to a stride the pixel crop
        # does not agree with, which assembles an image of the wrong size.
        self.assertIsNone(vae_tiling.tile_plan(legacy_pair_vae(), 200))
        self.assertIsNone(vae_tiling.tile_plan(overlap_hw_vae(), 200))
        self.assertIsNotNone(vae_tiling.tile_plan(legacy_pair_vae(), 192))

    def test_each_overlap_fraction_is_checked_against_its_own_axis(self):
        # 32 x 0.75 and 40 x 0.8 both land whole, so the window stands. Checking every fraction
        # against every latent window instead would fail it on 32 x 0.8 = 25.6.
        self.assertIsNotNone(vae_tiling.tile_plan(per_axis_overlap_vae(), 256))
        # 224px puts the width latent at 35, and 35 x 0.8 = 28 is whole, but the height latent
        # lands at 28 and 28 x 0.75 = 21 is whole too, so this one stands on both axes.
        self.assertIsNotNone(vae_tiling.tile_plan(per_axis_overlap_vae(), 224))

    def test_an_unkeyed_overlap_fraction_covers_both_axes(self):
        vae = StubVAE(
            tile_sample_min_height=256,
            tile_sample_min_width=256,
            tile_latent_min_height=16,
            tile_latent_min_width=16,
            tile_overlap_factor=0.25,
        )
        self.assertIsNotNone(vae_tiling.tile_plan(vae, 64))
        # 32px puts each latent window at 2, and 2 x 0.75 truncates to a stride of 1.
        self.assertIsNone(vae_tiling.tile_plan(vae, 32))

    def test_a_stride_below_one_latent_pixel_is_refused(self):
        # 8px would leave a 6px stride, under this VAE's 8px latent pixel, and diffusers steps
        # through the latents in a range() that would then be empty.
        self.assertIsNone(vae_tiling.tile_plan(stride_vae(), 8))

    def test_a_window_above_the_default_still_plans(self):
        # _apply_vae_tile_size declines these itself, having the config to say why.
        plan = vae_tiling.tile_plan(stride_vae(), 512)
        self.assertEqual(plan["tile_sample_stride_height"], 384)


class TestLatentRows(unittest.TestCase):
    """What a planned tile leaves for --use_parallel_vae, which splits those rows across ranks"""

    def test_rows_come_from_the_latent_window_where_the_vae_carries_one(self):
        vae = legacy_pair_vae()
        self.assertEqual(
            vae_tiling.latent_rows(vae, vae_tiling.tile_plan(vae, 128)), 16
        )

    def test_rows_come_from_the_compression_ratio_otherwise(self):
        vae = stride_vae()
        self.assertEqual(
            vae_tiling.latent_rows(vae, vae_tiling.tile_plan(vae, 128)), 16
        )

    def test_a_vae_that_says_neither_reports_none(self):
        vae = StubVAE(tile_sample_min_height=256, tile_sample_min_width=256)
        self.assertIsNone(vae_tiling.latent_rows(vae, vae_tiling.tile_plan(vae, 128)))

    def test_the_smallest_window_can_be_asked_to_hold_a_row_per_rank(self):
        vae = legacy_pair_vae()
        # This VAE tiles at multiples of 32px, so 32 is the smallest that works at all, but eight
        # ranks each need a latent row of their own and 32px only comes to four.
        self.assertEqual(vae_tiling.smallest_tile_window(vae, 8, 256), 32)
        self.assertEqual(
            vae_tiling.smallest_tile_window(vae, 8, 256, min_latent_rows=8), 64
        )


class TestSnapping(unittest.TestCase):

    def test_snapping_lands_on_the_next_workable_window_down(self):
        pixels, plan = vae_tiling.snap_tile_window(legacy_pair_vae(), 200)
        self.assertEqual(pixels, 192)
        self.assertEqual(plan["tile_latent_min_size"], 24)

    def test_snapping_keeps_a_window_that_already_works(self):
        pixels, _ = vae_tiling.snap_tile_window(stride_vae(), 128)
        self.assertEqual(pixels, 128)

    def test_snapping_never_returns_a_larger_window(self):
        for requested in range(1, 257):
            pixels, _ = vae_tiling.snap_tile_window(overlap_hw_vae(), requested)
            if pixels is not None:
                self.assertLessEqual(pixels, requested)

    def test_a_request_under_the_smallest_window_snaps_to_nothing(self):
        pixels, plan = vae_tiling.snap_tile_window(stride_vae(), 8)
        self.assertIsNone(pixels)
        self.assertIsNone(plan)

    def test_the_smallest_workable_window_is_reported_for_the_error_path(self):
        self.assertEqual(vae_tiling.smallest_tile_window(stride_vae(), 8, 256), 12)
        self.assertIsNone(vae_tiling.smallest_tile_window(StubVAE(), 8, 256))

    def test_a_vae_with_unequal_height_and_width_windows_takes_no_size(self):
        # One edge cannot describe a 240x360 window, so every size is refused and the caller is
        # told that rather than being sent looking for a smaller one.
        vae = asymmetric_vae()
        self.assertIsNone(vae_tiling.smallest_tile_window(vae, 1, 240))
        self.assertIsNone(vae_tiling.snap_tile_window(vae, 240)[0])


class TestEveryVAEARunnerLoads(unittest.TestCase):
    """Every VAE class the runner models load takes a --vae_tile_size and still decodes to the
    size an untiled decode gives"""

    # A tiny stand-in per class, small enough to decode on CPU. LTX2 pins its compression ratio
    # because the config default describes more encoder stages than its decoder upsamples.
    VAES = {
        "AutoencoderKL": (
            dict(
                block_out_channels=[8, 8, 16, 16],
                layers_per_block=1,
                latent_channels=4,
                norm_num_groups=8,
                sample_size=256,
                down_block_types=["DownEncoderBlock2D"] * 4,
                up_block_types=["UpDecoderBlock2D"] * 4,
            ),
            False,
            4,
        ),
        "AutoencoderKLFlux2": (
            dict(
                block_out_channels=[8, 8, 16, 16],
                layers_per_block=1,
                latent_channels=4,
                norm_num_groups=8,
                sample_size=256,
            ),
            False,
            4,
        ),
        "AutoencoderKLWan": (
            dict(base_dim=8, z_dim=4, dim_mult=[1, 2, 4, 4], num_res_blocks=1),
            True,
            4,
        ),
        "AutoencoderKLQwenImage": (
            dict(base_dim=8, z_dim=4, dim_mult=[1, 2, 4, 4], num_res_blocks=1),
            True,
            4,
        ),
        "AutoencoderKLHunyuanVideo": (
            dict(
                block_out_channels=(8, 8, 16, 16),
                layers_per_block=1,
                latent_channels=4,
                norm_num_groups=8,
            ),
            True,
            4,
        ),
        "AutoencoderKLHunyuanVideo15": (
            dict(
                block_out_channels=(8, 8, 16, 16, 16),
                layers_per_block=1,
                latent_channels=4,
            ),
            True,
            4,
        ),
        "AutoencoderKLLTX2Video": (
            dict(
                block_out_channels=(8, 16, 32, 32),
                latent_channels=8,
                layers_per_block=(1, 1, 1, 1, 1),
                spatial_compression_ratio=32,
            ),
            True,
            8,
        ),
    }
    # Large enough that the output is several tiles across once the window is halved, since a
    # decode that fits in one tile would pass without tiling anything.
    LATENT_GRID = 16

    def test_a_halved_window_decodes_to_the_same_size(self):
        import torch
        import diffusers

        for name, (kwargs, video, channels) in self.VAES.items():
            with self.subTest(vae=name):
                cls = getattr(diffusers, name, None)
                if cls is None:
                    self.skipTest(f"{name} is not in diffusers {diffusers.__version__}")
                vae = cls(**kwargs).eval()
                if not hasattr(vae, "enable_tiling"):
                    self.skipTest(
                        f"diffusers {diffusers.__version__} cannot tile {name}"
                    )

                grid = self.LATENT_GRID
                shape = (
                    (1, channels, 1, grid, grid) if video else (1, channels, grid, grid)
                )
                torch.manual_seed(0)
                latents = torch.randn(*shape)
                with torch.no_grad():
                    vae.disable_tiling()
                    expected = vae.decode(latents).sample.shape[-2:]

                # Same order as the runner: turn tiling on, then size its window.
                vae.enable_tiling()
                window = vae_tiling.tile_window(vae)
                self.assertIsNotNone(
                    window, f"{name} tiles but exposes no window this can read"
                )
                pixels, plan = vae_tiling.snap_tile_window(vae, window // 2)
                self.assertIsNotNone(
                    plan, f"{name} refused every window at or below {window // 2}"
                )
                for attr, value in plan.items():
                    setattr(vae, attr, value)
                with torch.no_grad():
                    got = vae.decode(latents).sample.shape[-2:]
                self.assertEqual(
                    got, expected, f"{name} decoded at a {pixels}px tile window"
                )


if __name__ == "__main__":
    unittest.main()
