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


def overlap_factor_vae():
    """AutoencoderKL and friends again, carrying the blending the batched decode reuses"""
    return StubVAE(
        tile_sample_min_size=256,
        tile_latent_min_size=32,
        tile_overlap_factor=0.25,
        blend_v=lambda above, tile, extent: tile,
        blend_h=lambda left, tile, extent: tile,
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


class TestTileBatching(unittest.TestCase):
    """Decoding same-shaped tiles in one call, which has to leave the image exactly as it was"""

    # The two VAE classes that tile by overlap fraction, reusing the stand-ins above.
    FAMILY = ("AutoencoderKL", "AutoencoderKLFlux2")
    # Three windows of latents across, so a run holds several tiles of the full shape alongside
    # the clipped ones at the right and bottom edges.
    WINDOWS_ACROSS = 3

    def test_only_the_overlap_factor_family_is_batched(self):
        # The others walk a stride they store outright, over a loop with its own blending and,
        # for the video VAEs, a frame axis this knows nothing about.
        self.assertTrue(vae_tiling.tiles_by_overlap_factor(overlap_factor_vae()))
        self.assertFalse(vae_tiling.tiles_by_overlap_factor(stride_vae()))
        self.assertFalse(vae_tiling.tiles_by_overlap_factor(overlap_hw_vae()))
        self.assertIsNone(vae_tiling.batched_tiled_decode(stride_vae(), 4096))
        self.assertIsNotNone(vae_tiling.batched_tiled_decode(overlap_factor_vae(), 4096))

    def test_the_tile_area_is_read_off_the_square_latent_window(self):
        self.assertEqual(vae_tiling.tile_latent_area(overlap_factor_vae()), 32 * 32)
        self.assertIsNone(vae_tiling.tile_latent_area(stride_vae()))

    def test_an_unnarrowed_window_budgets_exactly_one_tile(self):
        # The area a run gets when it asked for no window of its own, where batching has to be a
        # no-op: the two areas are the same, so their geometric mean is that area, so one tile.
        area = 128 * 128
        self.assertEqual(vae_tiling.tile_batch_budget(area, area), area)

    def test_halving_the_window_halves_the_area_a_call_carries(self):
        # What --vae_tile_size promises, in the units it is set in: it is an edge, so halving the
        # number halves the memory a decoder call needs. The tiles sharing each round of
        # collectives double at the same time, which is the other half of the bargain.
        default_edge = 128
        default_area = default_edge**2
        for edge, tiles, carried in ((64, 2, 8192), (32, 4, 4096), (16, 8, 2048), (8, 16, 1024)):
            with self.subTest(latent_window=edge):
                budget = vae_tiling.tile_batch_budget(default_area, edge * edge)
                self.assertEqual(budget, carried)
                self.assertEqual(budget // (edge * edge), tiles)
                # Linear in the edge: an eighth of the window is an eighth of the area per call.
                self.assertEqual(budget * default_edge, default_area * edge)

    def test_a_vae_with_no_square_window_is_not_batched(self):
        self.assertIsNone(vae_tiling.tile_batch_budget(None, 1024))
        self.assertIsNone(vae_tiling.tile_batch_budget(16384, None))

    def test_the_budget_never_falls_below_a_single_tile(self):
        # A tile larger than the VAE's own window cannot happen through --vae_tile_size, which
        # refuses one, but a budget under a tile would decode nothing at all rather than one tile.
        self.assertEqual(vae_tiling.tile_batch_budget(64, 4096), 4096)

    def _tiled_vae(self, name, batch=1):
        """A small VAE of class `name` at a narrowed window, and latents several tiles across"""
        import torch
        import diffusers

        cls = getattr(diffusers, name, None)
        if cls is None:
            self.skipTest(f"{name} is not in diffusers {diffusers.__version__}")
        kwargs, _, channels = TestEveryVAEARunnerLoads.VAES[name]
        vae = cls(**kwargs).eval()
        if not hasattr(vae, "use_tiling"):
            self.skipTest(f"diffusers {diffusers.__version__} cannot tile {name}")
        vae.enable_tiling()

        window = vae_tiling.tile_window(vae)
        pixels, plan = vae_tiling.snap_tile_window(vae, window // 4)
        self.assertIsNotNone(plan, f"{name} refused every window at or below {window // 4}")
        vae_tiling.apply_tile_plan(vae, plan)
        self.assertTrue(
            vae_tiling.tiles_by_overlap_factor(vae),
            f"{name} was expected to tile by overlap fraction",
        )

        grid = vae.tile_latent_min_size * self.WINDOWS_ACROSS
        torch.manual_seed(0)
        return vae, torch.randn(batch, channels, grid, grid)

    def _counted(self, vae):
        """Replace the decoder with one that records the shape of every call"""
        import torch.nn as nn

        class CountingDecoder(nn.Module):
            def __init__(self, decoder):
                super().__init__()
                self.decoder = decoder
                self.shapes = []

            def forward(self, x):
                self.shapes.append(tuple(x.shape))
                return self.decoder(x)

            @property
            def rows(self):
                """The rows each call carried: one tile each, at a latent batch of one"""
                return [shape[0] for shape in self.shapes]

        counted = CountingDecoder(vae.decoder)
        vae.decoder = counted
        return counted

    def test_a_batched_decode_gives_the_image_an_unbatched_one_gives(self):
        import torch

        for name in self.FAMILY:
            with self.subTest(vae=name):
                vae, latents = self._tiled_vae(name)
                with torch.no_grad():
                    expected = vae.tiled_decode(latents).sample
                    counted = self._counted(vae)
                    batched = vae_tiling.batched_tiled_decode(
                        vae, vae.tile_latent_min_size**2 * 8
                    )
                    got = batched(latents).sample
                # Nothing here changes what is summed, only how many rows a kernel is handed at
                # once, and a convolution picks its blocking off that. The residue is around
                # 1e-5 on an image in [-1, 1]; a tile put back in the wrong place would be off by
                # order one, which this still catches.
                self.assertEqual(got.shape, expected.shape)
                torch.testing.assert_close(got, expected, rtol=0, atol=1e-4)
                self.assertTrue(
                    any(rows > 1 for rows in counted.rows),
                    f"{name} decoded every tile on its own, so this proved nothing",
                )

    def test_a_latent_batch_is_taken_apart_again(self):
        import torch

        # Each tile carries every sample in the batch, so a call decodes tiles x samples rows and
        # the split back out has to step by the batch and not by one.
        vae, latents = self._tiled_vae("AutoencoderKL", batch=2)
        with torch.no_grad():
            expected = vae.tiled_decode(latents).sample
            got = vae_tiling.batched_tiled_decode(
                vae, vae.tile_latent_min_size**2 * 8
            )(latents).sample
        torch.testing.assert_close(got, expected, rtol=0, atol=1e-4)

    def test_a_budget_of_one_tile_decodes_exactly_as_upstream_does(self):
        import torch

        # This is what the default window gets: the budget is that window's own area, so nothing
        # about the decode changes for a run that did not ask for a smaller tile.
        vae, latents = self._tiled_vae("AutoencoderKL")
        with torch.no_grad():
            expected = vae.tiled_decode(latents).sample
            counted = self._counted(vae)
            got = vae_tiling.batched_tiled_decode(
                vae, vae.tile_latent_min_size**2
            )(latents).sample
        self.assertEqual(set(counted.rows), {1})
        torch.testing.assert_close(got, expected, rtol=0, atol=0)

    def test_a_budget_of_zero_decodes_exactly_as_upstream_does(self):
        import torch

        vae, latents = self._tiled_vae("AutoencoderKL")
        with torch.no_grad():
            expected = vae.tiled_decode(latents).sample
            counted = self._counted(vae)
            got = vae_tiling.batched_tiled_decode(vae, 0)(latents).sample
        self.assertEqual(set(counted.rows), {1})
        torch.testing.assert_close(got, expected, rtol=0, atol=0)

    def test_the_budget_caps_the_area_one_call_carries(self):
        import torch

        vae, latents = self._tiled_vae("AutoencoderKL")
        window = vae.tile_latent_min_size
        decoder = vae.decoder
        for tiles_per_call in (1, 2, 4):
            with self.subTest(tiles_per_call=tiles_per_call):
                budget = window**2 * tiles_per_call
                vae.decoder = decoder
                counted = self._counted(vae)
                with torch.no_grad():
                    vae_tiling.batched_tiled_decode(vae, budget)(latents)
                # No call may carry more latent area than the budget, which is what holds peak
                # memory where the VAE's own window put it.
                for shape in counted.shapes:
                    self.assertLessEqual(shape[0] * shape[-2] * shape[-1], budget)
                # The full-shape tiles spend the budget exactly, so the batch tracks it. Tiles
                # clipped by the latent bounds are smaller, and group larger for the same area.
                self.assertEqual(max(counted.rows), tiles_per_call)
        vae.decoder = decoder

    def test_the_narrowed_window_sets_the_batch_and_leaves_the_image_alone(self):
        import torch
        import diffusers

        # The two ends of the rule joined up: the budget the runner computes from the VAE's own
        # window and the narrowed one, spent by the decode that window produced.
        kwargs, _, _ = TestEveryVAEARunnerLoads.VAES["AutoencoderKL"]
        untouched = diffusers.AutoencoderKL(**kwargs).eval()
        untouched.enable_tiling()
        default_area = vae_tiling.tile_latent_area(untouched)

        vae, latents = self._tiled_vae("AutoencoderKL")
        narrowed_area = vae_tiling.tile_latent_area(vae)
        budget = vae_tiling.tile_batch_budget(default_area, narrowed_area)
        # Stated as the property rather than the numbers this particular stub lands on: the
        # budget is the geometric mean, so it sits between the two areas it was taken from.
        self.assertEqual(budget**2, default_area * narrowed_area)
        self.assertLess(narrowed_area, budget)
        self.assertLess(budget, default_area)

        with torch.no_grad():
            expected = vae.tiled_decode(latents).sample
            counted = self._counted(vae)
            got = vae_tiling.batched_tiled_decode(vae, budget)(latents).sample
        self.assertEqual(max(counted.rows), budget // narrowed_area)
        for shape in counted.shapes:
            self.assertLess(shape[0] * shape[-2] * shape[-1], default_area)
        torch.testing.assert_close(got, expected, rtol=0, atol=1e-4)

    def test_only_a_reimplemented_loop_can_have_its_tiles_dealt_out(self):
        # Choosing which rank makes which decoder call means owning the loop that makes them.
        self.assertTrue(vae_tiling.supports_tile_parallel(overlap_factor_vae()))
        self.assertFalse(vae_tiling.supports_tile_parallel(stride_vae()))
        self.assertFalse(vae_tiling.supports_tile_parallel(overlap_hw_vae()))

    def test_the_dispatcher_is_given_every_call_and_the_image_is_unchanged(self):
        import torch

        vae, latents = self._tiled_vae("AutoencoderKL")
        budget = vae.tile_latent_min_size**2 * 2
        seen = []

        def dispatch(calls):
            seen.append(len(calls))
            return [call() for call in calls]

        with torch.no_grad():
            expected = vae.tiled_decode(latents).sample
            counted = self._counted(vae)
            got = vae_tiling.batched_tiled_decode(vae, budget, dispatch)(latents).sample
        # One dispatch for the decode, holding every call it would have made itself, which is
        # what lets a group divide them and pay for one exchange rather than one per tile.
        self.assertEqual(seen, [len(counted.shapes)])
        # Batched, so the same residue as the batching tests above: a convolution blocks off the
        # rows it is handed, and a tile in the wrong place would be off by order one.
        torch.testing.assert_close(got, expected, rtol=0, atol=1e-4)

    def test_the_calls_can_be_made_in_any_order(self):
        import torch

        # What a rank split rests on: the tiles are independent, so which order the decoder sees
        # them in cannot matter. Only the assembly afterwards has an order, and it works off the
        # results rather than the calls.
        vae, latents = self._tiled_vae("AutoencoderKL")

        def backwards(calls):
            return list(reversed([call() for call in reversed(calls)]))

        with torch.no_grad():
            expected = vae.tiled_decode(latents).sample
            got = vae_tiling.batched_tiled_decode(vae, 0, backwards)(latents).sample
        torch.testing.assert_close(got, expected, rtol=0, atol=0)

    def test_a_tiled_decode_that_fits_in_one_tile_still_works(self):
        import torch

        # A single tile means one shape, one group, and no blending pass at all.
        vae, _ = self._tiled_vae("AutoencoderKL")
        latents = torch.randn(1, vae.config.latent_channels, 4, 4)
        with torch.no_grad():
            expected = vae.tiled_decode(latents).sample
            got = vae_tiling.batched_tiled_decode(vae, 1 << 20)(latents).sample
        torch.testing.assert_close(got, expected, rtol=0, atol=0)


class TestStrideTiledDecode(unittest.TestCase):
    """The video VAEs' own tiling loop, reimplemented so that its tiles can be handed round"""

    # The two whose loop walks a stride they store: a frame loop per tile, threading a feature
    # cache that is cleared where each tile starts.
    FAMILY = ("AutoencoderKLWan", "AutoencoderKLQwenImage")
    # Wide enough to be several tiles across once the window is halved, and two frames deep so
    # that the cache is threaded through more than the chunk the tile opens with.
    LATENT_GRID = 16
    FRAMES = 2

    def _tiled_vae(self, name, **extra):
        """A small video VAE of class `name` at a halved window, and latents a few tiles across"""
        import torch
        import diffusers

        cls = getattr(diffusers, name, None)
        if cls is None:
            self.skipTest(f"{name} is not in diffusers {diffusers.__version__}")
        kwargs, _, channels = TestEveryVAEARunnerLoads.VAES[name]
        vae = cls(**{**kwargs, **extra}).eval()
        if not hasattr(vae, "use_tiling"):
            self.skipTest(f"diffusers {diffusers.__version__} cannot tile {name}")
        vae.enable_tiling()

        window = vae_tiling.tile_window(vae)
        pixels, plan = vae_tiling.snap_tile_window(vae, window // 2)
        self.assertIsNotNone(plan, f"{name} refused every window at or below {window // 2}")
        vae_tiling.apply_tile_plan(vae, plan)
        self.assertTrue(
            vae_tiling.tiles_by_stored_stride(vae),
            f"{name} was expected to tile by a stride it stores",
        )

        torch.manual_seed(0)
        grid = self.LATENT_GRID
        return vae, torch.randn(1, channels, self.FRAMES, grid, grid)

    def test_only_the_families_whose_loop_this_is(self):
        # Hunyuan, LTX-2 and CogVideoX carry the same stride attributes and the same cache, and
        # walk them differently, so nothing about a VAE's attributes settles this on its own.
        self.assertFalse(vae_tiling.tiles_by_stored_stride(stride_vae()))
        self.assertFalse(vae_tiling.tiles_by_stored_stride(overlap_factor_vae()))

    def test_it_decodes_what_the_vae_decodes_for_itself(self):
        import torch

        for name, extra in (
            ("AutoencoderKLWan", {}),
            # Wan 2.2 folds a pixel unshuffle into the decode, which the assembly undoes at the
            # end and which moves every stride and blend the loop measures in.
            ("AutoencoderKLWan", {"patch_size": 2}),
            ("AutoencoderKLQwenImage", {}),
        ):
            with self.subTest(vae=name, **extra):
                vae, latents = self._tiled_vae(name, **extra)
                with torch.no_grad():
                    expected = vae.tiled_decode(latents).sample
                    got = vae_tiling.strided_tiled_decode(vae)(latents).sample
                # The same calls in the same order on the same tensors, so exactly the same
                # sample: this loop exists to hand the calls round, not to compute differently.
                self.assertEqual(got.shape, expected.shape)
                torch.testing.assert_close(got, expected, rtol=0, atol=0)

    def test_a_tile_is_a_call_and_they_can_be_made_in_any_order(self):
        import torch

        for name in self.FAMILY:
            with self.subTest(vae=name):
                vae, latents = self._tiled_vae(name)
                seen = []

                def backwards(calls):
                    seen.append(len(calls))
                    return list(reversed([call() for call in reversed(calls)]))

                with torch.no_grad():
                    expected = vae.tiled_decode(latents).sample
                    got = vae_tiling.strided_tiled_decode(vae, backwards)(latents).sample
                # One call per tile, and the order they are made in cannot reach the sample:
                # a tile's frames share a cache, and no two tiles share anything.
                stride = vae.tile_sample_stride_height // vae.spatial_compression_ratio
                across = len(range(0, latents.shape[-1], stride))
                self.assertEqual(seen, [across * across])
                torch.testing.assert_close(got, expected, rtol=0, atol=0)

    def test_the_loop_is_only_reimplemented_to_hand_it_round(self):
        # Without a group there is nothing to gain by replacing a loop that already does this,
        # so the VAE keeps its own and only a dispatcher brings this one in.
        vae, _ = self._tiled_vae("AutoencoderKLWan")
        self.assertTrue(vae_tiling.supports_tile_parallel(vae))
        self.assertIsNone(vae_tiling.tiled_decode_for(vae, 0))
        self.assertIsNotNone(vae_tiling.tiled_decode_for(vae, 0, lambda calls: []))


if __name__ == "__main__":
    unittest.main()
