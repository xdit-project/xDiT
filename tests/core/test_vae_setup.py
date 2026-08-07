"""The decisions a run makes about its VAE decode, and the order it has to make them in.

These were unreachable while they lived on the runner class: exercising one meant a loaded
pipeline, a distributed group and a config object, so the order between them - the part that is
actually fragile - had no test at all and was held in place only by the comments beside it.
"""

import unittest

import torch

from xfuser.core.utils import vae_setup, vae_tiling
from xfuser.core.utils.vae_setup import TilingRequest

from tests.core.test_vae_tiling import (
    StubVAE,
    TestEveryVAEARunnerLoads,
    legacy_pair_vae,
    overlap_factor_vae,
    stride_vae,
)


def request(**kwargs) -> TilingRequest:
    return TilingRequest(model_name="test-model", **kwargs)


def tileable(vae):
    """A stub the tiling path will accept: diffusers proves support by the state flag, not the
    method, so a stub has to carry the flag before configure() will tile it"""
    vae.use_tiling = False
    vae.use_slicing = False
    vae.enable_tiling = lambda: setattr(vae, "use_tiling", True)
    vae.enable_slicing = lambda: setattr(vae, "use_slicing", True)
    vae.decode = lambda z: z
    return vae


def wan_vae(test):
    """A real Wan VAE

    The stride-walking family is keyed by class name, not by the attributes a VAE carries, because
    the loop body differs per class - `stride_vae()` carries exactly Wan's stride spelling and is
    still deliberately not recognised. So the one family where the window and the overlap actually
    interact cannot be tested with a stub.
    """
    import diffusers

    cls = getattr(diffusers, "AutoencoderKLWan", None)
    if cls is None:
        test.skipTest(f"AutoencoderKLWan is not in diffusers {diffusers.__version__}")
    kwargs, _, _ = TestEveryVAEARunnerLoads.VAES["AutoencoderKLWan"]
    vae = cls(**kwargs).eval()
    if not hasattr(vae, "use_tiling"):
        test.skipTest(f"diffusers {diffusers.__version__} cannot tile AutoencoderKLWan")
    return vae


class TestWhichFlagAsked(unittest.TestCase):
    """Any of the three knobs is a request to tile, and the message has to name the one used"""

    def test_no_knob_asks_for_nothing(self):
        self.assertIsNone(request().flag)

    def test_each_knob_asks_by_itself(self):
        self.assertEqual(request(tiling=True).flag, "--enable_tiling")
        self.assertEqual(request(tile_size=128).flag, "--vae_tile_size")
        self.assertEqual(request(tile_overlap=0.125).flag, "--vae_tile_overlap")

    def test_a_zero_overlap_is_still_a_request(self):
        # 0.0 is a real ask - no redundancy at all - and must not read as "not given".
        self.assertEqual(request(tile_overlap=0.0).flag, "--vae_tile_overlap")

    def test_enable_tiling_is_named_first_where_several_were_given(self):
        self.assertEqual(request(tiling=True, tile_size=128).flag, "--enable_tiling")


class TestWhetherItTiles(unittest.TestCase):
    """Answered once, because the tiled decode and the parallel VAE have to agree"""

    def test_a_flag_makes_it_tile(self):
        self.assertTrue(vae_setup.tiles(StubVAE(), request(tiling=True)))

    def test_a_vae_already_tiling_makes_it_tile_with_no_flag(self):
        # LTX 2.3 turns tiling on for its stage-2 VAE at load, where no flag says so. Reading the
        # config alone left that VAE on the sharding path and hung the group.
        self.assertTrue(vae_setup.tiles(StubVAE(use_tiling=True), request()))

    def test_neither_means_no_tiling(self):
        self.assertFalse(vae_setup.tiles(StubVAE(use_tiling=False), request()))


class TestWindow(unittest.TestCase):

    def test_a_window_is_applied_and_reported(self):
        vae = legacy_pair_vae()
        self.assertEqual(vae_setup.apply_window(vae, request(tile_size=128)), 128)
        self.assertEqual(vae.tile_sample_min_size, 128)
        self.assertEqual(vae.tile_latent_min_size, 16)

    def test_no_ask_leaves_the_vae_alone(self):
        vae = legacy_pair_vae()
        self.assertIsNone(vae_setup.apply_window(vae, request()))
        self.assertEqual(vae.tile_sample_min_size, 256)

    def test_a_window_wider_than_the_vae_s_own_is_declined_not_applied(self):
        # Widening raises peak memory rather than lowering it, which is never what the flag is for.
        vae = legacy_pair_vae()
        self.assertIsNone(vae_setup.apply_window(vae, request(tile_size=512)))
        self.assertEqual(vae.tile_sample_min_size, 256)

    def test_a_pointlessly_narrow_window_is_clamped_rather_than_refused(self):
        vae = legacy_pair_vae()
        landed = vae_setup.apply_window(vae, request(tile_size=8))
        self.assertEqual(landed, vae_tiling.narrowest_useful_window(legacy_pair_vae()))

    def test_a_vae_with_no_single_window_says_so_by_name(self):
        with self.assertRaises(ValueError) as caught:
            vae_setup.apply_window(StubVAE(), request(tile_size=128))
        self.assertIn("--vae_tile_size", str(caught.exception))
        self.assertIn("test-model", str(caught.exception))


class TestOverlap(unittest.TestCase):

    def test_an_overlap_is_applied(self):
        vae = overlap_factor_vae()
        vae_setup.apply_overlap(vae, request(tile_overlap=0.125))
        self.assertAlmostEqual(vae.tile_overlap_factor, 0.125)

    def test_a_vae_that_does_not_say_how_it_steps_is_refused(self):
        with self.assertRaises(ValueError) as caught:
            vae_setup.apply_overlap(StubVAE(), request(tile_overlap=0.125))
        self.assertIn("--vae_tile_overlap", str(caught.exception))

    def test_an_overlap_the_vae_cannot_take_names_the_widest_it_can(self):
        # 0.99 leaves a step of nothing, so it is refused; the refusal has to name a step that
        # would have been accepted rather than leaving the caller to guess.
        vae = overlap_factor_vae()
        with self.assertRaises(ValueError) as caught:
            vae_setup.apply_overlap(vae, request(tile_overlap=0.99))
        self.assertIn("the most overlap it can step by", str(caught.exception))

    def test_a_vae_whose_loop_is_not_one_we_step_says_that_instead(self):
        # `stride_vae` carries Wan's exact stride spelling but is not a class whose loop body
        # xFuser reimplements, so the honest answer is that the loop is unknown, not that some
        # narrower overlap would work.
        with self.assertRaises(ValueError) as caught:
            vae_setup.apply_overlap(stride_vae(), request(tile_overlap=0.125))
        self.assertIn("not one xFuser knows how to step", str(caught.exception))


class TestTheOrderBetweenThem(unittest.TestCase):
    """Why configure() owns the order rather than leaving it to each caller

    Sizing the window rescales every tiling attribute by the same factor, the stride included, so
    the two knobs are not independent: the search for a workable window asks which sizes let the
    current stride land whole, and the overlap flag moves that stride. Window first means
    --vae_tile_size lands the same place whatever else was asked for.
    """

    def test_both_knobs_land_when_given_together(self):
        vae = wan_vae(self)
        window = vae_tiling.tile_window(vae)
        vae_setup.configure(vae, request(tile_size=window // 2, tile_overlap=0.125))
        self.assertEqual(vae.tile_sample_min_height, window // 2)
        down, across = vae_tiling.tile_overlap(vae)
        # Never steps wider than asked, so the overlap that lands is at least the one requested.
        self.assertGreaterEqual(down, 0.125 - 1e-9)
        self.assertGreaterEqual(across, 0.125 - 1e-9)

    def test_the_window_lands_the_same_whether_or_not_an_overlap_was_asked_for(self):
        # The invariant the fixed order buys. Reversed, the stride the overlap just set decides
        # which sizes the window search will accept, and --vae_tile_size lands somewhere else
        # without saying so.
        window = vae_tiling.tile_window(wan_vae(self))
        for size in (window // 4, window // 3, window // 2, window - 8):
            with self.subTest(size=size):
                alone, both = wan_vae(self), wan_vae(self)
                vae_setup.configure(alone, request(tile_size=size))
                vae_setup.configure(both, request(tile_size=size, tile_overlap=0.125))
                self.assertEqual(
                    both.tile_sample_min_height,
                    alone.tile_sample_min_height,
                    "--vae_tile_size landed differently because --vae_tile_overlap was also given",
                )


class TestRefusingATileThinnerThanTheGroup(unittest.TestCase):
    """A tile with fewer latent rows than ranks hangs the group rather than failing it"""

    def test_a_tile_short_of_a_row_per_rank_is_refused(self):
        vae = legacy_pair_vae()
        vae_setup.apply_window(vae, request(tile_size=128))  # 16 latent rows
        with self.assertRaises(ValueError) as caught:
            vae_setup.check_against_split(
                vae, request(splits_tiles=True, ranks=32), own_window=256
            )
        self.assertIn("--use_parallel_vae", str(caught.exception))

    def test_a_tile_with_rows_to_spare_passes(self):
        vae = legacy_pair_vae()
        vae_setup.apply_window(vae, request(tile_size=128))
        vae_setup.check_against_split(
            vae, request(splits_tiles=True, ranks=4), own_window=256
        )

    def test_nothing_is_checked_when_the_run_is_not_splitting_tiles(self):
        vae = legacy_pair_vae()
        vae_setup.apply_window(vae, request(tile_size=128))
        vae_setup.check_against_split(vae, request(ranks=32), own_window=256)

    def test_one_rank_divides_nothing(self):
        vae = legacy_pair_vae()
        vae_setup.apply_window(vae, request(tile_size=128))
        vae_setup.check_against_split(
            vae, request(splits_tiles=True, ranks=1), own_window=256
        )


class TestDecodeGuard(unittest.TestCase):

    @staticmethod
    def raising(error):
        def decode(z):
            raise error
        return decode

    def test_the_success_path_is_untouched(self):
        vae = StubVAE(decode=lambda z: ("decoded", z))
        vae_setup.install_decode_guard(vae, request())
        self.assertEqual(vae.decode(7), ("decoded", 7))

    def test_a_padding_failure_names_the_window_this_run_set(self):
        vae = StubVAE(decode=self.raising(RuntimeError(
            "Padding size should be less than the corresponding input dimension"
        )))
        vae_setup.install_decode_guard(vae, request(tile_size=128), window=128)
        with self.assertRaises(RuntimeError) as caught:
            vae.decode(None)
        self.assertIn("128px tile window", str(caught.exception))

    def test_a_padding_failure_at_the_vae_s_own_window_is_left_alone(self):
        # Nothing this run set, so nothing this run can suggest changing.
        original = RuntimeError("Padding size should be less than the corresponding input dimension")
        vae = StubVAE(decode=self.raising(original))
        vae_setup.install_decode_guard(vae, request(), window=None)
        with self.assertRaises(RuntimeError) as caught:
            vae.decode(None)
        self.assertIs(caught.exception, original)

    def test_an_unrelated_runtime_error_is_left_alone(self):
        original = RuntimeError("expected scalar type Half but found Float")
        vae = StubVAE(decode=self.raising(original))
        vae_setup.install_decode_guard(vae, request(tile_size=128), window=128)
        with self.assertRaises(RuntimeError) as caught:
            vae.decode(None)
        self.assertIs(caught.exception, original)

    def test_installing_again_updates_the_window_instead_of_nesting(self):
        # A runner's initialize() can run twice in a process. A second guard wrapping the first
        # would report whichever window was set first, forever.
        vae = StubVAE(decode=self.raising(RuntimeError(
            "Padding size should be less than the corresponding input dimension"
        )))
        vae_setup.install_decode_guard(vae, request(tile_size=256), window=256)
        once = vae.decode
        vae_setup.install_decode_guard(vae, request(tile_size=128), window=128)
        self.assertIs(vae.decode, once, "a second guard was stacked on the first")
        with self.assertRaises(RuntimeError) as caught:
            vae.decode(None)
        self.assertIn("128px", str(caught.exception))
        self.assertNotIn("256px", str(caught.exception))


class TestOomHint(unittest.TestCase):

    def test_a_model_that_can_tile_is_told_to(self):
        hint = vae_setup.oom_hint(StubVAE(use_tiling=False), request(model_tiles=True))
        self.assertIn("--enable_tiling", hint)

    def test_a_model_that_cannot_tile_is_not_sent_after_a_flag_it_lacks(self):
        hint = vae_setup.oom_hint(StubVAE(use_tiling=False), request(model_tiles=False))
        self.assertNotIn("--enable_tiling", hint)
        self.assertIn("test-model", hint)

    def test_a_tiling_vae_is_given_a_window_it_would_accept(self):
        vae = overlap_factor_vae()
        vae.use_tiling = True
        hint = vae_setup.oom_hint(vae, request())
        target = int(hint.split("--vae_tile_size ")[1].split(",")[0])
        # The hint must never name a size the next run would turn around and refuse.
        pixels, plan = vae_tiling.snap_tile_window(overlap_factor_vae(), target)
        self.assertIsNotNone(plan)
        self.assertEqual(pixels, target)


class TestConfigureEndToEnd(unittest.TestCase):

    def test_slicing_is_enabled_when_asked(self):
        vae = tileable(StubVAE())
        vae_setup.configure(vae, request(slicing=True))
        self.assertTrue(vae.use_slicing)

    def test_tiling_is_enabled_when_asked(self):
        vae = tileable(legacy_pair_vae())
        vae_setup.configure(vae, request(tiling=True))
        self.assertTrue(vae.use_tiling)

    def test_a_size_turns_tiling_on_by_itself(self):
        # Otherwise sizing the window means passing --enable_tiling too, which tiles every stage
        # to reach the one that ran out of memory.
        vae = tileable(legacy_pair_vae())
        vae_setup.configure(vae, request(tile_size=128))
        self.assertTrue(vae.use_tiling)
        self.assertEqual(vae.tile_sample_min_size, 128)

    def test_a_vae_that_cannot_tile_is_refused_by_the_flag_that_asked(self):
        # No `use_tiling`, which is how diffusers says it does not implement tiling for a class.
        with self.assertRaises(ValueError) as caught:
            vae_setup.configure(StubVAE(decode=lambda z: z), request(tile_size=128))
        self.assertIn("--vae_tile_size", str(caught.exception))

    def test_the_guard_is_installed_even_with_tiling_off(self):
        # A decode that runs out of memory with tiling off still needs to say what to turn on.
        vae = tileable(StubVAE())
        vae_setup.configure(vae, request())
        self.assertTrue(getattr(vae, "_xfuser_decode_guarded", False))

    def test_a_vae_that_is_not_asked_to_tile_keeps_its_window(self):
        vae = tileable(legacy_pair_vae())
        vae_setup.configure(vae, request())
        self.assertEqual(vae.tile_sample_min_size, 256)
        self.assertFalse(vae.use_tiling)


if __name__ == "__main__":
    unittest.main()
