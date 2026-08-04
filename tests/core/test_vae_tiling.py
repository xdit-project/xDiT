import unittest

from xfuser.core.utils import vae_tiling


class StubVAE:
    """Stands in for a diffusers VAE, carrying only the tiling attributes one would set"""

    def __init__(self, **attrs):
        for name, value in attrs.items():
            setattr(self, name, value)


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


if __name__ == "__main__":
    unittest.main()
