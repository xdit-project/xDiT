"""Structural guards against the VAE setup mistakes that fail quietly.

Both of the regressions guarded here produce no error and no warning: a run simply does less than
it was asked to. The first reaches for `self.pipe.vae` and so misses the full-resolution decode of
a staged model. The second calls `_setup_parallel_vae()` somewhere other than
`_post_load_and_state_initialization`, and so marks a VAE for tile dealing after the decode that
reads that mark has already been installed.

Neither is the kind of thing a behavioural test catches, because a single-stage model on one GPU
does the right thing either way.
"""

import ast
import inspect
import unittest
from pathlib import Path

import torch
from torch import nn

from xfuser.model_executor.models import runner_models
from xfuser.model_executor.models.runner_models import base_model
from xfuser.model_executor.models.runner_models.base_model import xFuserModel


class TinyVAE(nn.Module):
    """Enough of a VAE for the channels-last pass: something with a convolution, and a decode"""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(4, 4, kernel_size=3, padding=1)
        self.decoded = []

    def decode(self, z):
        self.decoded.append(z)
        return z


class Pipe:
    def __init__(self, vae):
        self.vae = vae


class Settings:
    model_output_type = "image"


class Staged:
    """A runner model with two stages, borrowing the methods under test from the real class"""

    _decoding_vaes = xFuserModel._decoding_vaes
    _convert_vae_to_channels_last = xFuserModel._convert_vae_to_channels_last
    _convert_one_vae_to_channels_last = xFuserModel._convert_one_vae_to_channels_last

    def __init__(self, first, second):
        self.pipe = Pipe(first)
        self.second_pipe = Pipe(second) if second is not None else None
        self.settings = Settings()


class TestStagedVAEsAreAllReached(unittest.TestCase):
    def test_a_second_stage_vae_is_converted_too(self):
        first, second = TinyVAE(), TinyVAE()
        Staged(first, second)._convert_vae_to_channels_last()

        for vae, which in ((first, "first"), (second, "second")):
            self.assertTrue(
                getattr(vae, "_xfuser_decode_channels_last", False),
                f"the {which} stage's VAE was left unconverted",
            )

    def test_the_one_stage_case_still_works(self):
        only = TinyVAE()
        Staged(only, None)._convert_vae_to_channels_last()
        self.assertTrue(getattr(only, "_xfuser_decode_channels_last", False))

    def test_a_vae_shared_by_both_stages_is_wrapped_once(self):
        # Stages sometimes share one VAE. Wrapping it twice would put a second copy of the same
        # conversion in front of every decode for the rest of the process.
        shared = TinyVAE()
        staged = Staged(shared, shared)
        staged._convert_vae_to_channels_last()
        once = shared.decode
        staged._convert_vae_to_channels_last()
        self.assertIs(shared.decode, once, "the shared VAE was wrapped a second time")

    def test_the_conversion_does_not_change_what_decode_returns(self):
        vae = TinyVAE()
        Staged(vae, None)._convert_vae_to_channels_last()
        # Random rather than zeros. A wrapper that returned zeros, or doubled what it was given,
        # or dropped the sign, all agree with the input when the input is zero, so the assertion
        # held for every way of getting this wrong as well as for getting it right.
        torch.manual_seed(0)
        z = torch.randn(1, 4, 8, 8)
        self.assertTrue(torch.equal(vae.decode(z), z))

    def test_the_decode_is_handed_a_channels_last_tensor(self):
        # The point of the pass, and the one thing none of the above asked about: they check the
        # marker attribute and the identity of the wrapper, so a conversion that set the flag,
        # wrapped the decode and then forgot to convert would satisfy all of them.
        vae = TinyVAE()
        Staged(vae, None)._convert_vae_to_channels_last()
        vae.decode(torch.randn(1, 4, 8, 8))
        self.assertTrue(
            vae.decoded[-1].is_contiguous(memory_format=torch.channels_last),
            "decode was given a tensor in the memory format the conversion exists to change",
        )


class TestNothingReachesPastTheList(unittest.TestCase):
    """The recurrence guard, which is the part worth keeping

    Fixing the one method that reached for `self.pipe.vae` fixes today's bug. The next VAE step
    someone adds will reach for it again, because it is the obvious thing to write and because
    nothing about a single-stage model ever complains.
    """

    # Methods that legitimately speak about the first pipeline's VAE alone. Empty on purpose: if
    # a case turns up, it goes here with a reason rather than the guard being dropped.
    ALLOWED = set()

    def _vae_methods(self):
        source = Path(inspect.getfile(base_model)).read_text(encoding="utf-8")
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "xFuserModel":
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        yield item
                return
        self.fail("xFuserModel not found in base_model")

    @staticmethod
    def _reaches_for_pipe_vae(method):
        """`self.pipe.vae` anywhere in the body"""
        for node in ast.walk(method):
            if (
                isinstance(node, ast.Attribute)
                and node.attr == "vae"
                and isinstance(node.value, ast.Attribute)
                and node.value.attr == "pipe"
                and isinstance(node.value.value, ast.Name)
                and node.value.value.id == "self"
            ):
                return True
        return False

    def test_no_vae_step_reaches_for_the_first_pipeline_alone(self):
        offenders = [
            method.name
            for method in self._vae_methods()
            if method.name not in self.ALLOWED and self._reaches_for_pipe_vae(method)
        ]
        self.assertEqual(
            offenders,
            [],
            "these reach for self.pipe.vae instead of walking self._decoding_vaes(), so they "
            f"miss the full-resolution decode of a staged model: {offenders}",
        )

    def test_the_walk_finds_the_methods_it_is_meant_to(self):
        # A guard that has stopped matching anything passes for the wrong reason.
        names = {method.name for method in self._vae_methods()}
        self.assertIn("_convert_vae_to_channels_last", names)
        self.assertIn("_decoding_vaes", names)


class TestTheParallelVAEIsSetUpBeforeTheOptions(unittest.TestCase):
    """Marking a VAE for tile dealing has to happen before the tiled decode is installed

    `initialize()` runs `_post_load_and_state_initialization()` and then `_enable_options()`.
    The first is where every runner calls `_setup_parallel_vae()`, which marks a VAE whose tiles
    will be dealt out to the group; the second reads that mark and installs the decode that does
    the dealing.

    Called the other way round, `_enable_options()` sees no mark and leaves upstream's own tiling
    loop in place, while `_setup_parallel_vae()` goes on to take the marking branch and so never
    shards the decoder either. Every rank then decodes every tile, identically. The output is
    correct and the run is silent; --use_parallel_vae has simply bought nothing.
    """

    HOME = "_post_load_and_state_initialization"

    def _call_sites(self):
        """Every `self._setup_parallel_vae()` in the runner models, and the method it sits in"""
        folder = Path(inspect.getfile(runner_models)).parent
        for path in sorted(folder.glob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                for inner in ast.walk(node):
                    if (
                        isinstance(inner, ast.Call)
                        and isinstance(inner.func, ast.Attribute)
                        and inner.func.attr == "_setup_parallel_vae"
                    ):
                        yield path.name, node.name, inner.lineno

    def test_every_runner_sets_the_parallel_vae_up_from_the_one_place(self):
        strays = [
            f"{name}:{line} in {method}()"
            for name, method, line in self._call_sites()
            if method != self.HOME
        ]
        self.assertEqual(
            strays,
            [],
            f"these call _setup_parallel_vae() outside {self.HOME}(), so the VAE may be marked "
            f"for tile dealing after the decode that reads the mark was installed: {strays}",
        )

    def test_the_walk_finds_the_call_sites_it_is_meant_to(self):
        # A guard that has stopped matching anything passes for the wrong reason.
        found = list(self._call_sites())
        self.assertGreater(len(found), 10, f"only found {len(found)} call sites; the walk broke")
        self.assertIn("wan.py", {name for name, _, _ in found})


if __name__ == "__main__":
    unittest.main()
