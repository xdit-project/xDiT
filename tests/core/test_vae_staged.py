"""Structural tests for VAE setup invariants.

Every stage VAE must be discovered, and parallel VAE setup must run before tiled-decode
installation. Violating either invariant silently disables requested processing in multi-stage
or multi-GPU runs.
"""

import ast
import inspect
import unittest
from pathlib import Path

import torch
from torch import nn

from xfuser.model_executor.models import runner_models
from xfuser.model_executor.models.runner_models import base_model
from xfuser.model_executor.models.runner_models import ltx
from xfuser.model_executor.models.runner_models import vae_manager
from xfuser.model_executor.models.runner_models.base_model import xFuserModel


class TinyVAE(nn.Module):
    """Minimal VAE containing a convolution and decode method for channels-last tests."""

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
    """A runner model with two stages, borrowing the lifecycle delegators under test."""

    _decoding_vaes = xFuserModel._decoding_vaes
    _convert_vae_to_channels_last = xFuserModel._convert_vae_to_channels_last

    def __init__(self, first, second):
        self.pipe = Pipe(first)
        self.second_pipe = Pipe(second) if second is not None else None
        self.settings = Settings()
        self._vae_manager = vae_manager.VAEManager(
            config=object(), capabilities=object(), settings=self.settings
        )


class TestAllStagedVAEsAreProcessed(unittest.TestCase):
    def test_a_second_stage_vae_is_converted_too(self):
        first, second = TinyVAE(), TinyVAE()
        Staged(first, second)._convert_vae_to_channels_last()

        for vae, which in ((first, "first"), (second, "second")):
            self.assertTrue(
                getattr(vae, "_xfuser_decode_channels_last", False),
                f"the {which} stage's VAE was left unconverted",
            )

    def test_channels_last_conversion_supports_single_stage_pipeline(self):
        only = TinyVAE()
        Staged(only, None)._convert_vae_to_channels_last()
        self.assertTrue(getattr(only, "_xfuser_decode_channels_last", False))

    def test_a_vae_shared_by_both_stages_is_wrapped_once(self):
        # When two stages share a VAE, conversion must wrap its decode method only once.
        shared = TinyVAE()
        staged = Staged(shared, shared)
        staged._convert_vae_to_channels_last()
        once = shared.decode
        staged._convert_vae_to_channels_last()
        self.assertIs(shared.decode, once, "the shared VAE was wrapped a second time")

    def test_the_conversion_does_not_change_what_decode_returns(self):
        vae = TinyVAE()
        Staged(vae, None)._convert_vae_to_channels_last()
        # Use nonzero signed values so the assertion detects wrappers that zero, scale, or remove
        # the sign from the input.
        torch.manual_seed(0)
        z = torch.randn(1, 4, 8, 8)
        self.assertTrue(torch.equal(vae.decode(z), z))

    def test_decode_receives_channels_last_tensor(self):
        # The marker and wrapper-identity tests do not verify the input memory format. This
        # assertion confirms that the wrapper converts the tensor before calling decode.
        vae = TinyVAE()
        Staged(vae, None)._convert_vae_to_channels_last()
        vae.decode(torch.randn(1, 4, 8, 8))
        self.assertTrue(
            vae.decoded[-1].is_contiguous(memory_format=torch.channels_last),
            "decode received a tensor that was not converted to channels-last memory format",
        )


class TestVAEMethodsUseDecodingVAEList(unittest.TestCase):
    """Prevent VAE-processing methods from accessing only self.pipe.vae.

    Direct access would omit VAEs used by later pipeline stages, and single-stage tests would not
    expose the omission.
    """

    # List methods that may intentionally access only the first pipeline's VAE. Each entry must
    # include a reason for the exception.
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

    def test_vae_processing_methods_do_not_access_only_the_first_pipeline(self):
        offenders = [
            method.name
            for method in self._vae_methods()
            if method.name not in self.ALLOWED and self._reaches_for_pipe_vae(method)
        ]
        self.assertEqual(
            offenders,
            [],
            "these methods access self.pipe.vae instead of iterating over "
            f"self._decoding_vaes(), so they omit later pipeline stages: {offenders}",
        )

    def test_ast_scan_finds_expected_vae_methods(self):
        # Confirm that the AST scan still finds representative VAE methods.
        names = {method.name for method in self._vae_methods()}
        self.assertIn("_convert_vae_to_channels_last", names)
        self.assertIn("_decoding_vaes", names)


class TestVAEOrchestrationBoundary(unittest.TestCase):
    def test_manager_defines_planning_and_guards_and_base_defines_delegators(self):
        manager_methods = {
            "_apply_vae_tile_shape",
            "_apply_vae_tile_overlap",
            "_check_tiles_against_parallel_vae",
            "_install_vae_tiled_decode",
            "_install_vae_decode_guard",
            "_vae_decode_oom_hint",
        }
        self.assertTrue(manager_methods <= set(vars(vae_manager.VAEManager)))
        self.assertTrue(
            {"_setup_parallel_vae", "_convert_vae_to_channels_last", "_decoding_vaes"}
            <= set(vars(xFuserModel))
        )
        self.assertTrue(manager_methods.isdisjoint(vars(xFuserModel)))
        base_source = Path(inspect.getfile(base_model)).read_text(encoding="utf-8")
        for implementation_name in (
            "load_distvae_vae",
            "load_distvae_parallel_context",
            "vae_tiling",
            "vae_tile_parallel",
            "vae_parallel",
        ):
            self.assertNotIn(implementation_name, base_source)

    def test_ltx_model_loading_does_not_enable_tiling_outside_the_vae_manager(self):
        source = Path(inspect.getfile(ltx)).read_text(encoding="utf-8")
        tree = ast.parse(source)
        offenders = [
            node.lineno
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "enable_tiling"
        ]
        self.assertEqual(
            offenders,
            [],
            "LTX model loading enables VAE tiling directly instead of honoring "
            "config.enable_tiling through VAEManager",
        )


class TestTheParallelVAEIsSetUpBeforeTheOptions(unittest.TestCase):
    """Require parallel VAE setup before tiled-decode installation.

    `initialize()` calls `_post_load_and_state_initialization()` before `_enable_options()`.
    Parallel setup marks each VAE for complete-tile assignment, and option setup uses that mark
    when installing tiled decode. Reversing the order leaves the original tiled decode active,
    which makes every rank decode every tile without reporting an error.
    """

    HOME = "_post_load_and_state_initialization"

    def _call_sites(self):
        """Yield each self._setup_parallel_vae() call and its containing runner method."""
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

    def test_every_runner_sets_up_parallel_vae_in_post_load_initialization(self):
        strays = [
            f"{name}:{line} in {method}()"
            for name, method, line in self._call_sites()
            if method != self.HOME
        ]
        self.assertEqual(
            strays,
            [],
            f"these methods call _setup_parallel_vae() outside {self.HOME}(), so tiled decode "
            f"may be installed before the VAE is marked for complete-tile assignment: {strays}",
        )

    def test_ast_scan_finds_expected_parallel_vae_call_sites(self):
        # Confirm that the AST scan still finds representative parallel VAE call sites.
        found = list(self._call_sites())
        self.assertGreater(len(found), 10, f"AST scan found only {len(found)} call sites")
        self.assertIn("wan.py", {name for name, _, _ in found})


if __name__ == "__main__":
    unittest.main()
