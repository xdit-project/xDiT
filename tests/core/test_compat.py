import ast
import importlib
import importlib.metadata
import pathlib
import shutil
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from xfuser.compat import (
    _import_optional,
    declared_floor,
    import_optional,
    is_diffusers_import_error,
    optional_exporter,
    reraise_unless_diffusers,
    version_at_least,
)


class TestVersionAtLeast(unittest.TestCase):
    """Floors must accept nightly and source builds of the version they name.

    A plain packaging comparison orders 0.36.0.dev0 before 0.36.0, which would lock
    out diffusers-from-source and torch nightlies. Compare release tuples instead.
    """

    def test_prerelease_satisfies_its_own_release(self):
        self.assertTrue(version_at_least("0.36.0.dev0", "0.36.0"))
        self.assertTrue(version_at_least("2.6.0.dev20250101+rocm6.3", "2.6.0"))
        self.assertTrue(version_at_least("5.0.0rc1", "5.0.0"))

    def test_local_version_segment_ignored(self):
        self.assertTrue(version_at_least("2.7.0+cu124", "2.6.0"))

    def test_older_versions_still_rejected(self):
        self.assertFalse(version_at_least("0.32.0", "0.33.0"))
        self.assertFalse(version_at_least("2.4.1", "2.6.0"))
        self.assertFalse(version_at_least("0.35.2", "0.36.0"))

    def test_unparseable_version_is_not_blocked(self):
        self.assertTrue(version_at_least("some-vendor-build", "0.33.0"))


class TestDiffusersImportErrors(unittest.TestCase):
    """Only diffusers-origin failures may be swallowed as version mismatches."""

    @staticmethod
    def _error(name):
        exc = ImportError("boom")
        exc.name = name
        return exc

    def test_diffusers_origin_is_recognised(self):
        self.assertTrue(is_diffusers_import_error(self._error("diffusers")))
        self.assertTrue(
            is_diffusers_import_error(
                self._error("diffusers.models.transformers.transformer_z_image")
            )
        )

    def test_xfuser_diffusers_adapters_is_not_diffusers(self):
        # Matching "diffusers" as a substring would hide real bugs in this package.
        exc = self._error("xfuser.model_executor.cache.diffusers_adapters.flux")
        self.assertFalse(is_diffusers_import_error(exc))
        with self.assertRaises(ImportError):
            reraise_unless_diffusers(exc)

    def test_bare_import_error_is_reraised(self):
        with self.assertRaises(ImportError):
            reraise_unless_diffusers(self._error(None))

    def test_diffusers_origin_is_swallowed(self):
        self.assertIsNone(reraise_unless_diffusers(self._error("diffusers")))


class TestDeclaredFloor(unittest.TestCase):
    """Floors are resolved from setup.py, never duplicated in runtime code."""

    def _with_requires(self, requires):
        return patch.object(
            importlib.metadata, "requires", lambda name: requires, create=False
        )

    def test_reads_install_requires(self):
        with self._with_requires(["diffusers>=0.33.0", "torch>=2.4.1", "einops"]):
            declared_floor.cache_clear()
            self.assertEqual(declared_floor("diffusers"), "0.33.0")
            self.assertEqual(declared_floor("torch"), "2.4.1")
        declared_floor.cache_clear()

    def test_reads_extras_with_markers(self):
        with self._with_requires(['flash-attn>=2.6.0; extra == "flash-attn"']):
            declared_floor.cache_clear()
            self.assertEqual(declared_floor("flash-attn"), "2.6.0")
        declared_floor.cache_clear()

    def test_unbounded_and_unknown_dependencies_yield_none(self):
        with self._with_requires(["einops", "diffusers>=0.33.0"]):
            declared_floor.cache_clear()
            self.assertIsNone(declared_floor("einops"))
            self.assertIsNone(declared_floor("scipy"))
        declared_floor.cache_clear()

    def test_uninstalled_source_tree_yields_none(self):
        def raise_missing(name):
            raise importlib.metadata.PackageNotFoundError(name)

        with patch.object(importlib.metadata, "requires", raise_missing):
            declared_floor.cache_clear()
            self.assertIsNone(declared_floor("diffusers"))
        declared_floor.cache_clear()

    def test_setup_py_floors_are_actually_resolvable(self):
        # Guards against setup.py being reshaped into something requires() can't
        # express, which would silently disable every runtime version check.
        setup_py = (_repo_root() / "setup.py").read_text()
        self.assertIn("diffusers>=", setup_py)
        self.assertIn("flash-attn>=", setup_py)


def _repo_root():
    return pathlib.Path(__file__).resolve().parents[2]


class TestOptionalExport(unittest.TestCase):
    """One mechanism gates every version-dependent feature: its module importing.

    Builds a throwaway package against a stub diffusers missing the newer symbols,
    which is what an old-diffusers install looks like from xfuser's side.
    """

    PACKAGE = "_xfuser_export_fixture"

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        root = pathlib.Path(self.tmp) / self.PACKAGE
        root.mkdir()
        (root / "__init__.py").write_text("__all__ = []\n")
        (root / "gated.py").write_text("from diffusers import NotInOldDiffusers\n")
        (root / "available.py").write_text("Wrapper = 'wrapper'\n")
        (root / "buggy.py").write_text("from .typo_module import thing\n")
        sys.path.insert(0, self.tmp)
        self._stub_diffusers = "diffusers" not in sys.modules
        if self._stub_diffusers:
            sys.modules["diffusers"] = types.ModuleType("diffusers")
        _import_optional.cache_clear()

    def tearDown(self):
        sys.path.remove(self.tmp)
        shutil.rmtree(self.tmp)
        for name in [m for m in sys.modules if m.startswith(self.PACKAGE)]:
            del sys.modules[name]
        if self._stub_diffusers:
            del sys.modules["diffusers"]
        _import_optional.cache_clear()

    def _exporter(self):
        namespace = {"__name__": self.PACKAGE, "__all__": []}
        return namespace, optional_exporter(namespace)

    def test_available_symbol_is_exported(self):
        namespace, optional = self._exporter()
        optional(".available", "Wrapper")
        self.assertEqual(namespace["Wrapper"], "wrapper")
        self.assertEqual(namespace["__all__"], ["Wrapper"])

    def test_gated_symbol_is_absent_rather_than_none(self):
        # Binding a None placeholder would turn a clear ImportError at the call site
        # into a confusing AttributeError deep inside a pipeline.
        namespace, optional = self._exporter()
        optional(".gated", "Anything")
        self.assertNotIn("Anything", namespace)
        self.assertEqual(namespace["__all__"], [])

    def test_one_gated_feature_does_not_hide_another(self):
        namespace, optional = self._exporter()
        optional(".gated", "Anything")
        optional(".available", "Wrapper")
        self.assertEqual(namespace["__all__"], ["Wrapper"])

    def test_symbol_missing_from_importable_module_is_skipped(self):
        # The same call has to work against a re-exporting package, where a name is
        # missing precisely when the gated module below it was unavailable. Raising
        # there would defeat the point.
        namespace, optional = self._exporter()
        optional(".available", "NotThere")
        self.assertNotIn("NotThere", namespace)
        self.assertEqual(namespace["__all__"], [])

    def test_xfuser_side_bug_is_not_swallowed(self):
        with self.assertRaises(ImportError):
            import_optional(f"{self.PACKAGE}.buggy")

    def test_repeated_gated_import_warns_once(self):
        # Python does not cache failed imports, so two packages re-exporting the same
        # unavailable module would each re-run it and each log a warning.
        with self.assertLogs("xfuser.compat", level="WARNING") as captured:
            self._exporter()[1](".gated", "Anything")
            self._exporter()[1](".gated", "Anything")
        self.assertEqual(len(captured.records), 1)

    def test_relative_and_absolute_names_share_a_result(self):
        absolute = import_optional(f"{self.PACKAGE}.available")
        relative = import_optional(".available", package=self.PACKAGE)
        self.assertIs(absolute, relative)


class TestReexportChain(unittest.TestCase):
    """The same call re-exports a gated symbol up through every package level.

    Mirrors how xfuser/__init__.py re-exports what model_executor.pipelines gated,
    against a fixture, so the chain is covered without needing torch installed.
    """

    PACKAGE = "_xfuser_chain_fixture"

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        root = pathlib.Path(self.tmp) / self.PACKAGE
        (root / "sub").mkdir(parents=True)
        (root / "sub" / "gated.py").write_text(
            "from diffusers import OnlyInNewDiffusers\nGated = 'gated'\n"
        )
        exporter = (
            "from xfuser.compat import optional_exporter\n"
            "__all__ = ['Stable']\n"
            "Stable = 'stable'\n"
            "_optional = optional_exporter(globals())\n"
        )
        (root / "sub" / "__init__.py").write_text(
            exporter + "_optional('.gated', 'Gated')\n"
        )
        (root / "__init__.py").write_text(exporter + "_optional('.sub', 'Gated')\n")
        sys.path.insert(0, self.tmp)
        self._stub_diffusers = "diffusers" not in sys.modules
        if self._stub_diffusers:
            sys.modules["diffusers"] = types.ModuleType("diffusers")

    def tearDown(self):
        sys.path.remove(self.tmp)
        shutil.rmtree(self.tmp)
        self._forget()
        if self._stub_diffusers:
            del sys.modules["diffusers"]
        _import_optional.cache_clear()

    def _forget(self):
        for name in [m for m in sys.modules if m.startswith(self.PACKAGE)]:
            del sys.modules[name]
        _import_optional.cache_clear()

    def _load(self, new_diffusers):
        self._forget()
        diffusers = sys.modules["diffusers"]
        if new_diffusers:
            diffusers.OnlyInNewDiffusers = object()
        elif hasattr(diffusers, "OnlyInNewDiffusers"):
            del diffusers.OnlyInNewDiffusers
        return importlib.import_module(self.PACKAGE)

    def test_gated_symbol_reaches_the_top_when_provided(self):
        top = self._load(new_diffusers=True)
        self.assertEqual(top.Gated, "gated")
        self.assertIn("Gated", top.__all__)

    def test_gated_symbol_absent_all_the_way_up_otherwise(self):
        top = self._load(new_diffusers=False)
        self.assertFalse(hasattr(top, "Gated"))
        self.assertNotIn("Gated", top.__all__)

    def test_stable_exports_unaffected_either_way(self):
        for new_diffusers in (True, False):
            with self.subTest(new_diffusers=new_diffusers):
                self.assertEqual(self._load(new_diffusers).Stable, "stable")


class TestRunnerModulesImportLazily(unittest.TestCase):
    """Runner modules must import transformer wrappers inside functions.

    The wrappers pull version-specific diffusers symbols. Importing one at module
    scope makes the whole runner module unimportable on an older diffusers, so its
    models never register and the user gets "unknown model". Kept lazy, the model
    registers and _load_model_checked names the symbol upstream has not shipped.
    """

    WRAPPER_PACKAGE = "xfuser.model_executor.models.transformers"

    def test_no_module_scope_wrapper_imports(self):
        runners = (
            _repo_root() / "xfuser" / "model_executor" / "models" / "runner_models"
        )
        offenders = []
        for path in sorted(runners.glob("*.py")):
            for node in ast.parse(path.read_text()).body:
                if (
                    isinstance(node, ast.ImportFrom)
                    and node.module
                    and node.module.startswith(self.WRAPPER_PACKAGE)
                ):
                    offenders.append(f"{path.name}:{node.lineno} imports {node.module}")
        self.assertEqual(
            offenders,
            [],
            "move these imports inside the method that uses them:\n  "
            + "\n  ".join(offenders),
        )


if __name__ == "__main__":
    unittest.main()
