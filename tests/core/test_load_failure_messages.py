"""How a model behaves when it is too new for the installed diffusers.

The load-time backstop is the only thing standing between "this model needs a newer
diffusers" and a bare ImportError from somewhere inside a wrapper, so the wording and
the origin check are worth pinning down. The same applies to what the package chooses to
export: a pipeline offered without the pieces it needs fails later and worse. Nothing
here touches the network.
"""

import inspect
import unittest

import diffusers

from xfuser.model_executor.models.runner_models.base_model import (
    DIFFUSERS_FROM_SOURCE,
    ModelSettings,
    xFuserModel,
)


def _import_error(message, name):
    error = ImportError(message)
    error.name = name
    return error


class _StubModel(xFuserModel):
    """Fails the way a model does when upstream has not shipped a symbol yet."""

    error = _import_error(
        "cannot import name 'SomeNewThing' from 'diffusers.models.transformers.x'",
        "diffusers.models.transformers.x",
    )

    def _load_model(self):
        raise self.error

    def _run_pipe(self, input_args):
        raise NotImplementedError


def _instance(cls, model_name="some-org/some-model"):
    """Build an instance without __init__, which wants a fully parsed CLI config.

    _load_model_checked only reads settings.model_name and min_diffusers_version.
    """
    obj = object.__new__(cls)
    obj.settings = ModelSettings(model_name=model_name)
    return obj


class TestLoadFailureMessages(unittest.TestCase):
    def test_floor_names_the_release_to_upgrade_to(self):
        class Model(_StubModel):
            min_diffusers_version = "0.99.0"

        with self.assertRaises(ImportError) as caught:
            _instance(Model)._load_model_checked()
        message = str(caught.exception)
        self.assertIn("some-org/some-model", message)
        self.assertIn(diffusers.__version__, message)
        self.assertIn("SomeNewThing", message)
        self.assertIn("Requires diffusers>=0.99.0.", message)

    def test_unreleased_support_says_to_build_from_source(self):
        """No version can be named, so send the user to a source install instead.

        Naming a guessed release here is how a floor ends up pointing at a version that
        never shipped.
        """

        class Model(_StubModel):
            min_diffusers_version = DIFFUSERS_FROM_SOURCE

        with self.assertRaises(ImportError) as caught:
            _instance(Model)._load_model_checked()
        message = str(caught.exception)
        self.assertIn("has not landed in a diffusers release yet", message)
        self.assertIn("installed from source", message)
        self.assertNotIn(f"diffusers>={DIFFUSERS_FROM_SOURCE}", message)

    def test_without_a_floor_the_advice_stays_generic(self):
        class Model(_StubModel):
            min_diffusers_version = None

        with self.assertRaises(ImportError) as caught:
            _instance(Model)._load_model_checked()
        message = str(caught.exception)
        self.assertIn("SomeNewThing", message)
        self.assertIn("A newer diffusers is required.", message)
        self.assertNotIn("Requires diffusers>=", message)

    def test_original_error_is_kept_as_the_cause(self):
        class Model(_StubModel):
            min_diffusers_version = "0.99.0"

        with self.assertRaises(ImportError) as caught:
            _instance(Model)._load_model_checked()
        self.assertIs(caught.exception.__cause__, Model.error)

    def test_an_xfuser_bug_is_not_relabelled_as_a_version_problem(self):
        """An ImportError from our own code must reach the developer untouched.

        Blaming diffusers for an xfuser typo sends people off to upgrade a dependency
        that was never the problem.
        """

        class Model(_StubModel):
            min_diffusers_version = "0.99.0"
            error = _import_error(
                "cannot import name 'typo' from 'xfuser.core.distributed'",
                "xfuser.core.distributed",
            )

        with self.assertRaises(ImportError) as caught:
            _instance(Model)._load_model_checked()
        self.assertIs(caught.exception, Model.error)
        self.assertNotIn("is unavailable with diffusers", str(caught.exception))

    def test_a_module_named_like_ours_is_not_mistaken_for_diffusers(self):
        """xfuser has modules with diffusers in the name; only the root counts."""

        class Model(_StubModel):
            error = _import_error(
                "cannot import name 'oops' from 'xfuser.model_executor.cache."
                "diffusers_adapters.registry'",
                "xfuser.model_executor.cache.diffusers_adapters.registry",
            )

        with self.assertRaises(ImportError) as caught:
            _instance(Model)._load_model_checked()
        self.assertIs(caught.exception, Model.error)

    def test_non_import_failures_are_left_alone(self):
        class Model(_StubModel):
            error = OSError("no space left on device")

        with self.assertRaises(OSError):
            _instance(Model)._load_model_checked()


class TestEveryModelRegisters(unittest.TestCase):
    """A model too new for the installed diffusers must still appear in the registry.

    Registration is what turns "unknown model" into a message that names the missing
    symbol, so it has to survive an unusable model. Krea2 and Cosmos3 exercise this
    for real: no released diffusers ships their transformers.
    """

    def test_registry_covers_models_diffusers_cannot_run(self):
        import pkgutil

        import xfuser.model_executor.models.runner_models as runners
        from xfuser.model_executor.models.runner_models.base_model import (
            MODEL_REGISTRY,
        )

        for _, modname, _ in pkgutil.iter_modules(runners.__path__):
            __import__(
                f"xfuser.model_executor.models.runner_models.{modname}",
                fromlist=["_"],
            )

        for name in ["Krea-2-Raw", "Krea-2-Turbo", "Cosmos3-Super", "Cosmos3-Nano"]:
            self.assertIn(
                name,
                MODEL_REGISTRY,
                f"{name} must register even when diffusers cannot provide it",
            )


class TestExportedPipelinesCanParallelise(unittest.TestCase):
    """An exported pipeline must have a wrapper for the backbone it parallelises.

    xFuserPipelineBaseWrapper.from_pretrained asks the transformer register for a
    wrapper whenever any parallelism is on, and a miss raises "is not supported by
    xFuser", which names no version and never mentions diffusers. So a pipeline gated
    on a lower diffusers release than its transformer wrapper is worse than one that is
    simply absent: gate both halves on the same release instead.
    """

    def test_every_exported_transformer_pipeline_has_a_backbone_wrapper(self):
        from xfuser.model_executor.models.transformers.register import (
            xFuserTransformerWrappersRegister,
        )
        from xfuser.model_executor.pipelines.register import (
            xFuserPipelineWrapperRegister,
        )

        registered = {
            cls
            for cls in xFuserTransformerWrappersRegister._XFUSER_TRANSFORMER_MAPPING
            if cls is not None
        }

        checked, missing = 0, []
        for pipe_cls in xFuserPipelineWrapperRegister._XFUSER_PIPE_MAPPING:
            try:
                parameters = inspect.signature(pipe_cls.__init__).parameters
            except (TypeError, ValueError):
                continue
            # UNet pipelines are wrapped through a different register, and some
            # pipelines take *args so there is nothing to introspect.
            backbone = parameters.get("transformer")
            if backbone is None or backbone.annotation is inspect.Parameter.empty:
                continue
            if not isinstance(backbone.annotation, type):
                continue
            checked += 1
            if not any(issubclass(backbone.annotation, r) for r in registered):
                missing.append(
                    f"{pipe_cls.__name__} is exported but "
                    f"{backbone.annotation.__name__} has no registered wrapper"
                )

        self.assertGreater(checked, 0, "no exported transformer pipelines were checked")
        self.assertEqual(
            missing,
            [],
            "import the transformer wrapper in the pipeline's module so both are gated "
            "on the same diffusers release:\n  " + "\n  ".join(missing),
        )


if __name__ == "__main__":
    unittest.main()
