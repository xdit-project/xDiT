"""The xfuser and xfuser.config exports load on first access."""

import importlib

import pytest


@pytest.mark.parametrize("package", ["xfuser", "xfuser.config"])
def test_star_import_binds_every_export(package):
    namespace = {}
    exec(f"from {package} import *", namespace)

    exported = importlib.import_module(package).__all__
    assert exported
    assert all(namespace[name] is not None for name in exported)


def test_package_exports_the_same_objects_as_their_modules():
    from xfuser import EngineConfig, xFuserArgs
    from xfuser.config.args import xFuserArgs as args_xFuserArgs
    from xfuser.config.config import EngineConfig as config_EngineConfig

    assert xFuserArgs is args_xFuserArgs
    assert EngineConfig is config_EngineConfig


def test_optional_export_the_pipelines_do_not_supply_fails_to_import(monkeypatch):
    import xfuser
    import xfuser.model_executor.pipelines as pipelines

    monkeypatch.delattr(pipelines, "xFuserFlux2Pipeline", raising=False)
    monkeypatch.delitem(vars(xfuser), "xFuserFlux2Pipeline", raising=False)
    monkeypatch.delitem(vars(xfuser), "__all__", raising=False)

    with pytest.raises(ImportError):
        from xfuser import xFuserFlux2Pipeline  # noqa: F401
    assert "xFuserFlux2Pipeline" not in xfuser.__all__


def test_unknown_name_is_an_attribute_error():
    import xfuser

    with pytest.raises(AttributeError):
        xfuser.not_an_export


@pytest.mark.parametrize(("package", "name"), [("xfuser", "xFuserFluxPipeline"), ("xfuser.config", "EngineConfig")])
def test_an_attribute_error_while_an_export_loads_stays_the_cause(monkeypatch, package, name):
    error = AttributeError("module 'diffusers' has no attribute 'NotInThisVersion'")

    def broken_import(*args, **kwargs):
        raise error

    module = importlib.import_module(package)
    monkeypatch.delitem(vars(module), name, raising=False)
    # The submodule that supplies the export fails while it loads.
    monkeypatch.setattr(importlib, "import_module", broken_import)

    with pytest.raises(ImportError, match="NotInThisVersion") as raised:
        exec(f"from {package} import {name}", {})
    assert raised.value.__cause__ is error
