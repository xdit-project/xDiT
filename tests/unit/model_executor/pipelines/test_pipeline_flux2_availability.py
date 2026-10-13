"""FLUX.2-dev PipeFusion must load on diffusers 0.36, its declared floor.

diffusers 0.36 ships Flux2Pipeline but not Flux2KleinPipeline, which arrived in 0.37.
The fixture hides Flux2KleinPipeline from the installed diffusers and re-imports the
FLUX.2 pipeline modules, so the check runs against any newer diffusers too.
"""

import importlib
import sys

import diffusers
import pytest

import xfuser.model_executor.pipelines as pipelines_package
from xfuser.compat import _import_optional, import_optional, is_diffusers_import_error
from xfuser.model_executor.pipelines.register import xFuserPipelineWrapperRegister

FLUX2 = "xfuser.model_executor.pipelines.pipeline_flux2"
KLEIN = "xfuser.model_executor.pipelines.pipeline_flux2_klein"


@pytest.fixture
def diffusers_without_klein(monkeypatch):
    if not hasattr(diffusers, "Flux2Pipeline"):
        pytest.skip("installed diffusers predates FLUX.2")

    lazy_module_type = type(diffusers)
    original_getattr = lazy_module_type.__getattr__

    def getattr_without_klein(module, name):
        if module is diffusers and name == "Flux2KleinPipeline":
            raise AttributeError(name)
        return original_getattr(module, name)

    monkeypatch.setattr(lazy_module_type, "__getattr__", getattr_without_klein)
    monkeypatch.delattr(diffusers, "Flux2KleinPipeline", raising=False)

    # Re-import from scratch, and put the real modules, package attributes and wrapper
    # registrations back afterwards.
    for dotted in (FLUX2, KLEIN):
        monkeypatch.delitem(sys.modules, dotted, raising=False)
        attr = dotted.rsplit(".", 1)[1]
        if hasattr(pipelines_package, attr):
            monkeypatch.setattr(pipelines_package, attr, getattr(pipelines_package, attr))
    monkeypatch.setattr(
        xFuserPipelineWrapperRegister,
        "_XFUSER_PIPE_MAPPING",
        dict(xFuserPipelineWrapperRegister._XFUSER_PIPE_MAPPING),
    )
    _import_optional.cache_clear()
    yield
    _import_optional.cache_clear()


def test_flux2_dev_pipefusion_wrapper_loads_without_klein(diffusers_without_klein):
    module = importlib.import_module(FLUX2)

    assert xFuserPipelineWrapperRegister.get_class(diffusers.Flux2Pipeline) is module.xFuserFlux2Pipeline


def test_klein_wrapper_is_unavailable_as_a_diffusers_version_gap(diffusers_without_klein):
    with pytest.raises(ImportError) as caught:
        importlib.import_module(KLEIN)

    # The loader turns a diffusers-origin ImportError into "Requires diffusers>=0.37.0".
    assert is_diffusers_import_error(caught.value)
    assert import_optional(KLEIN) is None
