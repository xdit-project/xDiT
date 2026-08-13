"""Dependency-floor and optional-import helpers.

Imports nothing heavy (no torch, no diffusers) so that package ``__init__`` files can
use it before those dependencies are known to work.
"""

import importlib
import importlib.metadata
import importlib.util
from functools import lru_cache
from types import ModuleType
from typing import Any, Callable, Dict, Optional

from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name
from packaging.version import InvalidVersion, Version

from xfuser.logger import init_logger

logger = init_logger(__name__)

@lru_cache(maxsize=None)
def declared_floor(name: str) -> Optional[str]:
    """The minimum version of ``name`` that setup.py declares xfuser needs.

    setup.py's ``install_requires``/``extras_require`` are the single place a floor is
    written down; setuptools copies them into xfuser's distribution metadata, which is
    what this reads back. Returns the bare version, e.g. ``"0.33.0"`` for
    ``"diffusers>=0.33.0"``.

    Returns None when no floor can be determined: an unrecognised dependency, a
    requirement with no lower bound, or a source tree that was never installed and has
    no metadata. Callers then skip their check. pip enforces these floors at install
    time anyway, so the runtime check only catches a dependency swapped out afterwards.
    """
    try:
        requirements = importlib.metadata.requires("xfuser") or ()
    except importlib.metadata.PackageNotFoundError:
        logger.debug(f"xfuser has no installed metadata; not checking {name} version")
        return None

    wanted = canonicalize_name(name)
    for entry in requirements:
        try:
            requirement = Requirement(entry)
        except InvalidRequirement:
            continue
        if canonicalize_name(requirement.name) != wanted:
            continue
        # Take the lower bound only. The comparison itself is version_at_least's job:
        # SpecifierSet.contains() rejects prereleases by default, and exactly how it
        # does so changed across packaging 24.2 / 25.0 / 26.0.
        for specifier in requirement.specifier:
            if specifier.operator in (">=", "==", "~="):
                return specifier.version

    logger.debug(f"setup.py declares no version floor for {name}")
    return None


def version_at_least(installed: str, floor: str) -> bool:
    """Whether ``installed`` is at least ``floor``, ignoring pre-release suffixes.

    Compares PEP 440 release tuples, so nightly and source builds count as the version
    they are heading towards. A plain comparison orders ``0.36.0.dev0`` before
    ``0.36.0``, which would lock out anyone running diffusers or torch from source,
    the install this project recommends.
    """
    try:
        return Version(installed).release >= Version(floor).release
    except InvalidVersion:
        # Let custom/vendor builds through; we have no basis for blocking them.
        logger.debug(
            f"Unparseable version {installed!r}; assuming it satisfies {floor}"
        )
        return True


def is_diffusers_import_error(exc: ImportError) -> bool:
    """Whether ``exc`` originated in the diffusers package.

    ``exc.name`` is the dotted path of the module that could not be imported; for
    ``from x import y`` it is ``x``. Match the top-level package exactly: a substring
    test would also swallow failures raised by xfuser's own ``diffusers_adapters``
    package, hiding real bugs as version mismatches.
    """
    return bool(exc.name) and exc.name.split(".")[0] == "diffusers"


def reraise_unless_diffusers(exc: ImportError) -> None:
    """Re-raise ``exc`` unless it is a diffusers version mismatch.

    Callers use this to skip optional, version-gated imports while letting genuine
    xfuser import bugs (typos, missing modules, circular imports) surface.
    """
    if not is_diffusers_import_error(exc):
        raise exc


def import_optional(name: str, package: Optional[str] = None) -> Optional[ModuleType]:
    """Import ``name``, returning None if the installed diffusers is too old for it.

    Every feature that needs diffusers symbols newer than the install floor lives in
    its own module, so importing that module is the availability test and no version
    number has to be written down. A failure originating in diffusers means the
    feature is unavailable here; an ImportError from xfuser is a bug and propagates.

    ``name`` may be relative if ``package`` is given, as for importlib.
    """
    return _import_optional(importlib.util.resolve_name(name, package))


@lru_cache(maxsize=None)
def _import_optional(name: str) -> Optional[ModuleType]:
    # Keyed on the absolute name and cached because Python does not cache failed
    # imports: without this, two packages re-exporting the same unavailable module
    # would each re-run the import and each log the warning.
    try:
        return importlib.import_module(name)
    except ImportError as e:
        reraise_unless_diffusers(e)
        logger.warning(
            f"Skipping {name}: {e}. What it provides will be unavailable, usually "
            "because the installed diffusers predates the feature. Set "
            "XDIT_LOGGING_LEVEL=DEBUG for the traceback."
        )
        logger.debug(f"Import of {name} failed", exc_info=True)
        return None


def optional_exporter(namespace: Dict[str, Any]) -> Callable[..., None]:
    """Build a package's re-exporter for symbols the installed diffusers may not have.

    Call once at the bottom of a package ``__init__``::

        _optional = optional_exporter(globals())
        _optional(".pipeline_flux2", "xFuserFlux2Pipeline")

    Each call binds the symbols it can and adds them to ``__all__``. The module may be
    relative to this package, and may be either the one defining the symbols or a
    package that re-exports them; either way the rule is to export what this install
    provides. Missing names stay unbound, so ``from xfuser import X`` raises
    ImportError at the import site.
    """
    package = namespace["__name__"]
    exported = namespace.setdefault("__all__", [])

    def optional(module: str, *symbols: str) -> None:
        resolved = import_optional(module, package)
        if resolved is None:
            return
        for symbol in symbols:
            if not hasattr(resolved, symbol):
                logger.debug(f"{module} does not provide {symbol} on this install")
                continue
            namespace[symbol] = getattr(resolved, symbol)
            exported.append(symbol)

    return optional
