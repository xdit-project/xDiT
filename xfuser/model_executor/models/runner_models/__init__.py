import importlib
import pkgutil

from xfuser.logger import init_logger

logger = init_logger(__name__)

# Import all model modules to trigger decorator registration. A module whose
# diffusers pipeline classes are unavailable on the installed diffusers version
# is skipped (its models stay unregistered) so one version-fragile module can't
# crash the whole package.
for importer, modname, ispkg in pkgutil.iter_modules(__path__):
    if modname != 'base_model':
        try:
            importlib.import_module(f'.{modname}', package=__name__)
        except ImportError as e:
            # Only swallow errors that originate from diffusers (missing symbol /
            # wrong version). Re-raise anything internal to xfuser so developer
            # bugs (typos, missing modules) aren't silently hidden.
            if e.name is None or "diffusers" not in e.name:
                raise
            logger.warning(
                f"Skipping runner module '{modname}': {e}. Models it provides will be "
                "unavailable (often a diffusers version mismatch; run with "
                "XDIT_LOGGING_LEVEL=DEBUG for the traceback)."
            )
            logger.debug(f"Import of runner module '{modname}' failed", exc_info=True)
