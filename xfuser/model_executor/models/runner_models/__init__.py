import importlib
import pkgutil

# Import all model modules to trigger decorator registration
for importer, modname, ispkg in pkgutil.iter_modules(__path__):
    if modname != 'base_model':
        importlib.import_module(f'.{modname}', package=__name__)