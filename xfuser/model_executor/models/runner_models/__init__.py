import pkgutil

from xfuser.compat import import_optional

# Import all model modules to trigger decorator registration. A module the installed
# diffusers is too old for is skipped, leaving its models unregistered. Runner modules
# import their pipelines and transformer wrappers lazily, so a module failing here
# needs a newer diffusers just to define its model classes.
for importer, modname, ispkg in pkgutil.iter_modules(__path__):
    if modname != "base_model":
        import_optional(f".{modname}", package=__name__)
