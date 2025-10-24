import diffusers
from packaging import version

DEFAULT_MINIMUM_DIFFUSERS_VERSION = "0.33.0"
MINIMUM_DIFFUSERS_VERSIONS = {
    "flux": "0.35.2",
}

def has_valid_diffusers_version(model_name: str|None = None) -> bool:
    diffusers_version = diffusers.__version__
    minimum_diffusers_version = MINIMUM_DIFFUSERS_VERSIONS.get(model_name, DEFAULT_MINIMUM_DIFFUSERS_VERSION)
    return version.parse(diffusers_version) >= version.parse(minimum_diffusers_version)


def get_minimum_diffusers_version(model_name: str|None = None) -> str:
    return MINIMUM_DIFFUSERS_VERSIONS.get(model_name, DEFAULT_MINIMUM_DIFFUSERS_VERSION)