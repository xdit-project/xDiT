import diffusers
from packaging.version import Version

DEFAULT_MINIMUM_DIFFUSERS_VERSION = "0.33.0"
MINIMUM_DIFFUSERS_VERSIONS = {
    "hunyuanvideo_15": "0.36.0",
    "zimage": "0.36.0",
    "flux2": "0.36.0",
    "flux": "0.35.2",
    "flux_kontext": "0.35.2",
    "hunyuanvideo": "0.35.2",
    "wan": "0.35.2",
}

def has_valid_diffusers_version(model_name: str|None = None) -> bool:
    diffusers_version = diffusers.__version__
    minimum_diffusers_version = MINIMUM_DIFFUSERS_VERSIONS.get(model_name, DEFAULT_MINIMUM_DIFFUSERS_VERSION)
    return Version(diffusers_version).release >= Version(minimum_diffusers_version).release


def get_minimum_diffusers_version(model_name: str|None = None) -> str:
    return MINIMUM_DIFFUSERS_VERSIONS.get(model_name, DEFAULT_MINIMUM_DIFFUSERS_VERSION)