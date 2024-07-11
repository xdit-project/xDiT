import os
import torch
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional
from packaging import version

if TYPE_CHECKING:
    MASTER_ADDR: str = ""
    MASTER_PORT: Optional[int] = None
    CUDA_HOME: Optional[str] = None
    LOCAL_RANK: int = 0
    CUDA_VISIBLE_DEVICES: Optional[str] = None
    PIPEFUSION_LOGGING_LEVEL: str = "INFO"
    CUDA_VERSION: version.Version
    TORCH_VERSION: version.Version


environment_variables: Dict[str, Callable[[], Any]] = {

    # ================== Runtime Env Vars ==================

    # used in distributed environment to determine the master address
    'MASTER_ADDR':
    lambda: os.getenv('MASTER_ADDR', ""),

    # used in distributed environment to manually set the communication port
    'MASTER_PORT':
    lambda: int(os.getenv('MASTER_PORT', '0'))
    if 'MASTER_PORT' in os.environ else None,

    # path to cudatoolkit home directory, under which should be bin, include,
    # and lib directories.
    "CUDA_HOME":
    lambda: os.environ.get("CUDA_HOME", None),

    # local rank of the process in the distributed setting, used to determine
    # the GPU device id
    "LOCAL_RANK":
    lambda: int(os.environ.get("LOCAL_RANK", "0")),

    # used to control the visible devices in the distributed setting
    "CUDA_VISIBLE_DEVICES":
    lambda: os.environ.get("CUDA_VISIBLE_DEVICES", None),

    # this is used for configuring the default logging level
    "PIPEFUSION_LOGGING_LEVEL":
    lambda: os.getenv("PIPEFUSION_LOGGING_LEVEL", "INFO"),
}

variables: Dict[str, Callable[[], Any]] = {

    # ================== Other Vars ==================

    # used in version checking
    'CUDA_VERSION': 
    lambda: version.parse(torch.version.cuda),

    'TORCH_VERSION':
    lambda: version.parse(version.parse(torch.__version__).base_version),
}

def __getattr__(name):
    # lazy evaluation of environment variables
    if name in environment_variables:
        return environment_variables[name]()
    if name in variables:
        return variables[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return list(environment_variables.keys())
