import os
import torch
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional
from packaging import version

from xfuser.logger import init_logger

logger = init_logger(__name__)

if TYPE_CHECKING:
    MASTER_ADDR: str = ""
    MASTER_PORT: Optional[int] = None
    CUDA_HOME: Optional[str] = None
    LOCAL_RANK: int = 0
    CUDA_VISIBLE_DEVICES: Optional[str] = None
    XDIT_LOGGING_LEVEL: str = "INFO"
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
    "XDIT_LOGGING_LEVEL":
    lambda: os.getenv("XDIT_LOGGING_LEVEL", "INFO"),
}

variables: Dict[str, Callable[[], Any]] = {

    # ================== Other Vars ==================

    # used in version checking
    'CUDA_VERSION': 
    lambda: version.parse(torch.version.cuda),

    'TORCH_VERSION':
    lambda: version.parse(version.parse(torch.__version__).base_version),
}

class PackagesEnvChecker:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(PackagesEnvChecker, cls).__new__(cls)
            cls._instance.initialize()
        return cls._instance

    def initialize(self):
        self.packages_info = {
            'has_flash_attn': self.check_flash_attn(),
            'has_long_ctx_attn': self.check_long_ctx_attn(),
        } 
        

    def check_flash_attn(self):
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            gpu_name = torch.cuda.get_device_name(device)
            if "Turing" in gpu_name or "Tesla" in gpu_name or "T4" in gpu_name:
                return False
            else:
                from flash_attn import flash_attn_func
                return True
        except ImportError:
            logger.warning(f'Flash Attention library "flash_attn" not found, '
                           f'using pytorch attention implementation')
            return False

    def check_long_ctx_attn(self):
        try:
            from yunchang import (
                set_seq_parallel_pg,
                ring_flash_attn_func,
                UlyssesAttention,
                LongContextAttention,
                LongContextAttentionQKVPacked,
            )
            return True
        except ImportError:
            logger.warning(f'Ring Flash Attention library "yunchang" not found, '
                           f'using pytorch attention implementation')
            return False

    def get_packages_info(self):
        return self.packages_info

PACKAGES_CHECKER = PackagesEnvChecker()


def __getattr__(name):
    # lazy evaluation of environment variables
    if name in environment_variables:
        return environment_variables[name]()
    if name in variables:
        return variables[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return list(environment_variables.keys())
