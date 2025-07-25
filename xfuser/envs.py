import os
import torch
import diffusers
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional
from packaging import version

try:
    import torch_musa
except ModuleNotFoundError:
    pass

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
    "MASTER_ADDR": lambda: os.getenv("MASTER_ADDR", ""),
    # used in distributed environment to manually set the communication port
    "MASTER_PORT": lambda: (
        int(os.getenv("MASTER_PORT", "0")) if "MASTER_PORT" in os.environ else None
    ),
    # path to cudatoolkit home directory, under which should be bin, include,
    # and lib directories.
    "CUDA_HOME": lambda: os.environ.get("CUDA_HOME", None),
    # local rank of the process in the distributed setting, used to determine
    # the GPU device id
    "LOCAL_RANK": lambda: int(os.environ.get("LOCAL_RANK", "0")),
    # used to control the visible devices in the distributed setting
    "CUDA_VISIBLE_DEVICES": lambda: os.environ.get("CUDA_VISIBLE_DEVICES", None),
    # this is used for configuring the default logging level
    "XDIT_LOGGING_LEVEL": lambda: os.getenv("XDIT_LOGGING_LEVEL", "INFO"),
}


def _is_hip():
    has_rocm = torch.version.hip is not None
    return has_rocm


def _is_cuda():
    has_cuda = torch.version.cuda is not None
    return has_cuda


def _is_musa():
    try:
        if hasattr(torch, "musa") and torch.musa.is_available():
            return True
    except ModuleNotFoundError:
        return False


def get_device(local_rank: int) -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda", local_rank)
    elif _is_musa():
        return torch.device("musa", local_rank)
    else:
        return torch.device("cpu")


def get_device_name() -> str:
    if torch.cuda.is_available():
        return "cuda"
    elif _is_musa():
        return "musa"
    else:
        return "cpu"


def get_device_version():
    if _is_hip():
        hip_version = torch.version.hip
        hip_version = hip_version.split("-")[0]
        return hip_version
    elif _is_cuda():
        return torch.version.cuda
    elif _is_musa():
        return torch.version.musa
    else:
        raise NotImplementedError(
            "No Accelerators(AMD/NV/MTT GPU, AMD MI instinct accelerators) available"
        )


def get_torch_distributed_backend() -> str:
    if torch.cuda.is_available():
        return "nccl"
    elif _is_musa():
        return "mccl"
    else:
        raise NotImplementedError(
            "No Accelerators(AMD/NV/MTT GPU, AMD MI instinct accelerators) available"
        )


variables: Dict[str, Callable[[], Any]] = {
    # ================== Other Vars ==================
    # used in version checking
    "CUDA_VERSION": lambda: version.parse(get_device_version()),
    "TORCH_VERSION": lambda: version.parse(
        version.parse(torch.__version__).base_version
    ),
}


def _setup_musa(environment_variables, variables):
    musa = getattr(torch, "musa", None)
    if musa is None:
        return
    try:
        if musa.is_available():
            environment_variables["MUSA_HOME"] = lambda: os.environ.get(
                "MUSA_HOME", None
            )
            environment_variables["MUSA_VISIBLE_DEVICES"] = lambda: os.environ.get(
                "MUSA_VISIBLE_DEVICES", None
            )
            musa_ver = getattr(getattr(torch, "version", None), "musa", None)
            if musa_ver:
                variables["MUSA_VERSION"] = lambda: version.parse(musa_ver)
    except Exception:
        pass


try:
    _setup_musa(environment_variables, variables)
except (AttributeError, ModuleNotFoundError):
    pass


class PackagesEnvChecker:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(PackagesEnvChecker, cls).__new__(cls)
            cls._instance.initialize()
        return cls._instance

    def initialize(self):
        self.packages_info = {
            "has_flash_attn": self.check_flash_attn(),
            "has_long_ctx_attn": self.check_long_ctx_attn(),
            "diffusers_version": self.check_diffusers_version(),
        }

    def check_flash_attn(self):
        if _is_musa():
            logger.info(
                "Flash Attention library is not supported on MUSA for the moment."
            )
            return False
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            gpu_name = torch.cuda.get_device_name(device)
            if "Turing" in gpu_name or "Tesla" in gpu_name or "T4" in gpu_name:
                return False
            else:
                from flash_attn import flash_attn_func
                from flash_attn import __version__

                if __version__ < "2.6.0":
                    raise ImportError(f"install flash_attn >= 2.6.0")
                return True
        except ImportError:
            logger.warning(
                f'Flash Attention library "flash_attn" not found, '
                f"using pytorch attention implementation"
            )
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
            logger.warning(
                f'Ring Flash Attention library "yunchang" not found, '
                f"using pytorch attention implementation"
            )
            return False

    def check_diffusers_version(self):
        if version.parse(
            version.parse(diffusers.__version__).base_version
        ) < version.parse("0.30.0"):
            raise RuntimeError(
                f"Diffusers version: {version.parse(version.parse(diffusers.__version__).base_version)} is not supported,"
                f"please upgrade to version > 0.30.0"
            )
        return version.parse(version.parse(diffusers.__version__).base_version)

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
