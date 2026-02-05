import os
import torch
import torch.distributed as dist
from packaging import version
from dataclasses import dataclass, fields

from torch import distributed as dist

from xfuser.logger import init_logger
import xfuser.envs as envs
from xfuser.envs import CUDA_VERSION, TORCH_VERSION, PACKAGES_CHECKER

logger = init_logger(__name__)

from typing import Union, Optional, List

env_info = PACKAGES_CHECKER.get_packages_info()
HAS_LONG_CTX_ATTN = env_info["has_long_ctx_attn"]
HAS_FLASH_ATTN = env_info["has_flash_attn"]


def check_packages():
    import diffusers

    if not version.parse(diffusers.__version__) > version.parse("0.30.2"):
        raise RuntimeError(
            "This project requires diffusers version > 0.30.2. Currently, you can not install a correct version of diffusers by pip install."
            "Please install it from source code!"
        )


def check_env():
    # https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/cudagraph.html
    if CUDA_VERSION < version.parse("11.3"):
        raise RuntimeError("NCCL CUDA Graph support requires CUDA 11.3 or above")
    if TORCH_VERSION < version.parse("2.2.0"):
        # https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/
        raise RuntimeError(
            "CUDAGraph with NCCL support requires PyTorch 2.2.0 or above. "
            "If it is not released yet, please install nightly built PyTorch "
            "with `pip3 install --pre torch torchvision torchaudio --index-url "
            "https://download.pytorch.org/whl/nightly/cu121`"
        )


@dataclass
class ModelConfig:
    model: str
    download_dir: Optional[str] = None
    trust_remote_code: bool = False


@dataclass
class RuntimeConfig:
    warmup_steps: int = 1
    dtype: torch.dtype = torch.float16
    use_cuda_graph: bool = False
    use_parallel_vae: bool = False
    use_profiler: bool = False
    use_torch_compile: bool = False
    use_onediff: bool = False
    use_fp8_t5_encoder: bool = False
    use_teacache: bool = False
    use_fbcache: bool = False
    attention_backend: Optional[str] = None

    def __post_init__(self):
        check_packages()
        if self.use_cuda_graph:
            check_env()


@dataclass
class FastAttnConfig:
    use_fast_attn: bool = False
    n_step: int = 20
    n_calib: int = 8
    threshold: float = 0.5
    window_size: int = 64
    coco_path: Optional[str] = None
    use_cache: bool = False

    def __post_init__(self):
        assert self.n_calib > 0, "n_calib must be greater than 0"
        assert self.threshold > 0.0, "threshold must be greater than 0"


@dataclass
class DataParallelConfig:
    dp_degree: int = 1
    use_cfg_parallel: bool = False
    dit_parallel_size: int = 1

    def __post_init__(self):
        assert self.dp_degree >= 1, "dp_degree must greater than or equal to 1"

        # set classifier_free_guidance_degree parallel for split batch
        if self.use_cfg_parallel:
            self.cfg_degree = 2
        else:
            self.cfg_degree = 1
        assert self.dp_degree * self.cfg_degree <= self.dit_parallel_size, (
            "dp_degree * cfg_degree must be less than or equal to "
            "dit_parallel_size because of classifier free guidance"
        )
        assert (
            self.dit_parallel_size % (self.dp_degree * self.cfg_degree) == 0
        ), "dit_parallel_size must be divisible by dp_degree * cfg_degree"


@dataclass
class SequenceParallelConfig:
    ulysses_degree: Optional[int] = None
    ring_degree: Optional[int] = None
    shard_dit: bool = False
    dit_parallel_size: int = 1

    def __post_init__(self):
        if self.ulysses_degree is None:
            self.ulysses_degree = 1
            logger.info(
                f"Ulysses degree not set, " f"using default value {self.ulysses_degree}"
            )
        if self.ring_degree is None:
            self.ring_degree = 1
            logger.info(
                f"Ring degree not set, " f"using default value {self.ring_degree}"
            )
        self.sp_degree = self.ulysses_degree * self.ring_degree
        if self.sp_degree == 1 and self.shard_dit:
            self.shard_dit = False
            logger.info(
                f"Sequence parallelism not set, model sharding disabled"
            )

        if not HAS_LONG_CTX_ATTN and self.sp_degree > 1:
            raise ImportError(
                f"Sequence Parallel kit 'yunchang' not found but "
                f"sp_degree is {self.sp_degree}, please set it "
                f"to 1 or install 'yunchang' to use it"
            )


@dataclass
class TensorParallelConfig:
    tp_degree: int = 1
    split_scheme: Optional[str] = "row"
    dit_parallel_size: int = 1

    def __post_init__(self):
        assert self.tp_degree >= 1, "tp_degree must greater than 1"
        assert (
            self.tp_degree <= self.dit_parallel_size
        ), "tp_degree must be less than or equal to dit_parallel_size"


@dataclass
class FullyShardConfig:
    fs_degree: int = 1
    dit_parallel_size: int = 1

    def __post_init__(self):
        assert self.fs_degree >= 1, "fully_shard_degree must greater than 1"
        assert (
            self.fs_degree <= self.dit_parallel_size
        ), "fully_shard_degree must be less than or equal to dit_parallel_size"

@dataclass
class PipeFusionParallelConfig:
    pp_degree: int = 1
    num_pipeline_patch: Optional[int] = None
    attn_layer_num_for_pp: Optional[List[int]] = (None,)
    dit_parallel_size: int = 1

    def __post_init__(self):
        assert (
            self.pp_degree is not None and self.pp_degree >= 1
        ), "pipefusion_degree must be set and greater than 1 to use pipefusion"
        assert (
            self.pp_degree <= self.dit_parallel_size
        ), "pipefusion_degree must be less than or equal to dit_parallel_size"
        if self.num_pipeline_patch is None:
            self.num_pipeline_patch = self.pp_degree
            logger.info(
                f"Pipeline patch number not set, "
                f"using default value {self.pp_degree}"
            )
        if self.attn_layer_num_for_pp is not None:
            logger.info(
                f"attn_layer_num_for_pp set, splitting attention layers"
                f"to {self.attn_layer_num_for_pp}"
            )
            assert len(self.attn_layer_num_for_pp) == self.pp_degree, (
                "attn_layer_num_for_pp must have the same "
                "length as pp_degree if not None"
            )
        if self.pp_degree == 1 and self.num_pipeline_patch > 1:
            logger.warning(
                f"Pipefusion degree is 1, pipeline will not be used,"
                f"num_pipeline_patch will be ignored"
            )
            self.num_pipeline_patch = 1


@dataclass
class ParallelConfig:
    dp_config: DataParallelConfig
    sp_config: SequenceParallelConfig
    pp_config: PipeFusionParallelConfig
    tp_config: TensorParallelConfig
    fs_config: FullyShardConfig
    world_size: int = 1 # FIXME: remove this
    dit_parallel_size: int = 1
    vae_parallel_size: int = 1 # 0 means the vae is in the same process with diffusion
    shard_t5_encoder: bool = False

    def __post_init__(self):
        assert self.tp_config is not None, "tp_config must be set"
        assert self.dp_config is not None, "dp_config must be set"
        assert self.sp_config is not None, "sp_config must be set"
        assert self.pp_config is not None, "pp_config must be set"
        parallel_world_size = (
            self.dp_config.dp_degree
            * self.dp_config.cfg_degree
            * self.sp_config.sp_degree
            * self.tp_config.tp_degree
            * self.pp_config.pp_degree
        )
        dit_parallel_size = self.dit_parallel_size
        assert parallel_world_size == dit_parallel_size, (
            f"parallel_world_size {parallel_world_size} "
            f"must be equal to dit_parallel_size {self.dit_parallel_size}"
        )
        assert (
            dit_parallel_size % (self.dp_config.dp_degree * self.dp_config.cfg_degree) == 0
        ), "dit_parallel_size must be divisible by dp_degree * cfg_degree"
        assert (
            dit_parallel_size % self.pp_config.pp_degree == 0
        ), "dit_parallel_size must be divisible by pp_degree"
        assert (
            dit_parallel_size % self.sp_config.sp_degree == 0
        ), "dit_parallel_size must be divisible by sp_degree"
        assert (
            dit_parallel_size % self.tp_config.tp_degree == 0
        ), "dit_parallel_size must be divisible by tp_degree"
        self.dp_degree = self.dp_config.dp_degree
        self.cfg_degree = self.dp_config.cfg_degree
        self.sp_degree = self.sp_config.sp_degree
        self.pp_degree = self.pp_config.pp_degree
        self.tp_degree = self.tp_config.tp_degree
        self.fs_degree = self.fs_config.fs_degree

        self.ulysses_degree = self.sp_config.ulysses_degree
        self.ring_degree = self.sp_config.ring_degree


@dataclass(frozen=True)
class EngineConfig:
    model_config: ModelConfig
    runtime_config: RuntimeConfig
    parallel_config: ParallelConfig
    fast_attn_config: FastAttnConfig

    def __post_init__(self):
        if self.fast_attn_config.use_fast_attn:
            assert self.parallel_config.dp_degree == self.parallel_config.dit_parallel_size, f"dit_parallel_size must be equal to dp_degree when using DiTFastAttn"

    def to_dict(self):
        """Return the configs as a dictionary, for use in **kwargs."""
        return dict((field.name, getattr(self, field.name)) for field in fields(self))


@dataclass
class InputConfig:
    height: int = 1024
    width: int = 1024
    num_frames: int = 49
    use_resolution_binning: bool = (True,)
    batch_size: Optional[int] = None
    img_file_path: Optional[str] = None
    prompt: Union[str, List[str]] = ""
    negative_prompt: Union[str, List[str]] = ""
    num_inference_steps: int = 20
    max_sequence_length: int = 256
    seed: int = 42
    output_type: str = "pil"
    guidance_scale: float = 3.5

    def __post_init__(self):
        if isinstance(self.prompt, list):
            assert (
                not self.negative_prompt
                or len(self.prompt) == len(self.negative_prompt)
                or len(self.negative_prompt) == 0
            ), "prompts and negative_prompts must have the same quantities"
            self.batch_size = self.batch_size or len(self.prompt)
        else:
            self.batch_size = self.batch_size or 1
        assert self.output_type in [
            "pil",
            "latent",
            "pt",
        ], "output_pil must be either 'pil' or 'latent'"
