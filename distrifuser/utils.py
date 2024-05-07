import torch
from packaging import version
from torch import distributed as dist

from distrifuser.logger import init_logger

logger = init_logger(__name__)

from typing import Union, Optional


def check_env():
    if version.parse(torch.version.cuda) < version.parse("11.3"):
        # https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/cudagraph.html
        raise RuntimeError("NCCL CUDA Graph support requires CUDA 11.3 or above")
    if version.parse(version.parse(torch.__version__).base_version) < version.parse(
        "2.2.0"
    ):
        # https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/
        raise RuntimeError(
            "CUDAGraph with NCCL support requires PyTorch 2.2.0 or above. "
            "If it is not released yet, please install nightly built PyTorch with "
            "`pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121`"
        )


def is_power_of_2(n: int) -> bool:
    return (n & (n - 1) == 0) and n != 0


class DistriConfig:
    def __init__(
        self,
        height: int = 1024,
        width: int = 1024,
        do_classifier_free_guidance: bool = True,
        split_batch: bool = True,
        warmup_steps: int = 4,
        comm_checkpoint: int = 1,
        mode: str = "corrected_async_gn",
        use_cuda_graph: bool = True,
        parallelism: str = "patch",
        split_scheme: str = "row",
        use_seq_parallel_attn: bool = False,
        batch_size: Optional[int] = None,
        num_micro_batch: int = 2,
        verbose: bool = False,
        use_resolution_binning: bool = True,
    ):
        try:
            # Initialize the process group
            dist.init_process_group("nccl")
            # Get the rank and world_size
            rank = dist.get_rank()
            world_size = dist.get_world_size()
        except Exception as e:
            rank = 0
            world_size = 1
            logger.warning(
                f"Failed to initialize process group: {e}, falling back to single GPU"
            )

        assert is_power_of_2(world_size) or parallelism == "pipeline"
        check_env()

        self.world_size = world_size
        self.rank = rank
        self.height = height
        self.width = width
        self.do_classifier_free_guidance = do_classifier_free_guidance
        self.split_batch = split_batch
        self.warmup_steps = warmup_steps
        self.comm_checkpoint = comm_checkpoint
        self.mode = mode
        self.use_cuda_graph = use_cuda_graph
        self.batch_size = batch_size

        self.parallelism = parallelism
        self.split_scheme = split_scheme
        self.use_seq_parallel_attn = use_seq_parallel_attn

        self.verbose = verbose
        self.use_resolution_binning = use_resolution_binning

        if do_classifier_free_guidance and split_batch:
            n_device_per_batch = world_size // 2
            if n_device_per_batch == 0:
                n_device_per_batch = 1
        else:
            n_device_per_batch = world_size

        self.n_device_per_batch = n_device_per_batch

        self.height = height
        self.width = width

        device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(device)
        self.device = device

        batch_group = None
        split_group = None
        if do_classifier_free_guidance and split_batch and world_size >= 2:
            batch_groups = []
            for i in range(2):
                batch_groups.append(
                    dist.new_group(
                        list(range(i * (world_size // 2), (i + 1) * (world_size // 2)))
                    )
                )
            batch_group = batch_groups[self.batch_idx()]
            split_groups = []
            for i in range(world_size // 2):
                split_groups.append(dist.new_group([i, i + world_size // 2]))
            split_group = split_groups[self.split_idx()]
        self.batch_group = batch_group
        self.split_group = split_group

        self.num_micro_batch = num_micro_batch
        # if self.parallelism == "pipeline":
        #     self.groups = []
        #     for _ in range(num_micro_batch):
        #         self.groups.append(dist.new_group())

        # pipeline variance
        self.num_inference_steps = None

    def batch_idx(self, rank: Optional[int] = None) -> int:
        if rank is None:
            rank = self.rank
        if self.do_classifier_free_guidance and self.split_batch:
            return 1 - int(rank < (self.world_size // 2))
        else:
            return 0  # raise NotImplementedError

    def split_idx(self, rank: Optional[int] = None) -> int:
        if rank is None:
            rank = self.rank
        return rank % self.n_device_per_batch


class PatchParallelismCommManager:
    def __init__(self, distri_config: DistriConfig):
        self.distri_config = distri_config

        self.torch_dtype = None
        self.numel = 0
        self.numel_dict = {}

        self.buffer_list = None

        self.starts = []
        self.ends = []
        self.shapes = []

        self.idx_queue = []

        self.handles = None

    def register_tensor(
        self,
        shape: Union[tuple[int, ...], list[int]],
        torch_dtype: torch.dtype,
        layer_type: str = None,
    ) -> int:
        if self.torch_dtype is None:
            self.torch_dtype = torch_dtype
        else:
            assert self.torch_dtype == torch_dtype
        self.starts.append(self.numel)
        numel = 1
        for dim in shape:
            numel *= dim
        self.numel += numel
        if layer_type is not None:
            if layer_type not in self.numel_dict:
                self.numel_dict[layer_type] = 0
            self.numel_dict[layer_type] += numel

        self.ends.append(self.numel)
        self.shapes.append(shape)
        return len(self.starts) - 1

    def create_buffer(self):
        distri_config = self.distri_config
        if distri_config.rank == 0 and distri_config.verbose:
            print(
                f"Create buffer with {self.numel / 1e6:.3f}M parameters for {len(self.starts)} tensors on each device."
            )
            for layer_type, numel in self.numel_dict.items():
                print(f"  {layer_type}: {numel / 1e6:.3f}M parameters")

        self.buffer_list = [
            torch.empty(
                self.numel, dtype=self.torch_dtype, device=self.distri_config.device
            )
            for _ in range(self.distri_config.n_device_per_batch)
        ]
        self.handles = [None for _ in range(len(self.starts))]

    def get_buffer_list(self, idx: int) -> list[torch.Tensor]:
        buffer_list = [
            t[self.starts[idx] : self.ends[idx]].view(self.shapes[idx])
            for t in self.buffer_list
        ]
        return buffer_list

    def communicate(self):
        distri_config = self.distri_config
        start = self.starts[self.idx_queue[0]]
        end = self.ends[self.idx_queue[-1]]
        tensor = self.buffer_list[distri_config.split_idx()][start:end]
        buffer_list = [t[start:end] for t in self.buffer_list]
        handle = dist.all_gather(
            buffer_list, tensor, group=self.distri_config.batch_group, async_op=True
        )
        for i in self.idx_queue:
            self.handles[i] = handle
        self.idx_queue = []

    def enqueue(self, idx: int, tensor: torch.Tensor):
        distri_config = self.distri_config
        if idx == 0 and len(self.idx_queue) > 0:
            self.communicate()
        assert len(self.idx_queue) == 0 or self.idx_queue[-1] == idx - 1
        self.idx_queue.append(idx)
        self.buffer_list[distri_config.split_idx()][
            self.starts[idx] : self.ends[idx]
        ].copy_(tensor.flatten())

        if len(self.idx_queue) == distri_config.comm_checkpoint:
            self.communicate()

    def clear(self):
        if len(self.idx_queue) > 0:
            self.communicate()
        if self.handles is not None:
            for i in range(len(self.handles)):
                if self.handles[i] is not None:
                    self.handles[i].wait()
                    self.handles[i] = None
