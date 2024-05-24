import torch
import torch.distributed as dist
from packaging import version
from torch import distributed as dist

from pipefuser.logger import init_logger

logger = init_logger(__name__)

from typing import Union, Optional, List


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
        pp_num_patch: int = 2,
        verbose: bool = False,
        use_resolution_binning: bool = True,
        attn_num: Optional[List[int]] = None,
        scheduler: str = "dpmsolver_multistep",
    ):
        f"""
        Configurations for distributed diffusion inference.
        
        Args:
            height (int, optional): height of generation image. Defaults to 1024.
            width (int, optional): width of generation image. Defaults to 1024.
            do_classifier_free_guidance (bool, optional): use classifier_free_guidance. Defaults to True.
            split_batch (bool, optional): first split the batch and then apply other parallelism. Defaults to True.
            warmup_steps (int, optional): sync timestep for DistriFusion and PipeFusion. Defaults to 4.
            comm_checkpoint (int, optional): _description_. Defaults to 1.
            mode (str, optional): sync mode. Defaults to "corrected_async_gn".
            use_cuda_graph (bool, optional): use cuda graph to accelerate speed. Defaults to True.
            parallelism (str, optional): parallel approches. Defaults to "patch".
            split_scheme (str, optional): how to split the image. Defaults to "row".
            use_seq_parallel_attn (bool, optional): sequence parallelism. Defaults to False.
            batch_size (Optional[int], optional): batch size. Defaults to None.
            pp_num_patch (int, optional): patch number. Defaults to 2.
            verbose (bool, optional): verbose print. Defaults to False.
            use_resolution_binning (bool, optional): image resolution bin. Defaults to True.
            attn_num (Optional[List[int]], optional): num of attn. Defaults to None.
            scheduler (str, optional): scheduler (sampler) for diffusion. Defaults to "dpmsolver_multistep".
        """
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

        self.pp_num_patch = pp_num_patch

        self.attn_num = attn_num
        self.scheduler = scheduler

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


class PipelineParallelismCommManager:
    def __init__(self, distri_config: DistriConfig):
        self.distri_config = distri_config
        self.recv_buffer: Optional[Union[List[torch.Tensor], torch.Tensor]] = None
        self.send_shape: Optional[torch.Size] = None
        self.recv_shape: Optional[List[torch.Size]] = None
        self.send_req = None
        self.recv_req = None
        self.recv_queue = []
        self.next_rank = (distri_config.rank + 1) % distri_config.world_size
        self.prev_rank = (
            distri_config.rank - 1 + distri_config.world_size
        ) % distri_config.world_size
        self.dtype = None
        if distri_config.world_size == 2:
            self.groups = [torch.distributed.new_group(), torch.distributed.new_group()]
        else:
            self.groups = None

    def _creat_recv_buffer(self):
        distri_config = self.distri_config
        assert self.recv_shape is not None
        shape = list(self.recv_shape)
        shape[-2] = shape[-2] // distri_config.pp_num_patch
        self.recv_buffer = [
            torch.zeros(
                shape,
                dtype=self.dtype,
                device=distri_config.device,
            )
            for _ in range(distri_config.pp_num_patch)
        ]
        self.recv_buffer.append(
            torch.zeros(self.recv_shape, dtype=self.dtype, device=distri_config.device)
        )

    def first_send_to_next(self, tensor: torch.Tensor):
        distri_config = self.distri_config
        if self.send_shape is None:
            shape = torch.tensor(
                tensor.shape, dtype=torch.int32, device=distri_config.device
            )
            dim = torch.tensor(
                len(shape), dtype=torch.int32, device=distri_config.device
            )
            dist.send(dim, dst=self.next_rank)
            dist.send(shape, dst=self.next_rank)
            self.send_shape = torch.Size(list(shape))
        else:
            logger.warning(f"Send buffer is already initialized, skip sending shape.")

    def first_recv_from_prev(self, dtype: Optional[torch.dtype] = None):
        distri_config = self.distri_config
        if self.recv_buffer is None:
            dim = torch.tensor(0, dtype=torch.int32, device=distri_config.device)
            dist.recv(dim, src=self.prev_rank)
            shape = torch.zeros(dim, dtype=torch.int32, device=distri_config.device)
            dist.recv(shape, src=self.prev_rank)
            self.recv_shape = torch.Size(list(shape))
            self.dtype = dtype
            self._creat_recv_buffer()
        else:
            logger.warning(f"Recv buffer is already initialized, skip receiving shape.")

    def isend_to_next(self, tensor: torch.Tensor):
        # logger.info(f"rank {self.distri_config.rank} is sending")
        if self.send_shape is None:
            self.first_send_to_next(tensor)
        group = (
            self.groups[(self.distri_config.rank + 1) % 2]
            if self.groups is not None
            else None
        )
        self.send_req = dist.isend(tensor, dst=self.next_rank, group=group)

    def _irecv_add_req(self):
        # if self.recv_req is not None and self.recv_req.is_completed():
        #     self.recv_req = None
        if self.recv_req is None and len(self.recv_queue) > 0:
            idx = self.recv_queue.pop(0)
            # logger.info(f"rank {self.distri_config.rank} is receiving {idx}")
            group = (
                self.groups[self.distri_config.rank % 2]
                if self.groups is not None
                else None
            )
            if idx is None:
                self.recv_req = dist.irecv(
                    self.recv_buffer[-1], src=self.prev_rank, group=group
                )
            else:
                self.recv_req = dist.irecv(
                    self.recv_buffer[idx], src=self.prev_rank, group=group
                )

    def irecv_from_prev(
        self, dtype: Optional[torch.dtype] = None, idx: Optional[int] = None
    ):
        if self.recv_buffer is None:
            self.first_recv_from_prev(dtype)
        self.recv_queue.append(idx)
        self._irecv_add_req()

    def get_data(self, idx: Optional[int] = None) -> torch.Tensor:
        if self.recv_req is not None:
            # logger.info(f"rank {self.distri_config.rank} : idx {idx} {self.recv_queue}")
            self.recv_req.wait()
            self.recv_req = None
            self._irecv_add_req()

        # logger.info(f"rank {self.distri_config.rank} is getting {idx}")
        if idx is None:
            return self.recv_buffer[-1]
        else:
            return self.recv_buffer[idx]


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
