from collections import namedtuple
from typing import Any, Dict, List, Optional, Tuple, Union
import pickle

import torch
from torch.distributed import Backend, ProcessGroup
import torch.distributed

from pipefuser.logger import init_logger

logger = init_logger(__name__)

TensorMetadata = namedtuple("TensorMetadata", ["device", "dtype", "size"])

def _split_tensor_dict(
        tensor_dict: Dict[str, Union[torch.Tensor, Any]],
        prefix: str = "") -> Tuple[List[Tuple[str, Any]], List[torch.Tensor]]:
    """Split the tensor dictionary into two parts:
    1. A list of (key, value) pairs. If the value is a tensor, it is replaced
         by its metadata.
    2. A list of tensors.

    If the Tensor is nested under `tensor_dict["key1"]["key2"]`, the key of its
    metadata will be "key1%key2".
    """
    metadata_list: List[Tuple[str, Any]] = []
    tensor_list = []
    for key, value in tensor_dict.items():
        assert "%" not in key, (
            "Avoid having '%' in key "
            "as it is used as a separator for nested entries.")
        if isinstance(value, torch.Tensor):
            # Note: we cannot use `value.device` here,
            # because it contains not only the device type but also the device
            # index (e.g. "cuda:0"). We only need the device type.
            # receiving side will set the device index.
            device = value.device.type
            metadata_list.append(
                (prefix + key, TensorMetadata(device, value.dtype,
                                              value.size())))
            tensor_list.append(value)
        elif isinstance(value, dict):
            if len(value) == 0:
                metadata_list.append((prefix + key, value))
            inner_metadata_list, inner_tensor_list = _split_tensor_dict(
                value, prefix + key + "%")
            metadata_list.extend(inner_metadata_list)
            tensor_list.extend(inner_tensor_list)
        else:
            metadata_list.append((prefix + key, value))
    return metadata_list, tensor_list


def _update_nested_dict(nested_dict, flattened_key, value):
    key_splits = flattened_key.split("%")
    cur_dict = nested_dict
    for k in key_splits[:-1]:
        if k not in cur_dict:
            cur_dict[k] = {}
        cur_dict = cur_dict[k]
    cur_dict[key_splits[-1]] = value


class GroupCoordinator:
    """
    PyTorch ProcessGroup wrapper for a group of processes.
    PyTorch ProcessGroup is bound to one specific communication backend,
        e.g. NCCL, Gloo, MPI, etc.
    GroupCoordinator takes charge of all the communication operations among
        the processes in the group. It can route the communication to
        a specific implementation (e.g. switch allreduce implementation
        based on the tensor size and cuda graph mode).
    """

    # available attributes:
    rank: int  # global rank
    ranks: List[int]  # global ranks in the group
    world_size: int  # size of the group
    # difference between `local_rank` and `rank_in_group`:
    # if we have a group of size 4 across two nodes:
    # Process | Node | Rank | Local Rank | Rank in Group
    #   0     |   0  |  0   |     0      |       0
    #   1     |   0  |  1   |     1      |       1
    #   2     |   1  |  2   |     0      |       2
    #   3     |   1  |  3   |     1      |       3
    local_rank: int  # local rank used to assign devices
    rank_in_group: int  # rank inside the group
    cpu_group: ProcessGroup  # group for CPU communication
    device_group: ProcessGroup  # group for device communication

    def __init__(
        self,
        group_ranks: List[List[int]],
        local_rank: int,
        torch_distributed_backend: Union[str, Backend],
    ):

        self.rank = torch.distributed.get_rank()
        self.local_rank = local_rank
        self.device_group = None
        self.cpu_group = None

        for ranks in group_ranks:
            device_group = torch.distributed.new_group(
                ranks, backend=torch_distributed_backend)
            # a group with `gloo` backend, to allow direct coordination between
            # processes through the CPU.
            cpu_group = torch.distributed.new_group(ranks, backend="gloo")
            if self.rank in ranks:
                self.ranks = ranks
                self.world_size = len(ranks)
                self.rank_in_group = ranks.index(self.rank)
                self.device_group = device_group
                self.cpu_group = cpu_group

        assert self.cpu_group is not None
        assert self.device_group is not None

        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:{local_rank}")
        else:
            self.device = torch.device("cpu")


    @property
    def first_rank(self):
        """Return the global rank of the first process in the group"""
        return self.ranks[0]

    @property
    def last_rank(self):
        """Return the global rank of the last process in the group"""
        return self.ranks[-1]

    @property
    def is_first_rank(self):
        """Return whether the caller is the first process in the group"""
        return self.rank == self.first_rank

    @property
    def is_last_rank(self):
        """Return whether the caller is the last process in the group"""
        return self.rank == self.last_rank

    @property
    def next_rank(self):
        """Return the global rank of the process that follows the caller"""
        rank_in_group = self.rank_in_group
        world_size = self.world_size
        return self.ranks[(rank_in_group + 1) % world_size]

    @property
    def prev_rank(self):
        """Return the global rank of the process that precedes the caller"""
        rank_in_group = self.rank_in_group
        world_size = self.world_size
        return self.ranks[(rank_in_group - 1) % world_size]

    @property
    def group_next_rank(self):
        """Return the group rank of the process that follows the caller"""
        rank_in_group = self.rank_in_group
        world_size = self.world_size
        return (rank_in_group + 1) % world_size

    @property
    def group_prev_rank(self):
        """Return the group rank of the process that precedes the caller"""
        rank_in_group = self.rank_in_group
        world_size = self.world_size
        return (rank_in_group - 1) % world_size

    def all_reduce(self, input_: torch.Tensor) -> torch.Tensor:
        """
        NOTE: This operation will be applied in-place or out-of-place. 
        Always assume this function modifies its input, but use the return
        value as the output.
        """
        # Bypass the function if we are using only 1 GPU.
        if self.world_size == 1:
            return input_
        else:
            torch.distributed.all_reduce(input_, group=self.device_group)
        return input_

    def all_gather(
        self, 
        input_: torch.Tensor, 
        dim: int = -1, 
        separate_tensors: bool = False
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        world_size = self.world_size
        # Bypass the function if we are using only 1 GPU.
        if world_size == 1:
            return input_
        assert -input_.dim() <= dim < input_.dim(), (
            f"Invalid dim ({dim}) for input tensor with shape {input_.size()}")
        input_size = input_.size()
        if not separate_tensors:
            if dim < 0:
                # Convert negative dim to positive.
                dim += input_.dim()
            # Allocate output tensor.
            output_tensor = torch.empty((world_size, ) + input_size,
                                        dtype=input_.dtype,
                                        device=input_.device)
            # All-gather.
            torch.distributed.all_gather_into_tensor(output_tensor,
                                                     input_,
                                                     group=self.device_group)
            # Reshape
            output_tensor = output_tensor.movedim(0, dim)
            output_tensor = output_tensor.reshape(input_size[:dim] +
                                                  (world_size *
                                                   input_size[dim], ) +
                                                  input_size[dim + 1:])
            return output_tensor
        else:
            # Allocate output tensors.
            output_tensors = [torch.empty_like(input_) 
                              for _ in range(world_size)]
            # All-gather.
            torch.distributed.all_gather(output_tensors,
                                        input_,
                                        group=self.device_group)
            return output_tensors

    def gather(self,
               input_: torch.Tensor,
               dst: int = 0,
               dim: int = -1) -> torch.Tensor:
        """
        NOTE: We assume that the input tensor is on the same device across
        all the ranks.
        NOTE: `dst` is the local rank of the destination rank.
        """
        world_size = self.world_size
        # Bypass the function if we are using only 1 GPU.
        if world_size == 1:
            return input_
        assert -input_.dim() <= dim < input_.dim(), (
            f"Invalid dim ({dim}) for input tensor with shape {input_.size()}")
        if dim < 0:
            # Convert negative dim to positive.
            dim += input_.dim()
        # Allocate output tensor.
        if self.rank_in_group == dst:
            gather_list = [torch.empty_like(input_) for _ in range(world_size)]
        else:
            gather_list = None
        # Gather.
        torch.distributed.gather(input_,
                                 gather_list,
                                 dst=self.ranks[dst],
                                 group=self.device_group)
        if self.rank_in_group == dst:
            output_tensor = torch.cat(gather_list, dim=dim)
        else:
            output_tensor = None
        return output_tensor

    def broadcast(self, input_: torch.Tensor, src: int = 0):
        """Broadcast the input tensor.
        NOTE: `src` is the local rank of the source rank.
        """
        assert src < self.world_size, f"Invalid src rank ({src})"

        # Bypass the function if we are using only 1 GPU.
        if self.world_size == 1:
            return input_
        # Broadcast.
        torch.distributed.broadcast(input_,
                                    src=self.ranks[src],
                                    group=self.device_group)
        return input_

    def broadcast_object(self, obj: Optional[Any] = None, src: int = 0):
        """Broadcast the input object.
        NOTE: `src` is the local rank of the source rank.
        """
        assert src < self.world_size, f"Invalid src rank ({src})"

        # Bypass the function if we are using only 1 GPU.
        if self.world_size == 1:
            return obj
        if self.shm_broadcaster is not None:
            assert src == 0, "Shared memory broadcaster only supports src=0"
            return self.shm_broadcaster.broadcast_object(obj)
        if self.rank_in_group == src:
            torch.distributed.broadcast_object_list([obj],
                                                    src=self.ranks[src],
                                                    group=self.cpu_group)
            return obj
        else:
            recv = [None]
            torch.distributed.broadcast_object_list(recv,
                                                    src=self.ranks[src],
                                                    group=self.cpu_group)
            return recv[0]

    def broadcast_object_list(self,
                              obj_list: List[Any],
                              src: int = 0,
                              group: Optional[ProcessGroup] = None):
        """Broadcast the input object list.
        NOTE: `src` is the local rank of the source rank.
        """
        assert src < self.world_size, f"Invalid src rank ({src})"

        # Bypass the function if we are using only 1 GPU.
        if self.world_size == 1:
            return obj_list
        # Broadcast.
        torch.distributed.broadcast_object_list(obj_list,
                                                src=self.ranks[src],
                                                group=self.device_group)
        return obj_list

    def send_object(self, obj: Any, dst: int) -> None:
        """Send the input object list to the destination rank."""
        """NOTE: `dst` is the local rank of the destination rank."""

        assert dst < self.world_size, f"Invalid dst rank ({dst})"

        assert dst != self.rank, (
            "Invalid destination rank. Destination rank is the same "
            "as the current rank.")

        # Serialize object to tensor and get the size as well
        object_tensor = torch.frombuffer(pickle.dumps(obj), dtype=torch.uint8)

        size_tensor = torch.tensor([object_tensor.numel()],
                                   dtype=torch.long,
                                   device="cpu")

        # Send object size

        torch.distributed.send(size_tensor,
                               dst=self.ranks[dst],
                               group=self.cpu_group)

        # Send object
        torch.distributed.send(object_tensor,
                               dst=self.ranks[dst],
                               group=self.cpu_group)

        return None

    def recv_object(self, src: int) -> Any:
        """Receive the input object list from the source rank."""
        """NOTE: `src` is the local rank of the source rank."""

        assert src < self.world_size, f"Invalid src rank ({src})"

        assert src != self.rank, (
            "Invalid source rank. Source rank is the same as the current rank."
        )

        size_tensor = torch.empty(1, dtype=torch.long, device="cpu")

        # Receive object size
        rank_size = torch.distributed.recv(size_tensor,
                                           src=self.ranks[src],
                                           group=self.cpu_group)

        # Tensor to receive serialized objects into.
        object_tensor = torch.empty(  # type: ignore[call-overload]
            size_tensor.item(),  # type: ignore[arg-type]
            dtype=torch.uint8,
            device="cpu")

        rank_object = torch.distributed.recv(object_tensor,
                                             src=self.ranks[src],
                                             group=self.cpu_group)

        assert rank_object == rank_size, (
            "Received object sender rank does not match the size sender rank.")

        obj = pickle.loads(object_tensor.numpy().tobytes())

        return obj

    def broadcast_tensor_dict(
        self,
        tensor_dict: Optional[Dict[str, Union[torch.Tensor, Any]]] = None,
        src: int = 0,
        group: Optional[ProcessGroup] = None,
        metadata_group: Optional[ProcessGroup] = None
    ) -> Optional[Dict[str, Union[torch.Tensor, Any]]]:
        """Broadcast the input tensor dictionary.
        NOTE: `src` is the local rank of the source rank.
        """
        # Bypass the function if we are using only 1 GPU.
        if (not torch.distributed.is_initialized() or self.world_size == 1):
            return tensor_dict

        group = self.device_group
        metadata_group = self.cpu_group
        assert src < self.world_size, f"Invalid src rank ({src})"
        src = self.ranks[src]

        rank = self.rank
        if rank == src:
            metadata_list: List[Tuple[Any, Any]] = []
            assert isinstance(
                tensor_dict,
                dict), (f"Expecting a dictionary, got {type(tensor_dict)}")
            metadata_list, tensor_list = _split_tensor_dict(tensor_dict)
            # `metadata_list` lives in CPU memory.
            # `broadcast_object_list` has serialization & deserialization,
            # all happening on CPU. Therefore, we can use the CPU group.
            self.broadcast_object(metadata_list, src=src)
            async_handles = []
            for tensor in tensor_list:
                if tensor.numel() == 0:
                    # Skip broadcasting empty tensors.
                    continue
                if tensor.is_cpu:
                    # use metadata_group for CPU tensors
                    handle = torch.distributed.broadcast(tensor,
                                                         src=src,
                                                         group=metadata_group,
                                                         async_op=True)
                else:
                    # use group for GPU tensors
                    handle = torch.distributed.broadcast(tensor,
                                                         src=src,
                                                         group=group,
                                                         async_op=True)
                async_handles.append(handle)
            for async_handle in async_handles:
                async_handle.wait()

        else:
            metadata_list = self.broadcast_object(None, src=src)
            tensor_dict = {}
            async_handles = []
            for key, value in metadata_list:
                if isinstance(value, TensorMetadata):
                    tensor = torch.empty(value.size,
                                         dtype=value.dtype,
                                         device=value.device)
                    if tensor.numel() == 0:
                        # Skip broadcasting empty tensors.
                        _update_nested_dict(tensor_dict, key, tensor)
                        continue
                    if tensor.is_cpu:
                        # use metadata_group for CPU tensors
                        handle = torch.distributed.broadcast(
                            tensor,
                            src=src,
                            group=metadata_group,
                            async_op=True)
                    else:
                        # use group for GPU tensors
                        handle = torch.distributed.broadcast(tensor,
                                                             src=src,
                                                             group=group,
                                                             async_op=True)
                    async_handles.append(handle)
                    _update_nested_dict(tensor_dict, key, tensor)
                else:
                    _update_nested_dict(tensor_dict, key, value)
            for async_handle in async_handles:
                async_handle.wait()
        return tensor_dict

    def send_tensor_dict(
        self,
        tensor_dict: Dict[str, Union[torch.Tensor, Any]],
        dst: Optional[int] = None
    ) -> Optional[Dict[str, Union[torch.Tensor, Any]]]:
        """Send the input tensor dictionary.
        NOTE: `dst` is the local rank of the source rank.
        """
        # Bypass the function if we are using only 1 GPU.
        if not torch.distributed.is_initialized() or self.world_size == 1:
            return tensor_dict

        group = self.device_group
        metadata_group = self.cpu_group

        if dst is None:
            dst = self.group_next_rank
        assert dst < self.world_size, f"Invalid dst rank ({dst})"

        metadata_list: List[Tuple[Any, Any]] = []
        assert isinstance(
            tensor_dict,
            dict), f"Expecting a dictionary, got {type(tensor_dict)}"
        metadata_list, tensor_list = _split_tensor_dict(tensor_dict)
        # `metadata_list` lives in CPU memory.
        # `send_object_list` has serialization & deserialization,
        # all happening on CPU. Therefore, we can use the CPU group.
        self.send_object(metadata_list, dst=dst)
        for tensor in tensor_list:
            if tensor.numel() == 0:
                # Skip sending empty tensors.
                continue
            if tensor.is_cpu:
                # use metadata_group for CPU tensors
                torch.distributed.send(tensor, 
                                       dst=self.ranks[dst], 
                                       group=metadata_group)
            else:
                # use group for GPU tensors
                torch.distributed.send(tensor, dst=self.ranks[dst], group=group)
        return None

    def recv_tensor_dict(
        self,
        src: Optional[int] = None
    ) -> Optional[Dict[str, Union[torch.Tensor, Any]]]:
        """Recv the input tensor dictionary.
        NOTE: `src` is the local rank of the source rank.
        """
        # Bypass the function if we are using only 1 GPU.
        if not torch.distributed.is_initialized() or self.world_size == 1:
            return None

        group = self.device_group
        metadata_group = self.cpu_group

        if src is None:
            src = self.group_prev_rank
        assert src < self.world_size, f"Invalid src rank ({src})"

        recv_metadata_list = self.recv_object(src=src)
        tensor_dict: Dict[str, Any] = {}
        for key, value in recv_metadata_list:
            if isinstance(value, TensorMetadata):
                tensor = torch.empty(value.size,
                                     dtype=value.dtype,
                                     device=value.device)
                if tensor.numel() == 0:
                    # Skip broadcasting empty tensors.
                    _update_nested_dict(tensor_dict, key, tensor)
                    continue
                if tensor.is_cpu:
                    # use metadata_group for CPU tensors
                    torch.distributed.recv(tensor,
                                           src=self.ranks[src],
                                           group=metadata_group)
                else:
                    # use group for GPU tensors
                    torch.distributed.recv(tensor, 
                                           src=self.ranks[src], 
                                           group=group)
                _update_nested_dict(tensor_dict, key, tensor)
            else:
                _update_nested_dict(tensor_dict, key, value)
        return tensor_dict

    def barrier(self):
        """Barrier synchronization among the group.
        NOTE: don't use `device_group` here! `barrier` in NCCL is
        terrible because it is internally a broadcast operation with
        secretly created GPU tensors. It is easy to mess up the current
        device. Use the CPU group instead.
        """
        torch.distributed.barrier(group=self.cpu_group)

    def send(self, tensor: torch.Tensor, dst: Optional[int] = None) -> None:
        """Sends a tensor to the destination rank in a non-blocking way"""
        """NOTE: `dst` is the rank_in_group of the destination rank."""
        if dst is None:
            dst = self.group_next_rank

        torch.distributed.send(
            tensor, 
            self.ranks[dst], 
            group=self.device_groups[self.rank_in_group % 2]
            if self.world_size == 2 else self.device_group
        )

    def recv(self,
             size: torch.Size,
             dtype: torch.dtype,
             src: Optional[int] = None) -> torch.Tensor:
        """Receives a tensor from the src rank."""
        """NOTE: `src` is the rank_in_group of the source rank."""
        if src is None:
            src = self.group_prev_rank

        tensor = torch.empty(size, dtype=dtype, device=self.device)
        torch.distributed.recv(
            tensor, 
            self.ranks[src], 
            self.device_groups[(self.rank_in_group + 1) % 2] 
            if self.world_size == 2 else self.device_group
        )
        return tensor

    def destroy(self):
        if self.device_group is not None:
            torch.distributed.destroy_process_group(self.device_group)
            self.device_group = None
        if self.cpu_group is not None:
            torch.distributed.destroy_process_group(self.cpu_group)
            self.cpu_group = None


class PipelineGroupCoordinator(GroupCoordinator):
    """
    available attributes:
    rank: int  # global rank
    ranks: List[int]  # global ranks in the group
    world_size: int  # size of the group
    difference between `local_rank` and `rank_in_group`:
    if we have a group of size 4 across two nodes:
    Process | Node | Rank | Local Rank | Rank in Group
      0     |   0  |  0   |     0      |       0
      1     |   0  |  1   |     1      |       1
      2     |   1  |  2   |     0      |       2
      3     |   1  |  3   |     1      |       3
    local_rank: int  # local rank used to assign devices
    rank_in_group: int  # rank inside the group
    cpu_group: ProcessGroup  # group for CPU communication
    device_group: ProcessGroup  # group for device communication
    """
    
    def __init__(
        self,
        group_ranks: List[List[int]],
        local_rank: int,
        torch_distributed_backend: Union[str, Backend],
    ):
        self.rank = torch.distributed.get_rank()
        self.local_rank = local_rank
        self.device_group = None
        self.cpu_group = None
        self.cpu_groups = []
        self.device_groups = []
        if len(group_ranks[0]) > 2 or len(group_ranks[0]) == 1:
            for ranks in group_ranks:
                device_group = torch.distributed.new_group(
                    ranks, backend=torch_distributed_backend)
                # a group with `gloo` backend, to allow direct coordination between
                # processes through the CPU.
                cpu_group = torch.distributed.new_group(ranks, backend="gloo")
                if self.rank in ranks:
                    self.ranks = ranks
                    self.world_size = len(ranks)
                    self.rank_in_group = ranks.index(self.rank)
                    self.device_group = device_group
                    self.cpu_group = cpu_group
        # when pipeline parallelism is 2, we need to create two groups to avoid
        #   communication stall.
        # *_group_0_1 represents the group for communication from device 0 to 
        #   device 1.
        # *_group_1_0 represents the group for communication from device 1 to
        #   device 0.
        elif len(group_ranks[0]) == 2:
            for ranks in group_ranks:
                device_group_0_1 = torch.distributed.new_group(
                    ranks, backend=torch_distributed_backend)
                device_group_1_0 = torch.distributed.new_group(
                    ranks, backend=torch_distributed_backend)
                # a group with `gloo` backend, to allow direct coordination between
                # processes through the CPU.
                cpu_group_0_1 = torch.distributed.new_group(ranks, backend="gloo")
                cpu_group_1_0 = torch.distributed.new_group(ranks, backend="gloo")
                if self.rank in ranks:
                    self.ranks = ranks
                    self.world_size = len(ranks)
                    self.rank_in_group = ranks.index(self.rank)
                    self.device_groups = [device_group_0_1, device_group_1_0]
                    self.cpu_groups = [cpu_group_0_1, cpu_group_1_0]
                    self.device_group = device_group_0_1
                    self.cpu_group = cpu_group_0_1

        assert self.cpu_group is not None
        assert self.device_group is not None

        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:{local_rank}")
        else:
            self.device = torch.device("cpu")

        self.hyper_parameters_set: bool = False
        self.recv_shape: Optional[torch.Size] = None
        self.send_shape: Optional[torch.Size] = None
        self.recv_tasks_queue: List[int] = []
        self.receiving_task: Optional[torch.distributed.Work] = None
        self.recv_buffer: Optional[Union[List[torch.Tensor], torch.Tensor]] \
            = None
        self.dtype: Optional[torch.dtype] = None
        self.num_pipefusion_patches: Optional[int] = None

    def reset_buffer(self):
        self.recv_shape = None
        self.send_shape = None
        self.recv_tasks_queue = []
        self.receiving_task = None
        self.recv_buffer = None
        
    def set_hyper_parameters(
        self, 
        dtype: Optional[torch.dtype] = None, 
        num_pipefusion_patches: Optional[int] = None,
    ):
        assert isinstance(dtype, torch.dtype), (
            "dtype must be a torch.dtype object")
        assert (isinstance(num_pipefusion_patches, int) and 
                num_pipefusion_patches >= 1), (
                    "num_pipefusion_patches must be greater than or equal to 1")
        self.dtype = dtype or self.dtype
        self.num_pipefusion_patches = num_pipefusion_patches or \
            self.num_pipefusion_patches
        self.hyper_parameters_set = True
    
    def pipeline_isend(self, tensor: torch.Tensor) -> None:
        tensor = tensor.contiguous()
        assert self.hyper_parameters_set, (
            "hyper parameters must be set before sending tensors")
        if self.send_shape is None:
            self._init_shape_and_send_metadata(tensor)
        self._pipeline_isend(tensor)

    def pipeline_recv(self, idx: Optional[int] = None) -> torch.Tensor:
        assert self.hyper_parameters_set, (
            "hyper parameters must be set before receiving tensors")
        if self.recv_buffer is None:
            self._recv_metadata_and_init_buffer() 
        idx = idx or -1
        self._pipeline_irecv(self.recv_buffer[idx]).wait()
        return self.recv_buffer[idx]

    def add_pipeline_recv_task(self, idx: Optional[int] = None):
        assert self.hyper_parameters_set, (
            "hyper parameters must be set before receiving tensors")
        if self.recv_buffer is None:
            self._recv_metadata_and_init_buffer()
        self.recv_tasks_queue.append(idx if idx is not None else -1)

    def get_pipeline_recv_data(self, idx: Optional[int] = None) -> torch.Tensor:
        if self.receiving_task is not None:
            self.receiving_task.wait()
            self.receiving_task = None
            self.recv_next()
        if idx is None:
            return self.recv_buffer[-1]
        else:
            return self.recv_buffer[idx]

    def recv_next(self):
        if len(self.recv_tasks_queue) == 0:
            raise ValueError("No more tasks to receive")
        elif len(self.recv_tasks_queue) > 0:
            if self.receiving_task is not None:
                logger.warning(f"Receiving task is not finished, waiting...")
                self.receiving_task.wait()
                self.receiving_task = None
            idx = self.recv_tasks_queue.pop(0)
            self.receiving_task = self._pipeline_irecv(self.recv_buffer[idx])
        
        
    # def _set_receiving_task(self):
    #     if len(self.recv_tasks_queue) > 0 and self.receiving_task is None:
    #         idx = self.recv_tasks_queue.pop(0)
    #         if torch.distributed.get_rank() == 0:
    #             print("size:", len(self.recv_tasks_queue))
    #         self.receiving_task = self._pipeline_irecv(
    #             self.recv_buffer[idx] if idx is not None 
    #             else self.recv_buffer[-1]
    #         )

    # def add_pipeline_recv_task_for_patch(self, idx: Optional[int] = None):
    #     assert self.hyper_parameters_set, (
    #         "hyper parameters must be set before receiving tensors")
    #     if self.recv_buffer is None:
    #         self._recv_metadata_and_init_buffer()
    #     self.recv_tasks_queue.append(idx)
    #     self._set_receiving_task()

    # def get_receiving_task_data(
    #     self, 
    #     idx: Optional[int] = None
    # ) -> torch.Tensor:
    #     if self.receiving_task is not None:
    #         self.receiving_task.wait()
    #         self.receiving_task = None
    #         self._set_receiving_task()
        
    #     if idx is None:
    #         return self.recv_buffer[-1]
    #     else:
    #         return self.recv_buffer[idx]

    def _init_shape_and_send_metadata(self, tensor: torch.Tensor) -> None:
        if self.send_shape is None:
            shape = torch.tensor(tensor.shape, dtype=torch.int32, 
                                 device=self.device)
            dim = torch.tensor(len(tensor.shape), dtype=torch.int32, 
                               device=self.device)
            self._pipeline_isend(dim).wait()
            self._pipeline_isend(shape).wait()
            self.send_shape = torch.Size(shape)
        else:
            logger.warning("Send shape is already initialized, "
                           "skip sending shape.")

    def _recv_metadata_and_init_buffer(self) -> Optional[torch.Tensor]:
        if self.recv_buffer is None:
            dim = self.recv(size=[1], dtype=torch.int32)
            shape = self.recv(size=dim, dtype=torch.int32)
            patch_shape = shape.clone()
            patch_shape[-2] = patch_shape[-2] // self.num_pipefusion_patches
            self.recv_buffer = [
                torch.zeros(*patch_shape, dtype=self.dtype, device=self.device)
                for _ in range(self.num_pipefusion_patches)
            ]
            self.recv_buffer.append(
                torch.zeros(*shape, dtype=self.dtype, device=self.device)
            )
            self.recv_shape = torch.Size(shape)
            self.recv_tasks_queue = []
        else:
            logger.warning("Recv buffer is already initialized, "
                           "skip receiving shape.")


                
    def _pipeline_irecv(self, tensor: torch.tensor):
        return torch.distributed.irecv(
            tensor,
            src=self.prev_rank,
            group=self.device_groups[(self.rank_in_group + 1) % 2] 
            if self.world_size == 2 else self.device_group
        )
    
    def _pipeline_isend(self, tensor: torch.tensor):
        return torch.distributed.isend(
            tensor,
            dst=self.next_rank,
            group=self.device_groups[self.rank_in_group % 2]
            if self.world_size == 2 else self.device_group
        )