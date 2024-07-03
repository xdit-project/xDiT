from contextlib import contextmanager
from typing import List, Optional

import torch
import torch.distributed
import pipefuser.envs as envs

from pipefuser.logger import init_logger
from pipefuser.distributed.group_coordinator import (
    GroupCoordinator,
    PipelineGroupCoordinator,
)


logger = init_logger(__name__)

_WORLD: Optional[GroupCoordinator] = None


def get_world_group() -> GroupCoordinator:
    assert _WORLD is not None, ("world group is not initialized")
    return _WORLD


def init_world_group(ranks: List[int], local_rank: int,
                     backend: str) -> GroupCoordinator:
    return GroupCoordinator(
        group_ranks=[ranks],
        local_rank=local_rank,
        torch_distributed_backend=backend,
    )


def init_model_parallel_group(group_ranks: List[List[int]], local_rank: int,
                              backend: str, parallel_mode: str) -> GroupCoordinator:
    assert parallel_mode in [
        "data", 
        "pipeline", 
        "tensor", 
        "sequence", 
        "classifier_free_guidance"
    ], f"parallel_mode {parallel_mode} is not supported"
    if parallel_mode == "pipeline":
        return PipelineGroupCoordinator(
            group_ranks=group_ranks,
            local_rank=local_rank,
            torch_distributed_backend=backend
        )
    else:
        return GroupCoordinator(
            group_ranks=group_ranks,
            local_rank=local_rank,
            torch_distributed_backend=backend,
        )


_TP: Optional[GroupCoordinator] = None

def get_tp_group() -> GroupCoordinator:
    assert _TP is not None, ("tensor model parallel group is not initialized")
    return _TP


_PP: Optional[GroupCoordinator] = None

def get_pp_group() -> GroupCoordinator:
    assert _PP is not None, (
        "pipeline model parallel group is not initialized")
    return _PP


_SP: Optional[GroupCoordinator] = None

def get_sp_group() -> GroupCoordinator:
    assert _SP is not None, (
        "pipeline model parallel group is not initialized")
    return _SP


_DP: Optional[GroupCoordinator] = None

def get_dp_group() -> GroupCoordinator:
    assert _DP is not None, (
        "pipeline model parallel group is not initialized")
    return _DP


_CFG: Optional[GroupCoordinator] = None
def get_cfg_group() -> GroupCoordinator:
    assert _CFG is not None, (
        "classifier_free_guidance parallel group is not initialized")
    return _CFG


@contextmanager
def graph_capture():
    """
    `graph_capture` is a context manager which should surround the code that
    is capturing the CUDA graph. Its main purpose is to ensure that the
    some operations will be run after the graph is captured, before the graph
    is replayed. It returns a `GraphCaptureContext` object which contains the
    necessary data for the graph capture. Currently, it only contains the
    stream that the graph capture is running on. This stream is set to the
    current CUDA stream when the context manager is entered and reset to the
    default stream when the context manager is exited. This is to ensure that
    the graph capture is running on a separate stream from the default stream,
    in order to explicitly distinguish the kernels to capture
    from other kernels possibly launched on background in the default stream.
    """
    with get_tp_group().graph_capture() as context, get_pp_group(
    ).graph_capture(context):
        yield context




def init_distributed_environment(
    world_size: int = -1,
    rank: int = -1,
    distributed_init_method: str = "env://",
    local_rank: int = -1,
    backend: str = "nccl",
):
    logger.debug(
        "world_size=%d rank=%d local_rank=%d "
        "distributed_init_method=%s backend=%s", world_size, rank, local_rank,
        distributed_init_method, backend)
    if not torch.distributed.is_initialized():
        assert distributed_init_method is not None, (
            "distributed_init_method must be provided when initializing "
            "distributed environment")
        # this backend is used for WORLD
        torch.distributed.init_process_group(
            backend=backend,
            init_method=distributed_init_method,
            world_size=world_size,
            rank=rank)
    # set the local rank
    # local_rank is not available in torch ProcessGroup,
    # see https://github.com/pytorch/pytorch/issues/122816
    if local_rank == -1:
        # local rank not set, this usually happens in single-node
        # setting, where we can use rank as local rank
        if distributed_init_method == "env://":
            local_rank = envs.LOCAL_RANK
        else:
            local_rank = rank
    global _WORLD
    if _WORLD is None:
        ranks = list(range(torch.distributed.get_world_size()))
        _WORLD = init_world_group(ranks, local_rank, backend)
    else:
        assert _WORLD.world_size == torch.distributed.get_world_size(), (
            "world group already initialized with a different world size")


def initialize_model_parallel(
    tensor_parallel_degree: int = 1,
    sequence_parallel_degree: int = 1,
    pipeline_parallel_degree: int = 1,
    classifier_free_guidance_degree: int = 1,
    data_parallel_degree: int = 1,
    backend: Optional[str] = None,
) -> None:
    """
    Initialize model parallel groups.

    Arguments:
        tensor_parallel_degree: number of GPUs used for tensor parallelism.
        sequence_parallel_degree: number of GPUs used for sequence parallelism.
        pipeline_parallel_degree: number of GPUs used for pipeline parallelism.
        data_parallel_degree: number of data parallelism groups.
        split_batch: whether to split the batch dimension to accelerate CFG.

    Let's say we have a total of 16 GPUs denoted by g0 ... g15 and we
    use 2 groups to parallelize the batch dim(dp), 2 groups to parallelize
    splited batch caused by CFG, and 2 GPUs to parallelize sequence. 
    The present function will create 2 data parallel-groups, 
    8 CFG group, 8 pipeline-parallel group, and 
    8 sequence-parallel groups:
        2 data-parallel groups:
            [g0, g1, g2, g3, g4, g5, g6, g7], 
            [g8, g9, g10, g11, g12, g13, g14, g15]
        8 CFG-parallel groups:
            [g0, g4], [g1, g5], [g2, g6], [g3, g7],
            [g8, g12], [g9, g13], [g10, g14], [g11, g15]
        8 sequence-parallel groups:
            [g0, g1], [g2, g3], [g4, g5], [g6, g7], 
            [g8, g9], [g10, g11], [g12, g13], [g14, g15]
        8 pipeline-parallel groups:
            [g0, g2], [g4, g6], [g8, g10], [g12, g14],
            [g1, g3], [g5, g7], [g9, g11], [g13, g15]
    Note that for efficiency, the caller should make sure adjacent ranks
    are on the same DGX box. For example if we are using 2 DGX-1 boxes
    with a total of 16 GPUs, rank 0 to 7 belong to the first box and
    ranks 8 to 15 belong to the second box.
    """
    # Get world size and rank. Ensure some consistencies.
    assert torch.distributed.is_initialized()
    world_size: int = torch.distributed.get_world_size()
    backend = backend or torch.distributed.get_backend(
        get_world_group().device_group)

    if (world_size !=
            data_parallel_degree * classifier_free_guidance_degree * 
            sequence_parallel_degree * tensor_parallel_degree * 
            pipeline_parallel_degree):
        raise RuntimeError(
            f"world_size ({world_size}) is not equal to "
            f"tensor_parallel_degree ({tensor_parallel_degree}) x "
            f"pipeline_parallel_degree ({pipeline_parallel_degree}) x"
            f"sequence_parallel_degree ({sequence_parallel_degree}) x"
            f"classifier_free_guidance_degree " 
            f"({classifier_free_guidance_degree}) x"
            f"data_parallel_degree ({data_parallel_degree})")

    # Build the data-parallel groups.
    num_data_parallel_devices: int = world_size // data_parallel_degree
    global _DP
    assert _DP is None, ("data parallel group is already initialized")
    group_ranks = []
    for i in range(data_parallel_degree):
        ranks = list(range(i * num_data_parallel_devices,
                          (i + 1) * num_data_parallel_devices))
        group_ranks.append(ranks)
    _DP = init_model_parallel_group(group_ranks,
                                    get_world_group().local_rank, backend)

    # Build the classifier_free_guidance parallel groups. (split batch)
    num_cfg_parallel_groups: int = (world_size //
                                    classifier_free_guidance_degree)
    num_splited_batch_devices: int = (num_data_parallel_devices // 
                                      classifier_free_guidance_degree)
    global _CFG
    assert _CFG is None, (
        "classifier_free_guidance group is already initialized")
    group_ranks = []
    for i in range(num_cfg_parallel_groups):
        start_rank = ((i // num_splited_batch_devices) * 
                      num_data_parallel_devices + 
                      i % num_splited_batch_devices)
        ranks = [
            start_rank + j * num_splited_batch_devices
            for j in range(classifier_free_guidance_degree)
        ]
        group_ranks.append(ranks)
    _CFG = init_model_parallel_group(group_ranks,
                                    get_world_group().local_rank, backend)
    
    # Build the sequence-parallel groups.
    num_sequence_parallel_devices: int = sequence_parallel_degree
    num_sequence_parallel_groups: int = (world_size // 
                                         num_sequence_parallel_devices)
    global _SP
    assert _SP is None, (
        "sequence parallel group is already initialized")
    group_ranks = []
    for i in range(num_sequence_parallel_groups):
        ranks = list(range(i * sequence_parallel_degree,
                          (i + 1) * sequence_parallel_degree))
        group_ranks.append(ranks)
    _SP = init_model_parallel_group(group_ranks,
                                    get_world_group().local_rank, backend)

    #TODO: implement tensor parallel groups
    assert tensor_parallel_degree == 1, "Tensor parallelism is not implemented"
    # # Build the tensor model-parallel groups.
    # num_tensor_model_parallel_groups: int = (world_size //
    #                                          tensor_parallel_degree)
    # global _TP
    # assert _TP is None, ("tensor model parallel group is already initialized")
    # group_ranks = []
    # for i in range(num_tensor_model_parallel_groups):
    #     ranks = list(
    #         range(i * tensor_parallel_degree,
    #               (i + 1) * tensor_parallel_degree))
    #     group_ranks.append(ranks)
    # _TP = init_model_parallel_group(group_ranks,
    #                                 get_world_group().local_rank, backend)

    # Build the pipeline model-parallel groups.
    num_pipeline_per_stage_devices: int = (sequence_parallel_degree * 
                                           tensor_parallel_degree)
    num_pipeline_parallel_devices: int = pipeline_parallel_degree
    num_pipeline_parallel_groups: int = (data_parallel_degree * 
                                         classifier_free_guidance_degree * 
                                         num_pipeline_per_stage_devices)
    global _PP
    assert _PP is None, (
        "pipeline model parallel group is already initialized")
    if num_pipeline_parallel_devices > 1:
        group_ranks = []
        for i in range(num_pipeline_parallel_groups):
            start_rank = (
                i // num_pipeline_per_stage_devices * num_splited_batch_devices 
                + i % num_pipeline_parallel_devices
            )
            ranks = [
                start_rank + j * num_pipeline_per_stage_devices
                for j in range(num_pipeline_parallel_devices)
            ]
            group_ranks.append(ranks)
        _PP = init_model_parallel_group(group_ranks,
                                        get_world_group().local_rank, backend)
    else:
        _PP = None


def ensure_model_parallel_initialized(
    tensor_parallel_degree: int,
    pipeline_parallel_degree: int,
    backend: Optional[str] = None,
) -> None:
    """Helper to initialize model parallel groups if they are not initialized,
    or ensure tensor-parallel and pipeline-parallel sizes are equal to expected
    values if the model parallel groups are initialized.
    """
    backend = backend or torch.distributed.get_backend(
        get_world_group().device_group)
    if not model_parallel_is_initialized():
        initialize_model_parallel(tensor_parallel_degree,
                                  pipeline_parallel_degree, backend)
        return

    assert (
        get_tensor_model_parallel_world_size() == tensor_parallel_degree
    ), ("tensor parallel group already initialized, but of unexpected size: "
        f"{get_tensor_model_parallel_world_size()=} vs. "
        f"{tensor_parallel_degree=}")
    pp_world_size = get_pp_group().world_size
    assert (pp_world_size == pipeline_parallel_degree), (
        "pipeline parallel group already initialized, but of unexpected size: "
        f"{pp_world_size=} vs. "
        f"{pipeline_parallel_degree=}")


def model_parallel_is_initialized():
    """Check if tensor and pipeline parallel groups are initialized."""
    return (_DP is not None and _CFG is not None)


_TP_STATE_PATCHED = False


@contextmanager
def patch_tensor_parallel_group(tp_group: GroupCoordinator):
    """Patch the tp group temporarily until this function ends.

    This method is for draft workers of speculative decoding to run draft model
    with different tp degree from that of target model workers.

    Args:
        tp_group (GroupCoordinator): the tp group coordinator
    """
    global _TP_STATE_PATCHED
    assert not _TP_STATE_PATCHED, "Should not call when it's already patched"

    _TP_STATE_PATCHED = True
    old_tp_group = get_tp_group()
    global _TP
    _TP = tp_group
    try:
        yield
    finally:
        # restore the original state
        _TP_STATE_PATCHED = False
        _TP = old_tp_group


def get_tensor_model_parallel_world_size():
    """Return world size for the tensor model parallel group."""
    return get_tp_group().world_size

def get_tensor_model_parallel_rank():
    """Return my rank for the tensor model parallel group."""
    return get_tp_group().rank_in_group


def get_pipeline_parallel_world_size():
    """Return world size for the pipeline model parallel group."""
    if get_pp_group() is None:
        return 1
    else:
        return get_pp_group().world_size

def get_pipeline_parallel_rank():
    """Return my rank for the pipeline model parallel group."""
    if get_pp_group() is None:
        return 0
    else:
        return get_pp_group().rank_in_group


def get_classifier_free_guidance_world_size():
    """Return world size for the classifier_free_guidance parallel group."""
    return get_cfg_group().world_size

def get_classifier_free_guidance_rank():
    """Return my rank for the classifier_free_guidance parallel group."""
    return get_cfg_group().rank_in_group


def get_sequence_parallel_world_size():
    """Return world size for the sequence parallel group."""
    return get_sp_group().world_size

def get_sequence_parallel_rank():
    """Return my rank for the sequence parallel group."""
    return get_sp_group().rank_in_group


def get_data_parallel_world_size():
    """Return world size for the data parallel group."""
    return get_dp_group().world_size

def get_data_parallel_rank():
    """Return my rank for the data parallel group."""
    return get_dp_group().rank_in_group


def destroy_model_parallel():
    """Set the groups to none and destroy them."""
    global _DP
    if _DP:
        _DP.destroy()
    _DP = None
    
    global _CFG
    if _CFG:
        _CFG.destroy()
    _CFG = None
    
    global _SP
    if _SP:
        _SP.destroy()
    _SP = None

    global _TP
    if _TP:
        _TP.destroy()
    _TP = None

    global _PP
    if _PP:
        _PP.destroy()
    _PP = None



def destroy_distributed_environment():
    global _WORLD
    if _WORLD:
        _WORLD.destroy()
    _WORLD = None
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()