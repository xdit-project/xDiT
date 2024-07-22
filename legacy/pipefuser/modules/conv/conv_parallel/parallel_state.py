# this file reference to https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/parallel_state.py
from datetime import timedelta

import torch
import torch.distributed as dist

# direction: rank0 -> rank1 -> ... -> rank-n
# group: rank0 [next_group]
# -> [previous_group] rank-i [next_group]
# -> [previous_group] rank-n
_PATCH_PARALLEL_PREVIOUS_GROUP = None

_PATCH_PARALLEL_NEXT_GROUP = None


def get_nccl_options(pg_name, nccl_comm_cfgs):
    """Set the NCCL process group options.

    Args:
        pg_name (str): process group name
        nccl_comm_cfgs (dict): nccl communicator configurations

    When an option (e.g., max_ctas) is not found in the config, use the NCCL default setting.
    """
    if pg_name in nccl_comm_cfgs:
        nccl_options = torch.distributed.ProcessGroupNCCL.Options()
        nccl_options.config.cga_cluster_size = nccl_comm_cfgs[pg_name].get(
            "cga_cluster_size", 4
        )
        nccl_options.config.max_ctas = nccl_comm_cfgs[pg_name].get("max_ctas", 32)
        nccl_options.config.min_ctas = nccl_comm_cfgs[pg_name].get("min_ctas", 1)
        return nccl_options
    else:
        return None


def init_patch_parallel(
    distributed_timeout_minutes: int = 30,
):
    global _PATCH_PARALLEL_PREVIOUS_GROUP
    global _PATCH_PARALLEL_NEXT_GROUP
    rank = 0
    world_size = 1
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    if world_size == 1:
        return
    timeout = timedelta(minutes=distributed_timeout_minutes)
    if rank == 0:
        _PATCH_PARALLEL_NEXT_GROUP = dist.new_group(
            [rank, rank + 1],
            timeout=timeout,
            pg_options=get_nccl_options("patch_parallel_next_group", {}),
        )
        return
    if rank + 1 == world_size:
        _PATCH_PARALLEL_PREVIOUS_GROUP = dist.new_group(
            [rank - 1, rank],
            timeout=timeout,
            pg_options=get_nccl_options("patch_parallel_previous_group", {}),
        )
        return
    _PATCH_PARALLEL_NEXT_GROUP = dist.new_group(
        [rank, rank + 1],
        timeout=timeout,
        pg_options=get_nccl_options("patch_parallel_next_group", {}),
    )
    _PATCH_PARALLEL_PREVIOUS_GROUP = dist.new_group(
        [rank - 1, rank],
        timeout=timeout,
        pg_options=get_nccl_options("patch_parallel_previous_group", {}),
    )


def get_patch_parallel_previous_group():
    global _PATCH_PARALLEL_PREVIOUS_GROUP
    return _PATCH_PARALLEL_PREVIOUS_GROUP


def get_patch_parallel_next_group():
    global _PATCH_PARALLEL_NEXT_GROUP
    return _PATCH_PARALLEL_NEXT_GROUP
