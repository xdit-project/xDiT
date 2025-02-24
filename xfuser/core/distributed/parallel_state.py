# Copyright 2024 xDiT team.
# Adapted from
# https://github.com/vllm-project/vllm/blob/main/vllm/distributed/parallel_state.py
# Copyright 2023 The vLLM team.
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
from typing import List, Optional

import torch
import torch.distributed
import xfuser.envs as envs
import os
from xfuser.logger import init_logger
from .group_coordinator import (
    GroupCoordinator,
    PipelineGroupCoordinator,
    SequenceParallelGroupCoordinator,
)
from .utils import RankGenerator, generate_masked_orthogonal_rank_groups

env_info = envs.PACKAGES_CHECKER.get_packages_info()
HAS_LONG_CTX_ATTN = env_info["has_long_ctx_attn"]
HAS_FLASH_ATTN = env_info["has_flash_attn"]

logger = init_logger(__name__)


_WORLD: Optional[GroupCoordinator] = None
_TP: Optional[GroupCoordinator] = None
_SP: Optional[SequenceParallelGroupCoordinator] = None
_PP: Optional[PipelineGroupCoordinator] = None
_CFG: Optional[GroupCoordinator] = None
_DP: Optional[GroupCoordinator] = None
_DIT: Optional[GroupCoordinator] = None
_VAE: Optional[GroupCoordinator] = None

# * QUERY
def get_world_group() -> GroupCoordinator:
    assert _WORLD is not None, "world group is not initialized"
    return _WORLD


# TP
def get_tp_group() -> GroupCoordinator:
    assert _TP is not None, "tensor model parallel group is not initialized"
    return _TP


def get_tensor_model_parallel_world_size():
    """Return world size for the tensor model parallel group."""
    return get_tp_group().world_size


def get_tensor_model_parallel_rank():
    """Return my rank for the tensor model parallel group."""
    return get_tp_group().rank_in_group


# SP
def get_sp_group() -> SequenceParallelGroupCoordinator:
    assert _SP is not None, "pipeline model parallel group is not initialized"
    return _SP


def get_sequence_parallel_world_size():
    """Return world size for the sequence parallel group."""
    return get_sp_group().world_size


def get_sequence_parallel_rank():
    """Return my rank for the sequence parallel group."""
    return get_sp_group().rank_in_group


def get_ulysses_parallel_world_size():
    return get_sp_group().ulysses_world_size


def get_ulysses_parallel_rank():
    return get_sp_group().ulysses_rank


def get_ring_parallel_world_size():
    return get_sp_group().ring_world_size


def get_ring_parallel_rank():
    return get_sp_group().ring_rank


# PP
def get_pp_group() -> PipelineGroupCoordinator:
    assert _PP is not None, "pipeline model parallel group is not initialized"
    return _PP


def get_pipeline_parallel_world_size():
    """Return world size for the pipeline model parallel group."""
    return get_pp_group().world_size


def get_pipeline_parallel_rank():
    """Return my rank for the pipeline model parallel group."""
    return get_pp_group().rank_in_group


def is_pipeline_first_stage():
    """Return True if in the first pipeline model parallel stage, False otherwise."""
    return get_pipeline_parallel_rank() == 0


def is_pipeline_last_stage():
    """Return True if in the last pipeline model parallel stage, False otherwise."""
    return get_pipeline_parallel_rank() == (get_pipeline_parallel_world_size() - 1)


# CFG
def get_cfg_group() -> GroupCoordinator:
    assert (
        _CFG is not None
    ), "classifier_free_guidance parallel group is not initialized"
    return _CFG


def get_classifier_free_guidance_world_size():
    """Return world size for the classifier_free_guidance parallel group."""
    return get_cfg_group().world_size


def get_classifier_free_guidance_rank():
    """Return my rank for the classifier_free_guidance parallel group."""
    return get_cfg_group().rank_in_group


# DP
def get_dp_group() -> GroupCoordinator:
    assert _DP is not None, "pipeline model parallel group is not initialized"
    return _DP


def get_data_parallel_world_size():
    """Return world size for the data parallel group."""
    return get_dp_group().world_size


def get_data_parallel_rank():
    """Return my rank for the data parallel group."""
    return get_dp_group().rank_in_group


def is_dp_last_group():
    """Return True if in the last data parallel group, False otherwise."""
    return (
        get_sequence_parallel_rank() == (get_sequence_parallel_world_size() - 1)
        and get_classifier_free_guidance_rank()
        == (get_classifier_free_guidance_world_size() - 1)
        and get_pipeline_parallel_rank() == (get_pipeline_parallel_world_size() - 1)
    )

def get_dit_world_size():
    """Return world size for the DiT model (excluding VAE)."""
    return (get_data_parallel_world_size() *
            get_classifier_free_guidance_world_size() *
            get_sequence_parallel_world_size() *
            get_pipeline_parallel_world_size() *
            get_tensor_model_parallel_world_size())

# Add VAE getter functions
def get_vae_parallel_group() -> GroupCoordinator:
    assert _VAE is not None, "VAE parallel group is not initialized"
    return _VAE

def get_vae_parallel_world_size():
    """Return world size for the VAE parallel group."""
    return get_vae_parallel_group().world_size

def get_vae_parallel_rank():
    """Return my rank for the VAE parallel group."""
    return get_vae_parallel_group().rank_in_group

# * SET


def init_world_group(
    ranks: List[int], local_rank: int, backend: str
) -> GroupCoordinator:
    return GroupCoordinator(
        group_ranks=[ranks],
        local_rank=local_rank,
        torch_distributed_backend=backend,
    )


def init_distributed_environment(
    world_size: int = -1,
    rank: int = -1,
    distributed_init_method: str = "env://",
    local_rank: int = -1,
    backend: str = "nccl",
):
    logger.debug(
        "world_size=%d rank=%d local_rank=%d " "distributed_init_method=%s backend=%s",
        world_size,
        rank,
        local_rank,
        distributed_init_method,
        backend,
    )
    if not torch.distributed.is_initialized():
        assert distributed_init_method is not None, (
            "distributed_init_method must be provided when initializing "
            "distributed environment"
        )
        # this backend is used for WORLD
        torch.distributed.init_process_group(
            backend=backend,
            init_method=distributed_init_method,
            world_size=world_size,
            rank=rank,
        )
        torch.cuda.set_device(torch.distributed.get_rank() % torch.cuda.device_count())
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
        assert (
            _WORLD.world_size == torch.distributed.get_world_size()
        ), "world group already initialized with a different world size"

def model_parallel_is_initialized():
    """Check if tensor and pipeline parallel groups are initialized."""
    return (
        _DP is not None
        and _CFG is not None
        and _SP is not None
        and _PP is not None
        and _TP is not None
    )


def init_model_parallel_group(
    group_ranks: List[List[int]],
    local_rank: int,
    backend: str,
    parallel_mode: str,
    **kwargs,
) -> GroupCoordinator:
    assert parallel_mode in [
        "data",
        "pipeline",
        "tensor",
        "sequence",
        "classifier_free_guidance",
    ], f"parallel_mode {parallel_mode} is not supported"
    if parallel_mode == "pipeline":
        return PipelineGroupCoordinator(
            group_ranks=group_ranks,
            local_rank=local_rank,
            torch_distributed_backend=backend,
        )
    elif parallel_mode == "sequence":
        return SequenceParallelGroupCoordinator(
            group_ranks=group_ranks,
            local_rank=local_rank,
            torch_distributed_backend=backend,
            **kwargs,
        )
    else:
        return GroupCoordinator(
            group_ranks=group_ranks,
            local_rank=local_rank,
            torch_distributed_backend=backend,
        )
 
def init_dit_group(
    dit_parallel_size: int,
    backend: str,
):
    global _DIT
    _DIT = torch.distributed.new_group(
                ranks=list(range(dit_parallel_size)), backend=backend
            )
    
def get_dit_group():
    assert _DIT is not None, "DIT group is not initialized"
    return _DIT

def init_vae_group(
    dit_parallel_size: int,
    vae_parallel_size: int,
    backend: str,
):
    # Initialize VAE group first
    global _VAE
    assert _VAE is None, "VAE parallel group is already initialized"
    vae_ranks = list(range(dit_parallel_size, dit_parallel_size + vae_parallel_size))
    _VAE = torch.distributed.new_group(
                ranks=vae_ranks, backend=backend
            )

def initialize_model_parallel(
    data_parallel_degree: int = 1,
    classifier_free_guidance_degree: int = 1,
    sequence_parallel_degree: int = 1,
    ulysses_degree: int = 1,
    ring_degree: int = 1,
    tensor_parallel_degree: int = 1,
    pipeline_parallel_degree: int = 1,
    vae_parallel_size: int = 0,
    backend: Optional[str] = None,
) -> None:
    """
    Initialize model parallel groups.

    Arguments:
        data_parallel_degree: number of data parallelism groups.
        classifier_free_guidance_degree: number of GPUs used for Classifier Free Guidance (CFG)
        sequence_parallel_degree: number of GPUs used for sequence parallelism.
        ulysses_degree: number of GPUs used for ulysses sequence parallelism.
        ring_degree: number of GPUs used for ring sequence parallelism.
        tensor_parallel_degree: number of GPUs used for tensor parallelism.
        pipeline_parallel_degree: number of GPUs used for pipeline parallelism.
        backend: distributed backend of pytorch collective comm.

    Let's say we have a total of 16 GPUs denoted by g0 ... g15 and we
    use 2 groups to parallelize the batch dim(dp), 2 groups to parallelize
    splited batch caused by CFG, and 2 GPUs to parallelize sequence.

    dp_degree (2) * cfg_degree (2) * sp_degree (2) * pp_degree (2) = 16.

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
    backend = backend or torch.distributed.get_backend(get_world_group().device_group)
    dit_parallel_size = (data_parallel_degree *
                     classifier_free_guidance_degree *
                     sequence_parallel_degree *
                     pipeline_parallel_degree *
                     tensor_parallel_degree)

    if world_size < dit_parallel_size:
        raise RuntimeError(
            f"world_size ({world_size}) is less than "
            f"tensor_parallel_degree ({tensor_parallel_degree}) x "
            f"pipeline_parallel_degree ({pipeline_parallel_degree}) x"
            f"sequence_parallel_degree ({sequence_parallel_degree}) x"
            f"classifier_free_guidance_degree "
            f"({classifier_free_guidance_degree}) x"
            f"data_parallel_degree ({data_parallel_degree})"
        )

    rank_generator: RankGenerator = RankGenerator(
        tensor_parallel_degree,
        sequence_parallel_degree,
        pipeline_parallel_degree,
        classifier_free_guidance_degree,
        data_parallel_degree,
        "tp-sp-pp-cfg-dp",
    )
    global _DP
    assert _DP is None, "data parallel group is already initialized"
    _DP = init_model_parallel_group(
        group_ranks=rank_generator.get_ranks("dp"),
        local_rank=get_world_group().local_rank,
        backend=backend,
        parallel_mode="data",
    )

    global _CFG
    assert _CFG is None, "classifier_free_guidance group is already initialized"
    _CFG = init_model_parallel_group(
        group_ranks=rank_generator.get_ranks("cfg"),
        local_rank=get_world_group().local_rank,
        backend=backend,
        parallel_mode="classifier_free_guidance",
    )

    global _PP
    assert _PP is None, "pipeline model parallel group is already initialized"
    _PP = init_model_parallel_group(
        group_ranks=rank_generator.get_ranks("pp"),
        local_rank=get_world_group().local_rank,
        backend=backend,
        parallel_mode="pipeline",
    )

    global _SP
    assert _SP is None, "sequence parallel group is already initialized"

    # if HAS_LONG_CTX_ATTN and sequence_parallel_degree > 1:
    if HAS_LONG_CTX_ATTN:
        from yunchang import set_seq_parallel_pg
        from yunchang.globals import PROCESS_GROUP

        set_seq_parallel_pg(
            sp_ulysses_degree=ulysses_degree,
            sp_ring_degree=ring_degree,
            rank=get_world_group().rank_in_group,
            world_size=dit_parallel_size
        )

        _SP = init_model_parallel_group(
            group_ranks=rank_generator.get_ranks("sp"),
            local_rank=get_world_group().local_rank,
            backend=backend,
            parallel_mode="sequence",
            ulysses_group=PROCESS_GROUP.ULYSSES_PG,
            ring_group=PROCESS_GROUP.RING_PG,
        )
    else:
        _SP = init_model_parallel_group(
            group_ranks=rank_generator.get_ranks("sp"),
            local_rank=get_world_group().local_rank,
            backend=backend,
            parallel_mode="sequence",
        )

    global _TP
    assert _TP is None, "Tensor parallel group is already initialized"
    _TP = init_model_parallel_group(
        group_ranks=rank_generator.get_ranks("tp"),
        local_rank=get_world_group().local_rank,
        backend=backend,
        parallel_mode="tensor",
    )

    if vae_parallel_size > 0:
        init_vae_group(dit_parallel_size, vae_parallel_size, backend)
    init_dit_group(dit_parallel_size, backend)

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

    global _VAE
    if _VAE:
        _VAE.destroy()
    _VAE = None


def destroy_distributed_environment():
    global _WORLD
    if _WORLD:
        _WORLD.destroy()
    _WORLD = None
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
