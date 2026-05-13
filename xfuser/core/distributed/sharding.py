"""
Sharding Utilities for Transformer Models.

This module provides functions to wrap transformer models with PyTorch's Fully Sharded
Data Parallel (FSDP) for distributed training. It enables efficient memory usage by
sharding model parameters across multiple GPUs while maintaining model parallelism.

Key Features:
    - Block-level FSDP wrapping for transformer architectures
    - Automatic handling of model conversion and device placement
    - Support for DiT and T5 encoder models

Functions:
    - shard_dit: Shard a Diffusion Transformer (DiT) model
    - shard_t5_encoder: Shard a T5 encoder model
    - shard_component: Generic transformer block sharding
"""
import logging
from functools import partial
from typing import Callable, Iterable, Optional

import torch
import functools
from torch.distributed.fsdp import (
    MixedPrecision,
    ShardingStrategy,
    FullyShardedDataParallel as FSDP,
)
from torch.distributed.fsdp.wrap import lambda_auto_wrap_policy
from torch.distributed.device_mesh import DeviceMesh


logger = logging.getLogger(__name__)


def _make_mesh(
    process_group: Optional[torch.distributed.ProcessGroup],
    device_type: str = "cuda",
):
    """Wrap an existing ProcessGroup as a 1-D DeviceMesh without creating a new NCCL communicator."""
    if process_group is None:
        return None
    return DeviceMesh.from_group(process_group, device_type)


def children_to_device(
    module: torch.nn.Module, device: str, excluded_children: Iterable[str] = []
) -> None:
    """
    Move immediate children of a module to the specified device.

    This helper function moves only the direct children (non-recursive) of a module
    to the target device. Since `.to(device)` is recursive, calling it on each
    immediate child will move that child and all its descendants.

    Args:
        module (torch.nn.Module): Parent module whose children should be moved.
        device (str): Target device string (e.g., 'cuda:0', 'cpu').
        excluded_children (Iterable[str], optional): Names of children to skip.
            Useful for excluding already-sharded modules (e.g., FSDP-wrapped blocks).
            Defaults to empty list.

    Note:
        - Uses `named_children()` not `named_modules()` because `.to()` is recursive
        - Each child's `.to()` call handles that child and all its descendants
        - Excluded children remain on their current device

    Example:
        >>> model = TransformerModel()
        >>> # Move all children except 'blocks' to GPU
        >>> children_to_device(model, 'cuda:0', excluded_children=['blocks'])
    """
    for name, child in module.named_children():
        if name not in excluded_children:
            child.to(device)


def shard_dit(
    transformer: torch.nn.Module,
    local_rank: int,
    process_group: Optional[torch.distributed.ProcessGroup] = None,
    block_attr: str = "blocks"
) -> torch.nn.Module:
    """
    Shard a DiT (Diffusion Transformer) model with FSDP block-by-block.

    This function wraps each transformer block with FSDP for distributed training,
    using bfloat16 dtype conversion and enabling forward prefetching for performance.
    Non-FSDP submodules are moved to the appropriate GPU device.

    Args:
        transformer (nn.Module): The transformer model to shard.
        local_rank (int): Local GPU rank/device ID for this process.
        process_group (ProcessGroup, optional): PyTorch distributed process group for
            FSDP communication. If None, uses the default process group. Note: pass
            `group.device_group` if using a GroupCoordinator wrapper.
        block_attr (str, optional): Name of the attribute containing transformer blocks.
            Defaults to 'blocks'.

    Returns:
        nn.Module: The FSDP-wrapped transformer model.

    Example:
        >>> from xfuser.core.distributed import get_sp_group
        >>> transformer = DiT(...)
        >>> # Pass the actual ProcessGroup, not the coordinator
        >>> sharded_model = shard_dit(
        ...     transformer,
        ...     local_rank=0,
        ...     process_group=get_sp_group().device_group,
        ...     block_attr='blocks'
        ... )
    """
    # Move any non-FSDP submodules to device (but NOT the blocks, they're already handled)
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    children_to_device(transformer, device, [block_attr])

    transformer = shard_component(
        transformer,
        wrap_attrs=[block_attr],
        device_id=local_rank,
        process_group=process_group,
        dtype=torch.bfloat16,
        use_orig_params=True,
        sync_module_states=True,
        forward_prefetch=True
    )


    return transformer


def shard_t5_encoder(
    transformer: torch.nn.Module,
    local_rank: int,
    process_group: Optional[torch.distributed.ProcessGroup] = None,
    block_attr: str = "block"
) -> torch.nn.Module:
    """
    Shard a T5 encoder model with FSDP block-by-block.

    This function specifically handles T5 encoder sharding by wrapping the encoder's
    transformer blocks with FSDP. Non-FSDP submodules are moved to the appropriate GPU.

    Args:
        transformer (nn.Module): The T5 transformer model containing an encoder.
        local_rank (int): Local GPU rank/device ID for this process.
        process_group (ProcessGroup, optional): PyTorch distributed process group for
            FSDP communication. If None, uses the default process group. Note: pass
            `group.device_group` if using a GroupCoordinator wrapper.
        block_attr (str, optional): Name of the attribute containing encoder blocks.
            Defaults to 'block' (T5 uses 'block' not 'blocks').

    Returns:
        nn.Module: The transformer with FSDP-wrapped encoder.

    Note:
        This function assumes the transformer has an 'encoder' attribute with transformer blocks.

    Example:
        >>> from xfuser.core.distributed import get_world_group
        >>> t5_model = T5EncoderModel(...)
        >>> sharded_model = shard_t5_encoder(
        ...     t5_model,
        ...     local_rank=0,
        ...     process_group=get_world_group().device_group,
        ...     block_attr='block'
        ... )
    """
    # Move any non-FSDP submodules to device (but NOT the block_attr, they're already handled)
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    children_to_device(transformer.encoder, device, [block_attr])
    children_to_device(transformer, device, ["encoder"])

    transformer.encoder = shard_component(
        transformer.encoder,
        wrap_attrs=[block_attr],
        device_id=local_rank,
        process_group=process_group,
        use_orig_params=True,
        sync_module_states=True,
        forward_prefetch=True
    )


    return transformer


def shard_component(
    component: torch.nn.Module,
    wrap_attrs: list[str],
    process_group: Optional[torch.distributed.ProcessGroup] = None,
    device_id: Optional[int] = None,
    dtype: Optional[torch.dtype] = None,
    use_orig_params: bool = True,
    sync_module_states: bool = True,
    forward_prefetch: bool = True,
    reshard_after_forward: bool = True,
    quantize_fn: Optional[Callable] = None,
) -> torch.nn.Module:
    """
    Wrap a component with FSDP, treating each block as a separate FSDP unit.

    Uses FSDP1 when quantize_fn is None (O(1) flat-param hooks, no DTensor bookkeeping,
    fastest for non-quantized inference). Uses FSDP2 (composable fully_shard) when
    quantize_fn is provided, required for torchao quantized tensor types that cannot
    be flattened to FSDP1's 1D FlatParameter.

    Args:
        component (nn.Module): The transformer model to wrap with FSDP.
        wrap_attrs (list[str]): Name of the model attributes containing elements
            to wrap in individual FSDP units.
        process_group (ProcessGroup, optional): PyTorch distributed process group for
            FSDP communication. If None, uses the default process group.
            **Important**: Pass `group.device_group` if using a GroupCoordinator wrapper
            (e.g., from `get_sp_group()` or `get_world_group()`), not the coordinator itself.
        device_id (int, optional): CUDA device ID to place the model on. If None,
            uses the current CUDA device.
        dtype (torch.dtype, optional): Target dtype to convert the model to before
            wrapping. If None, keeps the original dtype.
        use_orig_params (bool, optional): Whether to use the original parameters.
            Defaults to True.
        sync_module_states (bool, optional): Whether to sync module states.
            Defaults to True.
        forward_prefetch (bool, optional): Whether to use forward prefetch.
            Defaults to True.
        reshard_after_forward (bool, optional): If True (default), reshard parameters after each
            block's forward. Set False to keep params gathered post-forward, trading
            memory for latency. Maps to ShardingStrategy in FSDP1, reshard_after_forward
            in FSDP2.
            Defaults to True.
        quantize_fn (Callable, optional): Called as quantize_fn(block, idx) per block
            before FSDP2 wrapping. Selects FSDP2 path when provided.
            Defaults to None.

    Returns:
        nn.Module: The FSDP-wrapped component.

    Example:
        >>> from xfuser.core.distributed import get_sp_group
        >>> model = Transformer(...)
        >>> # Correct: extract device_group from coordinator
        >>> fsdp_model = shard_component(
        ...     model,
        ...     wrap_attrs=['blocks'],
        ...     device_id=0,
        ...     process_group=get_sp_group().device_group,  # NOT get_sp_group()
        ...     dtype=torch.bfloat16,
        ...     forward_prefetch=True,
        ...     reshard_after_forward=True,
        ...     quantize_fn=quantize_fn,
        ... )

    Note:
        - Each element in wrap_attrs becomes a separate FSDP unit
        - Requires PyTorch distributed to be initialized before calling
    """
    if device_id is None and torch.cuda.is_available():
        device_id = torch.cuda.current_device()

    wrapped_blocks = []
    for wrap_attr in wrap_attrs:
        wrapped_blocks.extend(rgetattr(component, wrap_attr))

    if dtype:
        component = component.to(dtype)

    if quantize_fn is None:
        # FSDP1: Fastest path for non-quantized inference.
        return FSDP(
            component,
            process_group=process_group,
            device_id=device_id,
            auto_wrap_policy=partial(lambda_auto_wrap_policy, lambda_fn=lambda m: m in wrapped_blocks),
            sharding_strategy=ShardingStrategy.FULL_SHARD if reshard_after_forward else ShardingStrategy.SHARD_GRAD_OP,
            sync_module_states=sync_module_states,
            use_orig_params=use_orig_params,
            forward_prefetch=forward_prefetch,
        )

    # FSDP2: Required for torchao quantized tensors.
    from torch.distributed._composable.fsdp import fully_shard  # noqa: PLC0415
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    device_str = f"{device_type}:{device_id}"
    mesh = _make_mesh(process_group, device_type)

    # Move non-block children first; block containers are handled block-by-block below.
    wrap_top_names = {attr.split(".")[0] for attr in wrap_attrs}
    for name, child in component.named_children():
        if name not in wrap_top_names:
            child.to(device_str)

    # Sequential: after fully_shard(block) each rank holds 1/N params, freeing memory
    # for the next block. At most one full block on GPU at a time.
    for i, block in enumerate(wrapped_blocks):
        block.to(device_str)
        quantize_fn(block, i)
        fully_shard(block, mesh=mesh, reshard_after_forward=reshard_after_forward)

    fully_shard(component, mesh=mesh, reshard_after_forward=reshard_after_forward)

    # FSDP2 forward prefetch: each block pre-fetches the next two blocks' all-gathers
    # so communication overlaps with compute. The first block has no predecessor to
    # trigger its prefetch, so a pre-hook manually unshards it before the forward begins.
    if forward_prefetch and len(wrapped_blocks) > 1:
        for i, block in enumerate(wrapped_blocks):
            lookahead = [
                wrapped_blocks[i + j]
                for j in range(1, 3)
                if i + j < len(wrapped_blocks)
            ]
            if lookahead:
                block.set_modules_to_forward_prefetch(lookahead)

        def _unshard_first_block(_module, _args, _kwargs):
            wrapped_blocks[0].unshard(async_op=True)
        component.register_forward_pre_hook(_unshard_first_block, with_kwargs=True)

    return component

def rgetattr(obj: object, attr: str) -> object:
    """ Recursive getattr to get nested attributes """
    return functools.reduce(getattr, [obj] + attr.split("."))