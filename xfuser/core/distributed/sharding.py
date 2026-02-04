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
    - shard_transformer_blocks: Generic transformer block sharding
"""
from functools import partial
from typing import Iterable, Optional, Any

import torch
import functools
from torch.distributed.fsdp import (
    MixedPrecision,
    ShardingStrategy,
    FullyShardedDataParallel as FSDP,
)
from torch.distributed.fsdp.wrap import (
    lambda_auto_wrap_policy,
)


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
        >>> _children_to_device(model, 'cuda:0', excluded_children=['blocks'])
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

    transformer = shard_transformer_blocks(
        transformer,
        block_attr=block_attr,
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

    transformer.encoder = shard_transformer_blocks(
        transformer.encoder,
        block_attr=block_attr,
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
) -> torch.nn.Module:
    """
    Wrap a component with FSDP, treating each block as a separate FSDP unit.

    This function applies Fully Sharded Data Parallel (FSDP) to a transformer model,
    automatically wrapping each transformer block separately for optimal memory
    distribution. Parameters and buffers are converted to the specified dtype.

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

    Returns:
        nn.Module: The FSDP-wrapped component.

    Raises:
        ValueError: If the component does not have the specified wrap_attrs attributes.

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
        ...     sync_module_states=True,
        ...     forward_prefetch=True
        ... )

    Note:
        - Uses FULL_SHARD strategy for maximum memory savings
        - Each element in 'wrap_attrs' becomes a separate FSDP unit
        - Requires PyTorch distributed to be initialized before calling
        - When passing a GroupCoordinator from get_sp_group() or get_world_group(),
          extract the ProcessGroup with `.device_group` attribute
    """
    # Determine device: use CUDA if available and device_id specified, else CPU
    if device_id is None:
        if torch.cuda.is_available():
            device_id = torch.cuda.current_device()
        else:
            device_id = None  # CPU mode

    wrapped_elements = []
    for wrap_attr in wrap_attrs:
        wrapped_elements.extend(rgetattr(component, wrap_attr))

    if dtype:
        component = component.to(dtype)

    component = FSDP(
        component,
        process_group=process_group,
        device_id=device_id,
        auto_wrap_policy=partial(lambda_auto_wrap_policy, lambda_fn=lambda module: module in wrapped_elements),
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        sync_module_states=sync_module_states,
        use_orig_params=use_orig_params,
        forward_prefetch=forward_prefetch,
    )

    return component

def rgetattr(obj: object, attr: str) -> object:
    """ Recursive getattr to get nested attributes """
    return functools.reduce(getattr, [obj] + attr.split("."))