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

from xfuser.core.utils.dtype_policy import (
    cast_preserving_fp32_modules,
    fp32_modules_for,
    pinned_fp32_parameters,
)


logger = logging.getLogger(__name__)


def _save_nonpersistent_buffers(module: torch.nn.Module, device: str):
    """Copy initialized runtime-only buffers before ``to_empty`` discards their storage."""
    saved = []
    for owner in module.modules():
        for name in owner._non_persistent_buffers_set:
            buffer = owner._buffers.get(name)
            if buffer is not None and not buffer.is_meta:
                saved.append(
                    (owner, name, buffer.detach().to(device=device, copy=True))
                )
    return saved


def _restore_nonpersistent_buffers(saved) -> None:
    """Restore saved buffers without changing their non-persistent registration."""
    for owner, name, buffer in saved:
        owner._buffers[name] = buffer


def _collective_quantize_call(operation, process_group, context):
    """Run quantization locally and make every process-group rank agree on failure."""
    dist = torch.distributed
    if not dist.is_available() or not dist.is_initialized():
        return operation()
    world_size = dist.get_world_size(group=process_group)
    if world_size <= 1:
        return operation()

    local_error = None
    local_exception = None
    try:
        result = operation()
    except Exception as error:
        result = None
        local_exception = error
        local_error = (type(error).__name__, str(error))

    failures = [None] * world_size
    dist.all_gather_object(failures, local_error, group=process_group)
    for rank, failure in enumerate(failures):
        if failure is not None:
            error_type, message = failure
            raise RuntimeError(
                f"{context} failed on rank {rank}: {error_type}: {message}"
            ) from local_exception
    return result


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


def _keep_recording_outputs(component: torch.nn.Module) -> None:
    """Re-key transformers' output recording onto the class fully_shard just rebound.

    A transformers model decides which submodule outputs to record — ``hidden_states``,
    ``attentions`` — by looking its own class up in a registry populated when the model was
    constructed. ``fully_shard`` rebinds ``__class__`` on what it wraps, so after sharding that
    lookup misses and a forward asked for ``output_hidden_states=True`` returns ``None`` instead of
    raising, which surfaces much later as a pipeline subscripting ``hidden_states[-2]``.

    The root has to be wrapped, so the recording has to follow it: the blocks share the root's
    lazily-initialized comm context, and leaving the root unwrapped makes each block its own root
    and breaks the cross-block prefetch. Registering the sharded class under the same spec is what
    lets both hold.

    Absence of the registry is not an error: it means the installed transformers does not resolve
    recording this way, in which case there is nothing to carry over. The end-to-end behaviour is
    pinned by tests/core/test_sharded_text_encoder_outputs.py, so a reworked mechanism fails there
    rather than silently costing a caller its hidden states.
    """
    try:
        from transformers.modeling_utils import (  # noqa: PLC0415
            _CAN_RECORD_REGISTRY,
        )
    except ImportError:
        return

    recordable = getattr(component, "_can_record_outputs", None)
    if recordable:
        _CAN_RECORD_REGISTRY[str(type(component))] = recordable


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
    memory_efficient_init: bool = False,
    offload_policy: Optional[str] = None,
    meta_init: bool = False,
    load_block_fn: Optional[Callable] = None,
    load_epilogue_fn: Optional[Callable] = None,
) -> torch.nn.Module:
    """
    Wrap a component with FSDP, treating each block as a separate FSDP unit.

    Uses FSDP1 when quantize_fn is None and memory_efficient_init is False (O(1)
    flat-param hooks, no DTensor bookkeeping, fastest for non-quantized inference).
    Uses FSDP2 (composable fully_shard) when quantize_fn is provided (required: FSDP1
    cannot flatten torchao quantized tensor subtypes) or memory_efficient_init is True)

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
            before FSDP2 wrapping. Automatically selects FSDP2; do not combine with
            memory_efficient_init=False to try to force FSDP1, it will be ignored.
            Defaults to None.
        memory_efficient_init (bool, optional): Initialize blocks sequentially one at a
            time to minimize peak GPU memory during model load. Selects FSDP2. Only use
            when the model OOMs during init with FSDP1; FSDP1 is faster at inference.
            Defaults to False.
        offload_policy (str, optional): "cpu" wraps params in FSDP2 CPUOffloadPolicy
            (params live on host, streamed to GPU per block); any other value / None keeps
            params on GPU. Selects FSDP2 when "cpu". Defaults to None.
        meta_init (bool, optional): The component's params are on the meta device (built from
            config, no weights). Skips all host/device moves and quantization; blocks are
            fully_shard'd while still meta. The caller must materialize real weights afterwards
            (e.g. rank0-broadcast set_model_state_dict). Selects FSDP2. Defaults to False.
        load_block_fn (Callable, optional): Called as load_block_fn(block, idx) per block, after the
            block is materialized empty on device (to_empty) and before quantize_fn/fully_shard, to
            fill that block's real weights (e.g. streamed per-block from disk). Unlike meta_init,
            which materializes the whole component from a rank0 state dict, this fills one block at a
            time, so no rank holds more than a block beyond the source's incremental reads; how the
            weights are obtained is the callback's business (xDiT reads on rank0 and broadcasts per
            block; see runner_models.loading.meta_load). Selects FSDP2. When set, quantize_fn still runs (on
            the now-real block) and the block is sharded normally. The component must already be on
            meta. Defaults to None.
        load_epilogue_fn (Callable, optional): Called as load_epilogue_fn(component) after the block
            loop but BEFORE the component-level fully_shard, to fill non-block params/buffers (which
            would otherwise become DTensors and reject a plain assignment). Pairs with load_block_fn.
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
    use_fsdp2 = quantize_fn is not None or memory_efficient_init or meta_init or load_block_fn is not None

    if device_id is None and torch.cuda.is_available():
        device_id = torch.cuda.current_device()

    wrapped_blocks = []
    for wrap_attr in wrap_attrs:
        wrapped_blocks.extend(rgetattr(component, wrap_attr))

    # The modules the model's loader keeps in fp32 have to stay out of every FSDP unit: a unit is one
    # flat allocation and FSDP rejects a mixture of dtypes within it. They are a rounding error in
    # size next to the weights they normalise (a few MB across all of Wan's blocks), so replicating
    # them costs nothing measurable, whereas casting them down loses precision the ordinary load of
    # the same checkpoint keeps.
    fp32_modules = fp32_modules_for(component, dtype) if dtype else ()
    if dtype and not meta_init:
        component = cast_preserving_fp32_modules(component, dtype)

    def ignore_fp32(module):
        """This module's pinned parameters, resolved now: a fill rebinds parameter slots, so a set
        collected earlier would name parameters the module no longer holds and FSDP would take them
        into a unit anyway. FSDP also leaves ignored parameters where they are, so they need the
        device move themselves; meta ones get it from the caller's fill.
        """
        pinned = pinned_fp32_parameters(module, fp32_modules)
        if pinned and device_id is not None:
            device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu", device_id
            )
            for parameter in pinned:
                if not parameter.is_meta and parameter.device != device:
                    parameter.data = parameter.data.to(device)
        return pinned or None

    if not use_fsdp2:
        # FSDP1: Fastest path for non-quantized inference.
        ignored = ignore_fp32(component)
        return FSDP(
            component,
            process_group=process_group,
            device_id=device_id,
            auto_wrap_policy=partial(lambda_auto_wrap_policy, lambda_fn=lambda m: m in wrapped_blocks),
            sharding_strategy=ShardingStrategy.FULL_SHARD if reshard_after_forward else ShardingStrategy.SHARD_GRAD_OP,
            sync_module_states=sync_module_states,
            use_orig_params=use_orig_params,
            forward_prefetch=forward_prefetch,
            ignored_states=list(ignored) if ignored else None,
        )

    # FSDP2: Required for torchao quantized tensors, or when use_fsdp2=True for
    # sequential block-by-block init to reduce peak GPU memory during model load.
    from torch.distributed._composable.fsdp import fully_shard, CPUOffloadPolicy  # noqa: PLC0415
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    device_str = f"{device_type}:{device_id}"
    mesh = _make_mesh(process_group, device_type)
    cpu_offload = CPUOffloadPolicy() if offload_policy == "cpu" else None

    # Move non-block children to device. With CPUOffloadPolicy params stay on CPU,
    # so skip this step — fully_shard handles placement.
    # meta_init: params can't be .to()'d (meta) and hold no real values to quantize; leave
    # them meta and let fully_shard build meta DTensors, filled later by the caller's broadcast.
    wrap_top_names = {attr.split(".")[0] for attr in wrap_attrs}
    # Skip the .to() move for both meta paths: broadcast (meta_init) fills DTensors later, and
    # self-fill (load_block_fn) materializes non-block children via load_epilogue_fn below.
    if cpu_offload is None and not meta_init and load_block_fn is None:
        for name, child in component.named_children():
            if name not in wrap_top_names:
                child.to(device_str)

    # Sequential: after fully_shard(block) each rank holds 1/N params, freeing memory
    # for the next block. At most one full block on GPU at a time.
    for i, block in enumerate(wrapped_blocks):
        if load_block_fn is not None:
            # Self-fill per rank: materialize the block empty on device, fill its real weights
            # from disk on this rank, quantize, then shard — the full model never lands anywhere.
            nonpersistent_buffers = _save_nonpersistent_buffers(block, device_str)
            block.to_empty(device=device_str, recurse=True)
            _restore_nonpersistent_buffers(nonpersistent_buffers)
            load_block_fn(block, i)
            if quantize_fn is not None:
                _collective_quantize_call(
                    lambda: quantize_fn(block, i),
                    process_group,
                    context=f"quantizing FSDP block {i}",
                )
        elif not meta_init:
            block.to(device_str)
            if quantize_fn is not None:
                _collective_quantize_call(
                    lambda: quantize_fn(block, i),
                    process_group,
                    context=f"quantizing FSDP block {i}",
                )
        fully_shard(
            block,
            mesh=mesh,
            reshard_after_forward=reshard_after_forward,
            offload_policy=cpu_offload,
            ignored_params=ignore_fp32(block),
        )

    # Fill non-block params/buffers before the component-level fully_shard turns them into DTensors
    # (a plain assignment onto a DTensor slot would diverge across ranks).
    if load_epilogue_fn is not None:
        load_epilogue_fn(component)

    fully_shard(
        component,
        mesh=mesh,
        reshard_after_forward=reshard_after_forward,
        offload_policy=cpu_offload,
        ignored_params=ignore_fp32(component),
    )
    _keep_recording_outputs(component)

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