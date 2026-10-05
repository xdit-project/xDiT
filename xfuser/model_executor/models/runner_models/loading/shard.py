"""FSDP-shard a loaded pipeline, quantizing each block as it is wrapped.

The last of the three load phases this package covers: ``quantization_plan`` decides what a run quantizes,
``meta_load`` builds components on meta and fills their real weights, and this wraps the result in
FSDP. It reads the fill decisions back off ``ModelLoader`` (whether a component is on meta,
and whether it fills itself from disk) because those select which collective the sharding does.

``build_block_quantize_fn`` serves both per-block quantizers: the sharded path below, and
``meta_load``'s replicated fill.
"""

import torch

from xfuser.core.distributed import get_world_group, shard_component
from xfuser.core.distributed.parallel_state import get_fs_group
from xfuser.core.utils.checkpoint_io import host_mem_gb
from xfuser.core.utils.runner_utils import (
    log,
    rgetattr,
)
from .quant_adapter import module_paths_overlap


def shard_pipeline_components(loader) -> None:
    """Shard every component the run's fsdp_strategy names, and move the rest to the local device."""
    model = loader.model
    local_rank = get_world_group().local_rank
    fs_local_rank = get_fs_group().local_rank
    device_group = get_fs_group().device_group
    for component_name, component in model.pipe.components.items():
        if component_name in model.settings.fsdp_strategy:
            log(
                f"Sharding {component_name} with FSDP... "
                f"(host cur/anon/file: {host_mem_gb()} GB, "
                f"VRAM: {torch.cuda.memory_allocated(local_rank)/1e9:.2f}GB)"
            )
            strategy = model.settings.fsdp_strategy[component_name]
            wrap_attrs = strategy.get("wrap_attrs", [])
            dtype = strategy.get("dtype", None)
            offload_policy = strategy.get("offload_policy", None)
            # A meta component was built on-config to avoid a full bf16 copy per rank. Two meta
            # paths: a component that can map its live names onto checkpoint keys self-fills each
            # block from disk (never full anywhere, quantized per block), which covers transformers
            # and the text encoders whose mapping was proven. Anything else is filled by a rank0
            # broadcast, with no per-block quantize since the source stays bf16/streamed-fp8 on
            # rank0.
            # Agreed across the fs group: this picks a collective branch, so a rank-local
            # answer that diverged would hang instead of raising.
            is_meta = loader.agreed_is_meta(
                component, component_name, get_fs_group(), f"cuda:{fs_local_rank}"
            )
            is_selffill = is_meta and loader.self_fills_from_disk(component)
            load_block_fn = load_epilogue_fn = None
            if is_selffill:
                quantize_fn = build_block_quantize_fn(
                    loader,
                    component_name,
                    wrap_attrs,
                    fs_local_rank,
                    component=component,
                )
                load_block_fn, load_epilogue_fn = loader.build_blockwise_disk_loaders(
                    component, wrap_attrs, component_name, f"cuda:{fs_local_rank}"
                )
            else:
                quantize_fn = (
                    None
                    if is_meta
                    else build_block_quantize_fn(
                        loader,
                        component_name,
                        wrap_attrs,
                        fs_local_rank,
                        component=component,
                    )
                )
            fsdp_object = shard_component(
                component,
                wrap_attrs,
                device_group,
                fs_local_rank,
                dtype,
                quantize_fn=quantize_fn,
                reshard_after_forward=model.config.reshard_after_forward,
                memory_efficient_init=model.config.memory_efficient_sharding,
                offload_policy=offload_policy,
                # Cache hits skip transformer blocks, so prefetching their all-gathers
                # wastes communication; executed blocks still gather on demand.
                forward_prefetch=not bool(model.config.cache_method),
                # All ranks load from the same checkpoint so states are already
                # identical. No broadcast needed regardless of offload policy.
                sync_module_states=False,
                meta_init=is_meta and not is_selffill,
                load_block_fn=load_block_fn,
                load_epilogue_fn=load_epilogue_fn,
            )
            if is_meta and not is_selffill:
                loader.broadcast_load(
                    fsdp_object, component_name, offload_policy == "cpu"
                )
            setattr(model.pipe, component_name, fsdp_object)
            torch.cuda.empty_cache()
            log(
                f"Sharded {component_name}. "
                f"(host cur/anon/file: {host_mem_gb()} GB, "
                f"VRAM: {torch.cuda.memory_allocated(local_rank)/1e9:.2f}GB)"
            )
        else:
            log(f"Skipping FSDP wrapping for {component_name}...")
            if hasattr(component, "to"):
                component.to(f"cuda:{local_rank}")
            else:
                log(
                    f"Component {component_name} has no .to() method, skipping device move."
                )

    _give_cpu_offloaded_components_an_exec_device_hook(model, local_rank)


def _give_cpu_offloaded_components_an_exec_device_hook(model, local_rank: int) -> None:
    """Keep diffusers' _execution_device from resolving to cpu on a cpu-offloaded pipeline.

    _execution_device short-circuits on the first nn.Module component that lacks _hf_hook, returning
    self.device (= that module's .device). With CPUOffloadPolicy, text_encoder.device is cpu, which
    breaks latent generation. Give every nn.Module component a minimal _hf_hook so the walk
    continues past them, with cpu-offloaded components advertising cuda.
    """
    cpu_offloaded = {
        name
        for name, s in model.settings.fsdp_strategy.items()
        if s.get("offload_policy") == "cpu"
    }
    if not cpu_offloaded:
        return
    cuda_device = f"cuda:{local_rank}"

    class _ExecDeviceHook:
        def __init__(self, execution_device):
            self.execution_device = execution_device

    for name, component in model.pipe.components.items():
        if not isinstance(component, torch.nn.Module):
            continue
        if not hasattr(component, "_hf_hook"):
            component._hf_hook = _ExecDeviceHook(
                cuda_device if name in cpu_offloaded else None
            )


def _wrapped_block_paths(component, component_name, wrap_attrs):
    paths = []
    for attr in wrap_attrs:
        paths.extend(
            f"{component_name}.{attr}.{index}"
            for index, _ in enumerate(rgetattr(component, attr))
        )
    return tuple(paths)


def _block_local_targets(targets, block_path):
    local = []
    for target in targets:
        if target == block_path or block_path.startswith(f"{target}."):
            return ("",)
        if target.startswith(f"{block_path}."):
            local.append(target[len(block_path) + 1 :])
    return tuple(dict.fromkeys(local)) or None


def build_block_quantize_fn(
    loader,
    component_name: str,
    wrap_attrs: list,
    local_rank: int,
    *,
    component=None,
):
    """Per-block quantize callable (block, block_idx) -> None, or None when
    this component is not quantized.

    Targets resolve against each block's actual ``component.wrap_attr.index``
    path; the callback index stays flattened across wrap attributes, which is
    what the checkpoint fill counts in.

    One pass per format the run named, rather than a boolean per format the
    loader knows about. Each converter's filter asks the plan what a leaf
    becomes, so a block whose format is split -- the whole block low, one
    submodule held high -- needs no target arithmetic to stay disjoint, and a
    pass whose format owns nothing in a block simply converts nothing.
    """

    plan = loader.quantization_plan.gemm_plan
    if plan is None or not plan.quantizes:
        return None

    model = loader.model
    device = f"cuda:{local_rank}"

    formats = plan.formats_in_play
    by_format = {name: plan.walk_roots(name) for name in formats}

    paths = [f"{component_name}.{attr}" for attr in wrap_attrs]
    if not any(
        module_paths_overlap(path, root)
        for path in paths
        for roots in by_format.values()
        for root in roots
    ):
        return None

    # Resolved once rather than per block: a format this run named needs a
    # converter whether or not this particular block has leaves for it.
    adapters = {}
    for format_name, roots in by_format.items():
        if not roots:
            continue
        adapter = loader.backends.adapter_for(format_name)
        if adapter is None:
            raise RuntimeError(
                f"{format_name.upper()} block conversion requested without "
                "a selected backend"
            )
        adapters[format_name] = adapter

    block_paths = (
        _wrapped_block_paths(component, component_name, wrap_attrs)
        if component is not None
        else None
    )
    if block_paths is None and len(wrap_attrs) != 1:
        raise ValueError(
            "multiple wrap_attrs require the component to resolve flattened "
            "block indices"
        )

    def quantize_fn(block, block_idx: int) -> None:
        block_path = (
            block_paths[block_idx]
            if block_paths is not None
            else f"{component_name}.{wrap_attrs[0]}.{block_idx}"
        )
        present = [
            format_name
            for format_name, adapter in adapters.items()
            if _block_local_targets(by_format[format_name], block_path) is not None
        ]

        for format_name in present:
            adapter = adapters[format_name]
            convert_kwargs = {}
            companion = _hybrid_companion(loader, plan, format_name)
            if companion is not None:
                convert_kwargs.update(companion=companion)
            adapter.convert_block(
                block,
                device=device,
                filter_fn=_plan_target_filter(plan, block_path, format_name),
                **convert_kwargs,
            )

    return quantize_fn


def _hybrid_companion(loader, plan, format_name):
    """The per-step alternate the hybrid schedule pairs with this format.

    The run names both formats and the plan says which is which, so the walk
    composes the pair. Neither converter learns about the other.
    """
    if not getattr(loader.model.config, "use_hybrid_gemm_schedule", False):
        return None
    if plan.high is None or format_name != plan.low:
        return None
    companion = loader.backends.adapter_for(plan.high)
    if companion is None:
        return None
    return companion.layer_factory(device=None)


def _plan_target_filter(plan, block_path, format_name):
    """Keep the leaves the plan gives to `format_name`, and no others."""

    def filter_fn(_module, fqn):
        path = f"{block_path}.{fqn}" if fqn else block_path
        return plan.format_for(path) == format_name

    return filter_fn


