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
from .format_backends import module_path_is_covered, module_paths_overlap


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


def _target_filter(targets, excluded_targets=(), include_suffixes=None):
    return lambda _module, fqn: any(
        not target or fqn == target or fqn.startswith(f"{target}.")
        for target in targets
    ) and not any(
        module_path_is_covered(fqn, target) for target in excluded_targets
    ) and (
        not include_suffixes or fqn.endswith(tuple(include_suffixes))
    )


def _has_unowned_target(targets, owners):
    return any(
        not any(module_path_is_covered(target, owner) for owner in owners)
        for target in targets
    )


def build_block_quantize_fn(
    loader,
    component_name: str,
    wrap_attrs: list,
    local_rank: int,
    *,
    component=None,
):
    """Return a per-block quantize callable (block, block_idx) -> None for this component, or None
    if no quantization is configured for it.

    Quantization targets are resolved against each block's actual
    ``component.wrap_attr.local_index`` path. The callback index remains flattened
    across wrap attributes for checkpoint loading and existing FP8 precision
    overrides; entries like "5." still apply to flattened block index 5.

    Suffix patterns (e.g. .net.0.proj) are block-local FQNs and are passed through unchanged on
    every block; only prefix patterns are stripped.
    """
    model = loader.model
    config, settings = model.config, model.settings
    if not (config.use_fp4_gemms or config.use_fp8_gemms or config.use_int8_gemms):
        return None

    device = f"cuda:{local_rank}"
    fp4_list = set(loader.quantization_plan.module_list("fp4"))
    fp8_list = set(loader.quantization_plan.module_list())
    fp8_overrides = settings.fp8_precision_overrides or ()
    fp8_suffix_overrides = settings.fp8_precision_override_suffixes
    int8_list = set(loader.quantization_plan.module_list("int8"))

    paths = [f"{component_name}.{a}" for a in wrap_attrs]

    def overlaps_any(targets):
        return any(
            module_paths_overlap(path, target) for path in paths for target in targets
        )

    use_fp4_here = config.use_fp4_gemms and overlaps_any(fp4_list)
    # fp8-only: in fp8 list but not fp4 list (e.g. transformer_2 in Wan2.2 FP4 mode)
    use_fp8_here = (config.use_fp8_gemms and overlaps_any(fp8_list)) or (
        config.use_fp4_gemms and overlaps_any(fp8_list) and not overlaps_any(fp4_list)
    )
    use_int8_here = config.use_int8_gemms and overlaps_any(int8_list)

    if not use_fp4_here and not use_fp8_here and not use_int8_here:
        return None

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
        local_fp4_targets = _block_local_targets(fp4_list, block_path)
        local_fp8_targets = _block_local_targets(fp8_list, block_path)
        local_int8_targets = _block_local_targets(int8_list, block_path)
        use_fp4_block = config.use_fp4_gemms and local_fp4_targets is not None
        use_fp8_block = (config.use_fp8_gemms and local_fp8_targets is not None) or (
            config.use_fp4_gemms
            and local_fp8_targets is not None
            and (
                local_fp4_targets is None
                or _has_unowned_target(local_fp8_targets, local_fp4_targets)
            )
        )
        use_int8_block = config.use_int8_gemms and local_int8_targets is not None
        if not use_fp4_block and not use_fp8_block and not use_int8_block:
            return

        block_prefix = f"{block_idx}."
        # Strip the block-index prefix so the quantize functions see local FQN paths.
        local_fp8 = (
            tuple(
                o[len(block_prefix) :]
                for o in fp8_overrides
                if o.startswith(block_prefix)
            )
            or None
        )
        if use_fp4_block:
            adapter = loader.backends.format
            if adapter is None:
                raise RuntimeError(
                    "FP4 block conversion requested without a selected backend"
                )
            adapter.convert_block(
                block,
                fp8_layers=local_fp8,
                fp8_suffix_layers=fp8_suffix_overrides,
                hybrid=config.use_hybrid_gemm_schedule,
                device=device,
                filter_fn=_target_filter(local_fp4_targets),
            )
        if use_fp8_block:
            adapter = loader.backends.blockwise_fp8
            if adapter is None:
                raise RuntimeError(
                    "FP8 block conversion requested without a selected backend"
                )
            adapter.convert_block(
                block,
                device=device,
                filter_fn=_target_filter(
                    local_fp8_targets,
                    (local_fp4_targets if use_fp4_block else ()),
                    getattr(settings, "fp8_gemm_include_suffixes", None),
                ),
            )
        elif use_int8_block:
            adapter = loader.backends.format
            if adapter is None:
                raise RuntimeError(
                    "INT8 block conversion requested without a selected backend"
                )
            adapter.convert_block(
                block,
                device=device,
                filter_fn=_target_filter(local_int8_targets),
            )

    return quantize_fn
