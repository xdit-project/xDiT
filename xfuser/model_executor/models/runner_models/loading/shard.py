"""FSDP-shard a loaded pipeline, quantizing each block as it is wrapped.

The last of the three load phases this package covers: ``fp8_plan`` decides what a run quantizes,
``meta_load`` builds components on meta and fills their real weights, and this wraps the result in
FSDP. It reads the fill decisions back off ``MemoryEfficientLoader`` (whether a component is on meta,
and whether it fills itself from disk) because those select which collective the sharding does.

``build_block_quantize_fn`` lives here rather than on the runner because both callers are in this
package: the sharded path below, and ``meta_load``'s replicated per-block fill.
"""

import torch

from xfuser.envs import _is_cuda
from xfuser.core.distributed import get_world_group, shard_component
from xfuser.core.distributed.parallel_state import get_fs_group
from xfuser.core.utils.checkpoint_io import host_mem_gb
from xfuser.core.utils.runner_utils import (
    log,
    quantize_linear_layers_to_int8,
    quantize_linear_layers_to_fp8,
    quantize_linear_layers_to_fp8_blockscale,
    quantize_linear_layers_to_fp4,
    quantize_linear_layers_to_nvfp4,
    _use_aiter_fp8_rdna4,
)


def shard_pipeline_components(model) -> None:
    """Shard every component the run's fsdp_strategy names, and move the rest to the local device."""
    if model.config.use_fp8_gemms and _is_cuda():
        from xfuser.core.utils.runner_utils import _TORCHAO_FLOAT8_FSDP2_PATCHES
        assert _TORCHAO_FLOAT8_FSDP2_PATCHES, (
            "FSDP2 + FP8 requires torchao Float8Tensor patches but they failed to apply at "
            "import time. Check for torchao import errors in runner_utils."
        )
    loader = model._loader
    local_rank = get_world_group().local_rank
    fs_local_rank = get_fs_group().local_rank
    device_group = get_fs_group().device_group
    for component_name, component in model.pipe.components.items():
        if component_name in model.settings.fsdp_strategy:
            log(f"Sharding {component_name} with FSDP... "
                f"(host cur/anon/file: {host_mem_gb()} GB, "
                f"VRAM: {torch.cuda.memory_allocated(local_rank)/1e9:.2f}GB)")
            strategy = model.settings.fsdp_strategy[component_name]
            wrap_attrs = strategy.get("wrap_attrs", [])
            dtype = strategy.get("dtype", None)
            offload_policy = strategy.get("offload_policy", None)
            # A meta component was built on-config to avoid a full bf16 copy per rank. Two meta
            # paths: a transformer we meta-built self-fills each block from disk (never full
            # anywhere, quantized per block). Anything else (text encoders, or a meta component
            # we did not build) is filled by a rank0 broadcast, with no per-block quantize since
            # the source stays bf16/streamed-fp8 on rank0.
            # Agreed across the fs group: this picks a collective branch, so a rank-local
            # answer that diverged would hang instead of raising.
            is_meta = loader.agreed_is_meta(
                component, component_name, get_fs_group(), f"cuda:{fs_local_rank}"
            )
            is_selffill = is_meta and loader.self_fills_from_disk(component)
            load_block_fn = load_epilogue_fn = None
            if is_selffill:
                quantize_fn = build_block_quantize_fn(
                    model, component_name, wrap_attrs, fs_local_rank
                )
                load_block_fn, load_epilogue_fn = loader.build_transformer_disk_loaders(
                    component, wrap_attrs, component_name, f"cuda:{fs_local_rank}"
                )
            else:
                quantize_fn = (
                    None if is_meta
                    else build_block_quantize_fn(
                        model, component_name, wrap_attrs, fs_local_rank
                    )
                )
            fsdp_object = shard_component(
                component, wrap_attrs, device_group, fs_local_rank, dtype,
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
            log(f"Sharded {component_name}. "
                f"(host cur/anon/file: {host_mem_gb()} GB, "
                f"VRAM: {torch.cuda.memory_allocated(local_rank)/1e9:.2f}GB)")
        else:
            log(f"Skipping FSDP wrapping for {component_name}...")
            if hasattr(component, "to"):
                component.to(f"cuda:{local_rank}")
            else:
                log(f"Component {component_name} has no .to() method, skipping device move.")

    _give_cpu_offloaded_components_an_exec_device_hook(model, local_rank)


def _give_cpu_offloaded_components_an_exec_device_hook(model, local_rank: int) -> None:
    """Keep diffusers' _execution_device from resolving to cpu on a cpu-offloaded pipeline.

    _execution_device short-circuits on the first nn.Module component that lacks _hf_hook, returning
    self.device (= that module's .device). With CPUOffloadPolicy, text_encoder.device is cpu, which
    breaks latent generation. Give every nn.Module component a minimal _hf_hook so the walk
    continues past them, with cpu-offloaded components advertising cuda.
    """
    cpu_offloaded = {
        name for name, s in model.settings.fsdp_strategy.items()
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


def build_block_quantize_fn(model, component_name: str, wrap_attrs: list, local_rank: int):
    """Return a per-block quantize callable (block, block_idx) -> None for this component, or None
    if no quantization is configured for it.

    fp8_precision_overrides entries like "5." apply to block index 5. We strip the block-index
    prefix before passing to the quantize functions so they see the same local FQN paths they would
    in the non-FSDP path.

    Suffix patterns (e.g. .net.0.proj) are block-local FQNs and are passed through unchanged on
    every block; only prefix patterns are stripped.
    """
    config, settings = model.config, model.settings
    if not (config.use_fp4_gemms or config.use_fp8_gemms or config.use_int8_gemms):
        return None

    device = f"cuda:{local_rank}"
    fp4_list = set(settings.fp4_gemm_module_list or [])
    fp8_list = set(model.fp8.module_list())
    fp8_overrides = settings.fp8_precision_overrides or ()
    fp8_suffix_overrides = settings.fp8_precision_override_suffixes
    int8_list = set(settings.int8_gemm_module_list or [])

    paths = [f"{component_name}.{a}" for a in wrap_attrs]

    use_fp4_here = config.use_fp4_gemms and any(p in fp4_list for p in paths)
    # fp8-only: in fp8 list but not fp4 list (e.g. transformer_2 in Wan2.2 FP4 mode)
    use_fp8_here = (
        config.use_fp8_gemms and any(p in fp8_list for p in paths)
    ) or (
        config.use_fp4_gemms and any(p in fp8_list and p not in fp4_list for p in paths)
    )
    use_int8_here = config.use_int8_gemms and any(p in int8_list for p in paths)

    if not use_fp4_here and not use_fp8_here and not use_int8_here:
        return None

    def quantize_fn(block, block_idx: int) -> None:
        block_prefix = f"{block_idx}."
        # Strip the block-index prefix so the quantize functions see local FQN paths.
        local_fp8 = tuple(
            o[len(block_prefix):] for o in fp8_overrides if o.startswith(block_prefix)
        ) or None
        if use_fp4_here:
            if _is_cuda():
                quantize_linear_layers_to_nvfp4(
                    block,
                    fp8_layers=local_fp8,
                    fp8_suffix_layers=fp8_suffix_overrides,
                    device=device,
                )
            else:
                quantize_linear_layers_to_fp4(
                    block,
                    fp8_layers=local_fp8,
                    fp8_suffix_layers=fp8_suffix_overrides,
                    use_hybrid_schedule=config.use_hybrid_gemm_schedule,
                    device=device,
                )
        elif use_fp8_here:
            if _use_aiter_fp8_rdna4():
                quantize_linear_layers_to_fp8_blockscale(block, device=device)
            else:
                quantize_linear_layers_to_fp8(block, device=device)
        else:
            # use_int8_here
            quantize_linear_layers_to_int8(block, device=device, min_layer_size=512)

    return quantize_fn
