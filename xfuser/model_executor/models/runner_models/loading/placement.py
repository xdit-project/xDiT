"""Place a loaded pipeline on the local device, quantizing whatever the load left in bf16.

The non-sharded counterpart to ``shard``: one rank owns the whole pipeline, so placement is a
``pipe.to`` rather than a collective. Components the load already quantized are skipped here, which
is what makes the walks below no-ops on the streaming paths rather than a second quantization.

Two of the walks differ only in which side of the device move they run on. AITER rewrites a module
layer-by-layer while it is still on the host, so it has to convert before the move or it needs the
bf16 module resident on the device first; torchao swaps in tensor subclasses that expect their final
device, so it converts after.
"""

from xfuser.core.distributed import get_world_group
from xfuser.core.utils.runner_utils import log, rgetattr
from xfuser.envs import _is_cuda
from .format_backends import (
    module_path_is_covered,
    module_paths_overlap,
    prepare_native_transformer_format_load,
)


def conversion_filter(module_path, excluded_paths, include_suffixes=None):
    """Decide whether to convert ``module_path``, and how to filter inside it.

    Returns ``(False, None)`` when an excluded path already covers the module, ``(True, None)`` when
    the whole module converts, and ``(True, filter_fn)`` when only part of it does because an
    excluded path sits below it or the run pins conversion to certain suffixes.
    """

    overlapping = tuple(
        path
        for path in excluded_paths
        if module_paths_overlap(module_path, path)
    )
    if any(
        module_path_is_covered(module_path, path)
        for path in overlapping
    ):
        return False, None
    descendants = tuple(
        path
        for path in overlapping
        if module_path_is_covered(path, module_path)
    )
    if not descendants and not include_suffixes:
        return True, None

    def filter_fn(_module, fqn):
        full_path = module_path if not fqn else f"{module_path}.{fqn}"
        if include_suffixes and not full_path.endswith(tuple(include_suffixes)):
            return False
        return not any(
            module_path_is_covered(full_path, path)
            for path in descendants
        )

    return True, filter_fn


def place_pipeline_components(loader) -> None:
    """Fill, quantize and move an unsharded pipeline to this rank's device."""

    model = loader.model
    local_rank = get_world_group().local_rank
    offload_requested = (
        model.config.enable_model_cpu_offload
        or model.config.enable_sequential_cpu_offload
        or model.config.enable_group_cpu_offload
    )

    loader.fill_eager_transformers()
    # Rank 0's real bf16 weights land on the peers' meta components over GPU->GPU and are
    # quantized per component in place, so VRAM holds one bf16 component rather than the pipeline.
    if loader.replicated_broadcast_load():
        loader.broadcast_fill_replicated(offload_requested)

    adapter = loader.backends.fp8
    if adapter is not None and adapter.converts_before_device_move:
        _convert_fp8_on_host(loader, adapter, local_rank, offload_requested)

    if not offload_requested:
        model.pipe = model.pipe.to(f"cuda:{local_rank}")

    if model.config.use_fp4_gemms:
        if _is_cuda():
            setup_nvfp4_gemms(loader, local_rank)
        else:
            setup_mxfp4_gemms(loader, local_rank)

    # FP4 setup owns its own hybrid FP8 path and any declared FP8-only modules, so the generic walk
    # would re-quantize inside the hybrid wrappers it just built.
    if (
        adapter is not None
        and not adapter.converts_before_device_move
        and not model.config.use_fp4_gemms
    ):
        _convert_fp8_on_device(loader, adapter, local_rank)

    if model.config.use_int8_gemms:
        _convert_int8_on_device(loader, local_rank)


def setup_mxfp4_gemms(loader, local_rank) -> None:
    """Quantize the FP4 modules to MXFP4, the format ROCm runs."""
    _setup_fp4_gemms(loader, local_rank, stream_quant=True)


def setup_nvfp4_gemms(loader, local_rank) -> None:
    """Quantize the FP4 modules to NVFP4, the format CUDA runs."""
    _setup_fp4_gemms(loader, local_rank, stream_quant=False)


def _setup_fp4_gemms(loader, local_rank, *, stream_quant) -> None:
    model = loader.model
    adapter = loader.backends.format
    for module_name in model.settings.fp4_gemm_module_list:
        component_name = module_name.partition(".")[0]
        convert, filter_fn = conversion_filter(
            module_name,
            loader.quantization_ledger.streaming_targets,
        )
        if not convert:
            continue
        # Some models balance performance against quality better by keeping some blocks at FP8
        # while the rest go to FP4, rather than quantizing uniformly.
        if loader.quantization_ledger.claim_description(component_name):
            descriptor = prepare_native_transformer_format_load(
                adapter,
                component_name=component_name,
                targets=loader.backends.format_targets_for(component_name),
                stream_quant=stream_quant,
                precision_prefixes=(model.settings.fp8_precision_overrides or ()),
                precision_suffixes=(
                    model.settings.fp8_precision_override_suffixes or ()
                ),
                hybrid=model.config.use_hybrid_gemm_schedule,
            ).descriptor
            log(descriptor.log_message())
        convert_kwargs = {}
        if filter_fn is not None:
            convert_kwargs["filter_fn"] = filter_fn
        adapter.convert_module(
            rgetattr(model.pipe, module_name),
            fp8_layers=model.settings.fp8_precision_overrides,
            fp8_suffix_layers=model.settings.fp8_precision_override_suffixes,
            hybrid=model.config.use_hybrid_gemm_schedule,
            device=f"cuda:{local_rank}",
            **convert_kwargs,
        )
    setup_fp8_only_gemm_modules(loader, local_rank)


def setup_fp8_only_gemm_modules(loader, local_rank) -> None:
    """Quantize to FP8 any module the run names for FP8 but not for FP4.

    MoE models such as Wan2.2 rely on this: the low-noise transformer generates the fine detail and
    needs FP8's precision, while the rest of the model can take FP4.
    """

    model = loader.model
    fp4_modules = set(loader.quantization_plan.module_list("fp4"))
    fp8_only_modules = [
        name
        for name in loader.quantization_plan.module_list()
        if not any(
            module_path_is_covered(name, fp4_module)
            for fp4_module in fp4_modules
        )
    ]
    if not fp8_only_modules:
        return
    adapter = loader.backends.blockwise_fp8
    for module_name in fp8_only_modules:
        excluded_paths = fp4_modules | loader.quantization_ledger.already_quantized(
            fp8=True
        )
        convert, filter_fn = conversion_filter(
            module_name,
            excluded_paths,
            include_suffixes=model.settings.fp8_gemm_include_suffixes,
        )
        if not convert:
            continue
        log(f"Quantizing linear layers in {module_name} to FP8...")
        convert_kwargs = {}
        if filter_fn is not None:
            convert_kwargs["filter_fn"] = filter_fn
        adapter.convert_module(
            rgetattr(model.pipe, module_name),
            device=f"cuda:{local_rank}",
            **convert_kwargs,
        )


def _convert_fp8_on_host(loader, adapter, local_rank, offload_requested) -> None:
    model = loader.model
    for module_name in loader.quantization_plan.module_list():
        convert, filter_fn = conversion_filter(
            module_name,
            loader.quantization_ledger.fp8_streaming_targets,
            include_suffixes=model.settings.fp8_gemm_include_suffixes,
        )
        if not convert:
            continue
        convert_kwargs = {}
        if filter_fn is not None:
            convert_kwargs["filter_fn"] = filter_fn
        replaced = adapter.convert_module(
            rgetattr(model.pipe, module_name),
            device=f"cuda:{local_rank}",
            offload_to_cpu=offload_requested,
            **convert_kwargs,
        )
        if replaced:
            log(
                f"Quantized {replaced} layers in {module_name} "
                f"to FP8 ({adapter.storage_semantics})."
            )
        else:
            log(
                f"{module_name} already FP8 (streamed quantize-on-load); "
                "post-load walk no-op."
            )


def _convert_fp8_on_device(loader, adapter, local_rank) -> None:
    model = loader.model
    for module_name in loader.quantization_plan.module_list():
        component_name = module_name.partition(".")[0]
        convert, filter_fn = conversion_filter(
            module_name,
            loader.quantization_ledger.fp8_streaming_targets,
            include_suffixes=model.settings.fp8_gemm_include_suffixes,
        )
        if not convert:
            continue
        if component_name.startswith(
            "transformer"
        ) and loader.quantization_ledger.claim_description(component_name, fp8=True):
            log(
                "Transformer quantization: requested=fp8, "
                f"backend={adapter.backend.value}, "
                f"storage={adapter.storage_semantics}, "
                "materialization=post_load; fallback=runner did "
                "not use the transformer construction seam"
            )
        convert_kwargs = {}
        if filter_fn is not None:
            convert_kwargs["filter_fn"] = filter_fn
        adapter.convert_module(
            rgetattr(model.pipe, module_name),
            device=f"cuda:{local_rank}",
            **convert_kwargs,
        )


def _convert_int8_on_device(loader, local_rank) -> None:
    model = loader.model
    adapter = loader.backends.format
    for module_name in model.settings.int8_gemm_module_list:
        component_name = module_name.partition(".")[0]
        convert, filter_fn = conversion_filter(
            module_name,
            loader.quantization_ledger.streaming_targets,
        )
        if not convert:
            continue
        if loader.quantization_ledger.claim_description(component_name):
            descriptor = prepare_native_transformer_format_load(
                adapter,
                component_name=component_name,
                targets=loader.backends.format_targets_for(component_name),
                stream_quant=False,
            ).descriptor
            log(descriptor.log_message())
        convert_kwargs = {}
        if filter_fn is not None:
            convert_kwargs["filter_fn"] = filter_fn
        adapter.convert_module(
            rgetattr(model.pipe, module_name),
            device=f"cuda:{local_rank}",
            **convert_kwargs,
        )
