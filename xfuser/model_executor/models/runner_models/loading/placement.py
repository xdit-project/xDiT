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


def place_pipeline_components(model) -> None:
    """Fill, quantize and move an unsharded pipeline to this rank's device."""

    local_rank = get_world_group().local_rank
    offload_requested = (
        model.config.enable_model_cpu_offload
        or model.config.enable_sequential_cpu_offload
        or model.config.enable_group_cpu_offload
    )

    fill_eager = getattr(
        getattr(model, "_loader", None),
        "fill_eager_transformers",
        None,
    )
    if fill_eager is not None:
        fill_eager()
    # Rank 0's real bf16 weights land on the peers' meta components over GPU->GPU and are
    # quantized per component in place, so VRAM holds one bf16 component rather than the pipeline.
    if model._replicated_broadcast_load():
        model._loader.broadcast_fill_replicated(offload_requested)

    adapter = model.fp8_backend
    if adapter is not None and adapter.converts_before_device_move:
        _convert_fp8_on_host(model, adapter, local_rank, offload_requested)

    if not offload_requested:
        model.pipe = model.pipe.to(f"cuda:{local_rank}")

    if model.config.use_fp4_gemms:
        if _is_cuda():
            model._setup_nvfp4_gemms(local_rank=local_rank)
        else:
            model._setup_mxfp4_gemms(local_rank=local_rank)

    # FP4 setup owns its own hybrid FP8 path and any declared FP8-only modules, so the generic walk
    # would re-quantize inside the hybrid wrappers it just built.
    if (
        adapter is not None
        and not adapter.converts_before_device_move
        and not model.config.use_fp4_gemms
    ):
        _convert_fp8_on_device(model, adapter, local_rank)

    if model.config.use_int8_gemms:
        _convert_int8_on_device(model, local_rank)


def _convert_fp8_on_host(model, adapter, local_rank, offload_requested) -> None:
    for module_name in model.fp8.module_list():
        convert, filter_fn = conversion_filter(
            module_name,
            model.quantization_ledger.fp8_streaming_targets,
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


def _convert_fp8_on_device(model, adapter, local_rank) -> None:
    for module_name in model.fp8.module_list():
        component_name = module_name.partition(".")[0]
        convert, filter_fn = conversion_filter(
            module_name,
            model.quantization_ledger.fp8_streaming_targets,
            include_suffixes=model.settings.fp8_gemm_include_suffixes,
        )
        if not convert:
            continue
        if component_name.startswith(
            "transformer"
        ) and model.quantization_ledger.claim_description(component_name, fp8=True):
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


def _convert_int8_on_device(model, local_rank) -> None:
    adapter = model.format_backend
    for module_name in model.settings.int8_gemm_module_list:
        component_name = module_name.partition(".")[0]
        convert, filter_fn = conversion_filter(
            module_name,
            model.quantization_ledger.streaming_targets,
        )
        if not convert:
            continue
        if model.quantization_ledger.claim_description(component_name):
            descriptor = prepare_native_transformer_format_load(
                adapter,
                component_name=component_name,
                targets=model.backends.format_targets_for(component_name),
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
