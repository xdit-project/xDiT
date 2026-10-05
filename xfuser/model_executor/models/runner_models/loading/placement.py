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
from .quant_adapter import descriptor_for, module_path_is_covered


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

    # One walk per side of the device move; the plan says what each module
    # becomes and the adapter says which side it belongs on.
    setup_gemm_quantization(
        loader,
        local_rank,
        offload_requested=offload_requested,
        before_device_move=True,
    )
    if not offload_requested:
        model.pipe = model.pipe.to(f"cuda:{local_rank}")
    setup_gemm_quantization(
        loader,
        local_rank,
        offload_requested=offload_requested,
        before_device_move=False,
    )


def _plan_conversion_filter(plan, module_path, format_name, already_quantized):
    """Keep the leaves under ``module_path`` the plan gives to ``format_name``.

    ``fqn`` arrives relative to the module being converted, so it is rejoined
    to that module's path before the plan is asked -- the plan speaks in full
    pipeline paths.
    """

    def filter_fn(_module, fqn):
        path = f"{module_path}.{fqn}" if fqn else module_path
        if any(module_path_is_covered(path, owner) for owner in already_quantized):
            return False
        return plan.format_for(path) == format_name

    return filter_fn


def setup_gemm_quantization(
    loader, local_rank, *, offload_requested, before_device_move
) -> None:
    """Quantize every declared target to the format the plan gives it.

    One walk per format over the declared subtrees. Which converter owns a leaf
    follows from its format, and which side of the device move a walk runs on
    follows from its converter, so neither is a branch here. Called once before
    the move and once after; each call skips the adapters belonging to the other
    side.

Each walk starts where ``walk_roots`` says -- the subtrees that format owns,
    widened only when it has a carve-out with no subtree of its own -- and asks
    ``format_for`` per leaf, which is the rule the sharded path applies too. A
    leaf belongs to exactly one format, so the walks cannot collide.
    """

    plan = loader.quantization_plan.gemm_plan
    if plan is None or not plan.quantizes:
        return

    model = loader.model
    ledger = loader.quantization_ledger
    formats = plan.formats_in_play

    for format_name in formats:
        roots = plan.walk_roots(format_name)
        adapter = loader.backends.adapter_for(format_name)
        if adapter is None:
            continue
        converts_first = bool(getattr(adapter, "converts_before_device_move", False))
        if converts_first != before_device_move:
            continue

        for module_name in roots:
            already = ledger.already_quantized()
            if any(module_path_is_covered(module_name, owner) for owner in already):
                continue

            convert_kwargs = {
                "device": f"cuda:{local_rank}",
                "filter_fn": _plan_conversion_filter(
                    plan, module_name, format_name, already
                ),
            }
            if before_device_move and offload_requested:
                convert_kwargs["offload_to_cpu"] = True
            # Claimed per (component, format), so a component whose descriptor
            # the load path already logged is not logged twice and each format
            # in play still gets a line of its own.
            component_name = module_name.partition(".")[0]
            if ledger.claim_description(component_name, format_name=format_name):
                # This walk *is* the post-load conversion, so the descriptor
                # says so outright. It used to be derived by asking
                # `prepare_native_load` what a streamed load would have done,
                # with `stream_quant` keyed off `_is_cuda()` -- a hardware test
                # in the one walk that has none, and one that let the line
                # claim `materialization=streaming` for a conversion happening
                # right here.
                descriptor = descriptor_for(
                    adapter,
                    component_name,
                    "post_load",
                    "converted after the load rather than on the way in",
                )
                log(descriptor.log_message())
            companion = loader.backends.hybrid_companion(
                format_name, device=convert_kwargs["device"]
            )
            if companion is not None:
                convert_kwargs["companion"] = companion

            adapter.convert_module(
                rgetattr(model.pipe, module_name), **convert_kwargs
            )


