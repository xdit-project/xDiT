"""Routing a transformer's weights onto the device.

One entry point, ``load_transformer``, which picks between the meta paths this package
implements and an ordinary ``from_pretrained``, and prepares whatever quantization the chosen
route needs. A runner asks for a transformer and gets one, knowing nothing about how the
weights arrive.

The order of the checks matters. A sharded or replicated meta build wins over native
quantize-on-load, because those paths quantize per block during the fill and a native
quantization config would ask ``from_pretrained`` to do it again. Below them sits the local
blockwise fallback, for a single-rank load whose checkpoint the standard loader cannot stream.
"""

from contextlib import ExitStack

import torch

from xfuser.core.distributed.parallel_state import get_world_group
from xfuser.core.utils.runner_utils import log
from .blockwise_ownership import (
    blockwise_transformer_descriptor,
    record_blockwise_ownership,
)
from .checkpoint import CheckpointManifest, CheckpointRequest
from .contracts import UnsupportedLoadContract


def build_transformer_structure(wrapper_cls, request: CheckpointRequest, init_kwargs):
    """Build only meta tensors so FP8 target prefixes can be mapped safely."""

    from accelerate import init_empty_weights

    config = wrapper_cls.load_config(
        request.model_name_or_path, **request.config_kwargs()
    )
    with ExitStack() as stack:
        try:
            stack.enter_context(init_empty_weights(include_buffers=True))
        except TypeError as exc:
            raise RuntimeError(
                "accelerate.init_empty_weights(include_buffers=True) is "
                "required for bounded structure inspection"
            ) from exc
        return wrapper_cls.from_config(config, **(init_kwargs or {}))


def native_quantization_device_map(model, adapter):
    """Choose where a native quantizer leaves each weight after conversion.

    AITER FP8 still performs the conversion on the local accelerator, but its quantizer uses the
    device map as the final placement and immediately evicts each converted leaf when that target
    is CPU.  This keeps only one transient bf16/FP8 layer on the accelerator while a pipeline that
    will receive CPU-offload hooks is loading.  Other native quantizers retain their accelerator
    placement requirements.
    """

    cpu_offload = any(
        getattr(model.config, flag, False)
        for flag in (
            "enable_model_cpu_offload",
            "enable_sequential_cpu_offload",
            "enable_group_cpu_offload",
        )
    )
    if (
        adapter.format.value == "fp8"
        and adapter.backend.value == "aiter"
        and cpu_offload
    ):
        return {"": "cpu"}
    return {"": get_world_group().local_rank}


def _resolve_request(loader, subfolder, checkpoint_request):
    request = checkpoint_request or loader.checkpoint_request(
        subfolder or "transformer"
    )
    if subfolder is not None and request.subfolder != subfolder:
        return request.with_subfolder(subfolder)
    if request.subfolder is None:
        return request.with_subfolder("transformer")
    return request


def _prepare_native_load(
    model, adapter, component_name, targets, stream_quant, model_factory
):
    """The quantization config an ordinary ``from_pretrained`` should carry, if any."""

    if adapter.format.value == "fp8":
        from .fp8_backends import prepare_native_transformer_fp8_load

        return prepare_native_transformer_fp8_load(
            adapter,
            component_name=component_name,
            targets=targets,
            # Native FP8 configs quantize every linear under each target. A suffix-restricted
            # policy must remain a post-load/blockwise conversion so the ledger does not claim
            # broader coverage than was requested.
            stream_quant=(
                stream_quant
                and not getattr(
                    model.settings, "fp8_gemm_include_suffixes", None
                )
            ),
            model_factory=model_factory,
        )
    from .format_backends import prepare_native_transformer_format_load

    is_fp4 = adapter.format.value in {"fp4", "fp8_fp4"}
    return prepare_native_transformer_format_load(
        adapter,
        component_name=component_name,
        targets=targets,
        stream_quant=stream_quant,
        precision_prefixes=(
            (model.settings.fp8_precision_overrides or ()) if is_fp4 else ()
        ),
        precision_suffixes=(
            (model.settings.fp8_precision_override_suffixes or ())
            if is_fp4
            else ()
        ),
        hybrid=(model.config.use_hybrid_gemm_schedule if is_fp4 else False),
        model_factory=model_factory,
    )


def _fp4_remainder(loader, component_name):
    """An FP4 blockwise fill also owns the FP8 remainder, whose targets it must be told."""

    if not loader.model.config.use_fp4_gemms:
        return {}
    return {
        "fp4_gemms": True,
        "fp8_targets": tuple(
            loader.quantization_plan.targets_for(component_name)
        ),
    }


def _record_native_quantization(ledger, adapter, component_name, prepared, targets):
    """Note what the native config will have quantized, so the post-load walk skips it."""

    log(prepared.descriptor.log_message())
    is_fp8 = adapter.format.value == "fp8"
    ledger.describe(component_name, fp8=is_fp8)
    if prepared.quantization_config is None:
        return
    # Only a format load narrows the streamed set; an FP8 one has no such field and owns
    # everything it was given.
    streamed = getattr(prepared, "streamed_targets", ()) or targets
    ledger.record_streamed(component_name, streamed, fp8=is_fp8)


def load_transformer(
    loader,
    wrapper_cls,
    subfolder: str | None = None,
    init_kwargs: dict | None = None,
    stream_quant: bool = True,
    checkpoint_request: CheckpointRequest | None = None,
    weight_source: CheckpointManifest | None = None,
):
    """Load a transformer through whichever materialization the run asked for.

    The meta paths keep xDiT's own build and blockwise filler. An ordinary load is handed a
    native Diffusers quantization config only where the format's exact semantics permit it,
    which is what ``stream_quant`` gates; the meta paths ignore it, since they always convert
    targeted blocks through the backend adapter.

    ``init_kwargs`` are extra wrapper ``__init__`` args (wan's attention kwargs, for one) and
    are forwarded on every route. ``weight_source`` names a resolved manifest for a checkpoint
    whose keys the standard loader cannot map, and is only honoured on a blockwise route.
    """
    model = loader.model
    ledger = loader.quantization_ledger
    request = _resolve_request(loader, subfolder, checkpoint_request)
    component_name = request.subfolder
    adapter, targets = loader.transformer_quantization_adapter(component_name)
    targets = tuple(targets)
    strategy = model.settings.fsdp_strategy.get(component_name, {})
    wrap_attrs = tuple(strategy.get("wrap_attrs", ()))
    build_kwargs = {"weight_source": weight_source} if weight_source is not None else {}

    fsdp_meta = loader.fsdp_meta_load()
    replicated_meta = False if fsdp_meta else loader.replicated_broadcast_load()
    if fsdp_meta or replicated_meta:
        if adapter is not None:
            record_blockwise_ownership(
                ledger,
                adapter,
                component_name,
                targets,
                wrap_attrs,
                blockwise_transformer_descriptor(
                    adapter, component_name, targets, wrap_attrs
                ),
                **_fp4_remainder(loader, component_name),
            )
        return loader.build_meta_transformer(
            wrapper_cls, request, init_kwargs, **build_kwargs
        )

    quantization_config = None
    if adapter is not None:
        prepared = _prepare_native_load(
            model,
            adapter,
            component_name,
            targets,
            stream_quant,
            lambda: build_transformer_structure(wrapper_cls, request, init_kwargs),
        )
        local_plan = loader.plan_eager_blockwise_fallback(prepared, targets, wrap_attrs)
        if local_plan is not None and local_plan.enabled:
            record_blockwise_ownership(
                ledger,
                adapter,
                component_name,
                targets,
                wrap_attrs,
                blockwise_transformer_descriptor(
                    adapter, component_name, targets, wrap_attrs, local=True
                ),
                **_fp4_remainder(loader, component_name),
            )
            component = loader.build_meta_transformer(
                wrapper_cls, request, init_kwargs, **build_kwargs
            )
            loader.mark_local_blockwise(component)
            return component
        if weight_source is not None:
            reason = (
                local_plan.reason
                if local_plan is not None
                else "local blockwise loading is unavailable"
            )
            raise UnsupportedLoadContract(
                f"{component_name} uses a mapped checkpoint source but "
                f"cannot enter local blockwise loading: {reason}"
            )
        _record_native_quantization(
            ledger, adapter, component_name, prepared, targets
        )
        quantization_config = prepared.quantization_config

    load_kwargs = request.from_pretrained_kwargs()
    if quantization_config is not None:
        load_kwargs.setdefault(
            "device_map", native_quantization_device_map(model, adapter)
        )
    return wrapper_cls.from_pretrained(
        request.model_name_or_path,
        torch_dtype=torch.bfloat16,
        quantization_config=quantization_config,
        **load_kwargs,
        **(init_kwargs or {}),
    )
