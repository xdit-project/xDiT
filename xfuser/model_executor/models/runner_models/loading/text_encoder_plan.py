"""Deciding how each text encoder is quantized and materialized.

``plan_text_encoders`` answers one question per encoder the runner declares FP8 targets for: is
it quantized on the way in from disk, streamed by the framework's own quantizer, or converted
after it lands. The answer follows from the materialization the run asked for, which is why the
decision sits beside the loader that made it.

It returns ``(pipe_component_kwargs, te_quant_config)``. On a meta path the kwargs carry meta
modules for the pipeline's ``from_pretrained`` to adopt instead of loading those components, and
the config is None: the meta module, FP8 where targeted and bf16 otherwise, is filled by the
sharded or broadcast load and then wrapped. The transformer is unaffected either way and keeps
its own route.
"""

from xfuser.core.utils.runner_utils import log
from .blockwise_ownership import (
    blockwise_transformer_descriptor,
    record_blockwise_ownership,
)


def _declared_components(model):
    """The encoders with FP8 targets, in declaration order and without repeats."""

    entries = model.settings.fp8_text_encoder_module_list or ()
    return tuple(
        dict.fromkeys(entry.partition(".")[0] for entry in entries if "." in entry)
    )


def plan_text_encoders(loader, existing_quantization_config=None):
    """Plan every declared text encoder, returning pipeline kwargs and a quantization config.

    The framework-level config is assembled last and only if some encoder needs one, because
    constructing it registers the Transformers quantizer process-globally and the meta paths
    have no use for that.
    """
    model = loader.model
    ledger = loader.quantization_ledger
    replicated_meta = loader.replicated_broadcast_load()
    fsdp_meta = False if replicated_meta else loader.fsdp_meta_load()
    adapter = loader.backends.fp8_adapter_for_contract()

    component_configs = {}
    if adapter is not None:
        from .fp8_backends import prepare_text_encoder_fp8_load

        for component_name in _declared_components(model):
            targets = tuple(loader.quantization_plan.targets_for(component_name))
            if not targets:
                continue
            # A blockwise-filled encoder is quantized per block on the way in from disk, before
            # FSDP wraps that block, so it needs neither a streaming config nor a post-load walk.
            # That is what lets TorchAO quantize a text encoder on the FSDP meta path at all: the
            # objection below is to converting a layout after wrapping, which this never does.
            if fsdp_meta and loader.will_fill_blockwise(component_name):
                wrap_attrs = tuple(
                    model.settings.fsdp_strategy.get(component_name, {}).get(
                        "wrap_attrs", ()
                    )
                )
                # Same bookkeeping the transformer's blockwise route records, so the post-load
                # walk knows these targets were already quantized during the fill.
                record_blockwise_ownership(
                    ledger,
                    adapter,
                    component_name,
                    targets,
                    wrap_attrs,
                    blockwise_transformer_descriptor(
                        adapter, component_name, targets, wrap_attrs
                    ),
                )
                continue
            # Existing meta layouts only mirror AITER's plain fp8+scale representation.
            # Replicated TorchAO falls back after broadcast; memory-efficient FSDP rejects
            # that layout-changing fallback.
            stream_quant = not (replicated_meta or fsdp_meta) or (
                adapter.backend.value == "aiter"
            )
            prepared = prepare_text_encoder_fp8_load(
                adapter,
                component_name=component_name,
                targets=targets,
                stream_quant=stream_quant,
                supports_post_load=not fsdp_meta,
                model_factory=lambda name=component_name: (
                    loader.build_meta_component(name, fp8=False)
                ),
            )
            log(prepared.descriptor.log_message())
            # Only the FP8 half: a text encoder never appears in the FP4 or INT8 module lists
            # the format-agnostic walks iterate.
            ledger.describe(component_name, fp8=True, any_format=False)
            if prepared.descriptor.materialization_mode == "streaming":
                ledger.record_streamed(
                    component_name, targets, fp8=True, any_format=False
                )
            if prepared.quantization_config is not None:
                component_configs[component_name] = prepared.quantization_config

    model._text_encoder_quantization_configs = dict(component_configs)

    pipeline_config = existing_quantization_config
    if component_configs:
        from .text_encoder_adapter import TextEncoderFrameworkAdapter

        pipeline_config = TextEncoderFrameworkAdapter().pipeline_quantization_config(
            component_configs, existing=existing_quantization_config
        )

    if replicated_meta:
        return loader.meta_te_kwargs_replicated(pipeline_config)
    if fsdp_meta:
        meta_kwargs = loader.meta_te_kwargs()
        if meta_kwargs is not None:
            return meta_kwargs
    return {}, pipeline_config
