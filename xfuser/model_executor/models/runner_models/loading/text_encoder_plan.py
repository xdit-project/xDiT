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
    """The encoders with declared targets, in declaration order, no repeats.

    The declaration rather than the run's plan: this lists the encoders to
    plan for, which is the same set whether or not the run quantizes them.
    """

    targets = model.settings.gemm_targets
    entries = targets.text_encoder.roots() if targets is not None else ()
    return tuple(
        dict.fromkeys(entry.partition(".")[0] for entry in entries if "." in entry)
    )




def plan_text_encoders(loader, existing_quantization_config=None):
    """Plan every declared text encoder, returning pipeline kwargs and a quantization config.

    Each encoder takes the format the plan gives its targets, which for a
    run whose format the encoder's loaders cannot follow is a pinned one.
    The framework-level config is assembled last and only if some
    encoder needs one, because constructing it registers the Transformers
    quantizer process-globally and the meta paths have no use for that.
    """
    model = loader.model
    ledger = loader.quantization_ledger
    replicated_meta = loader.replicated_broadcast_load()
    fsdp_meta = False if replicated_meta else loader.fsdp_meta_load()

    component_configs = {}
    # Resolved on first use rather than up front: asking up front asks this
    # machine to store a format a run that quantizes nothing never named.
    adapter = None
    encoder_format = None

    for component_name in _declared_components(model):
        # The declared encoder targets, not the targets of whatever format
        # the run named: this route places one format and the plan's answer
        # for these paths may be a different one.
        plan = loader.quantization_plan.gemm_plan
        targets = (
            plan.relative_to(
                component_name, plan.declared_roots(component="text_encoder")
            )
            if plan is not None
            else ()
        )
        if not targets:
            continue
        if adapter is None:
            from .backends import prepare_text_encoder_fp8_load

            # The plan's answer, not this route's own. The plan already knows
            # the encoder's loaders carry one format and has pinned it, so
            # asking is enough -- and every other walk asks the same way, which
            # is what stops them from disagreeing.
            encoder_format = loader.quantization_plan.component_format(
                component_name
            )
            adapter = loader.backends.adapter_for(encoder_format)
            if adapter is None:
                break
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
        # A meta component only carries the layout a converter that packs
        # in place would produce. Anything that swaps in a tensor subclass
        # has to convert after the broadcast, which the memory-efficient
        # FSDP path refuses -- so it does not claim quantize-on-load.
        stream_quant = (
            not (replicated_meta or fsdp_meta)
            or adapter.meta_layout_matches_storage
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
        # No "but the run asked for something else" note any more: nothing
        # is substituted here, because the plan decided it and said so on its
        # own line.
        log(prepared.descriptor.log_message())
        # A declared encoder is part of the plan the format-agnostic walks
        # iterate, so it is recorded under the format the plan gave it and
        # they leave it where this route put it.
        ledger.describe(component_name, format_name=encoder_format)
        if prepared.descriptor.materialization_mode == "streaming":
            ledger.record_streamed(
                component_name, targets, format_name=encoder_format
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
