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


#: The one format a text encoder is ever stored at, whatever the run names.
#:
#: Everything below the resolver is format-agnostic except this route. A
#: declared encoder is in the plan like any other target and `format_for` gives
#: it the run's format, but the meta and broadcast load paths this route goes
#: through have only ever carried FP8, so that is what it places. Generalising
#: it would run encoder formats nothing has measured, so it is stated here and
#: said out loud at load time rather than looking format-agnostic and quietly
#: substituting. See `plan_text_encoders`.
TEXT_ENCODER_FORMAT = "fp8"


def plan_text_encoders(loader, existing_quantization_config=None):
    """Plan every declared text encoder, returning pipeline kwargs and a quantization config.

    Every encoder is stored at ``TEXT_ENCODER_FORMAT`` regardless of what the
    run asked the transformer for, and says so in its log line when the two
    differ. The framework-level config is assembled last and only if some
    encoder needs one, because constructing it registers the Transformers
    quantizer process-globally and the meta paths have no use for that.
    """
    model = loader.model
    ledger = loader.quantization_ledger
    replicated_meta = loader.replicated_broadcast_load()
    fsdp_meta = False if replicated_meta else loader.fsdp_meta_load()
    adapter = loader.backends.adapter_for(TEXT_ENCODER_FORMAT)

    component_configs = {}
    if adapter is not None:
        from .fp8_backends import prepare_text_encoder_fp8_load

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
            message = prepared.descriptor.log_message()
            asked = plan.format_for(f"{component_name}.{targets[0]}".rstrip("."))
            if asked not in (None, TEXT_ENCODER_FORMAT):
                # Said out loud rather than substituted quietly: the run named
                # one format and this component is getting another.
                message += (
                    f" (text encoders are stored at {TEXT_ENCODER_FORMAT}; "
                    f"--gemm_quantization asked for {asked} here)"
                )
            log(message)
            # A declared encoder is part of the plan the format-agnostic walks
            # iterate, so it is recorded under the format this route actually
            # placed and they leave it where this route put it.
            ledger.describe(component_name, fp8=True)
            if prepared.descriptor.materialization_mode == "streaming":
                ledger.record_streamed(component_name, targets, fp8=True)
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
