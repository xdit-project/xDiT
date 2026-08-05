"""Loader configs for the AITER block-scale FP8 path, which quantizes weights as they load.

AITER is the only FP8 backend here with a quantize-on-load hook, so it is the only one that can keep
a full bf16 copy from ever materializing. Every other backend (torchao FP8, FP4) quantizes after the
weights are already on the GPU, which is why these configs have no equivalent elsewhere and why the
callers treat "no config" as "load normally, quantize later".

The runner side decides *whether* to quantize and *what* (see runner_models.loading.fp8_plan); this module
only knows how to express that decision to diffusers and transformers. It is imported lazily from
those call sites: reaching it means AITER is active, so the quantizer classes are needed anyway.
"""

from typing import List, Optional

from xfuser.model_executor.quant.aiter_fp8_quantizer import (
    AiterFp8BlockScaleConfig,
    AiterFp8BlockScaleTEConfig,
)


def stream_config(targets: List[str]) -> Optional[AiterFp8BlockScaleConfig]:
    """Config for streaming FP8 quantize-on-load of a transformer, or None when nothing is targeted.

    Passed to the transformer's from_pretrained, this quantizes each weight as it streams off disk,
    so the full bf16 transformer never materializes (peak ~= one weight + accumulating fp8) — cheaper
    than loading bf16 and quantizing in the post-load walk. ``targets`` are relative to the
    transformer (the pipe-level prefix already stripped), so the later post-load AITER walk is a safe
    no-op: those leaves are fp8 layers, not nn.Linear.

    Applies to both single-GPU and FSDP; the streamed fp8 module is what FSDP shards, so the
    per-block quantize_fn is a no-op on those leaves.
    """
    if not targets:
        return None
    return AiterFp8BlockScaleConfig(target_modules=list(targets))


def te_pipeline_config(entries: List[str]):
    """PipelineQuantizationConfig routing the streaming quantizer to the pipeline's text encoders,
    or None when nothing is targeted.

    A text encoder is a transformers model loaded by the diffusers pipeline, so it is quantized by
    component rather than by a single config. Streaming it to fp8 (instead of loading full bf16 then
    quantizing post-load) is the load-time host-RAM win on multi-GPU FSDP, where every node-local
    rank would otherwise hold a full bf16 copy.

    ``entries`` are pipe-level paths ("text_encoder.encoder.block"); they are grouped by leading
    component because that is the shape PipelineQuantizationConfig's quant_mapping takes, keyed by
    whatever the pipeline names each component. The DiT is not handled here; it streams via
    ``stream_config``.
    """
    component_targets: dict[str, list[str]] = {}
    for entry in entries or []:
        component, _, rest = entry.partition(".")
        if not rest:
            continue
        component_targets.setdefault(component, []).append(rest)
    if not component_targets:
        return None
    from diffusers.quantizers import PipelineQuantizationConfig
    return PipelineQuantizationConfig(
        quant_mapping={
            component: AiterFp8BlockScaleTEConfig(target_modules=targets)
            for component, targets in component_targets.items()
        },
    )
