"""Lazy Diffusers/Transformers boundary for text-encoder loading.

Backend selection and fallback policy live in ``fp8_backends``.  This module
only translates an already-selected backend and exact component targets into
framework configuration objects.
"""

from dataclasses import dataclass
import importlib.util
from typing import Callable, Mapping

_TRANSFORMERS_STREAMING_REQUIREMENT = (
    "transformers 5 (transformers>=5.0) with "
    "transformers.core_model_loading is required "
    "for text-encoder streaming quantize-on-load"
)


@dataclass(frozen=True)
class TransformersStreamingSupport:
    available: bool
    reason: str | None = None


def transformers_streaming_requirement() -> str:
    return _TRANSFORMERS_STREAMING_REQUIREMENT


def probe_transformers_streaming_loader(
    *,
    find_spec: Callable[[str], object | None] = importlib.util.find_spec,
) -> TransformersStreamingSupport:
    """Feature-detect the Transformers parameter conversion-op loader."""

    try:
        available = find_spec("transformers.core_model_loading") is not None
    except (ImportError, ModuleNotFoundError, ValueError) as exc:
        return TransformersStreamingSupport(
            False,
            f"{_TRANSFORMERS_STREAMING_REQUIREMENT}: " f"{type(exc).__name__}: {exc}",
        )
    return TransformersStreamingSupport(
        available,
        None if available else _TRANSFORMERS_STREAMING_REQUIREMENT,
    )


def _torchao_quant_type():
    from torchao.quantization.granularity import PerTensor
    from torchao.quantization.quant_api import (
        Float8DynamicActivationFloat8WeightConfig,
    )
    from xfuser.core.utils.runner_utils import (
        FP8_ACTIVATION_SCALE_FLOOR,
        _get_fp8_kernel_preference,
    )

    return Float8DynamicActivationFloat8WeightConfig(
        granularity=PerTensor(),
        set_inductor_config=False,
        kernel_preference=_get_fp8_kernel_preference(),
        activation_value_lb=FP8_ACTIVATION_SCALE_FLOOR,
    )


def _torchao_text_encoder_config(exclusions):
    from transformers import TorchAoConfig

    return TorchAoConfig(
        _torchao_quant_type(),
        modules_to_not_convert=list(exclusions),
    )


def _aiter_text_encoder_config(targets):
    support = probe_transformers_streaming_loader()
    if not support.available:
        raise RuntimeError(support.reason)
    from xfuser.model_executor.quant import AiterFp8BlockScaleTEConfig

    return AiterFp8BlockScaleTEConfig(target_modules=list(targets))


def _pipeline_config(mapping):
    from diffusers.quantizers import PipelineQuantizationConfig

    return PipelineQuantizationConfig(quant_mapping=dict(mapping))


class TextEncoderFrameworkAdapter:
    """Express selected text-encoder quantization through framework APIs."""

    def __init__(
        self,
        *,
        pipeline_config_factory=None,
        torchao_config_factory=None,
        aiter_config_factory=None,
    ) -> None:
        self._pipeline_config_factory = pipeline_config_factory or _pipeline_config
        self._torchao_config_factory = (
            torchao_config_factory or _torchao_text_encoder_config
        )
        self._aiter_config_factory = aiter_config_factory or _aiter_text_encoder_config

    def component_quantization_config(
        self,
        *,
        backend: str,
        targets,
        exclusions=(),
    ):
        """Build one Transformers component config from selected policy."""

        if backend == "torchao":
            return self._torchao_config_factory(list(exclusions))
        if backend == "aiter":
            return self._aiter_config_factory(list(targets))
        raise ValueError(f"unsupported text-encoder FP8 backend: {backend}")

    def pipeline_quantization_config(
        self,
        component_configs: Mapping[str, object],
        *,
        existing=None,
    ):
        """Create a granular pipeline mapping without replacing prior entries."""

        additions = dict(component_configs)
        if not additions:
            return existing
        mapping = {}
        if existing is not None:
            prior = getattr(existing, "quant_mapping", None)
            if prior is None:
                raise ValueError(
                    "existing pipeline quantization config is not granular; "
                    "cannot merge component mappings safely"
                )
            mapping.update(prior)
        overlap = mapping.keys() & additions.keys()
        if overlap:
            names = ", ".join(sorted(overlap))
            raise ValueError(
                "refusing to overwrite existing pipeline quantization "
                f"config for: {names}"
            )
        mapping.update(additions)
        return self._pipeline_config_factory(mapping)


def resolve_transformers_component(
    pipeline_cls,
    component_name: str,
    request,
    *,
    import_module=None,
):
    """Resolve one pipeline component's Transformers class and config."""

    if import_module is None:
        from importlib import import_module

    index = pipeline_cls.load_config(
        request.model_name_or_path,
        **request.config_kwargs(include_subfolder=False),
    )
    entry = index.get(component_name)
    if not (isinstance(entry, (list, tuple)) and len(entry) == 2):
        return None
    library, class_name = entry
    if library != "transformers":
        return None
    component_cls = getattr(import_module(library), class_name)
    component_request = request.with_subfolder(component_name)
    config = component_cls.config_class.from_pretrained(
        request.model_name_or_path,
        **component_request.config_kwargs(),
    )
    return component_cls, config


def _live_tensor_names(component):
    """Every tensor the blockwise fill will ask the checkpoint for, in fill order.

    remove_duplicate=False so tied names each appear, matching what the filler enumerates.
    Non-persistent buffers are excluded: they are recomputed on forward and never on disk.
    """
    names = [name for name, _ in component.named_parameters(remove_duplicate=False)]
    for name, _ in component.named_buffers(remove_duplicate=False):
        parent_name, _, local_name = name.rpartition(".")
        owner = component.get_submodule(parent_name) if parent_name else component
        if local_name not in owner._non_persistent_buffers_set:
            names.append(name)
    return names


def _alias_of(component, names):
    """Map each live name to the first name sharing its tensor object, for tied weights.

    Tied entries are absent from the checkpoint (only the tying target is stored), so they are
    resolved by reading the target's key rather than by a key of their own.
    """
    first_by_id: dict[int, str] = {}
    alias = {}
    for name in names:
        try:
            tensor = component.get_parameter(name)
        except AttributeError:
            tensor = component.get_buffer(name)
        owner = first_by_id.setdefault(id(tensor), name)
        if owner != name:
            alias[name] = owner
    return alias


def _renamed_live_name(checkpoint_key, conversions):
    """Apply the first matching Transformers renaming, as its loader does."""
    for conversion in conversions:
        renamed, matched = conversion.rename_source_key(checkpoint_key)
        if matched is not None:
            return renamed
    return checkpoint_key


def _candidate_live_names(checkpoint_key, conversions, prefix):
    """The live names a checkpoint key could denote, in the order Transformers would try them.

    Renaming runs first, then the base-model prefix, which is added per key rather than uniformly:
    Mistral3 stores `language_model.model.embed_tokens` for `model.language_model.embed_tokens`
    but `language_model.lm_head` for a bare `lm_head`, so one prefix rule cannot cover both.
    """
    renamed = _renamed_live_name(checkpoint_key, conversions)
    dotted = f"{prefix}." if prefix else ""
    candidates = [renamed]
    if dotted:
        candidates.append(dotted + renamed)
        if renamed.startswith(dotted):
            candidates.append(renamed[len(dotted) :])
    seen = {}
    return [seen.setdefault(name, name) for name in candidates if name not in seen]


def _declared_ties(component):
    """Live name -> the live name it is tied to, as the model declares them."""
    expand = getattr(component, "get_expanded_tied_weights_keys", None)
    tied = expand() if callable(expand) else getattr(component, "_tied_weights_keys", None)
    return dict(tied) if isinstance(tied, dict) else {}


def resolve_transformers_manifest(component, request, *, discover=None, conversions=None):
    """Map a meta-built transformers component's live tensor names onto its checkpoint keys.

    The blockwise fill reads one tensor at a time by live name, so it needs the key mapping
    from_pretrained would have applied. Reproducing that mapping in general is not safe: a
    Transformers WeightConverter may fuse or split tensors, so one live tensor can be a function of
    several checkpoint tensors and cannot be read by name at all.

    So the rules come from Transformers rather than from guesswork -- its own registered renamings,
    applied through its own rename_source_key -- and the result is then proven against the data.
    Live names and checkpoint keys are both known before any tensor is read, so a mapping is
    accepted only if it is an exact cover: every checkpoint key denotes one live tensor, and every
    live tensor is either mapped or declared tied to one that is. Anything needing a fuse, a split,
    or a rule this cannot reproduce fails that check and is refused rather than mapped wrong.

    Returns (manifest, None) when the mapping is proven, and (None, reason) when the caller should
    fall back to loading the component through from_pretrained.
    """
    # Imported here, not at module scope: this module is loaded standalone to prove it pulls in no
    # framework at import time.
    from .checkpoint import CheckpointManifest

    if discover is None:
        from .checkpoint import discover_checkpoint as discover

    if conversions is None:
        conversions, refusal = _model_renamings(component)
        if conversions is None:
            return None, refusal
    try:
        discovered = discover(request, basename="model")
    except FileNotFoundError as exc:
        return None, f"no transformers checkpoint to map: {exc}"
    checkpoint_paths = discovered.weight_map
    live_names = _live_tensor_names(component)
    if not live_names:
        return None, "component exposes no tensors to fill"
    live = set(live_names)
    prefix = getattr(component, "base_model_prefix", "") or ""

    mapping: dict[str, str] = {}
    for checkpoint_key in checkpoint_paths:
        hits = [
            name
            for name in _candidate_live_names(checkpoint_key, conversions, prefix)
            if name in live
        ]
        if not hits:
            return None, (
                f"checkpoint key {checkpoint_key!r} matches no tensor in the component, "
                f"so its layout needs more than a renaming"
            )
        if len(hits) > 1:
            return None, (
                f"checkpoint key {checkpoint_key!r} matches several tensors ({', '.join(hits)})"
            )
        claimed = mapping.get(hits[0])
        if claimed is not None:
            return None, (
                f"checkpoint keys {claimed!r} and {checkpoint_key!r} both map to {hits[0]!r}"
            )
        mapping[hits[0]] = checkpoint_key

    for name, target in _declared_ties(component).items():
        if name in live and name not in mapping and target in mapping:
            mapping[name] = mapping[target]
    for name, owner in _alias_of(component, live_names).items():
        if name not in mapping and owner in mapping:
            mapping[name] = mapping[owner]

    unmapped = [name for name in live_names if name not in mapping]
    if unmapped:
        preview = ", ".join(unmapped[:3])
        suffix = f" (+{len(unmapped) - 3} more)" if len(unmapped) > 3 else ""
        return None, (
            f"{len(unmapped)} tensor(s) have no checkpoint key and no declared tie: "
            f"{preview}{suffix}"
        )
    return (
        CheckpointManifest(
            weight_map={name: checkpoint_paths[key] for name, key in mapping.items()},
            checkpoint_keys=dict(mapping),
            strict=False,
            label=f"{request.model_name_or_path}/{request.subfolder or ''}",
        ),
        None,
    )


def _model_renamings(component):
    """Transformers' registered conversions, or a refusal if any is not a pure renaming."""
    try:
        from transformers.core_model_loading import WeightRenaming
        from transformers.modeling_utils import get_model_conversion_mapping
    except ImportError as exc:
        return None, f"transformers conversion mapping unavailable: {exc}"
    conversions = get_model_conversion_mapping(component)
    fused = [type(c).__name__ for c in conversions if not isinstance(c, WeightRenaming)]
    if fused:
        return None, (
            f"component needs {len(fused)} non-renaming conversion(s) ({fused[0]}), "
            f"whose tensors cannot be read one live name at a time"
        )
    return conversions, None


def load_transformers_component(
    component_cls,
    request,
    *,
    torch_dtype,
    quantization_config=None,
):
    """Call Transformers ``from_pretrained`` through the adapter boundary."""

    kwargs = request.from_pretrained_kwargs()
    if quantization_config is not None:
        kwargs["quantization_config"] = quantization_config
    return component_cls.from_pretrained(
        request.model_name_or_path,
        torch_dtype=torch_dtype,
        **kwargs,
    )
