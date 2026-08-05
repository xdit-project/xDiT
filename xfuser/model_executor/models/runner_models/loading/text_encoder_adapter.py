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
            f"{_TRANSFORMERS_STREAMING_REQUIREMENT}: "
            f"{type(exc).__name__}: {exc}",
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
    from xfuser.core.utils.runner_utils import _get_fp8_kernel_preference

    return Float8DynamicActivationFloat8WeightConfig(
        granularity=PerTensor(),
        set_inductor_config=False,
        kernel_preference=_get_fp8_kernel_preference(),
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
        self._pipeline_config_factory = (
            pipeline_config_factory or _pipeline_config
        )
        self._torchao_config_factory = (
            torchao_config_factory or _torchao_text_encoder_config
        )
        self._aiter_config_factory = (
            aiter_config_factory or _aiter_text_encoder_config
        )

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
