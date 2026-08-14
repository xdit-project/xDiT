"""Which modules a run quantizes to FP8, on any hardware.

One place answers "what does FP8 cover this run", so every consumer agrees: the post-load walks, the
per-block quantize during FSDP sharding, the streaming quantize-on-load configs, and the meta-init
paths in ``meta_load``. Keeping it out of ``base_model`` keeps the runner base free of quantizer
imports; ``xFuserModel.fp8`` is the only entry point.

``module_list`` and ``targets_for`` are hardware-independent: the same target list drives the AITER
block-scale path on RDNA4 and the torchao path everywhere else. (INT8 and FP4 are not routed through
here; they read their own ``int8_gemm_module_list`` / ``fp4_gemm_module_list`` directly.)

Everything named ``aiter_*`` is limited to AITER's block-scale format and its
pipeline/text-encoder integration. Transformer backend selection and native
Diffusers streaming for both AITER and TorchAO live in ``fp8_backends``.
"""

from typing import TYPE_CHECKING, List, Optional

from xfuser.core.utils.runner_utils import _use_aiter_fp8_rdna4

if TYPE_CHECKING:
    from xfuser.model_executor.quant import AiterFp8BlockScaleConfig


class Fp8Plan:
    """The run's FP8 coverage, derived from the model's declared targets and the CLI flags.

    Holds the ``xFuserModel`` (``model``) to read its settings and config; keeps no state of its
    own, so it stays correct across the settings edits some runners make while loading.
    """

    def __init__(self, model) -> None:
        self.model = model

    def module_list(self) -> List[str]:
        """Every pipe-level module path to quantize to FP8 for this run, whatever the backend.

        The model's transformer targets, plus its text-encoder targets when --use_fp8_text_encoder
        is set. Every consumer of the FP8 target list goes through here, so the opt-in is honoured
        identically by the post-load walks, the per-block FSDP quantize and the streaming configs.
        """
        settings = self.model.settings
        targets = list(settings.fp8_gemm_module_list or [])
        if self.model.config.use_fp8_text_encoder:
            targets += list(settings.fp8_text_encoder_module_list or [])
        return targets

    def targets_for(self, component_name: str) -> List[str]:
        """This run's FP8 targets under one pipeline component, with the component prefix stripped
        (e.g. "text_encoder.model.language_model.layers" -> "model.language_model.layers"), which is
        the form loaders and module walks take."""
        prefix = f"{component_name}."
        return [m[len(prefix) :] for m in self.module_list() if m.startswith(prefix)]

    @property
    def aiter_active(self) -> bool:
        """True when the AITER block-scale FP8 path applies: fp8 gemms requested, and hardware and
        library support for the only kernels that implement it (ROCm RDNA4 with AITER).
        """
        return bool(self.model.config.use_fp8_gemms and _use_aiter_fp8_rdna4())

    def aiter_covers(self, component_name: str) -> bool:
        """True when the AITER path is on for this run and covers part of this component, i.e. the
        component should be built, filled and sharded as FP8 rather than quantized afterwards.
        """
        return bool(self.aiter_active and self.targets_for(component_name))

    def aiter_stream_config(
        self, attr_prefix: str = "transformer"
    ) -> Optional["AiterFp8BlockScaleConfig"]:
        """Quantize-on-load config for this run's targets under ``attr_prefix`` (e.g. "transformer" /
        "transformer_2"), or None when the AITER path does not apply. See quant.aiter_load.
        """
        if not self.aiter_active:
            return None
        from xfuser.model_executor.quant.aiter_load import stream_config

        return stream_config(self.targets_for(attr_prefix))

    def aiter_te_pipeline_config(self):
        """Quantize-on-load config routing this run's text-encoder targets through the pipeline, or
        None when the AITER path or the --use_fp8_text_encoder opt-in does not apply.
        See quant.aiter_load."""
        if not (self.aiter_active and self.model.config.use_fp8_text_encoder):
            return None
        from xfuser.model_executor.quant.aiter_load import te_pipeline_config

        return te_pipeline_config(self.model.settings.fp8_text_encoder_module_list)
