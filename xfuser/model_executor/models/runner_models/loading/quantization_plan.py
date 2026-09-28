"""Backend-neutral quantization target planning for one model run."""

from dataclasses import replace
from typing import Optional

from xfuser.config.gemm import GemmQuantizationSpec
from xfuser.core.utils.runner_utils import log
from xfuser.model_executor.quant.targets import Select, resolve


def apply_fp8_override_cli_to_settings(config, settings) -> None:
    """Apply optional CLI FP8 override patterns (per-slot) to model settings."""

    def _parse_csv_patterns(raw: Optional[str]) -> Optional[tuple[str, ...]]:
        if raw is None or not raw.strip():
            return None
        patterns = tuple(p.strip() for p in raw.split(",") if p.strip())
        return patterns or None

    spec = getattr(config, "gemm_quantization_spec", None)
    if spec is not None and spec.is_pure("fp4"):
        settings.fp4_gemm_module_list = list(
            dict.fromkeys(
                list(settings.fp4_gemm_module_list or ())
                + list(settings.fp8_gemm_module_list or ())
            )
        )
        settings.fp8_gemm_module_list = []
        settings.fp8_precision_overrides = None
        settings.fp8_precision_override_suffixes = None
        return

    if getattr(config, "_gemm_config_loaded", False):
        eligible = list(
            dict.fromkeys(
                list(settings.fp4_gemm_module_list or ())
                + list(settings.fp8_gemm_module_list or ())
            )
        )
        if config.gemm_high_precision_targets == "none":
            settings.fp8_gemm_module_list = []
            settings.fp4_gemm_module_list = eligible
            settings.fp8_precision_overrides = None
            settings.fp8_precision_override_suffixes = None

        modules = _parse_csv_patterns(config.gemm_high_precision_module_patterns)
        if modules:
            unknown = set(modules) - set(eligible)
            if unknown:
                raise ValueError(
                    "GEMM high-precision modules must exactly match declared "
                    f"model targets; unknown: {sorted(unknown)}"
                )
            high_modules = list(settings.fp8_gemm_module_list or ())
            low_modules = list(settings.fp4_gemm_module_list or ())
            for module in modules:
                high_modules.append(module)
                low_modules = [low for low in low_modules if low != module]
            settings.fp8_gemm_module_list = list(dict.fromkeys(high_modules))
            settings.fp4_gemm_module_list = low_modules
        prefixes = _parse_csv_patterns(config.gemm_high_precision_prefix_patterns)
        if prefixes:
            settings.fp8_precision_overrides = tuple(
                dict.fromkeys(
                    tuple(settings.fp8_precision_overrides or ()) + prefixes
                )
            )
        suffixes = _parse_csv_patterns(config.gemm_high_precision_suffix_patterns)
        if suffixes:
            settings.fp8_precision_override_suffixes = tuple(
                dict.fromkeys(
                    tuple(settings.fp8_precision_override_suffixes or ()) + suffixes
                )
            )
        return

    if config.fp8_precision_override_prefix_patterns is not None:
        settings.fp8_precision_overrides = _parse_csv_patterns(
            config.fp8_precision_override_prefix_patterns
        )
    if config.fp8_precision_override_suffix_patterns is not None:
        settings.fp8_precision_override_suffixes = _parse_csv_patterns(
            config.fp8_precision_override_suffix_patterns
        )


def _csv(raw) -> tuple:
    if raw is None or not str(raw).strip():
        return ()
    return tuple(part.strip() for part in str(raw).split(",") if part.strip())


def _with_config_overrides(targets, config):
    """Apply an advanced GEMM config file to the model's declaration.

    The file says which modules to hold at the better format, which is exactly
    what `keep_high` means -- and its three pattern kinds are Select's three
    fields. So it overrides that one field and nothing else.

    Unlike the legacy mechanism its prefixes are absolute paths, matching the
    rest of Select, rather than block-relative indices.
    """
    if not getattr(config, "_gemm_config_loaded", False):
        return targets

    keep = targets.keep_high
    if getattr(config, "gemm_high_precision_targets", "model") == "none":
        keep = Select()

    modules = _csv(getattr(config, "gemm_high_precision_module_patterns", None))
    prefixes = _csv(getattr(config, "gemm_high_precision_prefix_patterns", None))
    suffixes = _csv(getattr(config, "gemm_high_precision_suffix_patterns", None))
    if modules or prefixes or suffixes:
        keep = Select(
            modules=keep.modules + modules,
            prefixes=keep.prefixes + prefixes,
            suffixes=keep.suffixes + suffixes,
        )
    if keep == targets.keep_high:
        return targets
    # GemmTargets validates that keep_high stays inside the target set, so a
    # pattern naming something undeclared is refused here rather than ignored.
    return replace(targets, keep_high=keep)


class QuantizationPlan:
    """Resolve declared FP8, FP4, FP6, and INT8 targets from one runner."""

    def __init__(self, model) -> None:
        self.model = model

    @property
    def gemm_plan(self):
        """This run's resolved GEMM plan, or None for an unmigrated model.

        The formats come from the run, the modules from the model, and the
        components from whichever switch enables them.
        """
        targets = getattr(self.model.settings, "gemm_targets", None)
        if targets is None:
            return None
        targets = _with_config_overrides(targets, self.model.config)
        spec = getattr(self.model.config, "gemm_quantization_spec", None)
        if spec is None:
            spec = GemmQuantizationSpec()
        enable = ["transformer"]
        if getattr(self.model.config, "quantize_text_encoder", False):
            enable.append("text_encoder")
        return resolve(targets, spec, enable=tuple(enable))

    def _declared_targets(self, targets, format_name: str) -> list[str]:
        """The per-format list that one `gemm_targets` declaration stands in for.

        Temporary scaffolding, in the old dialect on purpose: consumers
        recover the high tier by subtracting one list from another, and read a
        list before the run gates it. Until they are rewritten onto GemmPlan,
        handing them a narrower list would change what they compute.

        A format the model cannot run never reaches here -- `_validate_config`
        refuses it against ModelCapabilities first.
        """
        entries = list(targets.transformer.roots())
        spec = getattr(self.model.config, "gemm_quantization_spec", None)
        tiered = spec is not None and spec.is_tiered

        if tiered and format_name == spec.low:
            # Held-high modules are absent from the low list, so the
            # subtraction the consumers do recovers exactly keep_high.
            entries = [e for e in entries if not targets.keep_high.matches(e)]
        if format_name != "fp8":
            return entries

        if spec is not None and spec.is_pure("fp4"):
            # A pure fp4 run folds the fp8 targets into fp4 and empties fp8.
            entries = []
        if self.model.config.quantize_text_encoder:
            entries += list(targets.text_encoder.roots())
        return entries

    def module_list(self, format_name: str = "fp8") -> list[str]:
        settings = self.model.settings
        targets = getattr(settings, "gemm_targets", None)
        if targets is not None:
            if format_name not in ("fp8", "fp4", "fp6", "int8"):
                raise ValueError(
                    f"unsupported quantization target format: {format_name}"
                )
            return self._declared_targets(targets, format_name)
        if format_name == "fp8":
            targets = list(settings.fp8_gemm_module_list or ())
            if self.model.config.quantize_text_encoder:
                targets += list(settings.fp8_text_encoder_module_list or ())
            return targets
        if format_name == "fp4":
            return list(settings.fp4_gemm_module_list or ())
        if format_name == "fp6":
            return list(
                dict.fromkeys(
                    list(getattr(settings, "fp4_gemm_module_list", None) or ())
                    + list(getattr(settings, "fp8_gemm_module_list", None) or ())
                )
            )
        if format_name == "int8":
            return list(settings.int8_gemm_module_list or ())
        raise ValueError(f"unsupported quantization target format: {format_name}")

    def targets_for(self, component_name: str, format_name: str = "fp8") -> list[str]:
        prefix = f"{component_name}."
        return [
            "" if target == component_name else target[len(prefix) :]
            for target in self.module_list(format_name)
            if target == component_name or target.startswith(prefix)
        ]

    def log_fp8_overrides(self) -> None:
        """Log FP8 precision overrides once when the FP4 plan is materialized."""
        settings = self.model.settings
        prefixes = settings.fp8_precision_overrides
        suffixes = settings.fp8_precision_override_suffixes
        use_fp6 = bool(getattr(self.model.config, "use_fp6_gemms", False))
        if use_fp6 and not self.model.config.use_fp4_gemms:
            return
        override_format = "MXFP6" if use_fp6 else "FP8"
        if prefixes:
            log(
                f"The following layers will be quantized to {override_format}, "
                "to maintain output quality: "
                f"{prefixes} (prefix match)"
            )
        if suffixes:
            log(
                f"The following layers will be quantized to {override_format}, "
                "to maintain output quality: "
                f"{suffixes} (suffix match)"
            )

    def _log_resolved_plan(self, plan) -> None:
        """Say what each declared target becomes, straight from the plan.

        The legacy version recovered the high tier by subtracting one list
        from another; the plan already knows, so this only has to read it out.
        """
        if not plan.quantizes:
            return

        high_roots = plan.roots(plan.high) if plan.high else ()
        if high_roots:
            log(
                f"GEMM high-precision tier: format={plan.high}, "
                f"modules={tuple(high_roots)}"
            )

        hybrid = getattr(self.model.config, "use_hybrid_gemm_schedule", False)
        low_detail = plan.low.upper()
        if hybrid and plan.high:
            low_detail = f"{plan.high.upper()} endpoints / {plan.low.upper()} middle"
        for root in plan.roots(plan.low):
            log(f"GEMM quantization: {root} -> {low_detail}")
        for root in high_roots:
            log(f"GEMM quantization: {root} -> {plan.high.upper()}")

    def log_gemm_plan(self) -> None:
        """Log the resolved transformer target-to-format mapping."""

        plan = self.gemm_plan
        if plan is not None:
            self._log_resolved_plan(plan)
            return

        spec = getattr(self.model.config, "gemm_quantization_spec", None)
        if spec is None:
            if self.model.config.use_fp4_gemms:
                self.log_fp8_overrides()
            return
        if spec.is_pure("none"):
            return

        settings = self.model.settings
        if not spec.is_tiered:
            targets = (
                self.module_list("fp6")
                if spec.is_pure("fp6")
                else list(
                    getattr(
                        settings,
                        f"{spec.low}_gemm_module_list",
                        None,
                    )
                    or ()
                )
            )
            for target in targets:
                log(f"GEMM quantization: {target} -> {spec.low.upper()}")
            return

        low_targets = list(settings.fp4_gemm_module_list or ())
        high_targets = list(settings.fp8_gemm_module_list or ())
        high_only_targets = [
            target
            for target in high_targets
            if not any(
                target == low or target.startswith(f"{low}.")
                for low in low_targets
            )
        ]
        prefixes = tuple(settings.fp8_precision_overrides or ())
        suffixes = tuple(settings.fp8_precision_override_suffixes or ())
        baseline = getattr(
            self.model.config,
            "gemm_high_precision_targets",
            "model",
        )
        log(
            "GEMM high-precision tier: "
            f"format={spec.high}, baseline={baseline}, "
            f"modules={tuple(high_only_targets)}, "
            f"prefixes={prefixes}, suffixes={suffixes}"
        )

        hybrid = self.model.config.use_hybrid_gemm_schedule
        for target in list(dict.fromkeys(low_targets + high_only_targets)):
            if target in high_only_targets:
                detail = spec.high.upper()
            elif hybrid:
                detail = (
                    f"{spec.high.upper()} endpoints / "
                    f"{spec.low.upper()} middle"
                )
            else:
                detail = spec.low.upper()
            if target in low_targets and (prefixes or suffixes):
                detail += f"; selected high-tier layers -> {spec.high.upper()}"
            log(f"GEMM quantization: {target} -> {detail}")
