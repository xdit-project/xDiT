"""Backend-neutral quantization target planning for one model run."""

from dataclasses import replace
from typing import Optional

from xfuser.config.gemm import GemmQuantizationSpec
from xfuser.core.utils.runner_utils import log
from xfuser.model_executor.quant.targets import Select, resolve


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
    """Resolve one runner's declared GEMM targets against this run."""

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
        config = self.model.config
        sp_world_size = (getattr(config, "ulysses_degree", 1) or 1) * (
            getattr(config, "ring_degree", 1) or 1
        )
        return resolve(
            targets,
            spec,
            enable=tuple(enable),
            sp_world_size=sp_world_size,
        )

    def _log_resolved_plan(self, plan) -> None:
        """Say what each declared target becomes, straight from the plan.

        The legacy version recovered the high tier by subtracting one list
        from another; the plan already knows, so this only has to read it out.
        """
        if not plan.quantizes:
            return

        high_roots = plan.roots(plan.high) if plan.high else ()
        # A carve-out named by suffix has no module path to report, so naming
        # only the roots would leave those layers held at the better format
        # with nothing in the log to say so.
        high_leaves = plan.keep_high.suffixes if plan.high else ()
        if high_roots or high_leaves:
            detail = f"GEMM high-precision tier: format={plan.high}"
            if high_roots:
                detail += f", modules={tuple(high_roots)}"
            if high_leaves:
                detail += f", leaves={tuple(high_leaves)} (suffix match)"
            log(detail)

        hybrid = getattr(self.model.config, "use_hybrid_gemm_schedule", False)
        low_detail = plan.low.upper()
        if hybrid and plan.high:
            low_detail = f"{plan.high.upper()} endpoints / {plan.low.upper()} middle"
        for root in plan.roots(plan.low):
            detail = low_detail
            if high_leaves:
                detail += f"; selected layers -> {plan.high.upper()}"
            log(f"GEMM quantization: {root} -> {detail}")
        for root in high_roots:
            log(f"GEMM quantization: {root} -> {plan.high.upper()}")

    def log_gemm_plan(self) -> None:
        """Log the resolved target-to-format mapping."""

        plan = self.gemm_plan
        if plan is not None:
            self._log_resolved_plan(plan)
