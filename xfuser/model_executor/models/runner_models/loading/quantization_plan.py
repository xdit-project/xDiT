"""Backend-neutral quantization target planning for one model run."""


class QuantizationPlan:
    """Resolve declared FP8, FP4, and INT8 targets from one runner."""

    def __init__(self, model) -> None:
        self.model = model

    def module_list(self, format_name: str = "fp8") -> list[str]:
        settings = self.model.settings
        if format_name == "fp8":
            targets = list(settings.fp8_gemm_module_list or ())
            if self.model.config.use_fp8_text_encoder:
                targets += list(settings.fp8_text_encoder_module_list or ())
            return targets
        if format_name == "fp4":
            return list(settings.fp4_gemm_module_list or ())
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
