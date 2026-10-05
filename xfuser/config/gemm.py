from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml


#: The format names this build knows, for catching a typo at the command line
#: rather than minutes later. It is not a capability list and not a list of
#: pairs: whether a machine can store a format is one registry lookup at load
#: time, and any two distinct names may be tiered. Kept here because argument
#: parsing must not import the registry, which pulls in torch -- so
#: ``test_the_parser_knows_exactly_the_formats_the_registry_does`` pins the two
#: together instead.
_KNOWN_FORMATS = frozenset({"none", "fp8", "fp4", "fp6", "int8"})
#: Formats whose GEMM kernel refuses a small M. TorchAO's INT8 path lowers to
#: ``torch._int_mm`` under ``torch.compile``, which needs M >= 16; the FP8 and
#: MX kernels have no such floor. A module a model declares ``short_sequence``
#: is left alone by these formats once sequence parallelism can chunk it below
#: that -- see ``GemmTargets.short_sequence``.
MIN_M_FORMATS = frozenset({"int8"})

_CONFIG_SETTING_KEYS = frozenset(
    {
        "gemm_high_precision_targets",
        "gemm_high_precision_module_patterns",
        "gemm_high_precision_prefix_patterns",
        "gemm_high_precision_suffix_patterns",
        "hybrid_gemm_schedule",
    }
)


@dataclass(frozen=True)
class GemmQuantizationSpec:
    """Canonical transformer GEMM quantization selection."""

    low: str = "none"
    high: str | None = None

    def __post_init__(self) -> None:
        low = self.low.strip().lower()
        high = None if self.high is None else self.high.strip().lower()
        object.__setattr__(self, "low", low)
        object.__setattr__(self, "high", high)

        unknown = [
            name
            for name in (low, high)
            if name is not None and name not in _KNOWN_FORMATS
        ]
        if unknown:
            raise ValueError(
                f"unknown GEMM quantization format(s) {unknown}; expected one of "
                f"{', '.join(sorted(_KNOWN_FORMATS))}"
            )
        if high is None:
            return
        if low == high:
            raise ValueError("GEMM low and high quantization formats must differ")
        if "none" in (low, high):
            raise ValueError(
                "a GEMM tier names a format to quantize to; use "
                "--gemm_quantization none to quantize nothing"
            )

    @classmethod
    def parse(cls, value: GemmQuantizationSpec | str | None) -> GemmQuantizationSpec:
        if isinstance(value, cls):
            return value
        if value is None:
            return cls()
        if not isinstance(value, str):
            raise TypeError(
                "gemm_quantization must be a string or GemmQuantizationSpec"
            )

        raw = value.strip().lower()
        if not raw:
            raise ValueError("gemm_quantization cannot be empty")
        if "=" not in raw:
            if "," in raw:
                raise ValueError(
                    "pure GEMM quantization accepts one format; tiered "
                    "quantization must use low=<format>,high=<format>"
                )
            return cls(low=raw)

        values: dict[str, str] = {}
        for token in raw.split(","):
            token = token.strip()
            if not token or token.count("=") != 1:
                raise ValueError(
                    "tiered GEMM quantization must use " "low=<format>,high=<format>"
                )
            key, format_name = (part.strip() for part in token.split("=", 1))
            if key not in {"low", "high"}:
                raise ValueError(
                    f"unknown GEMM quantization tier {key!r}; expected low or high"
                )
            if key in values:
                raise ValueError(f"duplicate GEMM quantization tier {key!r}")
            if not format_name:
                raise ValueError(f"GEMM quantization tier {key!r} has no format")
            values[key] = format_name

        missing = {"low", "high"} - values.keys()
        if missing:
            raise ValueError(
                "tiered GEMM quantization requires both low and high; missing "
                + ", ".join(sorted(missing))
            )
        return cls(low=values["low"], high=values["high"])

    @property
    def is_tiered(self) -> bool:
        return self.high is not None

    @property
    def formats(self) -> frozenset[str]:
        return frozenset({self.low} if self.high is None else {self.low, self.high})

    def is_pure(self, format_name: str) -> bool:
        return self.high is None and self.low == format_name

    def __str__(self) -> str:
        if self.high is None:
            return self.low
        return f"low={self.low},high={self.high}"


@dataclass(frozen=True)
class GemmAdvancedConfig:
    path: str
    settings: Mapping[str, Any]


def _normalize_config_setting(key: str, value):
    if key == "gemm_high_precision_targets":
        if not isinstance(value, str):
            raise TypeError("gemm_high_precision_targets must be a string")
        normalized = value.lower()
        if normalized not in {"model", "none"}:
            raise ValueError("gemm_high_precision_targets must be 'model' or 'none'")
        return normalized
    if key in {
        "gemm_high_precision_module_patterns",
        "gemm_high_precision_prefix_patterns",
        "gemm_high_precision_suffix_patterns",
    }:
        if value is None or isinstance(value, str):
            return value
        if isinstance(value, list) and all(isinstance(item, str) for item in value):
            return ",".join(value) or None
        raise TypeError(f"{key} must be a string, list of strings, or null")
    if key == "hybrid_gemm_schedule":
        if isinstance(value, list):
            if not all(isinstance(item, str) for item in value):
                raise TypeError(f"{key} list entries must be strings")
            value = ",".join(value)
        if value is not None and not isinstance(value, str):
            raise TypeError(f"{key} must be a string, list of strings, or null")
        if value is not None:
            tokens = tuple(token.strip().lower() for token in value.split(","))
            # Any format this build knows, rather than the three that existed
            # when the schedule was written: which pair can actually drive a
            # per-step schedule is measured at load time, not listed here.
            schedulable = _KNOWN_FORMATS - {"none"}
            if not tokens or any(token not in schedulable for token in tokens):
                raise ValueError(
                    f"{key} entries must name a quantization format "
                    f"({', '.join(sorted(schedulable))}), got {value!r}"
                )
            return ",".join(tokens)
        return value
    raise ValueError(f"unsupported GEMM config key {key!r}")


def load_gemm_config(path: str) -> GemmAdvancedConfig:
    """Load advanced GEMM controls from a user-specified YAML file."""

    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"GEMM config file does not exist: {resolved}")
    with resolved.open("r", encoding="utf-8") as handle:
        source = handle.read()
        try:
            raw = yaml.safe_load(source) if source.strip() else {}
        except yaml.YAMLError as exc:
            raise ValueError(f"Invalid GEMM YAML in {resolved}: {exc}") from exc
    if not isinstance(raw, dict):
        raise TypeError(f"GEMM config {resolved} must contain a YAML mapping")

    unknown = set(raw) - _CONFIG_SETTING_KEYS
    if unknown:
        raise ValueError(
            f"GEMM config {resolved} contains unknown key(s): "
            + ", ".join(sorted(unknown))
        )
    try:
        settings = {
            key: _normalize_config_setting(key, value) for key, value in raw.items()
        }
    except (TypeError, ValueError) as exc:
        raise type(exc)(f"GEMM config {resolved}: {exc}") from exc
    return GemmAdvancedConfig(
        path=str(resolved),
        settings=settings,
    )


def parse_gemm_quantization(value: str) -> GemmQuantizationSpec:
    """ArgumentParser-compatible precision specification parser."""

    return GemmQuantizationSpec.parse(value)
