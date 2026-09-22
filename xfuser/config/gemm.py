from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml


_PURE_FORMATS = frozenset({"none", "fp8", "fp4", "fp6", "int8"})
_TIERED_FORMATS = frozenset(
    {
        ("fp4", "fp8"),
        ("fp4", "fp6"),
    }
)
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

        if high is None:
            if low not in _PURE_FORMATS:
                raise ValueError(
                    f"unknown GEMM quantization format {low!r}; expected one of "
                    f"{', '.join(sorted(_PURE_FORMATS))}"
                )
            return

        if low == high:
            raise ValueError("GEMM low and high quantization formats must differ")
        if (low, high) not in _TIERED_FORMATS:
            supported = ", ".join(
                f"low={pair[0]},high={pair[1]}" for pair in sorted(_TIERED_FORMATS)
            )
            raise ValueError(
                f"unsupported GEMM quantization pair low={low},high={high}; "
                f"supported pairs: {supported}"
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
            if not tokens or any(
                token not in {"fp8", "fp6", "fp4"} for token in tokens
            ):
                raise ValueError(
                    f"{key} entries must be fp8, fp6, or fp4, got {value!r}"
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
