"""Every migrated model must target what its old lists targeted.

`data/legacy_gemm_declarations.json` records what each model declared before
migration. A model that has moved to `gemm_targets` is checked against its own
recorded entry, through the unchanged legacy code -- so the oracle survives the
deletion of the fields it describes.

Models still on the legacy path are skipped, and so are ones migrated before
the snapshot was taken (FLUX.2-dev, which carries its own frozen copy in
test_flux2_gemm_equivalence.py).
"""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from xfuser.config.gemm import GemmQuantizationSpec
from xfuser.model_executor.models.runner_models.base_model import MODEL_REGISTRY
from xfuser.model_executor.models.runner_models.loading.quantization_plan import (
    QuantizationPlan,
    apply_fp8_override_cli_to_settings,
)

SNAPSHOT = json.loads(
    (Path(__file__).parent / "data" / "legacy_gemm_declarations.json").read_text()
)
LEGACY_FIELDS = (
    "fp8_gemm_module_list",
    "fp4_gemm_module_list",
    "int8_gemm_module_list",
    "fp8_text_encoder_module_list",
    "fp8_precision_overrides",
    "fp8_precision_override_suffixes",
    "fp8_gemm_include_suffixes",
)


def _checkable():
    """Migrated models whose pre-migration declaration was recorded."""
    for cls in dict.fromkeys(MODEL_REGISTRY.values()):
        entry = SNAPSHOT.get(cls.__name__)
        if cls.settings.gemm_targets is None or entry is None:
            continue
        if not any(entry.get(f) for f in LEGACY_FIELDS):
            continue  # migrated before the snapshot; covered elsewhere
        yield cls, entry


CHECKABLE = list(_checkable())
IDS = [cls.__name__ for cls, _ in CHECKABLE]


def _profiles(entry):
    caps = entry["capabilities"]
    found = [f for f in ("fp8", "fp4", "fp6", "int8") if caps.get(f"use_{f}_gemms")]
    profiles = list(found)
    if "fp4" in found and "fp8" in found:
        profiles.append("low=fp4,high=fp8")
    return profiles


def _plan(cls, entry, raw, *, text_encoder, legacy):
    spec = GemmQuantizationSpec.parse(raw)
    settings = copy.deepcopy(cls.settings)
    if legacy:
        settings.gemm_targets = None
        for field in LEGACY_FIELDS:
            value = entry.get(field) or None
            setattr(settings, field, list(value) if value else None)
    config = SimpleNamespace(
        gemm_quantization_spec=spec,
        _gemm_config_loaded=False,
        fp8_precision_override_prefix_patterns=None,
        fp8_precision_override_suffix_patterns=None,
        quantize_text_encoder=text_encoder,
        use_fp4_gemms=spec.low == "fp4",
        use_hybrid_gemm_schedule=False,
        gemm_high_precision_targets="model",
    )
    if legacy:
        apply_fp8_override_cli_to_settings(config, settings)
    return QuantizationPlan(
        SimpleNamespace(settings=settings, config=config, capabilities=cls.capabilities)
    )


@pytest.mark.skipif(not CHECKABLE, reason="no migrated models recorded yet")
@pytest.mark.parametrize("cls, entry", CHECKABLE, ids=IDS)
def test_a_migrated_model_targets_what_it_used_to(cls, entry):
    for raw in _profiles(entry):
        for text_encoder in (False, True):
            if text_encoder and not entry["quantize_text_encoder"]:
                continue
            old = _plan(cls, entry, raw, text_encoder=text_encoder, legacy=True)
            new = _plan(cls, entry, raw, text_encoder=text_encoder, legacy=False)
            spec = GemmQuantizationSpec.parse(raw)
            for fmt in spec.formats:
                if fmt == "none":
                    continue
                assert sorted(new.module_list(fmt)) == sorted(old.module_list(fmt)), (
                    f"{cls.__name__} {raw} te={text_encoder} {fmt}"
                )


def test_the_snapshot_still_describes_unmigrated_models():
    """Guards the oracle: a model must not lose its lists without migrating."""
    lost = [
        cls.__name__
        for cls in dict.fromkeys(MODEL_REGISTRY.values())
        if cls.settings.gemm_targets is None
        and any(SNAPSHOT.get(cls.__name__, {}).get(f) for f in LEGACY_FIELDS)
        and not any(getattr(cls.settings, f, None) for f in LEGACY_FIELDS)
    ]
    assert not lost, f"declarations vanished without a gemm_targets migration: {lost}"
