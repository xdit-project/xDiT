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


# ---------------------------------------------------------------------------
# what the oracle above cannot see: which block a pattern picks out
# ---------------------------------------------------------------------------

def _translated():
    """Migrated models that held blocks back by block-relative prefix.

    Keyed on the prefixes rather than on any pattern, because the suffix-only
    models -- FastH3 and MiniMax -- had nothing to translate: their override
    suffixes named leaves their own include-suffixes already excluded, and only
    the legacy primary-format walk's failure to apply that narrowing ever let
    them fire. `test_only_narrows_every_format_the_same_way` covers those.
    """
    for cls, entry in CHECKABLE:
        if entry["fp8_precision_overrides"]:
            yield cls, entry


TRANSLATED = list(_translated())
TRANSLATED_IDS = [cls.__name__ for cls, _ in TRANSLATED]


@pytest.mark.skipif(not TRANSLATED, reason="no pattern models migrated yet")
@pytest.mark.parametrize("cls, entry", TRANSLATED, ids=TRANSLATED_IDS)
def test_a_translated_pattern_holds_back_exactly_what_it_used_to(cls, entry):
    """The oracle compares root lists, which cannot tell block 3 from block 30.

    The legacy patterns were block-relative ("3.", ".net.2") and the
    declaration is absolute, so this replays both dialects over every block of
    a generous depth and requires the same answer for every leaf.
    """
    from xfuser.core.utils.runner_utils import _layer_uses_fp8_override
    from xfuser.model_executor.quant.targets import resolve

    root = (entry["fp4_gemm_module_list"] or entry["fp8_gemm_module_list"])[0]
    prefixes = tuple(entry["fp8_precision_overrides"] or ())
    suffixes = tuple(entry["fp8_precision_override_suffixes"] or ())
    plan = resolve(
        cls.settings.gemm_targets, GemmQuantizationSpec.parse("low=fp4,high=fp8")
    )

    for index in range(100):
        for leaf in ("attn.to_q", "ffn.net.0.proj", "ffn.net.2", "norm.linear"):
            local = f"{index}.{leaf}"
            expected = (
                "fp8"
                if _layer_uses_fp8_override(local, prefixes, suffixes)
                else "fp4"
            )
            assert plan.format_for(f"{root}.{local}") == expected, (
                f"{cls.__name__}: {root}.{local}"
            )


@pytest.mark.parametrize(
    ("raw", "sp_world_size", "quantized"),
    [
        ("int8", 1, True),   # one rank sees the whole caption
        ("int8", 8, False),  # chunked below torch._int_mm's minimum M
        ("int8", 4, False),
        ("fp8", 8, True),    # no floor, so the chunking does not matter
    ],
)
def test_z_image_context_refiner_follows_the_kernels_floor(
    raw, sp_world_size, quantized
):
    """What `_customize_settings` used to do by editing the INT8 list.

    Nothing else in the suite covered it, and the oracle cannot: it compares
    declarations, and this one only differs once a run names both a format and
    a parallel degree.
    """
    from xfuser.model_executor.models.runner_models.z_image import xFuserZImageModel

    config = SimpleNamespace(
        gemm_quantization_spec=GemmQuantizationSpec.parse(raw),
        _gemm_config_loaded=False,
        quantize_text_encoder=False,
        ulysses_degree=sp_world_size,
        ring_degree=1,
        use_hybrid_gemm_schedule=False,
        gemm_high_precision_targets="model",
    )
    plan = QuantizationPlan(
        SimpleNamespace(
            settings=copy.deepcopy(xFuserZImageModel.settings), config=config
        )
    ).gemm_plan

    refiner = plan.format_for("transformer.context_refiner.0.attn.to_q")
    assert (refiner is not None) == quantized
    # its neighbours are quantized whatever the parallel degree
    assert plan.format_for("transformer.layers.0.attn.to_q") == raw
    assert plan.format_for("transformer.noise_refiner.0.attn.to_q") == raw
