"""Optional cache-dit imports and DBCache configuration construction."""

import dataclasses
import json
import logging
from typing import Any, Dict, Optional

import torch.distributed as dist

from xfuser.model_executor.cache.presets import DBCachePreset

logger = logging.getLogger(__name__)


def _is_rank0() -> bool:
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0


def _import_cache_dit():
    try:
        from cache_dit import BlockAdapter, DBCacheConfig, ForwardPattern, enable_cache

        return enable_cache, DBCacheConfig, BlockAdapter, ForwardPattern
    except ImportError:
        raise ImportError(
            "cache-dit is required for --cache_method dbcache. "
            "Install: pip install cache-dit  or  pip install 'xdit[cache-dit]'"
        )


def _build_calibrator_config(enable_encoder_calibrator: Optional[bool] = None) -> Optional[Any]:
    """Build TaylorSeerCalibratorConfig (the default calibrator)."""
    try:
        from cache_dit import TaylorSeerCalibratorConfig

        kwargs: Dict[str, Any] = {"taylorseer_order": 1}
        if enable_encoder_calibrator is not None:
            kwargs["enable_encoder_calibrator"] = enable_encoder_calibrator
        return TaylorSeerCalibratorConfig(**kwargs)
    except ImportError:
        if _is_rank0():
            logger.warning(
                "TaylorSeerCalibratorConfig not available in this cache-dit version; running without calibrator"
            )
        return None


def _build_scm_mask(policy: Optional[str], num_steps: int) -> Optional[Any]:
    """Build a computation mask from the SCM policy."""
    if not policy:
        return None
    try:
        import cache_dit as cd

        return cd.steps_mask(mask_policy=policy, total_steps=num_steps)
    except (ImportError, AttributeError):
        if _is_rank0():
            logger.warning("cache_dit.steps_mask not available; scm_policy ignored")
        return None


def _resolve_enable_separate_cfg(requested: bool) -> bool:
    """Resolve the requested cache mode against the active CFG topology."""
    if not requested:
        return False

    from xfuser.core.distributed import get_classifier_free_guidance_world_size

    return get_classifier_free_guidance_world_size() == 1


def _build_config(
    num_steps: int,
    preset_kwargs,
    cache_config_json: Optional[str],
    enable_separate_cfg: bool,
    DBCacheConfig: Any,
):
    """Build a DBCacheConfig and optional calibrator from a preset and JSON overrides."""
    overrides: Dict[str, Any] = {}
    if cache_config_json:
        try:
            overrides = json.loads(cache_config_json)
            if not isinstance(overrides, dict):
                raise TypeError("cache_config must be a JSON object")
        except (json.JSONDecodeError, TypeError) as e:
            raise ValueError(f"--cache_config is not valid JSON: {e}") from e

    if isinstance(preset_kwargs, DBCachePreset):
        preset_fields = {field.name for field in dataclasses.fields(DBCachePreset)}
        preset_overrides = {key: value for key, value in overrides.items() if key in preset_fields}
        overrides = {key: value for key, value in overrides.items() if key not in preset_fields}
        preset = dataclasses.replace(preset_kwargs, **preset_overrides) if preset_overrides else preset_kwargs
        config_kwargs: Dict[str, Any] = {
            "Fn_compute_blocks": preset.Fn_compute_blocks,
            "Bn_compute_blocks": preset.Bn_compute_blocks,
            "residual_diff_threshold": preset.residual_diff_threshold,
            "max_warmup_steps": preset.max_warmup_steps,
            "max_cached_steps": preset.max_cached_steps,
        }
        if preset.enable_separate_cfg is not None:
            enable_separate_cfg = preset.enable_separate_cfg
        scm_mask = _build_scm_mask(preset.scm_policy, num_steps)
        calibrator_config = (
            None if preset.enable_taylorseer is False else _build_calibrator_config(preset.enable_encoder_calibrator)
        )
    else:
        config_kwargs = dict(preset_kwargs or {})
        scm_mask = None
        calibrator_config = _build_calibrator_config()

    config_kwargs["num_inference_steps"] = num_steps
    config_kwargs.update(overrides)

    if scm_mask is not None and "steps_computation_mask" not in config_kwargs:
        config_kwargs["steps_computation_mask"] = scm_mask
        config_kwargs.setdefault("steps_computation_policy", "dynamic")

    valid_fields = {field.name for field in dataclasses.fields(DBCacheConfig)}
    enable_separate_cfg = _resolve_enable_separate_cfg(config_kwargs.get("enable_separate_cfg", enable_separate_cfg))
    if enable_separate_cfg:
        config_kwargs.setdefault("enable_separate_cfg", True)
    elif "enable_separate_cfg" in valid_fields:
        config_kwargs["enable_separate_cfg"] = False
    else:
        config_kwargs.pop("enable_separate_cfg", None)

    unknown = set(config_kwargs) - valid_fields
    if unknown:
        raise ValueError(f"Unknown --cache_config keys for DBCacheConfig: {sorted(unknown)}")

    config_kwargs = {key: value for key, value in config_kwargs.items() if key in valid_fields}
    return DBCacheConfig(**config_kwargs), calibrator_config
