"""
cache-dit adapter (optional dependency).
Provides DBCache step-caching via cache-dit's enable_cache() + BlockAdapter.

Model block layout is declared in ModelSettings.cache_config.adapter (CacheDitAdapterConfig)
No model-specific dispatch in this file.
"""
import dataclasses
import json
import logging
from typing import Any, Dict, List, Optional

import torch
import torch.distributed as dist
from xfuser.model_executor.cache.presets import CacheDitAdapterConfig, DBCachePreset

logger = logging.getLogger(__name__)


def _is_rank0() -> bool:
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0


def _import_cache_dit():
    try:
        from cache_dit import enable_cache, DBCacheConfig, BlockAdapter, ForwardPattern
        return enable_cache, DBCacheConfig, BlockAdapter, ForwardPattern
    except ImportError:
        raise ImportError(
            "cache-dit is required for --cache_method dbcache. "
            "Install: pip install cache-dit  or  pip install 'xdit[cache-dit]'"
        )


def _build_calibrator_config(enable_encoder_calibrator: Optional[bool] = None) -> Optional[Any]:
    """Build TaylorSeerCalibratorConfig (default calibrator)."""
    try:
        from cache_dit import TaylorSeerCalibratorConfig
        kwargs: Dict[str, Any] = {"taylorseer_order": 1}
        if enable_encoder_calibrator is not None:
            kwargs["enable_encoder_calibrator"] = enable_encoder_calibrator
        return TaylorSeerCalibratorConfig(**kwargs)
    except ImportError:
        if _is_rank0():
            logger.warning("TaylorSeerCalibratorConfig not available in this cache-dit version; running without calibrator")
        return None


def _build_scm_mask(
    policy: Optional[str],
    num_steps: int
) -> Optional[Any]:
    """Build steps_computation_mask from scm_policy field."""
    if not policy:
        return None
    try:
        import cache_dit as cd
        return cd.steps_mask(mask_policy=policy, total_steps=num_steps)
    except (ImportError, AttributeError):
        if _is_rank0():
            logger.warning(
                "cache_dit.steps_mask not available; scm_policy ignored")
        return None


def _build_config(
    num_steps: int,
    preset_kwargs,
    cache_config_json: Optional[str],
    enable_separate_cfg: bool,
    DBCacheConfig: Any,
):
    """Build (DBCacheConfig, calibrator_config) from a DBCachePreset or plain dict.

    cache_config JSON overrides apply at the preset level first (so preset-only fields
    like scm_policy / enable_encoder_calibrator are overridable), then any remaining
    keys pass through as raw DBCacheConfig overrides.
    """
    overrides: Dict[str, Any] = {}
    if cache_config_json:
        try:
            overrides = json.loads(cache_config_json)
        except json.JSONDecodeError as e:
            raise ValueError(f"--cache_config is not valid JSON: {e}") from e

    if isinstance(preset_kwargs, DBCachePreset):
        preset_fields = {f.name for f in dataclasses.fields(DBCachePreset)}
        preset_overrides = {k: v for k, v in overrides.items() if k in preset_fields}
        overrides = {k: v for k, v in overrides.items() if k not in preset_fields}
        p = dataclasses.replace(preset_kwargs, **preset_overrides) if preset_overrides else preset_kwargs
        config_kwargs: Dict[str, Any] = {
            "Fn_compute_blocks": p.Fn_compute_blocks,
            "Bn_compute_blocks": p.Bn_compute_blocks,
            "residual_diff_threshold": p.residual_diff_threshold,
            "max_warmup_steps": p.max_warmup_steps,
            "max_cached_steps": p.max_cached_steps,
        }
        if p.enable_separate_cfg is not None:
            enable_separate_cfg = p.enable_separate_cfg
        scm_mask = _build_scm_mask(p.scm_policy, num_steps)
        calibrator_config = _build_calibrator_config(p.enable_encoder_calibrator)
    else:
        config_kwargs = dict(preset_kwargs or {})
        scm_mask = None
        calibrator_config = _build_calibrator_config()

    config_kwargs["num_inference_steps"] = num_steps

    config_kwargs.update(overrides)

    if scm_mask is not None and "steps_computation_mask" not in config_kwargs:
        config_kwargs["steps_computation_mask"] = scm_mask
        config_kwargs.setdefault("steps_computation_policy", "dynamic")

    if enable_separate_cfg:
        config_kwargs.setdefault("enable_separate_cfg", True)

    valid_fields = {f.name for f in dataclasses.fields(DBCacheConfig)}
    unknown = set(config_kwargs) - valid_fields
    if unknown:
        if _is_rank0():
            logger.warning(
                f"--cache_config keys not recognized by DBCacheConfig and will be ignored: {sorted(unknown)}")
    config_kwargs = {k: v for k, v in config_kwargs.items()
                     if k in valid_fields}

    return DBCacheConfig(**config_kwargs), calibrator_config


def _unwrap_fsdp(transformer):
    """Return the real transformer when wrapped by FSDP1.

    shard_component uses FSDP1 for non-quantized models (e.g. Wan), wrapping the module
    in FullyShardedDataParallel. Block containers like `blocks` may not be accessible on
    the shell, causing _build_adapter block lookup to return None. FSDP2 (fully_shard)
    shards in place and keeps the original type, so only FSDP1 needs unwrapping.
    The wrapper.forward delegates to the inner module's (cache-patched) forward.
    """
    inner = getattr(transformer, "_fsdp_wrapped_module", None)
    if inner is not None:
        return inner
    if type(transformer).__name__ == "FullyShardedDataParallel":
        return getattr(transformer, "module", transformer)
    return transformer


def _build_adapter(transformer, pipe, adapter_cfg, BlockAdapter, ForwardPattern):
    """Build a BlockAdapter from a CacheDitAdapterConfig.

    Skips block attrs that don't exist on the transformer so a single config can
    cover models where e.g. single_transformer_blocks is optional.
    All builders pass check_forward_pattern=False: the runner per-block-compiles the
    transformer before cache application, replacing blocks with OptimizedModule whose
    forward signature is (*args, **kwargs), breaking cache-dit's pattern introspection.
    We pin the correct ForwardPattern per model so the check is redundant.
    """
    found = [
        (attr, getattr(ForwardPattern, pat))
        for attr, pat in adapter_cfg.blocks
        if getattr(transformer, attr, None) is not None
    ]
    if not found:
        raise RuntimeError(
            f"CacheDitAdapterConfig specifies blocks {[a for a, _ in adapter_cfg.blocks]!r} "
            f"but none exist on {type(transformer).__name__}. Check DBCacheConfig.adapter."
        )
    attrs, patterns = zip(*found)
    if len(attrs) == 1:
        return BlockAdapter(
            pipe=pipe, transformer=transformer,
            blocks=getattr(transformer, attrs[0]),
            blocks_name=attrs[0],
            forward_pattern=patterns[0],
            check_forward_pattern=False,
        )
    return BlockAdapter(
        pipe=pipe, transformer=transformer,
        blocks=[getattr(transformer, a) for a in attrs],
        blocks_name=list(attrs),
        forward_pattern=list(patterns),
        check_forward_pattern=False,
    )


_SP_SYNC_PATCHED = False


def _install_sp_can_cache_sync() -> None:
    """Force cache_dit's skip decision to agree bit-for-bit across all ranks.

    cache_dit's CachedContextManager.can_cache derives a bool from an AVG all_reduce over the default
    (world) process group; AVG is not bit-identical across RCCL ranks, so near-threshold the
    bool can flip per-rank. One divergent step desyncs collective counts (FSDP all-gather,
    ulysses all-to-all) -> NCCL hang. We wrap can_cache to broadcast the rank-0 result over
    the world group -- matching the group cache_dit reduces over, so it also covers the FSDP
    dimension (fully_shard) and pure-FSDP configs with ulysses=1, which an SP-only broadcast
    missed. dbcache bans PP, and these paths run no data parallelism, so world-wide agreement
    is correct. Idempotent; patched once.
    """
    global _SP_SYNC_PATCHED
    if _SP_SYNC_PATCHED:
        return
    try:
        from cache_dit.caching.cache_contexts.cache_manager import CachedContextManager
    except ImportError:
        return
    from xfuser.core.distributed import get_world_group

    orig_can_cache = CachedContextManager.can_cache

    @torch.compiler.disable
    def can_cache(self, *args, **kwargs):
        result = orig_can_cache(self, *args, **kwargs)
        t = torch.tensor(
            [1 if result else 0], device=torch.cuda.current_device(), dtype=torch.int32)
        get_world_group().broadcast(t, src=0)
        return bool(t.item())

    CachedContextManager.can_cache = can_cache
    _SP_SYNC_PATCHED = True


def _is_parallelized_flag() -> bool:
    from xfuser.core.distributed import (
        get_sequence_parallel_world_size,
        get_pipeline_parallel_world_size,
    )
    return get_sequence_parallel_world_size() > 1 or get_pipeline_parallel_world_size() > 1


def apply_cache_dit_cache(
    transformer: torch.nn.Module,
    num_steps: int,
    pipe: Optional[Any] = None,
    preset_kwargs: Optional[Dict[str, Any]] = None,
    cache_config: Optional[str] = None,
    adapter_config: Optional[CacheDitAdapterConfig] = None,
) -> torch.nn.Module:
    """Apply DBCache to a single transformer via cache-dit's enable_cache() with BlockAdapter."""
    enable_cache, DBCacheConfig, BlockAdapter, ForwardPattern = _import_cache_dit()
    _install_sp_can_cache_sync()

    # Route on the unwrapped module (FSDP1 wrapper hides the real type), but return
    # the original transformer so the pipe keeps its FSDP wrapper.
    routing_transformer = _unwrap_fsdp(transformer)
    routing_transformer._is_parallelized = _is_parallelized_flag()

    if adapter_config is not None:
        enable_separate_cfg = adapter_config.enable_separate_cfg
        adapter = _build_adapter(
            routing_transformer, pipe, adapter_config, BlockAdapter, ForwardPattern)
    else:
        if _is_rank0():
            logger.warning(
                f"No CacheDitAdapterConfig for {type(routing_transformer).__name__}; "
                "falling back to auto=True. Set DBCacheConfig.adapter in the model runner."
            )
        adapter = BlockAdapter(pipe=pipe, auto=True)
        enable_separate_cfg = False

    db_config, calibrator_config = _build_config(
        num_steps=num_steps,
        preset_kwargs=preset_kwargs,
        cache_config_json=cache_config,
        enable_separate_cfg=enable_separate_cfg,
        DBCacheConfig=DBCacheConfig,
    )

    enable_cache_kwargs: Dict[str, Any] = {"cache_config": db_config}
    if calibrator_config is not None:
        enable_cache_kwargs["calibrator_config"] = calibrator_config

    enable_cache(adapter, **enable_cache_kwargs)

    cls_name = type(routing_transformer).__name__
    calib_name = type(calibrator_config).__name__ if calibrator_config else "none"
    if _is_rank0():
        logger.info(
            f"Applied dbcache to {cls_name}: "
            f"F{db_config.Fn_compute_blocks}B{db_config.Bn_compute_blocks} "
            f"threshold={db_config.residual_diff_threshold} "
            f"calibrator={calib_name} "
            f"enable_separate_cfg={enable_separate_cfg}"
        )
    return transformer


def apply_cache_dit_cache_multi(
    pipe: Any,
    num_steps: int,
    adapter_configs: List[CacheDitAdapterConfig],
    presets: List[DBCachePreset],
    cache_config: Optional[str] = None,
) -> None:
    """Apply DBCache to multiple transformers in one enable_cache() call using ParamsModifier.

    presets maps 1:1 to adapter_configs; each carries per-transformer cache params.
    """
    enable_cache, DBCacheConfig, BlockAdapter, ForwardPattern = _import_cache_dit()
    _install_sp_can_cache_sync()
    from cache_dit import ParamsModifier

    if len(adapter_configs) != len(presets):
        raise ValueError(
            f"adapter_configs ({len(adapter_configs)}) and presets ({len(presets)}) must have same length"
        )

    # Resolve transformers from pipe via each adapter's transformer_attr
    transformers = []
    for cfg in adapter_configs:
        t = getattr(pipe, cfg.transformer_attr, None)
        if t is None:
            raise RuntimeError(
                f"apply_cache_dit_cache_multi: pipe has no attribute {cfg.transformer_attr!r}"
            )
        transformers.append(t)

    routing_transformers = [_unwrap_fsdp(t) for t in transformers]
    parallelized = _is_parallelized_flag()
    for rt in routing_transformers:
        rt._is_parallelized = parallelized

    cfg_flags = {c.enable_separate_cfg for c in adapter_configs}
    if len(cfg_flags) > 1:
        raise ValueError(
            f"All adapter_configs must agree on enable_separate_cfg; got {cfg_flags}"
        )
    enable_separate_cfg = adapter_configs[0].enable_separate_cfg

    # Build a full config per transformer so every per-preset field (compute blocks,
    # threshold, scm policy, calibrator), not just warmup/cached steps, pipes through.
    # cache_manager applies each ParamsModifier over the shared base via update() (non-None
    # fields win), so presets[0] is the base and presets[i] overrides for transformer i.
    # Each preset is built through _build_config so --cache_config JSON stays applied.
    configs = []
    calibrators = []
    for p in presets:
        c, cal = _build_config(
            num_steps=num_steps,
            preset_kwargs=p,
            cache_config_json=cache_config,
            enable_separate_cfg=enable_separate_cfg,
            DBCacheConfig=DBCacheConfig,
        )
        configs.append(c)
        calibrators.append(cal)
    db_config, calibrator_config = configs[0], calibrators[0]

    # Per-transformer block resolution and ParamsModifiers
    found_blocks = []
    found_attrs = []
    found_patterns = []
    params_modifiers = []
    for rt, cfg, c, cal in zip(routing_transformers, adapter_configs, configs, calibrators):
        for attr, pat_name in cfg.blocks:
            blocks_obj = getattr(rt, attr, None)
            if blocks_obj is not None:
                found_blocks.append(blocks_obj)
                found_attrs.append(attr)
                found_patterns.append(getattr(ForwardPattern, pat_name))
                break
        else:
            raise RuntimeError(
                f"CacheDitAdapterConfig blocks {[a for a, _ in cfg.blocks]!r} "
                f"not found on {type(rt).__name__} (pipe.{cfg.transformer_attr})"
            )
        params_modifiers.append(ParamsModifier(cache_config=c, calibrator_config=cal))

    adapter = BlockAdapter(
        pipe=pipe,
        transformer=routing_transformers,
        blocks=found_blocks,
        blocks_name=found_attrs,
        forward_pattern=found_patterns,
        params_modifiers=params_modifiers,
        check_forward_pattern=False,
    )

    enable_cache_kwargs: Dict[str, Any] = {"cache_config": db_config}
    if calibrator_config is not None:
        enable_cache_kwargs["calibrator_config"] = calibrator_config

    enable_cache(adapter, **enable_cache_kwargs)

    names = [type(rt).__name__ for rt in routing_transformers]
    calib_name = type(calibrator_config).__name__ if calibrator_config else "none"
    if _is_rank0():
        logger.info(
            f"Applied dbcache to [{', '.join(names)}] (multi): "
            f"F{db_config.Fn_compute_blocks}B{db_config.Bn_compute_blocks} "
            f"threshold={db_config.residual_diff_threshold} "
            f"calibrator={calib_name} "
            f"enable_separate_cfg={enable_separate_cfg} "
            f"warmup_steps={[p.max_warmup_steps for p in presets]}"
        )
