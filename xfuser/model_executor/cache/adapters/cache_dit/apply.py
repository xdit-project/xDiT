"""Cache-dit adapter construction and cache application."""

import dataclasses
import logging
from typing import Any, Dict, List, Optional

import torch

from xfuser.model_executor.cache.adapters.cache_dit.config import (
    _build_config,
    _import_cache_dit,
    _is_rank0,
)
from xfuser.model_executor.cache.adapters.cache_dit.context import (
    install_cache_decision_sync,
    install_pipefusion_patch_contexts,
    install_pipefusion_static_fn_bypass,
    is_parallelized_flag,
)
from xfuser.model_executor.cache.presets import CacheDitAdapterConfig, DBCachePreset

logger = logging.getLogger(__name__)


def _unwrap_fsdp(transformer):
    """Return the real transformer when an FSDP1 wrapper encloses it."""
    inner = getattr(transformer, "_fsdp_wrapped_module", None)
    if inner is not None:
        return inner
    if type(transformer).__name__ == "FullyShardedDataParallel":
        return getattr(transformer, "module", transformer)
    return transformer


def _build_adapter(transformer, pipe, adapter_cfg, BlockAdapter, ForwardPattern):
    """Build a BlockAdapter from a CacheDitAdapterConfig."""
    found = [
        (attribute, getattr(ForwardPattern, pattern))
        for attribute, pattern in adapter_cfg.blocks
        if (getattr(transformer, attribute, None) is not None and len(getattr(transformer, attribute)) > 0)
    ]
    if not found:
        raise RuntimeError(
            f"CacheDitAdapterConfig specifies blocks {[attribute for attribute, _ in adapter_cfg.blocks]!r} "
            f"but none exist on {type(transformer).__name__}. Check DBCacheSettings.adapter."
        )
    attributes, patterns = zip(*found)
    if len(attributes) == 1:
        return BlockAdapter(
            pipe=pipe,
            transformer=transformer,
            blocks=getattr(transformer, attributes[0]),
            blocks_name=attributes[0],
            forward_pattern=patterns[0],
            check_forward_pattern=False,
        )
    return BlockAdapter(
        pipe=pipe,
        transformer=transformer,
        blocks=[getattr(transformer, attribute) for attribute in attributes],
        blocks_name=list(attributes),
        forward_pattern=list(patterns),
        check_forward_pattern=False,
    )


def apply_cache_dit_cache(
    transformer: torch.nn.Module,
    num_steps: int,
    pipe: Optional[Any] = None,
    preset_kwargs: Optional[Dict[str, Any]] = None,
    cache_config: Optional[str] = None,
    adapter_config: Optional[CacheDitAdapterConfig] = None,
) -> torch.nn.Module:
    """Apply DBCache to one transformer with cache-dit's enable_cache."""
    from xfuser.core.distributed import get_pipeline_parallel_world_size

    enable_cache, DBCacheConfig, BlockAdapter, ForwardPattern = _import_cache_dit()
    pipefusion = get_pipeline_parallel_world_size() > 1

    routing_transformer = _unwrap_fsdp(transformer)
    routing_transformer._is_parallelized = is_parallelized_flag()

    if adapter_config is not None:
        enable_separate_cfg = adapter_config.enable_separate_cfg
        adapter = _build_adapter(
            routing_transformer,
            pipe,
            adapter_config,
            BlockAdapter,
            ForwardPattern,
        )
    else:
        if _is_rank0():
            logger.warning(
                f"No CacheDitAdapterConfig for {type(routing_transformer).__name__}; "
                "falling back to auto=True. Set DBCacheSettings.adapter in the model runner."
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
    static_pipefusion = (
        pipefusion
        and getattr(db_config, "steps_computation_policy", None) == "static"
        and bool(getattr(db_config, "steps_computation_mask", None))
    )
    if static_pipefusion:
        # A predetermined schedule does not consume adaptive TaylorSeer
        # decisions. The stage-level fast path installed below caches the full
        # local residual and skips Fn on reuse steps.
        calibrator_config = None
    share_pipefusion_decisions = bool(getattr(db_config, "_xdit_share_pipefusion_decisions", False))
    if pipefusion and adapter_config is not None:
        local_block_counts = [
            len(getattr(routing_transformer, attribute))
            for attribute, _pattern in adapter_config.blocks
            if (
                getattr(routing_transformer, attribute, None) is not None
                and len(getattr(routing_transformer, attribute)) > 0
            )
        ]
        local_fn_blocks = min(
            db_config.Fn_compute_blocks,
            min(local_block_counts),
        )
        local_bn_blocks = min(
            db_config.Bn_compute_blocks,
            min(max(block_count - local_fn_blocks, 0) for block_count in local_block_counts),
        )
        db_config = dataclasses.replace(
            db_config,
            Fn_compute_blocks=local_fn_blocks,
            Bn_compute_blocks=local_bn_blocks,
        )
        db_config._xdit_share_pipefusion_decisions = share_pipefusion_decisions

    enable_cache_kwargs: Dict[str, Any] = {"cache_config": db_config}
    if calibrator_config is not None:
        enable_cache_kwargs["calibrator_config"] = calibrator_config
    enable_cache(adapter, **enable_cache_kwargs)
    static_fn_bypass = False
    if static_pipefusion:
        static_fn_bypass = install_pipefusion_static_fn_bypass(
            routing_transformer,
            db_config,
        )
    routing_transformer._xdit_share_pipefusion_decisions = bool(
        getattr(db_config, "_xdit_share_pipefusion_decisions", False)
    )
    install_cache_decision_sync(routing_transformer)

    if pipefusion:
        install_pipefusion_patch_contexts(routing_transformer)

    if _is_rank0():
        calibrator_name = type(calibrator_config).__name__ if calibrator_config else "none"
        logger.info(
            f"Applied dbcache to {type(routing_transformer).__name__}: "
            f"F{0 if static_fn_bypass else db_config.Fn_compute_blocks}"
            f"B{db_config.Bn_compute_blocks} "
            f"threshold={db_config.residual_diff_threshold} "
            f"calibrator={calibrator_name} "
            f"enable_separate_cfg={getattr(db_config, 'enable_separate_cfg', False)}"
        )
    return transformer


def apply_cache_dit_cache_multi(
    pipe: Any,
    num_steps: int,
    adapter_configs: List[CacheDitAdapterConfig],
    presets: List[DBCachePreset],
    cache_config: Optional[str] = None,
) -> None:
    """Apply DBCache to multiple transformers in one coordinated call."""
    from xfuser.core.distributed import get_pipeline_parallel_world_size

    if get_pipeline_parallel_world_size() > 1:
        raise ValueError("PipeFusion step caching does not yet support multi-transformer DBCache adapters.")

    enable_cache, DBCacheConfig, BlockAdapter, ForwardPattern = _import_cache_dit()

    from cache_dit import ParamsModifier

    if len(adapter_configs) != len(presets):
        raise ValueError(f"adapter_configs ({len(adapter_configs)}) and presets ({len(presets)}) must have same length")

    transformers = []
    for config in adapter_configs:
        transformer = getattr(pipe, config.transformer_attr, None)
        if transformer is None:
            raise RuntimeError(f"apply_cache_dit_cache_multi: pipe has no attribute {config.transformer_attr!r}")
        transformers.append(transformer)

    routing_transformers = [_unwrap_fsdp(transformer) for transformer in transformers]
    parallelized = is_parallelized_flag()
    for transformer in routing_transformers:
        transformer._is_parallelized = parallelized

    cfg_flags = {config.enable_separate_cfg for config in adapter_configs}
    if len(cfg_flags) > 1:
        raise ValueError(f"All adapter_configs must agree on enable_separate_cfg; got {cfg_flags}")
    enable_separate_cfg = adapter_configs[0].enable_separate_cfg

    configs = []
    calibrators = []
    for preset in presets:
        config, calibrator = _build_config(
            num_steps=num_steps,
            preset_kwargs=preset,
            cache_config_json=cache_config,
            enable_separate_cfg=enable_separate_cfg,
            DBCacheConfig=DBCacheConfig,
        )
        configs.append(config)
        calibrators.append(calibrator)
    db_config, calibrator_config = configs[0], calibrators[0]

    found_blocks = []
    found_attributes = []
    found_patterns = []
    params_modifiers = []
    for transformer, adapter_config, config, calibrator in zip(
        routing_transformers,
        adapter_configs,
        configs,
        calibrators,
    ):
        for attribute, pattern_name in adapter_config.blocks:
            blocks = getattr(transformer, attribute, None)
            if blocks is not None:
                found_blocks.append(blocks)
                found_attributes.append(attribute)
                found_patterns.append(getattr(ForwardPattern, pattern_name))
                break
        else:
            raise RuntimeError(
                f"CacheDitAdapterConfig blocks {[attribute for attribute, _ in adapter_config.blocks]!r} "
                f"not found on {type(transformer).__name__} (pipe.{adapter_config.transformer_attr})"
            )
        params_modifiers.append(ParamsModifier(cache_config=config, calibrator_config=calibrator))

    adapter = BlockAdapter(
        pipe=pipe,
        transformer=routing_transformers,
        blocks=found_blocks,
        blocks_name=found_attributes,
        forward_pattern=found_patterns,
        params_modifiers=params_modifiers,
        check_forward_pattern=False,
    )
    enable_cache_kwargs: Dict[str, Any] = {"cache_config": db_config}
    if calibrator_config is not None:
        enable_cache_kwargs["calibrator_config"] = calibrator_config
    enable_cache(adapter, **enable_cache_kwargs)
    for transformer, config in zip(routing_transformers, configs):
        transformer._xdit_share_pipefusion_decisions = bool(getattr(config, "_xdit_share_pipefusion_decisions", False))
        install_cache_decision_sync(transformer)

    if _is_rank0():
        calibrator_name = type(calibrator_config).__name__ if calibrator_config else "none"
        names = [type(transformer).__name__ for transformer in routing_transformers]
        logger.info(
            f"Applied dbcache to [{', '.join(names)}] (multi): "
            f"F{db_config.Fn_compute_blocks}B{db_config.Bn_compute_blocks} "
            f"threshold={db_config.residual_diff_threshold} "
            f"calibrator={calibrator_name} "
            f"enable_separate_cfg={getattr(db_config, 'enable_separate_cfg', False)} "
            f"warmup_steps={[preset.max_warmup_steps for preset in presets]}"
        )
