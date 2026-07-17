"""
Step-caching presets and configuration dataclasses.

DBCachePreset: typed config for cache-dit DBCache.

CacheDitAdapterConfig: block layout for one transformer in a cache-dit BlockAdapter.
    transformer_attr: pipe attribute name for this transformer (e.g. "transformer", "transformer_2").
    blocks: (block_attr, ForwardPattern_name) pairs to try on the transformer.
    enable_separate_cfg: True for two-pass CFG models.

DBCacheConfig: bundles adapter + preset for one model.
    adapter: CacheDitAdapterConfig (single transformer) or List[CacheDitAdapterConfig]
             (multi-transformer, e.g. Wan2.2). Must map 1:1 with preset list.
    preset: DBCachePreset or List[DBCachePreset] mapping 1:1 with adapter list.

ModelCacheConfig: dict[str, method_config] keyed by cache method name,
    e.g. {"dbcache": DBCacheConfig(...)}. .get(cache_method) returns None for
    methods the model doesn't support.
"""
import dataclasses
from typing import Dict, List, Optional, Tuple, Union


@dataclasses.dataclass
class DBCachePreset:
    # the first N blocks to use to calculate L1 difference, increase to improve accuracy at the cost of performance
    Fn_compute_blocks: int = 8
    # set to 0 explicitly since we exclusively use the TaylorSeer calibrator which overrides this
    Bn_compute_blocks: int = 0
    # increase to allow more caching
    residual_diff_threshold: float = 0.08
    # steps before cache kicks in
    max_warmup_steps: int = 8
    max_cached_steps: int = -1
    # SCM steps_computation_mask policy: None | "slow" | "medium" | "fast" | "ultra"
    scm_policy: Optional[str] = "fast"
    # enable_separate_cfg: True for models with two separate CFG forward passes (Wan, Qwen-Image-Edit).
    # False for fused-CFG or no-CFG models (FLUX, HunyuanVideo, Qwen-Image).
    # None = infer from CacheDitAdapterConfig.enable_separate_cfg.
    enable_separate_cfg: Optional[bool] = None
    # enable_encoder_calibrator: set False for MMDiT models (SD3.5) with Bn=0
    # Bn-residual buffer is never populated, causing the calibrator to assert.
    enable_encoder_calibrator: Optional[bool] = None


@dataclasses.dataclass(frozen=True)
class CacheDitAdapterConfig:
    """Block layout for one transformer in a cache-dit BlockAdapter.

    transformer_attr: pipe attribute name used to retrieve this transformer
        (e.g. "transformer", "transformer_2"). Default "transformer" covers all
        single-transformer models.

    blocks: sequence of (block_attr_name, ForwardPattern_name) pairs tried in order;
        first match on the transformer wins. Allows one config to cover model variants
        where optional block groups exist (e.g. Flux1 vs Flux2 single_transformer_blocks).

    enable_separate_cfg: True for models where CFG runs as two separate forward passes
        (Wan, Qwen-Image-Edit). False for fused or no-CFG models (FLUX, SD3, ZImage).
    """
    blocks: Tuple[Tuple[str, str], ...]
    enable_separate_cfg: bool = False
    transformer_attr: str = "transformer"


AdapterValue = Union["CacheDitAdapterConfig", List["CacheDitAdapterConfig"]]
PresetValue = Union[DBCachePreset, List[DBCachePreset]]


@dataclasses.dataclass
class DBCacheConfig:
    """Bundles CacheDitAdapterConfig(s) and DBCachePreset(s) for a model.

    Single transformer: adapter=CacheDitAdapterConfig(...), preset=DBCachePreset(...)
    Multi-transformer:  adapter=[cfg_t1, cfg_t2], preset=[preset_t1, preset_t2]
        Lists must be the same length; index i covers transformer i.
    """
    adapter: AdapterValue
    preset: Optional[PresetValue] = None


# dict keyed by cache method name, e.g. {"dbcache": DBCacheConfig(...)}.
ModelCacheConfig = Dict[str, object]
