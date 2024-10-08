from typing import Optional
from diffusers import DiffusionPipeline
from xfuser.config.config import (
    ParallelConfig,
    RuntimeConfig,
    InputConfig,
    FastAttnConfig,
    EngineConfig,
)
from xfuser.logger import init_logger

logger = init_logger(__name__)


class FastAttnState:
    enable: bool = False
    n_step: int = 20
    n_calib: int = 8
    threshold: float = 0.5
    window_size: int = 64
    coco_path: Optional[str] = None
    use_cache: bool = False
    config_file: str
    layer_name: str

    def __init__(self, pipe: DiffusionPipeline, config: FastAttnConfig):
        self.enable = config.use_fast_attn
        if self.enable:
            self.n_step = config.n_step
            self.n_calib = config.n_calib
            self.threshold = config.threshold
            self.window_size = config.window_size
            self.coco_path = config.coco_path
            self.use_cache = config.use_cache
            self.config_file = self.config_file_path(pipe, config)
            self.layer_name = self.attn_name_to_wrap(pipe)

    def config_file_path(self, pipe: DiffusionPipeline, config: FastAttnConfig):
        """Return the config file path."""
        return f"cache/{pipe.config._name_or_path.replace('/', '_')}_{config.n_step}_{config.n_calib}_{config.threshold}_{config.window_size}.json"

    def attn_name_to_wrap(self, pipe: DiffusionPipeline):
        """Return the attr name of attention layer to wrap."""
        names = ["attn1", "attn"]  # names of self attention layer
        assert hasattr(pipe, "transformer"), "transformer is not found in pipeline."
        assert hasattr(pipe.transformer, "transformer_blocks"), "transformer_blocks is not found in pipeline."
        block = pipe.transformer.transformer_blocks[0]
        for name in names:
            if hasattr(block, name):
                return name
        raise AttributeError(f"Attention layer name is not found in {names}.")


_FASTATTN: Optional[FastAttnState] = None


def get_fast_attn_state() -> FastAttnState:
    # assert _FASTATTN is not None, "FastAttn state is not initialized"
    return _FASTATTN


def get_fast_attn_enable() -> bool:
    """Return whether fast attention is enabled."""
    return get_fast_attn_state().enable


def get_fast_attn_step() -> int:
    """Return the fast attention step."""
    return get_fast_attn_state().n_step


def get_fast_attn_calib() -> int:
    """Return the fast attention calibration."""
    return get_fast_attn_state().n_calib


def get_fast_attn_threshold() -> float:
    """Return the fast attention threshold."""
    return get_fast_attn_state().threshold


def get_fast_attn_window_size() -> int:
    """Return the fast attention window size."""
    return get_fast_attn_state().window_size


def get_fast_attn_coco_path() -> Optional[str]:
    """Return the fast attention coco path."""
    return get_fast_attn_state().coco_path


def get_fast_attn_use_cache() -> bool:
    """Return the fast attention use_cache."""
    return get_fast_attn_state().use_cache


def get_fast_attn_config_file() -> str:
    """Return the fast attention config file."""
    return get_fast_attn_state().config_file


def get_fast_attn_layer_name() -> str:
    """Return the fast attention layer name."""
    return get_fast_attn_state().layer_name


def initialize_fast_attn_state(pipeline: DiffusionPipeline, single_config: FastAttnConfig):
    global _FASTATTN
    if _FASTATTN is not None:
        logger.warning("FastAttn state is already initialized, reinitializing with pipeline...")
    _FASTATTN = FastAttnState(pipe=pipeline, config=single_config)
