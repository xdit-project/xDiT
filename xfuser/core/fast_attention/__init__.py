"""DiTFastAttn state, attention layer, and compression config file.

This package must not import ``xfuser.model_executor`` when it loads:
``xfuser.model_executor.base_wrapper`` imports this package. The calibration
is in ``xfuser.model_executor.fast_attention_calibration``.
``fast_attention_compression`` stays an export of this package for
compatibility, and loads on first access.
"""

from xfuser.compat import lazy_exports

__getattr__, __dir__ = lazy_exports(
    globals(),
    {
        ".fast_attn_state": (
            "get_fast_attn_state",
            "get_fast_attn_enable",
            "get_fast_attn_step",
            "get_fast_attn_calib",
            "get_fast_attn_threshold",
            "get_fast_attn_window_size",
            "get_fast_attn_coco_path",
            "get_fast_attn_use_cache",
            "get_fast_attn_config_file",
            "get_fast_attn_layer_name",
            "initialize_fast_attn_state",
        ),
        ".attn_layer": ("xFuserFastAttention", "FastAttnMethod"),
        "xfuser.model_executor.fast_attention_calibration": ("fast_attention_compression",),
    },
)
