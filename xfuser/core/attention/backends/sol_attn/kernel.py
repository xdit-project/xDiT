"""Sol-Attn forward.

USP hands backends BHSD. Sol-Attn wants contiguous BF16 BTHD with head
dimension 128 and equal Q, K, and V shapes. ``accepts`` refuses anything
else, and the spec's SDPA fallback serves those calls.
"""

import torch
from sol_attn import sol_attn
from torch.library import custom_op

from xfuser.core.attention.numerics.layout import from_bshd, to_bshd
from xfuser.core.attention.spec import AttnCall

_AUTO_KV_TOKENS = 65536
_DEFAULTS = {
    "sol_attn_tau": 0.2,
    "sol_attn_thresh_type": "exact",
    "sol_attn_kv_splits": "auto",
    "sol_attn_sink_tokens": 0,
    "sol_attn_sink_start": None,
}
_THRESH = {"diag": 0, "exact": 1}


def _maybe_runtime_config():
    """Runtime knobs, when xDiT has been initialized. Absent otherwise.

    Imported lazily: this module is loaded when the backend is selected, and
    the runtime module is not needed to resolve the kernel.
    """

    try:
        from xfuser.core.distributed.runtime_state import (
            get_runtime_state,
            runtime_state_is_initialized,
        )
    except ImportError:
        return None
    if not runtime_state_is_initialized():
        return None
    return get_runtime_state().runtime_config


def _configured(call: AttnCall) -> dict:
    """Per-call kwargs win, then the runtime config, then the kernel defaults."""

    kwargs = call.attention_kwargs
    cfg = None
    values = {}
    for key, default in _DEFAULTS.items():
        if key in kwargs and kwargs[key] is not None:
            values[key] = kwargs[key]
            continue
        if key in kwargs and key == "sol_attn_sink_start":
            values[key] = None
            continue
        if cfg is None:
            cfg = _maybe_runtime_config()
        if cfg is not None and hasattr(cfg, key):
            found = getattr(cfg, key)
            values[key] = default if found is None and key != "sol_attn_sink_start" else found
        else:
            values[key] = default
    return values


def _resolve_kv_splits(query: torch.Tensor, kv_splits) -> int:
    """Match Sana's integration policy: 4 only for long SM90 CuTe sequences."""

    if kv_splits not in (None, "auto"):
        value = int(kv_splits)
        if value not in (1, 2, 4):
            raise ValueError(f"sol_attn_kv_splits must be auto, 1, 2, or 4, got {kv_splits}")
        return value
    if query.device.type == "cuda" and query.shape[1] >= _AUTO_KV_TOKENS and _cute_runtime_available():
        arch = tuple(torch.cuda.get_device_capability(query.device))
        if arch == (9, 0):
            return 4
    return 1


def _cute_runtime_available() -> bool:
    """True only when the CuTe kernels can actually be imported.

    ``sol_attn`` treats any importable ``cutlass.cute`` as CuTe-capable. CUTLASS
    DSL 4.3 imports, then the SM90 kernel fails on ``PipelineClcFetchAsync``.
    That decision stays here so the Sana package is used unchanged: without
    the symbol, this backend calls the Triton entry point instead.
    """

    try:
        import cuda.bindings.driver  # noqa: F401
        import cutlass.cute  # noqa: F401
        from cutlass.pipeline import PipelineClcFetchAsync  # noqa: F401
        from cutlass.utils import ClcDynamicPersistentTileScheduler  # noqa: F401
    except ImportError:
        return False
    return True


def _forward(query, key, value, *, tau, thresh_type, kv_splits, sink_tokens, sink_start):
    if _cute_runtime_available():
        return sol_attn(
            query,
            key,
            value,
            tau=tau,
            thresh_type=thresh_type,
            kv_splits=kv_splits,
            sink_tokens=sink_tokens,
            sink_start=sink_start,
        )
    # Triton has no KV split. The public dispatcher would still select CuTe
    # on this CUTLASS and crash while importing the SM90 kernel.
    from sol_attn.triton_ref import sol_attn as triton_sol_attn

    return triton_sol_attn(
        query,
        key,
        value,
        tau=tau,
        thresh_type=thresh_type,
        sink_tokens=sink_tokens,
        sink_start=sink_start,
    )


@custom_op("xfuser::sol_attn", mutates_args=())
def _sol_attn_op(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    tau: float,
    thresh_type: int,
    kv_splits: int,
    sink_tokens: int,
    sink_start: int,
) -> torch.Tensor:
    kind = "exact" if thresh_type else "diag"
    splits = kv_splits if kv_splits > 0 else _resolve_kv_splits(query, "auto")
    return _forward(
        query,
        key,
        value,
        tau=tau,
        thresh_type=kind,
        kv_splits=splits,
        sink_tokens=sink_tokens,
        sink_start=None if sink_start < 0 else sink_start,
    )


@_sol_attn_op.register_fake
def _sol_attn_fake(query, key, value, tau, thresh_type, kv_splits, sink_tokens, sink_start):
    return torch.empty_like(query)


def sol_attention(query, key, value, call: AttnCall):
    q, k, v = to_bshd(query, key, value, contiguous=True)
    cfg = _configured(call)
    thresh = cfg["sol_attn_thresh_type"]
    if thresh not in _THRESH:
        raise ValueError(f"sol_attn_thresh_type must be 'diag' or 'exact', got {thresh!r}")
    kv_splits = cfg["sol_attn_kv_splits"]
    splits = -1 if kv_splits in (None, "auto") else _resolve_kv_splits(q, kv_splits)
    sink_start = cfg["sol_attn_sink_start"]
    out = _sol_attn_op(
        q,
        k,
        v,
        float(cfg["sol_attn_tau"]),
        _THRESH[thresh],
        splits,
        int(cfg["sol_attn_sink_tokens"]),
        -1 if sink_start is None else int(sink_start),
    )
    return from_bshd(out), None
