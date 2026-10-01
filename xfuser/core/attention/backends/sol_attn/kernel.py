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
from xfuser.core.distributed.runtime_state import get_runtime_state

_AUTO_KV_TOKENS = 65536
_THRESH = {"diag": 0, "exact": 1}


def _resolve_kv_splits(query: torch.Tensor, kv_splits) -> int:
    """Match Sana's integration policy: 4 only for long SM90 CuTe sequences."""

    if kv_splits not in (None, "auto"):
        value = int(kv_splits)
        if value not in (1, 2, 4):
            raise ValueError(f"sol_attn_kv_splits must be auto, 1, 2, or 4, got {kv_splits}")
        return value
    if query.device.type == "cuda" and query.shape[1] >= _AUTO_KV_TOKENS and _CUTE_RUNTIME_AVAILABLE:
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


_CUTE_RUNTIME_AVAILABLE = _cute_runtime_available()

if _CUTE_RUNTIME_AVAILABLE:

    def _forward(query, key, value, *, tau, thresh_type, kv_splits, sink_tokens, sink_start):
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

else:
    # Triton has no KV split. The public dispatcher would still select CuTe
    # on this CUTLASS and crash while importing the SM90 kernel.
    from sol_attn.triton_ref import sol_attn as triton_sol_attn

    def _forward(query, key, value, *, tau, thresh_type, kv_splits, sink_tokens, sink_start):
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
    runtime_config = get_runtime_state().runtime_config
    thresh = runtime_config.sol_attn_thresh_type
    if thresh not in _THRESH:
        raise ValueError(f"sol_attn_thresh_type must be 'diag' or 'exact', got {thresh!r}")
    kv_splits = runtime_config.sol_attn_kv_splits
    splits = -1 if kv_splits in (None, "auto") else _resolve_kv_splits(q, kv_splits)
    sink_start = runtime_config.sol_attn_sink_start
    out = _sol_attn_op(
        q,
        k,
        v,
        float(runtime_config.sol_attn_tau),
        _THRESH[thresh],
        splits,
        int(runtime_config.sol_attn_sink_tokens),
        -1 if sink_start is None else int(sink_start),
    )
    return from_bshd(out), None
