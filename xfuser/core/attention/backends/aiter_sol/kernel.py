"""AITER Sol-Attn launcher. Imported when one of the family is selected."""

from xfuser.core.attention.sparsity.head_balance import COST_SINK_KEY
from xfuser.core.attention.sparsity.sol import SolAttnUnsupported, check_sol_attn_recipe, sol_attn_bhsd
from xfuser.core.attention.spec import AttnCall
from xfuser.core.distributed.runtime_state import (
    get_runtime_state,
    get_scheduled_solattn_beta,
    runtime_state_is_initialized,
)


def _beta(kwargs):
    """This call's routing threshold: the step's scheduled beta if one is driving it, else the flag.

    The schedule wins over the caller's dict rather than the other way round, because the dict is
    built once per run from --solattn_beta and would otherwise pin every step to the same value --
    the very thing a schedule exists to stop. A caller with no schedule is unaffected.
    """
    scheduled = get_scheduled_solattn_beta()
    if scheduled is not None:
        # Handed on as the 0-d tensor it is. float() here would read it on the host inside the
        # compiled forward, which both breaks the graph and pins this step's beta into the graph as
        # a constant, recompiling for every beta the schedule holds. The routing consumes it as a
        # scalar operand of the threshold either way.
        return scheduled
    return float(kwargs.get("solattn_beta", 0.5))


def _block_tile():
    """--sparseattn_block_size, or None for the recipe's default."""
    if not runtime_state_is_initialized():
        return None
    return get_runtime_state().runtime_config.sparseattn_block_size


def check_sparse_block_size(runtime_config, *, recipe: str) -> None:
    """Refuse a --sparseattn_block_size that `recipe`'s Sol-Attn row is not built at."""
    block_tile = runtime_config.sparseattn_block_size
    if block_tile is None:
        return
    try:
        check_sol_attn_recipe(recipe, block_tile=block_tile)
    except SolAttnUnsupported as error:
        raise ValueError(str(error)) from None


def sol(query, key, value, call: AttnCall, *, recipe: str):
    """Sol-Attn on the mode-2 row for `recipe`.

    Unlike the Sparge backends this builds no block mask: the adaptive-threshold routing is part of
    the algorithm, and aiter derives the LUT, the pooled K/V and the selection bitmap together from
    one mask so they cannot disagree.

    A declared trailing key pad, which `accepts` has checked, is dropped by length rather than
    gathered. The kernels take the remaining key count as it is, so a sequence that is no multiple
    of the KV tile is attended exactly as long as it is.
    """
    kwargs = call.attention_kwargs
    cost_sink = kwargs.get(COST_SINK_KEY)
    output, head_cost = sol_attn_bhsd(
        query,
        key,
        value,
        is_causal=call.is_causal,
        beta=_beta(kwargs),
        ring_world_size=call.ring_world_size,
        return_head_cost=cost_sink is not None,
        recipe=recipe,
        key_seqlen=None if call.varlen is None else int(kwargs["valid_kv_len"]),
        block_tile=_block_tile(),
    )
    if cost_sink is not None:
        cost_sink.copy_(head_cost)
    return output, None
