"""AITER Sol-Attn launcher. Imported when one of the family is selected."""

from typing import Optional

from xfuser.core.attention.sparsity.head_balance import COST_SINK_KEY
from xfuser.core.attention.sparsity.sol import SolAttnUnsupported, check_sol_attn_recipe, sol_attn_bhsd
from xfuser.core.attention.spec import AttnCall
from xfuser.core.distributed.attention_schedule import SolAttnBetaSchedule
from xfuser.core.distributed.runtime_state import (
    get_runtime_state,
    runtime_state_is_initialized,
)
from xfuser.logger import init_logger

logger = init_logger(__name__)

# --solattn_beta_schedule for this process's run, or None when --solattn_beta stands.
_BETA_SCHEDULE: Optional[SolAttnBetaSchedule] = None


def install_beta_schedule(spec: str, denoising_steps: int, forwards_per_step: int) -> SolAttnBetaSchedule:
    """Route each forward at its step's beta from `spec`, in place of --solattn_beta.

    The spec is read per denoising step and held across that step's forwards_per_step forwards,
    the same multiplier the hybrid schedules use. The step counter advances per forward, so the
    expansion is what makes a step's conditional and unconditional branch share a beta; spreading
    the ramp over forwards instead would route one branch more exactly than the other for the
    whole run, and guidance subtracts the two rather than averaging them.
    """
    global _BETA_SCHEDULE
    schedule = SolAttnBetaSchedule.from_spec(spec, denoising_steps, forwards_per_step=forwards_per_step)
    get_runtime_state().track_steps()
    _BETA_SCHEDULE = schedule
    logger.warning(
        "Per-step Sol-Attn beta schedule enabled (%d steps, beta %.4f to %.4f).",
        denoising_steps,
        schedule.get_beta(0),
        schedule.get_beta(schedule.total_steps - 1),
    )
    return schedule


def clear_beta_schedule() -> None:
    """Go back to --solattn_beta for every forward."""
    global _BETA_SCHEDULE
    _BETA_SCHEDULE = None


def _beta(call: AttnCall, device):
    """This call's routing threshold: the step's scheduled beta if one is driving it, else the flag.

    The schedule wins over the caller's dict rather than the other way round, because the dict is
    built once per run from --solattn_beta and would otherwise pin every step to the same value --
    the very thing a schedule exists to stop. A call with no step, from outside a pipeline run,
    takes the flag.
    """
    schedule = _BETA_SCHEDULE
    if schedule is not None and call.step is not None:
        # Handed on as a 0-d tensor on the routing's device. float() here would read it on the host
        # inside the compiled forward, which both breaks the graph and pins this step's beta into
        # the graph as a constant, recompiling for every beta the schedule holds.
        return schedule.get_beta_tensor(call.step.to(device))
    return float(call.attention_kwargs.get("solattn_beta", 0.5))


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
        beta=_beta(call, query.device),
        ring_world_size=call.ring_world_size,
        return_head_cost=cost_sink is not None,
        recipe=recipe,
        key_seqlen=None if call.varlen is None else int(kwargs["valid_kv_len"]),
        block_tile=_block_tile(),
    )
    if cost_sink is not None:
        cost_sink.copy_(head_cost)
    return output, None
