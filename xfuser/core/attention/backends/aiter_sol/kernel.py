"""AITER Sol-Attn launcher. Imported when one of the family is selected."""

from typing import Optional

import torch

from xfuser.core.attention.sparsity.head_balance import COST_SINK_KEY
from xfuser.core.attention.sparsity.sol import (
    SolAttnUnsupported,
    check_sol_attn_recipe,
    check_sol_tilt,
    sol_attn_bhsd,
)
from xfuser.core.attention.spec import EXACT_KV_TOKENS_KEY, EXACT_Q_TOKENS_KEY, AttnCall
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

    Each beta is held across its step's forwards_per_step forwards, so a guided step's conditional
    and unconditional branches share it.
    """
    global _BETA_SCHEDULE
    schedule = SolAttnBetaSchedule.from_spec(spec, denoising_steps, forwards_per_step=forwards_per_step)
    device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else None
    if device is not None:
        # A table first built inside the compiled forward fails the guard its trace recorded.
        schedule.betas_on(device)
    get_runtime_state().track_steps(device)
    _BETA_SCHEDULE = schedule
    logger.warning(
        "Per-step Sol-Attn beta schedule enabled (%d steps, beta %.4f to %.4f).",
        denoising_steps,
        schedule.get_beta(0),
        schedule.get_beta(schedule.total_steps - 1),
    )
    return schedule


def configure_run(runtime_config, denoising_steps: Optional[int], forwards_per_step: int) -> None:
    """Install --solattn_beta_schedule for a run of `denoising_steps` steps, if one is set.

    Called by the runtime once the step plan is known, before the forward is compiled.
    """
    if runtime_config.solattn_beta_schedule is None:
        return
    if denoising_steps is None:
        raise ValueError("--solattn_beta_schedule needs the run's number of denoising steps, and this run has none.")
    schedule = install_beta_schedule(runtime_config.solattn_beta_schedule, denoising_steps, forwards_per_step)
    logger.debug("Sol-Attn beta schedule: %s", schedule.per_denoising_step)


def clear_beta_schedule() -> None:
    """Go back to --solattn_beta for every forward."""
    global _BETA_SCHEDULE
    _BETA_SCHEDULE = None


def _beta(call: AttnCall, device):
    """This call's routing threshold: the step's scheduled beta if one is installed, else the flag.

    A call with no step, from outside a pipeline run, takes the flag. The flag is the caller's
    `solattn_beta`, else --solattn_beta from the run's config, so a model need not pass it.
    """
    schedule = _BETA_SCHEDULE
    if schedule is not None and call.step is not None:
        # Kept a tensor: a Python float would break the graph and recompile it for every beta.
        return schedule.get_beta_tensor(call.step.to(device))
    beta = call.attention_kwargs.get("solattn_beta")
    if beta is None:
        beta = get_runtime_state().runtime_config.solattn_beta if runtime_state_is_initialized() else 0.5
    return float(beta)


def _block_tile():
    """--sparseattn_block_size, or None for the recipe's default."""
    if not runtime_state_is_initialized():
        return None
    return get_runtime_state().runtime_config.sparseattn_block_size


def _tilt_rank() -> int:
    """--solattn_tilt_rank, or 0 outside a run."""
    if not runtime_state_is_initialized():
        return 0
    return get_runtime_state().runtime_config.solattn_tilt_rank


def check_run_config(runtime_config, *, recipe: str) -> None:
    """Refuse a --sparseattn_block_size or --solattn_tilt_rank that `recipe`'s Sol-Attn row cannot serve."""
    try:
        check_sol_tilt(recipe, runtime_config.solattn_tilt_rank)
        if runtime_config.sparseattn_block_size is not None:
            check_sol_attn_recipe(recipe, block_tile=runtime_config.sparseattn_block_size)
    except SolAttnUnsupported as error:
        raise ValueError(str(error)) from None


def sol(query, key, value, call: AttnCall, *, recipe: str):
    """Sol-Attn on the mode-2 row for `recipe`.

    Builds no block mask: the routing selects the blocks itself. A declared trailing key pad, which
    `accepts` has checked, is dropped by length, and so is a query pad the caller declares as
    `valid_q_len`. EXACT_KV_TOKENS_KEY names keys whose blocks are always computed exactly, and
    EXACT_Q_TOKENS_KEY query rows whose tiles compute every block exactly.
    """
    kwargs = call.attention_kwargs
    valid_q_len = kwargs.get("valid_q_len") if call.varlen is not None else None
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
        query_seqlen=None if valid_q_len is None else int(valid_q_len),
        exact_tokens=kwargs.get(EXACT_KV_TOKENS_KEY),
        exact_query_tokens=kwargs.get(EXACT_Q_TOKENS_KEY),
        tilt_rank=_tilt_rank(),
    )
    if cost_sink is not None:
        cost_sink.copy_(head_cost)
    return output, None
