import torch
import torch.distributed._functional_collectives as ft_c

import xfuser.envs as envs

if torch.cuda.is_available() or envs._is_npu():
    from yunchang.globals import PROCESS_GROUP
else:
    PROCESS_GROUP = None

# Key under which USP injects the per-call "cost sink" tensor into a shallow copy
# of attention_kwargs. The sparge backend writes this rank's per-head cost into
# it; the key's PRESENCE is what activates cost publishing in the backend (so no
# separate global enable flag is needed).
COST_SINK_KEY: str = "_hb_cost_sink"

# Key under which the output-revert bookkeeping (applied + inverse head
# permutation) is stashed in attention_kwargs, ignored by the backend.
_HB_STATE_KEY: str = "_hb_revert_state"


def _maybe_wait(tensor: torch.Tensor) -> torch.Tensor:
    """Resolve an async functional-collective tensor. No-op while tracing, where
    the result is not an ``AsyncCollectiveTensor``."""
    if isinstance(tensor, ft_c.AsyncCollectiveTensor):
        return tensor.wait()
    return tensor

def compute_perm(cost: torch.Tensor, world_size: int) -> torch.Tensor:
    """Balanced equal-cardinality head permutation from per-head costs.

    ``cost`` is a 1-D tensor (length H, global head order). Returns a long
    permutation (length H) that groups heads into ``world_size`` equal bins
    (H/world_size heads each) so the per-bin cost sums are balanced, using a
    sorted-snake assignment (sort by cost desc, then deal heads to bins in a
    boustrophedon order so heavy and light heads pair up).
    """
    H = cost.shape[0]
    order = torch.argsort(cost, descending=True)            # densest-first head ids
    pos = torch.arange(H, device=cost.device)
    rnd = pos // world_size                                  # which "deal round"
    col = pos % world_size                                   # position within the round
    # Snake: even rounds deal left->right, odd rounds right->left.
    rank_of_sorted = torch.where(rnd % 2 == 0, col, (world_size - 1) - col)
    # Stable group-by-rank preserves the (descending-cost) order within each bin.
    group_order = torch.argsort(rank_of_sorted, stable=True)
    return order[group_order]


def scatter_to_global(full_in_applied_order: torch.Tensor,
                      applied_perm: torch.Tensor) -> torch.Tensor:
    """Map per-head costs gathered in applied-permutation (rank, local) order
    back to global head order: global[applied_perm[j]] = full[j]. Traceable."""
    H = full_in_applied_order.shape[0]
    glob = torch.zeros(H, dtype=full_in_applied_order.dtype,
                       device=full_in_applied_order.device)
    return glob.index_copy(0, applied_perm.to(full_in_applied_order.device),
                           full_in_applied_order)


def apply_head_balance(query, key, value, head_balance_layer, *,
                       enabled, ulysses_world_size, ring_world_size,
                       is_sparge_backend, joint_strategy, attention_kwargs):
    """Conditionally apply Ulysses block-sparse head balancing before the input
    all-to-all.

    Balancing engages only when the feature flag is set and we are on the
    Ulysses-only sparge path with a per-layer balancing buffer present. When it
    does, Q,K,V heads are permuted by this step's plan so each rank receives a
    cost-balanced subset, and a per-head "cost sink" (for the backend to fill)
    plus the revert bookkeeping (applied + inverse permutation) are stashed in a
    shallow-copied attention_kwargs -- keeping the head cost OUT of the
    (output, softmax_lse) contract and leaving USP with no extra locals.

    Returns ``(query, key, value, hb_applied, attention_kwargs)``; when not
    applied the inputs are returned unchanged with ``hb_applied=False``. The gate
    is composed of trace-time constants so it folds away cleanly under compile.
    """
    head_perm = getattr(head_balance_layer, "head_perm", None)
    hb_applied = (
        enabled
        and ulysses_world_size > 1
        and ring_world_size == 1
        and head_perm is not None
        and is_sparge_backend
        and joint_strategy is None
        and query.shape[1] % ulysses_world_size == 0
    )
    if not hb_applied:
        return query, key, value, False, attention_kwargs

    applied_perm = head_perm.clone()  # snapshot the permutation applied this step
    inv_perm = torch.argsort(applied_perm)
    query = query.index_select(1, applied_perm)
    key = key.index_select(1, applied_perm)
    value = value.index_select(1, applied_perm)
    cost_sink = query.new_zeros(query.shape[1] // ulysses_world_size,
                                dtype=torch.float32)
    attention_kwargs = {
        **(attention_kwargs or {}),
        COST_SINK_KEY: cost_sink,
        _HB_STATE_KEY: (applied_perm, inv_perm),
    }
    return query, key, value, True, attention_kwargs


def revert_head_balance(out, attention_kwargs, head_balance_layer,
                        ulysses_world_size):
    """Undo head balancing on the attention output and plan the next step.

    Reads the cost sink and applied/inverse permutation back out of
    ``attention_kwargs`` (stashed by ``apply_head_balance``). Restores the
    original (global) head order on ``out``, all-gathers this step's per-head
    costs across the Ulysses group, maps them back to global head order, and
    stores next step's balanced permutation into the per-layer ``head_perm``
    buffer. Returns the reordered ``out``.
    """
    applied_perm, inv_perm = attention_kwargs[_HB_STATE_KEY]
    cost_sink = attention_kwargs[COST_SINK_KEY]
    out = out.index_select(1, inv_perm)
    full = _maybe_wait(
        ft_c.all_gather_tensor(cost_sink, 0, PROCESS_GROUP.ULYSSES_PG)
    )
    glob = scatter_to_global(full, applied_perm)
    head_balance_layer.head_perm.copy_(compute_perm(glob, ulysses_world_size))
    return out
