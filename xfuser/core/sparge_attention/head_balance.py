import torch

# Gate for the feature. Sourced from the --use_spargeattn_head_balance CLI flag
# via set_enabled(), which the runner calls ONCE before torch.compile
ENABLED: bool = False

# Key under which USP injects the per-call "cost sink" tensor into a shallow
# copy of attention_kwargs.
COST_SINK_KEY: str = "_hb_cost_sink"


def set_enabled(flag: bool) -> None:
    """Enable/disable head balancing. Must be called before torch.compile so the
    flag is baked as a trace-time constant."""
    global ENABLED
    ENABLED = bool(flag)


def enabled() -> bool:
    return ENABLED


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
