"""AITER VSA: Jenga block-sparse self-attention over a spatial layout."""

from xfuser.core.attention.backends.aiter import AITER_ARCH, aiter_attention
from xfuser.core.attention.constraints import NO_DROPOUT, NO_VARLEN, NON_CAUSAL
from xfuser.core.attention.requirements import SYMBOL
from xfuser.core.attention.spec import AttentionBackendType, AttnCall, Spec

_VSA = "aiter.ops.jenga_sparse_attention:vsa_sparse_attention"


def vsa_attention(query, key, value, call: AttnCall):
    """Sparse when the model publishes a spatial layout, dense otherwise.

    Cross-attention and MiniMax-H3's token refiner carry no `thw`, and the
    drop-rate schedule marks early/late steps dense; all three run dense AITER
    rather than failing.
    """
    from xfuser.core.vsa_attention import aiter_vsa_attention, jenga_scheduled_drop_rate

    kwargs = call.attention_kwargs
    thw = kwargs.get("thw")
    is_self_attention = query.shape == key.shape == value.shape
    if thw is None or not is_self_attention:
        return aiter_attention(query, key, value, call)

    drop_rate = None
    drop_rates = kwargs.get("vsa_drop_rates")
    if drop_rates:
        drop_rate = kwargs.get("vsa_effective_drop_rate")
        if drop_rate is None:
            drop_rate = jenga_scheduled_drop_rate(
                int(kwargs.get("vsa_step_index", 0)),
                int(kwargs.get("vsa_num_steps", 1)),
                drop_rates,
            )
        use_dense = bool(kwargs.get("vsa_use_dense", drop_rate <= 0.25))
        kwargs["vsa_effective_drop_rate"] = drop_rate
        kwargs["vsa_use_dense"] = use_dense
        if use_dense:
            return aiter_attention(query, key, value, call)

    collect_density = bool(kwargs.get("vsa_collect_density", False))
    output, density = aiter_vsa_attention(
        query, key, value,
        thw=tuple(thw),
        sp_size=call.ctx.ulysses_world_size,
        block_size=int(kwargs.get("vsa_block_size", 128)),
        top_k=int(kwargs.get("vsa_top_k", 1)),
        top_k_ratio=float(kwargs.get("vsa_top_k_ratio", 0.0)),
        drop_rate=drop_rate,
        prob_threshold=float(kwargs.get("vsa_prob_threshold", 0.9)),
        reorder_sequence=bool(kwargs.get("vsa_reorder_sequence", True)),
        use_static_block_mask=bool(kwargs.get("use_vsa_static_block_mask", True)),
        use_first_frame_mask=bool(kwargs.get("use_vsa_first_frame_mask", True)),
        collect_density=collect_density,
    )
    if density is not None:
        kwargs["vsa_last_density"] = density.detach()
    return output, None


SPECS = [
    Spec(
        AttentionBackendType.AITER_VSA,
        impl=vsa_attention,
        sparsity="vsa",
        low_precision=True,
        # Causal and dropout are refused outright; the dense-routing cases are
        # handled inside the kernel because they depend on the metadata, not
        # the tensors.
        accepts=NON_CAUSAL & NO_DROPOUT & NO_VARLEN,
        requires=SYMBOL(_VSA) & AITER_ARCH,
    ),
]
