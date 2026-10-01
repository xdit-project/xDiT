"""What a blockwise load already quantized, recorded so nothing quantizes it twice.

A component filled block by block is converted on the way in, before FSDP wraps the block. The
post-load walks cannot see that happened, so each blockwise load records the module paths it owns and
the walks skip them. Getting this wrong is quiet in both directions: an unrecorded target is converted
a second time, and an over-recorded one is never converted at all.

Ownership is recorded against the *wrapped* paths rather than the declared targets, because those are
what the fill actually covered. A target naming a whole component is narrowed to the blocks that were
wrapped, and a target below a block is kept as-is; ``blockwise_owned_targets`` reduces the pair to the
shortest paths that cover it.
"""

from dataclasses import replace

from xfuser.core.utils.runner_utils import log
from .format_backends import module_path_is_covered


def blockwise_owned_targets(targets, wrap_attrs):
    """The shortest paths covering what the block fill quantized.

    A target and a wrap_attr can contain one another either way round: "transformer_blocks" covers a
    target of "transformer_blocks.0.attn", and a target of "transformer_blocks" covers the wrap_attr.
    Both directions count as owned, then the result is reduced so no path is listed under another.
    """
    owned = []
    for target in targets:
        for wrap_attr in wrap_attrs:
            if module_path_is_covered(target, wrap_attr):
                owned.append(target)
            elif module_path_is_covered(wrap_attr, target):
                owned.append(wrap_attr)
    minimal = []
    for target in sorted(set(owned), key=lambda path: (path.count("."), path)):
        if not any(module_path_is_covered(target, owner) for owner in minimal):
            minimal.append(target)
    return tuple(minimal)


def record_blockwise_ownership(
    ledger,
    adapter,
    component_name,
    targets,
    wrap_attrs,
    descriptor,
    *,
    remainder=(),
):
    """Log the plan and record what it will have quantized by the time the fill finishes.

    ``remainder`` is what the same fill converts besides its own targets, as
    ``(format_name, targets)`` pairs: a blockwise fill walks each block once and
    places every format that block holds, so the ledger has to learn about all
    of them. Which formats those are is the plan's business -- this used to ask
    whether the run was FP4 and then pick between an ``fp8_targets`` and an
    ``mxfp6_targets`` argument, which stopped being true the moment any format
    could be the low one.
    """
    log(descriptor.log_message())
    ledger.describe(component_name, format_name=adapter.format.value)
    if descriptor.materialization_mode not in {"streaming", "blockwise"}:
        return
    ledger.record_streamed(
        component_name,
        blockwise_owned_targets(targets, wrap_attrs),
        format_name=adapter.format.value,
    )
    if descriptor.materialization_mode != "blockwise":
        return
    for format_name, remainder_targets in remainder:
        if not remainder_targets:
            continue
        ledger.record_streamed(
            component_name,
            blockwise_owned_targets(tuple(remainder_targets), wrap_attrs),
            format_name=format_name,
        )


def blockwise_transformer_descriptor(
    adapter,
    component_name,
    targets,
    wrap_attrs,
):
    """How one component's blockwise load will be performed, for logging and ownership.

    A single-rank fill reaches the same per-block conversion without the
    collective, and is described the same way, so the ownership rules above
    need no special case for it.
    """
    from .quant_adapter import describe_blockwise_load

    return describe_blockwise_load(
        adapter,
        component_name=component_name,
        targets=targets,
        wrap_attrs=wrap_attrs,
    )
