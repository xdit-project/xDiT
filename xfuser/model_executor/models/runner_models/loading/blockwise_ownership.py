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
    fp4_gemms=False,
    fp8_targets=(),
):
    """Log the plan and record what it will have quantized by the time the fill finishes.

    ``fp8_targets`` are the component's FP8 targets, needed only for the FP4 case below, so a
    caller that is not running FP4 gemms need not look them up.
    """
    log(descriptor.log_message())
    is_fp8 = adapter.format.value == "fp8"
    ledger.describe(component_name, fp8=is_fp8)
    if descriptor.materialization_mode not in {"streaming", "blockwise"}:
        return
    owned_targets = blockwise_owned_targets(targets, wrap_attrs)
    ledger.record_streamed(component_name, owned_targets, fp8=is_fp8)
    # An fp4 blockwise fill also converts the fp8 remainder its adapter leaves behind, so those
    # targets are owned too even though they are not the ones the descriptor names.
    if descriptor.materialization_mode == "blockwise" and fp4_gemms:
        ledger.record_streamed(
            component_name,
            blockwise_owned_targets(tuple(fp8_targets), wrap_attrs),
            fp8=True,
        )


def blockwise_transformer_descriptor(
    adapter,
    component_name,
    targets,
    wrap_attrs,
    *,
    local=False,
):
    """How one component's blockwise load will be performed, for logging and ownership.

    local marks a single-rank fill, which reaches the same per-block conversion without the
    collective; the fp8 planner describes that as streaming, so it is relabelled to match what the
    ownership rules above key off.
    """
    if adapter.format.value == "fp8":
        from .fp8_backends import plan_blockwise_transformer_fp8_load

        descriptor = plan_blockwise_transformer_fp8_load(
            adapter,
            component_name=component_name,
            targets=targets,
            wrap_attrs=wrap_attrs,
        )
        return (
            replace(descriptor, materialization_mode="blockwise")
            if local and descriptor.materialization_mode == "streaming"
            else descriptor
        )
    from .format_backends import describe_blockwise_format_load

    return describe_blockwise_format_load(
        adapter,
        component_name=component_name,
        targets=targets,
        wrap_attrs=wrap_attrs,
    )
