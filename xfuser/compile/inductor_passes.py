"""
Custom Inductor post-grad passes for working around scheduler issues.

Currently provides:

- ``fix_sage_joint_cat_fusion``: defeats a bad Inductor scheduling decision
  where a ``cat``-then-reduction pattern (used by AITER ``sage_quant`` in
  Hunyuan's asymmetric joint attention path) gets inlined into a strided-
  gather reduction kernel, causing a large per-call slowdown.

The pass is pattern-matched on the FX graph and is a no-op when the bad
pattern is not present, so it is safe to install globally; non-matching
configurations are untouched.
"""

from __future__ import annotations

import logging

import torch
import torch.fx as fx


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Realize marker custom op
# ---------------------------------------------------------------------------
# Inductor treats torch.library.custom_op calls as opaque kernels: their
# inputs must be materialized in real buffers, their outputs are realized
# buffers, and the op is not fused with neighbours. We use it here only as
# a marker, inserted by the pass below between an aten.cat node and its
# reduction consumers, to break a specific fusion that Inductor's scheduler
# handles badly.
#
# The op itself is semantically a clone. In the unlikely event that the
# pass fires outside torch.compile (eager fallback), it still produces a
# correct result; the cost is one extra clone.

_REALIZE_OP_NAME = "xfuser::_realize_for_inductor"


def _maybe_register_realize_op() -> None:
    """Register the realize marker once. Safe to call multiple times."""
    ns, name = _REALIZE_OP_NAME.split("::")
    if hasattr(torch.ops, ns) and hasattr(getattr(torch.ops, ns), name):
        return

    @torch.library.custom_op(_REALIZE_OP_NAME, mutates_args=())
    def _realize_for_inductor(x: torch.Tensor) -> torch.Tensor:
        return x.clone()

    @_realize_for_inductor.register_fake
    def _(x: torch.Tensor) -> torch.Tensor:
        return torch.empty_like(x)


# ---------------------------------------------------------------------------
# Post-grad pass
# ---------------------------------------------------------------------------

def _is_reduction_target(target) -> bool:
    """Targets that participate in the bad fusion as cat consumers.

    Sage's ``v.abs().amax(dim=2)`` lowers via ``aten.abs.default`` followed
    by ``aten.amax.default``; Sage's ``k.mean(dim=2, keepdim=True)`` lowers
    via ``aten.mean.dim``. We match on the immediate consumer of the cat,
    so the relevant targets are ``aten.abs.default`` and ``aten.mean.dim``.
    """
    return target in {
        torch.ops.aten.abs.default,
        torch.ops.aten.mean.dim,
    }


def fix_sage_joint_cat_fusion(graph: fx.Graph) -> fx.Graph:
    """Insert an opaque realize marker between an ``aten.cat`` node and its
    reduction consumers, only when the cat has >= 2 users and at least one
    is ``aten.abs.default`` or ``aten.mean.dim``.

    This is the precise FX pattern produced by xDiT's asymmetric joint
    attention path (``_concat_joint_tensor`` after Ulysses all-to-all) when
    combined with the AITER Sage backend. In that pattern, Inductor's
    scheduler picks "re-derive the cat from sources" for the reduction
    consumer, producing a strided-gather kernel over the post-A2A view
    chain. With this pass, the reduction consumer is forced through an opaque op,
    so Inductor must use the materialized cat buffer 
    (which it is already producing for the sage kernel) and emits a coalesced read.

    The pass is a no-op for graphs that don't contain the pattern.
    """
    _maybe_register_realize_op()

    cat_target = torch.ops.aten.cat.default
    realize_target = torch.ops.xfuser._realize_for_inductor.default

    # Diagnostic counters
    num_cat_nodes = 0
    num_cat_with_reduction = 0
    num_cat_with_multi_users = 0
    num_matches = 0
    sample_user_targets: list[str] = []

    for node in list(graph.nodes):
        if node.op != "call_function" or node.target != cat_target:
            continue

        num_cat_nodes += 1
        users = list(node.users.keys())

        # Diagnostic: record the targets of the first few cat nodes' users
        if len(sample_user_targets) < 7:
            user_target_strs = [
                f"{u.op}:{getattr(u.target, '__qualname__', repr(u.target))}"
                for u in users
            ]
            sample_user_targets.append(
                f"cat#{num_cat_nodes}(n_users={len(users)}) users={user_target_strs}"
            )

        reduction_users = [
            u for u in users
            if u.op == "call_function" and _is_reduction_target(u.target)
        ]
        if reduction_users:
            num_cat_with_reduction += 1
        if len(users) >= 2:
            num_cat_with_multi_users += 1

        # Match the bad pattern: cat is consumed by a reduction (sage's eager
        # v.abs().amax() or k.mean()) AND has at least one other user (the
        # view chain that feeds the sage kernel's V_Input/K_Input). When both
        # are true, Inductor inlines the cat into the reduction kernel instead
        # of reusing the cat buffer that's being materialized for the other
        # consumer path.
        if not reduction_users or len(users) < 2:
            continue

        # Pattern matched: insert realize marker between cat and reduction users.
        with graph.inserting_after(node):
            realize_node = graph.call_function(realize_target, args=(node,))
            if "val" in node.meta:
                realize_node.meta["val"] = node.meta["val"]
            if "tensor_meta" in node.meta:
                realize_node.meta["tensor_meta"] = node.meta["tensor_meta"]

        for ru in reduction_users:
            ru.replace_input_with(node, realize_node)

        num_matches += 1

    if num_matches > 0:
        logger.info(
            "fix_sage_joint_cat_fusion: scanned graph: %d cat nodes total, "
            "%d with multiple users, %d with reduction user, %d matches inserted. "
            "Sample cat-user targets: %s",
            num_cat_nodes,
            num_cat_with_multi_users,
            num_cat_with_reduction,
            num_matches,
            " | ".join(sample_user_targets) if sample_user_targets else "(none)",
        )
        graph.lint()

    return graph


# ---------------------------------------------------------------------------
# Installation
# ---------------------------------------------------------------------------

def install() -> None:
    """Register :func:`fix_sage_joint_cat_fusion` as Inductor's post-grad
    custom_post_pass.

    Idempotent and compositional:

    - If nothing is registered, installs ``fix_sage_joint_cat_fusion`` directly.
    - If another pass is already registered, installs a wrapper that runs the
      existing pass first and then ``fix_sage_joint_cat_fusion``. The existing
      pass's behavior is preserved.
    - If a chain we previously created (or ``fix_sage_joint_cat_fusion`` itself)
      is already installed, this call is a no-op — re-installing does not grow
      the chain across repeated ``_compile_model`` invocations.

    Must be called before :func:`torch.compile` so the hook is in place when
    Inductor lowers any subsequent graph.
    """
    import torch._inductor.config as inductor_config

    _maybe_register_realize_op()

    existing = getattr(inductor_config, "post_grad_custom_post_pass", None)
    if existing is fix_sage_joint_cat_fusion:
        return

    if existing is not None:
        if getattr(existing, "_xfuser_composed", False):
            return

        def composed_pass(graph: fx.Graph) -> fx.Graph:
            result = existing(graph)
            # FX passes may mutate in-place and return None.
            # Normalize so our pass always receives a Graph.
            if result is None:
                result = graph
            return fix_sage_joint_cat_fusion(result)

        composed_pass._xfuser_composed = True
        inductor_config.post_grad_custom_post_pass = composed_pass
        logger.warning(
            "post_grad_custom_post_pass is already set to %r; chained "
            "fix_sage_joint_cat_fusion with it.",
            existing,
        )
    else:
        inductor_config.post_grad_custom_post_pass = fix_sage_joint_cat_fusion
        logger.warning(
            "fix_sage_joint_cat_fusion: registered as post_grad_custom_post_pass"
        )
