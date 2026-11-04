"""AITER MHA v4 Sol-Attn: one kernel, one routing algorithm, a table of recipes.

Sol-Attn (arXiv 2607.24027) computes the KV blocks its routing selects exactly
and recovers the rest from pooled K/V under the same softmax, so a skipped
block costs its higher-order terms rather than all of its mass. Every recipe
shares the routing, the pooled correction and the per-head cost; they differ
only in how Q/K/V are quantized, which is all a row of the table says.

Not to be confused with SOL_ATTN, the NVIDIA kernel from the sol_attn package.
"""

from dataclasses import dataclass
from typing import Optional

from xfuser.core.attention.constraints import (
    BF16,
    HEAD_DIM,
    NO_DROPOUT,
    NON_CAUSAL,
    PACKED_KEYS,
    CallConstraint,
)
from xfuser.core.attention.requirements import ARCH, NEVER, SYMBOL, Requirement
from xfuser.core.attention.spec import AttentionBackendType, Impl, Sparsity, Spec

GFX950 = ARCH("gfx950")


@dataclass(frozen=True)
class SolRecipe:
    name: str  # AITER_<name>_SOL
    recipe: str  # the row's id in xfuser.core.attention.sparsity.sol
    archs: Requirement
    low_precision: bool


# fmt: off
# Q/K format then V format, as the dense MHA v4 table names them: F8F6 is FP8
# Q/K with MXFP6 V, F6F6 MXFP6 throughout. The MX-V rows are gfx950's FP6-P
# kernels.
RECIPES = [
    #          name       recipe     on      low precision
    SolRecipe("BF16",    "bf16",    GFX950, False),
    SolRecipe("BF16FP8", "bf16fp8", GFX950, False),
    SolRecipe("I8FP8",   "i8fp8",   GFX950, True),
    SolRecipe("FP8",     "fp8",     GFX950, True),
    SolRecipe("MXFP8",   "mxfp8",   GFX950, True),
    SolRecipe("F8F6",    "f8f6",    GFX950, True),
    SolRecipe("F6F6",    "f6f6",    GFX950, True),
    SolRecipe("F6F4",    "f6f4",    GFX950, True),
    SolRecipe("MXFP4",   "mxfp4",   GFX950, True),
]
# fmt: on

# The Sol rows by backend, for checking --sparseattn_block_size against the
# recipe a selected backend dispatches.
SOL_RECIPES = {AttentionBackendType[f"AITER_{row.name}_SOL"]: row.recipe for row in RECIPES}


@dataclass(frozen=True)
class _SolRow(Requirement):
    """The device has this recipe's mode-2 row. Whether it also has one at a
    --sparseattn_block_size the run asks for is checked by ``validates``."""

    recipe: str

    def unmet(self) -> Optional[str]:
        from xfuser.core.attention.sparsity.sol import SolAttnUnsupported, check_sol_attn_recipe

        try:
            check_sol_attn_recipe(self.recipe)
        except SolAttnUnsupported as error:
            return str(error)
        return None


class _TrailingPadKeys(CallConstraint):
    """Packed keys only as one sequence's declared trailing pad.

    Sol-Attn has no mask, so the pad is dropped from K/V rather than attended,
    and every row this backend runs takes the shorter length as it is. Keys
    gathered from anywhere else -- several segments, or a pad that is not a
    trailing block -- would be attended across their boundaries, and the pooled
    blocks would straddle them too.
    """

    def unmet(self, query, key, value, call) -> Optional[str]:
        if call.varlen is None:
            return None
        if call.attention_kwargs.get("valid_kv_len") is None:
            return "serves packed keys only as a trailing pad declared through valid_kv_len"
        if key.shape[0] != 1:
            return f"serves a trailing key pad on one sequence only, got batch size {key.shape[0]}"
        return PACKED_KEYS.unmet(query, key, value, call)


TRAILING_PAD_KEYS = _TrailingPadKeys()

CALLS = NO_DROPOUT & NON_CAUSAL & BF16 & HEAD_DIM(128) & TRAILING_PAD_KEYS


def _spec(row: SolRecipe) -> Spec:
    return Spec(
        AttentionBackendType[f"AITER_{row.name}_SOL"],
        impl=Impl("kernel:sol", {"recipe": row.recipe}),
        # The kernel's LSE is exact, but routing is not shard invariant: the
        # threshold is taken over the blocks one rank holds.
        ring=NEVER,
        sparsity=Sparsity.SOL,
        head_balanced=True,
        low_precision=row.low_precision,
        accepts=CALLS,
        requires=(
            SYMBOL("aiter.ops.mha_v4:mha_v4_sol")
            & SYMBOL("aiter.ops.triton.attention.utils:sol_prepare")
            & row.archs
            & _SolRow(row.recipe)
        ),
        validates=Impl("kernel:check_sparse_block_size", {"recipe": row.recipe}),
    )


SPECS = [_spec(row) for row in RECIPES]
