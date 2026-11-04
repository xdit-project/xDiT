"""The AITER Sol-Attn backends on the device.

Sol-Attn (arXiv 2607.24027) runs an AITER MHA v4 mode-2 row over the KV blocks its routing selects
and recovers the rest from pooled per-block K/V under the same online softmax, so a dropped block
costs its higher-order terms rather than all of its mass.

Nine rows are built, differing only in how Q/K/V are quantized: bf16, bf16fp8, fp8, i8fp8 and
mxfp8, and the four MX-V rows f8f6, f6f6, f6f4 and mxfp4 on gfx950's FP6-P kernels. Where a test is
about the mechanism rather than one row's precision it is parametrized over all of them, because
the rows reach the kernel by different routes: MXFP8 has no raw entry point, and the block-scaled
rows pool in dequantized space and hand the kernel a pooled scale of their own.

The CPU-side contract -- what the backends refuse, and what reaches sol_attn_bhsd -- is in
tests/unit/core/test_aiter_sol_contracts.py.
"""

import pytest
import torch
import torch.nn.functional as F

from xfuser.core.attention import registry
from xfuser.core.attention.spec import AttentionBackendType, AttnCall

pytestmark = pytest.mark.rocm

ALL_RECIPES = ["bf16", "bf16fp8", "fp8", "i8fp8", "mxfp8", "f8f6", "f6f6", "f6f4", "mxfp4"]

# Each recipe's dense MHA v4 row: the same quantization over every block, so what is left in the
# gap is the routing plus the pooled correction. f6f6 has none -- the dense table's MXFP6 row keeps
# V in FP8 -- so it is graded against fp32 only.
DENSE_SIBLING = {
    "bf16": "AITER_BF16",
    "bf16fp8": "AITER_BF16FP8",
    "fp8": "AITER_FP8",
    "i8fp8": "AITER_I8FP8",
    "mxfp8": "AITER_MXFP8",
    "f8f6": "AITER_F8F6",
    "f6f4": "AITER_F6F4",
    "mxfp4": "AITER_MXFP4",
}


def _backend(recipe):
    from xfuser.core.attention.backends.aiter_sol.spec import RECIPES

    return next(AttentionBackendType[f"AITER_{row.name}_SOL"] for row in RECIPES if row.recipe == recipe)


def _require(backend):
    """Skip unless this machine can run the backend, per its own spec."""
    if not torch.cuda.is_available():
        pytest.skip("requires a GPU")
    unavailable = registry.get(backend).unavailable()
    if unavailable is not None:
        pytest.skip(f"{backend.name}: {unavailable}")


def _require_sol_recipe(recipe):
    """A plain skip, not an expected failure: aiter's MX quantizer aborts the process rather than
    raising on an unsupported device, so reaching one of those rows here would take the whole
    session down instead of failing a test."""
    _require(_backend(recipe))


def _run(backend, query, key, value, **attention_kwargs):
    spec = registry.get(backend)
    spec.resolved()
    return spec.run(query, key, value, AttnCall(attention_kwargs=attention_kwargs))


def _recipes_on_this_device():
    """The Sol-Attn recipe ids this arch builds rows for, in declaration order."""
    from xfuser.core.attention.sparsity import sol

    return sol.sol_attn_recipes_served()


def _serves(recipe, tile):
    """Whether this GPU has a Sol-Attn kernel at `tile` for `recipe`'s operands."""
    from xfuser.core.attention.sparsity import sol

    operands = sol._recipe_operands(sol._RECIPES[recipe])
    return sol._kv_tile_for_q_tile(tile[0], operands, sol._AITER.sol_mode) == tile[1]


def _default_tile(recipe):
    """The (q_tile, kv_tile) `recipe`'s kernel defaults to.

    Asked per recipe because the arch has no one answer to give: on gfx950 the BF16 rows route on
    a 64-token block and every other recipe on 128. Asked of aiter directly, so the tests below
    can check sol_attn_block_tile against it.
    """
    from xfuser.core.attention.sparsity import sol

    operands = sol._recipe_operands(sol._RECIPES[recipe])
    return sol._default_block_tile(operands, sol._AITER.sol_mode)


def _kv_blocks(key, recipe):
    """Number of KV blocks in a BHSD key at the tile this recipe's row uses, a short last one included."""
    return -(-key.shape[2] // _default_tile(recipe)[1])


def _operands(seqlen=1024, heads=2, head_dim=128, seed=1234, sharpness=2.0, cluster=128):
    """BHSD bf16 operands with real block structure, which is the layout the backends take.

    Independent Gaussian noise is the wrong input for judging any block-sparse kernel: attention
    over it is close to uniform, so every block carries similar mass, no selection can exploit
    anything, and the measured gap to dense says more about the data than the kernel. Here the keys
    sit in contiguous per-block clusters and each query tile targets one of them, so there really is
    a block to find.

    sharpness sets cluster separation relative to the noise, and 2.0 is deliberate. Push it higher
    and the per-tensor fp8 scale is set by the cluster centers, which coarsens everything else --
    measured at 2048 tokens, fp8 falls from 0.98 cosine against fp32 at sharpness 2 to 0.82 at 8,
    and MXFP4 from 0.82 to 0.12. So a test that reads an absolute cosine wants the default, and one
    that raises sharpness has to compare each row against its own dense sibling instead, which
    moves with it.

    cluster is how wide one group of related tokens is, and it defaults to the default kernel's KV
    tile so that the structure lands on tile boundaries. Lowering it below a tile is what separates
    the geometries: selection is per tile, so a cluster narrower than one drags in its neighbours.
    """
    g = torch.Generator(device="cuda").manual_seed(seed)
    # Round the cluster counts up so a seqlen that is not a whole number of clusters still gets a
    # center for its short last one; _spread trims the overhang. Exact for an aligned seqlen.
    num_clusters = -(-seqlen // cluster)
    centers = torch.randn(num_clusters, head_dim, generator=g, device="cuda") * sharpness

    def _spread(rows):
        x = rows[:seqlen] + torch.randn(seqlen, head_dim, generator=g, device="cuda")
        return x.unsqueeze(0).unsqueeze(0).expand(1, heads, seqlen, head_dim).contiguous()

    key = _spread(centers.repeat_interleave(cluster, 0))
    # Queries change target half as often as the keys change cluster, so that a run of queries has
    # one block to find rather than every query wanting its own.
    targets = torch.arange(-(-seqlen // (2 * cluster)), device="cuda") % num_clusters
    query = _spread(centers[targets].repeat_interleave(2 * cluster, 0))
    value = torch.randn(1, heads, seqlen, head_dim, generator=g, device="cuda")
    return query.bfloat16(), key.bfloat16(), value.bfloat16()


def _fp32_attention(query, key, value):
    q, k, v = query.float(), key.float(), value.float()
    scores = torch.einsum("bhqd,bhkd->bhqk", q, k) / (q.shape[-1] ** 0.5)
    return torch.einsum("bhqk,bhkd->bhqd", scores.softmax(dim=-1), v)


def _cosine(a, b):
    return F.cosine_similarity(a.float().flatten(), b.float().flatten(), dim=0).item()


def _relative_error(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


# ---------------------------------------------------------------------------
# the key length
# ---------------------------------------------------------------------------


def test_sol_attn_drops_a_caller_alignment_pad():
    """key_seqlen must reproduce attention over the real tokens, whatever the pad holds.

    A model that pads its sequence for alignment reports the real count as varlen metadata.
    Sol-Attn takes no mask, so the pad is dropped rather than attended. The non-zero case is the
    one a bias on the QKV projection would produce.
    """
    _require_sol_recipe("fp8")
    from xfuser.core.attention.sparsity.sol import sol_attn_bhsd

    real, align = 3970, 64
    padded = -(-real // align) * align
    torch.manual_seed(0)

    def bhsd(seq):
        return torch.randn(1, 4, seq, 128, device="cuda", dtype=torch.bfloat16)

    query = bhsd(512)
    for fill in ("zeros", "nonzero"):
        key, value = bhsd(padded), bhsd(padded)
        if fill == "zeros":
            key[:, :, real:] = 0
            value[:, :, real:] = 0
        dropped, _ = sol_attn_bhsd(query, key, value, beta=0.5, key_seqlen=real)
        unpadded, _ = sol_attn_bhsd(query, key[:, :, :real], value[:, :, :real], beta=0.5)
        assert torch.equal(dropped, unpadded), f"dropping a {fill} pad did not reproduce attention over the real tokens"


@pytest.mark.parametrize("recipe", ALL_RECIPES)
@pytest.mark.parametrize("seqlen", [1023, 1008, 960, 897])
def test_sol_attn_takes_a_seqlen_that_is_not_a_whole_number_of_kv_blocks(recipe, seqlen):
    """Wan is ragged at every standard size, so this is the common case rather than a corner.

    720p is 21 latent frames x 80 x 45 = 75600 tokens, i.e. 590.6 KV blocks. A ragged row is
    handed that length as it is; a row that is not has K/V padded to its tile. Either way the
    output keeps Q's real length and accuracy does not sag as the remainder grows. 897 leaves 127
    of 1024 over, far past anything Wan asks for.
    """
    _require_sol_recipe(recipe)
    from xfuser.core.attention.sparsity.sol import sol_attn_bhsd

    query, key, value = _operands(seqlen=seqlen)
    reference = _fp32_attention(query, key, value)
    with torch.no_grad():
        out, _ = sol_attn_bhsd(query, key, value, beta=0.5, recipe=recipe)

    assert out.shape == query.shape, "the pad must not reach the output"
    assert out.dtype == torch.bfloat16
    assert torch.isfinite(out).all()
    if recipe in DENSE_SIBLING:
        dense, _ = _run(AttentionBackendType[DENSE_SIBLING[recipe]], query, key, value)
        assert _cosine(out, reference) > _cosine(dense, reference) - 0.01
    else:
        assert _cosine(out, reference) > 0.9


def test_a_ragged_row_attends_no_padding_keys():
    """A zero pad key scores 0 rather than -inf, so it takes softmax mass from every query.

    On unclustered operands with every block selected that mass is a plain fraction of the row,
    and BF16 is the row with no quantization error to hide it behind. Matching exact attention to
    well under that fraction is what shows the row was handed the true key length.
    """
    _require_sol_recipe("bf16")
    from xfuser.core.attention.sparsity import sol

    recipe = sol._RECIPES["bf16"]
    if not sol._takes_ragged_kv(recipe, sol.sol_attn_block_tile(recipe)):
        pytest.skip("this GPU's bf16 Sol-Attn row needs its keys padded")
    # One key into a last block of 64, so padding it would add 63 zero keys.
    query, key, value = _operands(seqlen=961, sharpness=0.0)
    with torch.no_grad():
        # A threshold far below every block's score selects them all, leaving nothing pooled.
        out, _ = sol.sol_attn_bhsd(query, key, value, beta=-10.0, recipe="bf16")

    assert _relative_error(out, _fp32_attention(query, key, value)) < 2e-2


# ---------------------------------------------------------------------------
# accuracy against the dense rows
# ---------------------------------------------------------------------------


def test_sol_attn_tracks_the_dense_fp8_sibling():
    """Against AITER_FP8, which is the same fp8 row over every block.

    That sibling is the right target rather than an fp32 reference: it shares the quantization and
    the rotation, so what is left in the gap is the routing plus the pooled correction.
    """
    _require_sol_recipe("fp8")
    _require(AttentionBackendType.AITER_FP8)

    query, key, value = _operands()
    with torch.no_grad():
        sol, sol_lse = _run(AttentionBackendType.AITER_FP8_SOL, query, key, value, solattn_beta=0.5)
        dense, _ = _run(AttentionBackendType.AITER_FP8, query, key, value)

    assert sol_lse is None, "Sol-Attn returns no LSE; ring parallelism is refused instead"
    assert sol.shape == dense.shape
    assert sol.dtype == torch.bfloat16
    assert torch.isfinite(sol).all()
    cosine = _cosine(sol, dense)
    assert cosine > 0.99, f"cosine to the dense fp8 sibling {cosine}"


# Per-format distance a row is allowed from its dense sibling, measured at the geometry below.
# BF16's is four orders tighter than the rest because nothing rounds on that row: with no
# quantization the pooled correction reproduces the dense result outright, so anything it loses is
# float accumulation order and not the algorithm. The quantized rows are bounded by how far their
# pooled means round away from the blocks they stand in for, which is why the rows that put V into
# MXFP4 -- eight magnitude levels -- get an order more room.
_DENSE_SIBLING_TOLERANCE = {
    "bf16": 1e-6,
    "bf16fp8": 2e-3,
    "fp8": 2e-3,
    "i8fp8": 2e-3,
    "mxfp8": 2e-3,
    "f8f6": 2e-3,
    "f6f4": 2e-2,
    "mxfp4": 2e-2,
}


# The rows with block-scaled Q/K lose their exact blocks at sharpness 8: their dense sibling falls to
# about 0.33 cosine against fp32, and Sol-Attn with every block selected falls with it, while Sol-Attn
# at beta 0.5 holds about 0.80 because its pooled blocks are averaged in dequantized space. That is
# AITER's MX Q/K kernel at large scores rather than the routing, so these stay strict and flip back
# when it is fixed.
_MX_QK_COLLAPSE = pytest.mark.xfail(
    strict=True, reason="AITER's MX Q/K rows lose accuracy at large scores, so dense and Sol-Attn part ways"
)


def _sibling_cases():
    for recipe in sorted(DENSE_SIBLING):
        for sharpness in (2.0, 4.0, 8.0):
            marks = _MX_QK_COLLAPSE if sharpness == 8.0 and recipe in ("mxfp8", "f6f4", "mxfp4") else ()
            yield pytest.param(recipe, sharpness, marks=marks, id=f"{recipe}-{sharpness:g}")


@pytest.mark.parametrize("recipe,sharpness", _sibling_cases())
def test_every_sol_row_holds_its_dense_sibling_across_cluster_separations(recipe, sharpness):
    """Each row against its own dense sibling at three cluster separations, not just one.

    Cluster separation is what sets the per-tensor scales, so raising it coarsens everything
    outside the cluster centers and drives each format toward the regime where its pooled means
    stop representing the blocks they stand in for. A row that is wired correctly and still
    degrades only there would pass at one separation and fail here.

    BF16 is the row that makes this readable. It has no quantization to blame, and it reproduces
    its dense sibling to within a bf16 ulp at every separation, which says the selection and the
    correction are together exact on these operands -- so the distance every other row shows is
    its format rounding and nothing else. That is also why this asserts a two-sided band against
    fp32 rather than that Sol-Attn beats dense: a one-sided claim would be asserting a coincidence.

    Run at 2048 tokens: 16 KV blocks of 128, which is _MIN_USEFUL_KV_BLOCKS. Below it sol.py itself
    warns that a mean-plus-sigma threshold over the blocks says little.
    """
    _require_sol_recipe(recipe)
    dense_backend = AttentionBackendType[DENSE_SIBLING[recipe]]
    _require(dense_backend)
    from xfuser.core.attention.sparsity.sol import sol_attn_bhsd

    tolerance = _DENSE_SIBLING_TOLERANCE[recipe]
    query, key, value = _operands(seqlen=2048, sharpness=sharpness)
    reference = _fp32_attention(query, key, value)
    with torch.no_grad():
        sol, _ = sol_attn_bhsd(query, key, value, beta=0.5, recipe=recipe)
        dense, _ = _run(dense_backend, query, key, value)

    assert sol.shape == dense.shape and sol.dtype == torch.bfloat16
    assert torch.isfinite(sol).all()

    sibling = _cosine(sol, dense)
    assert 1.0 - sibling < tolerance, (
        f"at sharpness {sharpness} the {recipe} Sol-Attn row tracks its dense sibling at {sibling:.8f}"
    )

    sol_cosine, dense_cosine = _cosine(sol, reference), _cosine(dense, reference)
    assert abs(sol_cosine - dense_cosine) < tolerance, (
        f"at sharpness {sharpness} the {recipe} Sol-Attn row tracks fp32 at {sol_cosine:.6f} "
        f"against its dense sibling's {dense_cosine:.6f}"
    )


def test_the_all_mxfp6_row_holds_exact_attention():
    """f6f6 has no dense sibling to be graded against, so it is graded against fp32, at the bound
    its neighbours on either side meet: f8f6 shares its V and f6f4 its Q/K."""
    _require_sol_recipe("f6f6")
    from xfuser.core.attention.sparsity.sol import sol_attn_bhsd

    query, key, value = _operands(seqlen=2048)
    reference = _fp32_attention(query, key, value)
    with torch.no_grad():
        f6f6, _ = sol_attn_bhsd(query, key, value, beta=0.5, recipe="f6f6")
        f6f4, _ = sol_attn_bhsd(query, key, value, beta=0.5, recipe="f6f4")

    assert torch.isfinite(f6f6).all()
    # Same Q/K and a finer V than f6f4, so it may not land further from the truth.
    assert _cosine(f6f6, reference) >= _cosine(f6f4, reference) - 1e-3


# ---------------------------------------------------------------------------
# the head cost, and the packed path it moves a call onto
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("recipe", ALL_RECIPES)
def test_sol_attn_publishes_a_head_cost_without_changing_the_output(recipe):
    """Requesting the head cost moves a call onto aiter's packed API.

    That is a different entry point reached with operands quantized here rather than inside the raw
    one, so the two have to be shown to agree -- otherwise turning head balancing on would silently
    change what the model computes. MXFP8 has no raw path and goes packed either way.
    """
    _require_sol_recipe(recipe)
    from xfuser.core.attention.sparsity.head_balance import COST_SINK_KEY

    heads = 2
    query, key, value = _operands(heads=heads)
    backend = _backend(recipe)
    cost_sink = torch.zeros(heads, device="cuda", dtype=torch.float32)

    with torch.no_grad():
        plain, _ = _run(backend, query, key, value, solattn_beta=0.5)
        balanced, _ = _run(backend, query, key, value, solattn_beta=0.5, **{COST_SINK_KEY: cost_sink})

    assert torch.equal(plain, balanced), "the packed path taken for head cost must reproduce the raw path exactly"
    assert (cost_sink > 0).all(), "every head should select at least one block"
    num_q_tiles = -(-query.shape[2] // 256)
    assert (cost_sink <= num_q_tiles * _kv_blocks(key, recipe)).all()


@pytest.mark.parametrize("recipe", ALL_RECIPES)
@pytest.mark.parametrize("seqlen", [1024, 1008])
@pytest.mark.parametrize("head_cost", [False, True])
def test_sol_attn_holds_one_graph(recipe, seqlen, head_cost):
    """fullgraph=True, because a break here lands in the middle of every attention layer.

    xDiT compiles without fullgraph, so a break does not fail the run -- it splits the block into
    two compiled regions and quietly gives back the fusion, which is only visible in a profile.
    Every row is covered because they take different branches: the ragged seqlen reaches the K/V
    tail, the head cost swaps the raw entry point for the packed one, and the MX rows carry their
    own quantizers, packed views and pooled scales through it.
    """
    _require_sol_recipe(recipe)
    from xfuser.core.attention.sparsity.sol import sol_attn_bhsd

    query, key, value = _operands(seqlen=seqlen)

    def run(q, k, v):
        out, _ = sol_attn_bhsd(q, k, v, beta=0.5, return_head_cost=head_cost, recipe=recipe)
        return out

    torch._dynamo.reset()
    with torch.no_grad():
        compiled = torch.compile(run, fullgraph=True)(query, key, value)
        eager = run(query, key, value)

    assert torch.equal(compiled, eager), "compiling must not change the result"


def test_sol_attn_is_equivariant_under_a_head_permutation():
    """Permuting heads must permute the cost and the output and change nothing else.

    This is the contract head balancing rests on. It reorders heads before the input all-to-all so
    each rank gets a cost-balanced subset, then inverts that on the output, which is only sound if
    the permutation is a pure relabelling. Sol-Attn routes per head and the routing never mixes
    them, so this should hold to the bit rather than approximately.
    """
    _require_sol_recipe("fp8")
    from xfuser.core.attention.sparsity.sol import sol_attn_bhsd

    heads = 8
    query, key, value = _operands(heads=heads)
    perm = torch.randperm(heads, device="cuda")
    inverse = torch.argsort(perm)

    with torch.no_grad():
        out, cost = sol_attn_bhsd(query, key, value, beta=0.5, return_head_cost=True)
        permuted, permuted_cost = sol_attn_bhsd(
            query.index_select(1, perm),
            key.index_select(1, perm),
            value.index_select(1, perm),
            beta=0.5,
            return_head_cost=True,
        )

    assert cost.shape == (heads,) and cost.dtype == torch.float32, (
        "the cost sink apply_head_balance allocates is float32 (local_nheads_q,)"
    )
    assert torch.equal(cost[perm], permuted_cost), "cost must follow its head"
    assert torch.equal(out.index_select(1, perm), permuted), "output must follow its head"
    assert torch.equal(permuted.index_select(1, inverse), out), "revert must be exact"


# ---------------------------------------------------------------------------
# beta
# ---------------------------------------------------------------------------


def _selected_blocks(query, key, value, backend, **attention_kwargs):
    from xfuser.core.attention.sparsity.head_balance import COST_SINK_KEY

    sink = torch.zeros(query.shape[1], device="cuda", dtype=torch.float32)
    with torch.no_grad():
        out, _ = _run(backend, query, key, value, **attention_kwargs, **{COST_SINK_KEY: sink})
    return out, sink


def test_sol_attn_beta_controls_sparsity():
    """A higher threshold has to select strictly fewer blocks, or the knob is not wired through.

    Deliberately run on UNCLUSTERED operands, the one place they are the right input. The clustered
    ones the accuracy tests use are so cleanly separated that every beta lands on the same single
    matching block, which is the floor sol_prepare enforces anyway, leaving the threshold nothing
    to move. A smooth score distribution is what makes the knob observable.
    """
    _require_sol_recipe("fp8")
    query, key, value = _operands(sharpness=0.0)

    costs = {
        beta: _selected_blocks(query, key, value, AttentionBackendType.AITER_FP8_SOL, solattn_beta=beta)[1].sum().item()
        for beta in (0.0, 1.5)
    }

    assert costs[1.5] < costs[0.0], f"beta did not tighten the selection: {costs}"


def test_a_scheduled_beta_routes_exactly_as_the_same_number_would(monkeypatch):
    """--solattn_beta_schedule hands the backends a 0-d tensor where --solattn_beta hands a float.

    It has to be the same threshold to the last bit, or a schedule would silently mean something
    other than the betas it was given. A tensor is what the schedule passes because reading it as a
    number happens inside the compiled forward, where it costs a graph break and a recompile per
    distinct beta.
    """
    _require_sol_recipe("fp8")
    from types import SimpleNamespace

    from xfuser.core.distributed import runtime_state

    query, key, value = _operands(sharpness=0.0)

    _use_block_size(monkeypatch, None)

    def run(scheduled):
        monkeypatch.setattr(runtime_state, "_RUNTIME", SimpleNamespace(scheduled_solattn_beta=scheduled), raising=False)
        return _selected_blocks(query, key, value, AttentionBackendType.AITER_FP8_SOL, solattn_beta=0.25)

    # The first Sol-Attn call of a process autotunes the pooling kernels, and a config picked on
    # timing noise moves a near-threshold block, which would read as a difference between the betas.
    run(None)

    from_flag, cost_from_flag = run(None)
    from_schedule, cost_from_schedule = run(torch.tensor(0.25, dtype=torch.float32))

    assert torch.equal(cost_from_flag, cost_from_schedule), (
        "the scheduled beta selected a different number of blocks than the float: "
        f"{cost_from_flag.tolist()} vs {cost_from_schedule.tolist()}"
    )
    assert torch.equal(from_flag, from_schedule), "same threshold, so the output must be identical"


# ---------------------------------------------------------------------------
# the block tile
# ---------------------------------------------------------------------------


def _use_block_size(monkeypatch, tile):
    """Point the backends at one geometry, as --sparseattn_block_size does through the runtime."""
    from xfuser.core.attention.backends.aiter_sol import kernel

    monkeypatch.setattr(kernel, "_block_tile", lambda: tile)


def test_an_unset_block_tile_takes_each_recipes_kernel_default():
    """Unset is the shipped configuration, and it must not pin a geometry of its own: gfx950's BF16
    rows route on a 64-token block and the rest on 128, so one pinned here would be right for at
    most some of them."""
    _require_sol_recipe("fp8")
    from aiter.ops.mha_v4 import mha_v4_block_tile

    from xfuser.core.attention.sparsity import sol

    for recipe in _recipes_on_this_device():
        operands = sol._recipe_operands(sol._RECIPES[recipe])
        assert sol.sol_attn_block_tile(sol._RECIPES[recipe]) == mha_v4_block_tile(operands, sol._AITER.sol_mode), (
            f"the '{recipe}' recipe does not take aiter's default for its own row"
        )
        assert _serves(recipe, _default_tile(recipe))


def test_the_64x64_block_size_routes_and_dispatches_at_64(monkeypatch):
    """End to end at the finer geometry: it has to route at 64 and hold accuracy.

    Accuracy alone would pass with the block size ignored, so the block count carries the proof. It
    is reported in (query tile x KV block) pairs, so naming the same content on the finer grid
    costs strictly more of them; ignoring the block size would return the default run's number
    exactly. What is deliberately NOT asserted is that 64x64 is more accurate: on clustered
    operands the pooled correction already recovers most of what coarse selection drops.

    Every backend whose recipe has a 64x64 row is exercised, read off the manifest rather than
    named, so a precision that gains one is covered the day it lands.
    """
    _require_sol_recipe("fp8")
    serving = [recipe for recipe in _recipes_on_this_device() if _serves(recipe, (64, 64))]
    if not serving:
        pytest.skip("this build has no 64x64 Sol-Attn row.")

    # Clustered at 64, which only the finer geometry can resolve; see _operands.
    query, key, value = _operands(seqlen=2048, cluster=64)
    reference = _fp32_attention(query, key, value)

    for recipe in serving:
        backend = _backend(recipe)
        default_tile = _default_tile(recipe)

        def run(tile):
            _use_block_size(monkeypatch, tile)
            out, sink = _selected_blocks(query, key, value, backend, solattn_beta=0.25)
            return out, sink.sum().item()

        default, default_blocks = run(None)
        fine, fine_blocks = run((64, 64))

        assert fine_blocks > default_blocks, (
            f"{recipe}: 64x64 reported {fine_blocks:.0f} selected blocks and "
            f"{default_tile[0]}x{default_tile[1]} reported {default_blocks:.0f}; an equal or smaller "
            "count means the block size never reached routing"
        )
        cosine, baseline = _cosine(fine, reference), _cosine(default, reference)
        # Floored at what the recipe's own format reaches dense, since no fixed number suits both
        # BF16 and MXFP4, whose dense row lands near 0.83 on these operands.
        if recipe in DENSE_SIBLING:
            dense, _ = _run(AttentionBackendType[DENSE_SIBLING[recipe]], query, key, value)
            floor = _cosine(dense, reference) - 0.02
        else:
            floor = 0.9
        assert cosine > floor, f"{recipe}: 64x64 Sol-Attn fell to {cosine:.4f} against fp32, below {floor:.4f}"
        assert cosine > baseline - 0.02, f"{recipe}: 64x64 lost ground to the default: {cosine:.4f} vs {baseline:.4f}"


def test_a_block_size_with_no_kernel_is_rejected_at_setup_by_name():
    """Named at the flag that set it, at setup: this is the check that stands between a mistyped
    launch and a run that loads a model for minutes before dying."""
    _require_sol_recipe("fp8")
    from xfuser.core.attention.sparsity.sol import SolAttnUnsupported, check_sol_attn_recipe

    unserved = (512, 512)
    for recipe in _recipes_on_this_device():
        if _serves(recipe, unserved):
            continue
        with pytest.raises(SolAttnUnsupported, match=rf"--sparseattn_block_size 512x512 .*'{recipe}'"):
            check_sol_attn_recipe(recipe, block_tile=unserved)
        check_sol_attn_recipe(recipe, block_tile=_default_tile(recipe))
