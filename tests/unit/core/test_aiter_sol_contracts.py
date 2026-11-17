"""AITER Sol-Attn: what the backends promise before any kernel runs.

Sol-Attn (arXiv 2607.24027) routes for itself rather than through a Sparge
block mask, so what reaches the kernel is the call's own tensors plus the
attention_kwargs a model publishes. These pin that hand-off, and the calls the
backends refuse. The kernel itself is covered in
tests/integration/accelerator/core/test_aiter_sol_attn.py.
"""

from types import SimpleNamespace

import pytest
import torch

from xfuser.core.attention import registry
from xfuser.core.attention.spec import AttentionBackendType, AttnCall, Sparsity, VarlenPacking


def _sol_specs():
    return [registry.get(backend) for backend in sorted(registry.types_where(sparsity=Sparsity.SOL), key=str)]


def _run(backend, query, key, value, **kwargs):
    spec = registry.get(backend)
    spec.resolved()
    return spec.run(query, key, value, AttnCall(**kwargs))


def _trailing_pad(batch, valid, padded):
    """The varlen metadata for one sequence padded from valid to padded keys."""
    indices = torch.cat([torch.arange(valid) + row * padded for row in range(batch)])
    cu_seqlens = torch.arange(batch + 1, dtype=torch.int32) * valid
    return {"indices_k": indices, "cu_seqlens_k": cu_seqlens, "max_seqlen_k": valid, "valid_kv_len": valid}


def test_every_sol_recipe_is_one_selectable_backend():
    """--attention_backend resolves by enum member, so each recipe needs exactly one, and every one
    of them has to land on a recipe sol.py actually builds."""
    from xfuser.core.attention.sparsity.sol import SOL_ATTN_RECIPES

    recipes = [spec.impl.bound["recipe"] for spec in _sol_specs()]
    assert sorted(recipes) == sorted(SOL_ATTN_RECIPES)


def test_sol_backends_are_head_balanced_and_never_a_ring_rank():
    """They publish the per-head cost the balancer consumes, and their routing threshold is taken
    over the blocks one rank holds, so no merge of per-rank partials could be exact."""
    from xfuser.model_executor.layers.usp import _HEAD_BALANCE_BACKENDS

    for spec in _sol_specs():
        assert spec.type in _HEAD_BALANCE_BACKENDS
        assert spec.ring.unmet() is not None, f"{spec.type.name} would be allowed onto a ring"


@pytest.mark.parametrize(
    "call, dtype",
    [
        (AttnCall(is_causal=True), torch.bfloat16),
        (AttnCall(dropout_p=0.1), torch.bfloat16),
        (AttnCall(), torch.float16),
    ],
    ids=["causal", "dropout", "fp16"],
)
def test_sol_refuses_what_its_kernel_cannot_compute(call, dtype):
    """Causal is a conflict rather than a missing feature: the pooled correction assumes every
    skipped block is fully attendable."""
    tensor = torch.zeros((1, 2, 8, 128), dtype=dtype)
    for spec in _sol_specs():
        assert spec.rejects(tensor, tensor, tensor, call) is not None


def test_sol_serves_packed_keys_only_as_one_sequences_trailing_pad():
    """The pad is dropped by length, so it has to be one trailing block on one sequence. Keys
    gathered from several would be attended across their boundaries."""
    key = torch.zeros((1, 2, 256, 128), dtype=torch.bfloat16)
    spec = registry.get(AttentionBackendType.AITER_FP8_SOL)

    def call(kwargs):
        return AttnCall(varlen=VarlenPacking.from_kwargs(kwargs), attention_kwargs=kwargs)

    declared = _trailing_pad(1, 200, 256)
    assert spec.rejects(key, key, key, call(declared)) is None

    undeclared = {k: v for k, v in declared.items() if k != "valid_kv_len"}
    assert "valid_kv_len" in spec.rejects(key, key, key, call(undeclared))

    batch = torch.zeros((2, 2, 256, 128), dtype=torch.bfloat16)
    assert "batch size 2" in spec.rejects(batch, batch, batch, call(_trailing_pad(2, 200, 256)))


def test_sol_refuses_ring_parallelism_at_the_kernel(monkeypatch):
    """Ring is not a call constraint, so the kernel entry has to refuse it rather than merge
    partials each rank routed against its own shard."""
    from xfuser.core.attention.sparsity import sol
    from xfuser.core.attention.sparsity.sol import SolAttnUnsupported

    if sol._AITER is None:
        # Refused before any AITER call; only the availability check stands in front of it.
        monkeypatch.setattr(sol, "_AITER", object())
    tensor = torch.zeros((1, 2, 8, 128), dtype=torch.bfloat16)

    with pytest.raises(SolAttnUnsupported, match="ring"):
        _run(AttentionBackendType.AITER_FP8_SOL, tensor, tensor, tensor, ring_world_size=2)


def _record_sol_calls(monkeypatch, block_size=None):
    from xfuser.core.attention.backends.aiter_sol import kernel

    seen = {}

    def record(query, key, value, **kwargs):
        seen.update(kwargs)
        return torch.zeros_like(query), None

    monkeypatch.setattr(kernel, "sol_attn_bhsd", record)
    if block_size is not None:
        state = SimpleNamespace(runtime_config=SimpleNamespace(sparseattn_block_size=block_size))
        monkeypatch.setattr(kernel, "runtime_state_is_initialized", lambda: True)
        monkeypatch.setattr(kernel, "get_runtime_state", lambda: state)
    return seen


def test_the_backend_hands_beta_and_recipe_to_the_routing(monkeypatch):
    """Every piece correct on its own, and one connection between them never made, is the bug
    this guards: the model's beta has to reach sol_attn_bhsd, on the recipe the backend names."""
    seen = _record_sol_calls(monkeypatch)
    query = torch.zeros((1, 2, 256, 128), dtype=torch.bfloat16)

    _run(AttentionBackendType.AITER_F6F4_SOL, query, query, query, attention_kwargs={"solattn_beta": 0.25})

    assert seen["beta"] == 0.25
    assert seen["recipe"] == "f6f4"
    assert seen["key_seqlen"] is None
    assert seen["block_tile"] is None


def test_the_backend_hands_sparseattn_block_size_to_the_kernel(monkeypatch):
    """--sparseattn_block_size picks the Sol tile the same way it picks the Sparge one."""
    seen = _record_sol_calls(monkeypatch, block_size=(64, 64))
    query = torch.zeros((1, 2, 256, 128), dtype=torch.bfloat16)

    _run(AttentionBackendType.AITER_FP8_SOL, query, query, query, attention_kwargs={})

    assert seen["block_tile"] == (64, 64)
    assert seen["beta"] == 0.5


def test_a_block_size_no_sol_kernel_serves_is_refused_at_selection(monkeypatch):
    from xfuser.config.config import RuntimeConfig
    from xfuser.core.attention.backends.aiter_sol import kernel

    checked = []

    def refuse(recipe, block_tile=None):
        checked.append((recipe, block_tile))
        raise kernel.SolAttnUnsupported("--sparseattn_block_size 64x64 has no Sol-Attn kernel")

    monkeypatch.setattr(kernel, "check_sol_attn_recipe", refuse)
    spec = registry.get(AttentionBackendType.AITER_F6F4_SOL)

    spec.validate(RuntimeConfig())
    with pytest.raises(ValueError, match="has no Sol-Attn kernel"):
        spec.validate(RuntimeConfig(sparseattn_block_size=(64, 64)))

    assert checked == [("f6f4", (64, 64))]


def test_the_backend_drops_a_declared_trailing_pad_by_length(monkeypatch):
    """Sol-Attn takes no mask. The pad's length is handed over rather than the pad gathered away,
    and the kernel attends the remaining keys as they are, however they fall on its tile."""
    seen = _record_sol_calls(monkeypatch)
    query = torch.zeros((1, 2, 256, 128), dtype=torch.bfloat16)
    kwargs = _trailing_pad(1, 201, 256)

    _run(
        AttentionBackendType.AITER_FP8_SOL,
        query,
        query,
        query,
        varlen=VarlenPacking.from_kwargs(kwargs),
        attention_kwargs=kwargs,
    )

    assert seen["key_seqlen"] == 201
