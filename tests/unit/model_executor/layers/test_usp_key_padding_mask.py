"""A key-padding mask keeps masked keys out of USP attention, or refuses a setup that cannot honour it."""

import dataclasses
from types import SimpleNamespace

import pytest
import torch

from xfuser.config.config import RuntimeConfig
from xfuser.core.attention import registry as attention_registry
from xfuser.core.attention.spec import AttentionBackendType
from xfuser.core.distributed.runtime_state import DiTRuntimeState
from xfuser.model_executor.layers import usp


@pytest.fixture
def runtime(monkeypatch):
    """USP's view of the runtime: FP8 communication state, ring degree and the selected backend.

    Returns a namespace with ``configure`` and the ``warnings`` the helper logged.
    """
    warnings = []
    monkeypatch.setattr(usp, "warn_once", lambda _logger, message, *args: warnings.append(message % args))

    def configure(fp8_comms=None, ring_degree=1, backend=AttentionBackendType.SDPA):
        state = SimpleNamespace(fp8_comms=fp8_comms, attention_backend=backend)
        monkeypatch.setattr(usp, "get_runtime_state", lambda: state)
        monkeypatch.setattr(usp, "get_ring_parallel_world_size", lambda: ring_degree)

    configure()
    return SimpleNamespace(configure=configure, warnings=warnings)


def test_without_a_mask_the_call_keeps_its_kwargs_and_its_backend(runtime):
    runtime.configure(fp8_comms=object(), ring_degree=2, backend=AttentionBackendType.AITER_SPARGE)
    kwargs = {"valid_kv_len": 5}

    assert usp.key_padding_mask_attention_kwargs(None, kwargs) == (kwargs, None)
    assert usp.key_padding_mask_attention_kwargs(None) == (None, None)


def test_a_mask_reaches_the_backend_as_a_broadcastable_bool_mask_next_to_the_declared_pad(runtime):
    kwargs = {"valid_kv_len": 5}
    mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]])

    updated, backend = usp.key_padding_mask_attention_kwargs(mask, kwargs)

    assert backend is None
    assert updated["valid_kv_len"] == 5
    assert updated["attn_mask"].dtype == torch.bool
    assert updated["attn_mask"].shape == (2, 1, 1, 5)
    assert updated["attn_mask"][:, 0, 0].equal(mask.bool())
    # The wrapper's shared dict is not written to: it outlives the call.
    assert kwargs == {"valid_kv_len": 5}
    assert runtime.warnings == []


@pytest.mark.parametrize(
    "selected",
    [
        AttentionBackendType.SDPA,
        AttentionBackendType.SDPA_MATH,
        AttentionBackendType.SDPA_EFFICIENT,
        # The default on NVIDIA without Flash Attention 3 or 4.
        AttentionBackendType.CUDNN,
    ],
)
def test_a_backend_that_applies_the_mask_is_kept(runtime, selected):
    runtime.configure(backend=selected)

    updated, backend = usp.key_padding_mask_attention_kwargs(torch.ones(1, 2, dtype=torch.bool))

    assert backend is None
    assert "attn_mask" in updated
    assert runtime.warnings == []


@pytest.mark.parametrize("selected", [AttentionBackendType.FLASH, AttentionBackendType.SDPA_FLASH])
def test_a_dense_backend_that_cannot_apply_the_mask_is_replaced_by_sdpa_and_logged(runtime, selected):
    runtime.configure(backend=selected)

    updated, backend = usp.key_padding_mask_attention_kwargs(torch.ones(1, 2, dtype=torch.bool))

    assert backend is AttentionBackendType.SDPA
    assert "attn_mask" in updated
    assert len(runtime.warnings) == 1
    assert selected.value in runtime.warnings[0] and "cannot apply the key mask" in runtime.warnings[0]


@pytest.mark.parametrize(
    "setup, match",
    [
        pytest.param(dict(fp8_comms=object()), r"--use_fp8_comms cannot be combined with batched prompts", id="fp8"),
        pytest.param(
            dict(ring_degree=2), r"cannot run with ring attention.*ring degree 2.*--use_cfg_parallel", id="ring"
        ),
        pytest.param(
            dict(backend=AttentionBackendType.AITER_SPARGE),
            r"cannot be combined with batched prompts.*only applied by dense backends",
            id="sparse",
        ),
    ],
)
def test_a_setup_that_cannot_apply_the_mask_is_refused_instead_of_running_without_it(runtime, setup, match):
    runtime.configure(**setup)

    with pytest.raises((ValueError, NotImplementedError), match=match):
        usp.key_padding_mask_attention_kwargs(torch.tensor([[1, 0]]))


def test_a_masked_call_replaced_by_sdpa_compiles_without_a_graph_break(monkeypatch):
    """A selection that cannot apply the mask sends masked calls to SDPA, which
    must already be resolved when the first one is traced: resolving imports the
    kernel module, a graph break (a failure under fullgraph=True)."""
    pytest.importorskip("torch._dynamo")
    # Unresolved copies of the specs, as in a fresh process, whatever earlier tests resolved.
    specs = [dataclasses.replace(spec, _resolved=None, _fallback=None) for spec in attention_registry.REGISTRY.values()]
    with attention_registry.using(specs):
        state = DiTRuntimeState.__new__(DiTRuntimeState)
        state.runtime_config = RuntimeConfig()
        state.parallel_config = SimpleNamespace(ring_degree=1)
        state.set_attention_backend(AttentionBackendType.SDPA_FLASH)
        assert not attention_registry.get(AttentionBackendType.SDPA_FLASH).applies_attn_mask
        # One rank: no process group, and the SP getters answer 1, as without parallelism.
        for name in (
            "get_sequence_parallel_world_size",
            "get_ulysses_parallel_world_size",
            "get_ring_parallel_world_size",
        ):
            monkeypatch.setattr(usp, name, lambda: 1)
        monkeypatch.setattr(usp, "get_runtime_state", lambda: state)

        def masked_usp(query, key, value, key_mask):
            attention_kwargs, backend = usp.key_padding_mask_attention_kwargs(key_mask)
            return usp.USP(query, key, value, backend=backend, attention_kwargs=attention_kwargs)

        generator = torch.Generator().manual_seed(0)
        query, key, value = (torch.randn(2, 2, 8, 16, generator=generator) for _ in range(3))
        key_mask = torch.ones(2, 8, dtype=torch.bool)
        key_mask[0, 5:] = False
        torch._dynamo.reset()
        torch._dynamo.utils.counters.clear()
        output = torch.compile(masked_usp, backend="aot_eager")(query, key, value, key_mask)
        graph_breaks = dict(torch._dynamo.utils.counters["graph_break"])
        torch._dynamo.reset()

    expected = torch.nn.functional.scaled_dot_product_attention(query, key, value, attn_mask=key_mask[:, None, None])
    assert graph_breaks == {}
    torch.testing.assert_close(output, expected)
