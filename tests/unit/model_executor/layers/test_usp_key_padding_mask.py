"""A key-padding mask keeps masked keys out of USP attention, or refuses a setup that cannot honour it."""

from types import SimpleNamespace

import pytest
import torch

from xfuser.core.attention.spec import AttentionBackendType
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


def test_a_mask_reaches_sdpa_as_a_broadcastable_bool_mask_next_to_the_declared_pad(runtime):
    kwargs = {"valid_kv_len": 5}
    mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]])

    updated, backend = usp.key_padding_mask_attention_kwargs(mask, kwargs)

    assert backend is AttentionBackendType.SDPA
    assert updated["valid_kv_len"] == 5
    assert updated["attn_mask"].dtype == torch.bool
    assert updated["attn_mask"].shape == (2, 1, 1, 5)
    assert updated["attn_mask"][:, 0, 0].equal(mask.bool())
    # The wrapper's shared dict is not written to: it outlives the call.
    assert kwargs == {"valid_kv_len": 5}
    assert runtime.warnings == []


def test_overriding_another_dense_backend_is_logged(runtime):
    runtime.configure(backend=AttentionBackendType.FLASH)

    _, backend = usp.key_padding_mask_attention_kwargs(torch.ones(1, 2, dtype=torch.bool))

    assert backend is AttentionBackendType.SDPA
    assert len(runtime.warnings) == 1 and AttentionBackendType.FLASH.value in runtime.warnings[0]


@pytest.mark.parametrize(
    "setup, match",
    [
        pytest.param(dict(fp8_comms=object()), r"--use_fp8_comms cannot be combined with batched prompts", id="fp8"),
        pytest.param(
            dict(ring_degree=2), r"cannot run with ring attention.*ring degree 2.*--use_cfg_parallel", id="ring"
        ),
        pytest.param(
            dict(backend=AttentionBackendType.AITER_SPARGE),
            r"cannot be combined with batched prompts.*only applied by dense SDPA",
            id="sparse",
        ),
    ],
)
def test_a_setup_that_cannot_apply_the_mask_is_refused_instead_of_running_without_it(runtime, setup, match):
    runtime.configure(**setup)

    with pytest.raises((ValueError, NotImplementedError), match=match):
        usp.key_padding_mask_attention_kwargs(torch.tensor([[1, 0]]))
