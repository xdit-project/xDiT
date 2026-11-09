from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from xfuser.core.attention.constraints import ANY_CALL
from xfuser.core.attention.requirements import ALWAYS
from xfuser.core.attention.spec import AttentionBackendType, Spec
from xfuser.model_executor.layers import usp
from xfuser.model_executor.layers.attention_mask import make_attn_mask_with_meta


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@mock.patch("xfuser.model_executor.layers.usp.attention_registry.find")
@mock.patch("xfuser.model_executor.layers.usp._compile_capture_bypasses_attention", return_value=True)
def test_compile_capture_bypasses_attention_backend(bypass, find, dtype):
    query = torch.randn(1, 4, 8, 16, dtype=dtype)

    output = usp.attention(query, query, query, backend=object())

    torch.testing.assert_close(output, torch.zeros_like(query))
    assert output.dtype == dtype
    bypass.assert_called_once()
    find.assert_not_called()


@mock.patch("xfuser.model_executor.layers.usp._compile_capture_bypasses_attention", return_value=True)
def test_compile_capture_bypass_preserves_joint_output_shape(bypass):
    query = torch.randn(1, 4, 8, 16)
    joint_query = torch.randn(1, 4, 3, 16)

    output = usp.USP(
        query,
        query,
        query,
        joint_query=joint_query,
        joint_key=joint_query,
        joint_value=joint_query,
        joint_strategy="front",
    )

    assert output.shape == (1, 4, 11, 16)
    assert output.dtype == query.dtype
    bypass.assert_called_once()


@mock.patch("xfuser.model_executor.layers.usp.get_cache_manager")
def test_cache_update_requires_registered_layer(get_cache_manager):
    cache_manager = get_cache_manager.return_value
    layer = object()

    cache_manager.has_cache_entry.return_value = False
    assert not usp._has_kv_cache(layer)

    cache_manager.has_cache_entry.return_value = True
    assert usp._has_kv_cache(layer)

    cache_manager.has_cache_entry.reset_mock()
    assert not usp._has_kv_cache(None)
    cache_manager.has_cache_entry.assert_not_called()


@mock.patch("xfuser.model_executor.layers.usp.get_ulysses_parallel_world_size")
@mock.patch("xfuser.model_executor.layers.usp._sdpa_all_to_all_single")
def test_combined_qkv_all_to_all(all_to_all, world_size):
    world_size.return_value = 2
    all_to_all.side_effect = lambda tensor: tensor
    tensors = [torch.randn(2, 4, 8, 4) for _ in range(4)]

    expected = tuple(usp._ft_c_input_all_to_all(tensor) for tensor in tensors)
    actual = usp._combined_qkv_all_to_all(*tensors)

    for expected_tensor, actual_tensor in zip(expected, actual):
        torch.testing.assert_close(actual_tensor, expected_tensor)


@mock.patch("xfuser.model_executor.layers.usp.get_ulysses_parallel_world_size")
@mock.patch("xfuser.model_executor.layers.usp._sdpa_all_to_all_single")
def test_combined_gqa_qkv_all_to_all(all_to_all, world_size):
    world_size.return_value = 2
    all_to_all.side_effect = lambda tensor: tensor
    query = torch.randn(2, 6, 8, 4)
    key = torch.randn(2, 2, 8, 4)
    value = torch.randn_like(key)
    extra = torch.randn_like(query)

    expected = (
        usp._ft_c_input_all_to_all(query),
        usp._ft_c_input_all_to_all(key),
        usp._ft_c_input_all_to_all(value),
        usp._ft_c_input_all_to_all(extra),
    )
    actual = usp._combined_gqa_qkv_all_to_all(query, key, value, extra)

    for expected_tensor, actual_tensor in zip(expected, actual):
        torch.testing.assert_close(actual_tensor, expected_tensor)


def test_repeat_kv_heads_preserves_gqa_order():
    key = torch.tensor([[[[0.0]], [[1.0]]]])
    value = key + 10

    repeated_key, repeated_value = usp._repeat_kv_heads(key, value, repeats=3)

    torch.testing.assert_close(repeated_key.flatten(), torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0, 1.0]))
    torch.testing.assert_close(
        repeated_value.flatten(),
        torch.tensor([10.0, 10.0, 10.0, 11.0, 11.0, 11.0]),
    )


def test_ulysses_extra_inputs_are_named_by_the_caller():
    query = torch.randn(1, 2, 8, 4)
    gate = torch.randn_like(query)
    attention_kwargs = {
        usp.ULYSSES_EXTRA_INPUTS_KEY: ("some_backend_tensor",),
        "some_backend_tensor": gate,
    }

    assert usp._ulysses_extra_inputs(attention_kwargs, query) == [("some_backend_tensor", gate)]
    assert usp._ulysses_extra_inputs({}, query) == []
    assert usp._ulysses_extra_inputs({usp.ULYSSES_EXTRA_INPUTS_KEY: ("missing",)}, query) == []

    attention_kwargs["some_backend_tensor"] = torch.randn(1, 2, 8, 5)
    with pytest.raises(ValueError, match="some_backend_tensor"):
        usp._ulysses_extra_inputs(attention_kwargs, query)


@pytest.mark.parametrize("backend", [AttentionBackendType.SDPA, AttentionBackendType.SDPA_MATH])
def test_sdpa_serves_a_key_padding_mask_that_carries_varlen_packing(backend):
    """Krea-2 and LTX-2 pass a key-padding mask together with the packing
    derived from it. SDPA applies the mask and attends over exactly the valid
    keys of each sequence, as if the padded keys were never there."""
    torch.manual_seed(0)
    batch, heads, q_len, kv_len, head_dim = 2, 2, 5, 7, 8
    query = torch.randn(batch, heads, q_len, head_dim)
    key = torch.randn(batch, heads, kv_len, head_dim)
    value = torch.randn(batch, heads, kv_len, head_dim)
    valid = torch.tensor([[1, 1, 1, 0, 1, 0, 0], [1, 1, 1, 1, 1, 1, 0]])
    meta = make_attn_mask_with_meta(valid)
    attention_kwargs = {
        "attn_mask": meta.attn_mask,
        "indices_k": meta.indices_k,
        "cu_seqlens_k": meta.cu_seqlens_k,
        "max_seqlen_k": meta.max_seqlen_k,
    }

    out = usp.attention(query, key, value, backend=backend, attention_kwargs=attention_kwargs)

    for b in range(batch):
        keep = valid[b].bool()
        expected = torch.nn.functional.scaled_dot_product_attention(
            query[b : b + 1], key[b : b + 1, :, keep], value[b : b + 1, :, keep]
        )
        torch.testing.assert_close(out[b : b + 1], expected)


def test_sdpa_still_refuses_packing_with_no_mask_to_apply():
    query = torch.randn(1, 2, 4, 8)
    packing = make_attn_mask_with_meta(torch.tensor([[1, 1, 0, 0]]))
    attention_kwargs = {
        "indices_k": packing.indices_k,
        "cu_seqlens_k": packing.cu_seqlens_k,
        "max_seqlen_k": packing.max_seqlen_k,
    }

    with pytest.raises(NotImplementedError, match="varlen packed keys"):
        usp.attention(query, query, query, backend=AttentionBackendType.SDPA, attention_kwargs=attention_kwargs)


def _attention_with_lse(query, key, value):
    scores = query @ key.transpose(-1, -2) * query.shape[-1] ** -0.5
    return torch.softmax(scores, dim=-1) @ value, torch.logsumexp(scores, dim=-1)


def _recording_attention_function(kv_lens):
    """USP's attention function around a kernel that notes each call's K length."""

    def kernel(query, key, value, call):
        kv_lens.append(key.shape[2])
        return _attention_with_lse(query, key, value)

    spec = Spec(AttentionBackendType.SDPA, impl=kernel, ring=ALWAYS, accepts=ANY_CALL, requires=ALWAYS)
    return usp.concat_joint_tensors_decorator(usp._spec_adapter(spec))


def _merge_ring_steps(outputs, lses):
    lse = torch.logsumexp(torch.stack(lses), dim=0)
    return sum(output * torch.exp(step_lse - lse).unsqueeze(-1) for output, step_lse in zip(outputs, lses))


@pytest.mark.parametrize("ring_size, pad", [(2, 1), (3, 5)])
def test_ring_steps_drop_the_pad_from_whichever_shard_holds_it(monkeypatch, ring_size, pad):
    shard_len = 8
    valid_kv_len = ring_size * shard_len - pad
    generator = torch.Generator().manual_seed(0)
    query, key, value = (
        torch.randn(1, 2, ring_size * shard_len, 4, generator=generator, dtype=torch.float64) for _ in range(3)
    )
    expected, _ = _attention_with_lse(query, key[:, :, :valid_kv_len], value[:, :, :valid_kv_len])
    monkeypatch.setattr(usp, "get_ring_parallel_world_size", lambda: ring_size)

    def shard(tensor, ring_rank):
        return tensor[:, :, ring_rank * shard_len : (ring_rank + 1) * shard_len]

    for rank in range(ring_size):
        monkeypatch.setattr(usp, "get_ring_parallel_rank", lambda rank=rank: rank)
        kv_lens = []
        step = usp._ring_trim_trailing_kv_padding(_recording_attention_function(kv_lens), valid_kv_len, shard_len)
        # torch's ring hands step i the K/V shard of ring rank (rank - i) % size.
        sources = [(rank - i) % ring_size for i in range(ring_size)]
        outputs, lses = zip(
            *(
                step(shard(query, rank), shard(key, source), shard(value, source), is_causal=False, attention_kwargs={})
                for source in sources
            )
        )

        torch.testing.assert_close(_merge_ring_steps(outputs, lses), shard(expected, rank))
        last_shard_len = valid_kv_len - (ring_size - 1) * shard_len
        assert kv_lens == [last_shard_len if source == ring_size - 1 else shard_len for source in sources]


def test_ring_steps_refuse_a_shard_holding_nothing_but_pad(monkeypatch):
    monkeypatch.setattr(usp, "get_ring_parallel_world_size", lambda: 2)
    monkeypatch.setattr(usp, "get_ring_parallel_rank", lambda: 0)
    with pytest.raises(NotImplementedError, match="more than one ring shard"):
        usp._ring_trim_trailing_kv_padding(_recording_attention_function([]), valid_kv_len=8, shard_len=8)


_REAL_LEN, _PADDED_LEN = 45, 48


@pytest.fixture
def usp_kv_lens(monkeypatch):
    """Run USP on one rank, recording the K length of every attention call."""
    kv_lens = []
    runtime_state = SimpleNamespace(
        fp8_comms=None,
        attention_backend=AttentionBackendType.SDPA,
        runtime_config=SimpleNamespace(use_spargeattn_head_balance=False),
    )
    monkeypatch.setattr(usp, "get_runtime_state", lambda: runtime_state)
    for degree in ("sequence", "ulysses", "ring"):
        monkeypatch.setattr(usp, f"get_{degree}_parallel_world_size", lambda: 1)
    monkeypatch.setattr(usp, "_get_attention_function", lambda backend=None: _recording_attention_function(kv_lens))
    return kv_lens


def _padded_qkv():
    return [torch.randn(1, 2, _PADDED_LEN, 8) for _ in range(3)]


@pytest.mark.parametrize(
    "attention_kwargs, backend, is_causal, expected_kv_len",
    [
        pytest.param(None, AttentionBackendType.SDPA, False, _PADDED_LEN, id="no-declared-pad"),
        pytest.param({"valid_kv_len": _REAL_LEN}, AttentionBackendType.SDPA, False, _REAL_LEN, id="declared-pad"),
        # Causality already hides a trailing pad from every real query.
        pytest.param({"valid_kv_len": _REAL_LEN}, AttentionBackendType.SDPA, True, _PADDED_LEN, id="causal"),
        # setup_sparge strips the pad against the published layout, and needs equal Q/K/V lengths to do it.
        pytest.param(
            {"valid_kv_len": _REAL_LEN, "thw": (3, 3, 5)},
            AttentionBackendType.AITER_SPARGE,
            False,
            _PADDED_LEN,
            id="sparge-strips-it",
        ),
        pytest.param(
            {"valid_kv_len": _REAL_LEN, "thw": (3, 3, 5)},
            AttentionBackendType.AITER_VSA,
            False,
            _PADDED_LEN,
            id="vsa-strips-it",
        ),
        pytest.param(
            {"valid_kv_len": _REAL_LEN, "thw": (3, 3, 5), "encoder_sp_padding": 1},
            AttentionBackendType.FLEX_BLOCK_ATTN,
            False,
            _PADDED_LEN,
            id="ssta-strips-it",
        ),
        pytest.param(
            {"valid_kv_len": _REAL_LEN, "thw": None},
            AttentionBackendType.AITER_SPARGE,
            False,
            _REAL_LEN,
            id="sparge-without-layout",
        ),
    ],
)
def test_usp_drops_a_declared_pad_unless_something_else_hides_it(
    usp_kv_lens, attention_kwargs, backend, is_causal, expected_kv_len
):
    usp.USP(*_padded_qkv(), is_causal=is_causal, backend=backend, attention_kwargs=attention_kwargs)
    assert usp_kv_lens == [expected_kv_len]


@pytest.fixture
def ring_of_two(monkeypatch, usp_kv_lens):
    """USP as ring rank 0 of two, with each ring step served the local K/V."""
    monkeypatch.setattr(usp, "get_sequence_parallel_world_size", lambda: 2)
    monkeypatch.setattr(usp, "get_ring_parallel_world_size", lambda: 2)
    monkeypatch.setattr(usp, "get_ring_parallel_rank", lambda: 0)

    def ring_attn(attention_function, query, key, value, **kwargs):
        for _ in range(2):
            out, _ = attention_function(query, key, value, **kwargs)
        return out

    monkeypatch.setattr(usp, "ring_attn", ring_attn)
    return usp_kv_lens


def test_usp_drops_the_pad_inside_the_ring_step_that_attends_to_it(ring_of_two):
    local = [tensor[:, :, : _PADDED_LEN // 2] for tensor in _padded_qkv()]
    usp.USP(*local, attention_kwargs={"valid_kv_len": _REAL_LEN})
    # Step 0 attends to rank 0's own shard, which holds no pad; step 1 to the last rank's.
    assert ring_of_two == [_PADDED_LEN // 2, _REAL_LEN - _PADDED_LEN // 2]


def test_usp_refuses_packed_keys_together_with_ring_attention(ring_of_two):
    local = [tensor[:, :, : _PADDED_LEN // 2] for tensor in _padded_qkv()]
    attention_kwargs = {
        "valid_kv_len": _REAL_LEN,
        "indices_k": torch.arange(_REAL_LEN),
        "cu_seqlens_k": torch.tensor([0, _REAL_LEN], dtype=torch.int32),
        "max_seqlen_k": _REAL_LEN,
    }
    with pytest.raises(ValueError, match="valid_kv_len must be in"):
        usp.USP(*local, attention_kwargs=attention_kwargs)


def _joint(length=3):
    return {name: torch.randn(1, 2, length, 8) for name in ("joint_query", "joint_key", "joint_value")}


def _vsa_like_function(seen):
    """An attention function whose kernel crops K to `valid_kv_len` as AITER_VSA's dense fallback does."""

    def kernel(query, key, value, call):
        valid = call.attention_kwargs.get("valid_kv_len")
        if valid is not None and valid < key.shape[2]:
            key, value = key[:, :, :valid], value[:, :, :valid]
        seen.append(key.shape[2])
        return _attention_with_lse(query, key, value)

    spec = Spec(AttentionBackendType.SDPA, impl=kernel, ring=ALWAYS, accepts=ANY_CALL, requires=ALWAYS)
    return usp.concat_joint_tensors_decorator(usp._spec_adapter(spec))


@pytest.mark.parametrize("strategy", ["front", "rear"])
def test_usp_publishes_no_pad_length_for_the_keys_joined_after_the_trim(monkeypatch, usp_kv_lens, strategy):
    # The declared length counts the main keys only. Left in the kwargs, a backend that crops K to it
    # (the dense fallback of AITER_VSA) would cut the joint tokens' keys off again.
    seen = []
    monkeypatch.setattr(usp, "get_ulysses_parallel_rank", lambda: 0)
    monkeypatch.setattr(usp, "_get_attention_function", lambda backend=None: _vsa_like_function(seen))
    attention_kwargs = {"valid_kv_len": _REAL_LEN}

    usp.USP(*_padded_qkv(), joint_strategy=strategy, **_joint(), attention_kwargs=attention_kwargs)

    assert seen == [_REAL_LEN + 3]
    # The wrapper-owned dict keeps its declaration for the next layer.
    assert attention_kwargs == {"valid_kv_len": _REAL_LEN}


@pytest.mark.parametrize(
    "attention_kwargs, backend, is_causal",
    [
        pytest.param({"valid_kv_len": _REAL_LEN}, AttentionBackendType.SDPA, True, id="causal"),
        pytest.param(
            {"valid_kv_len": _REAL_LEN, "indices_k": torch.arange(_REAL_LEN)},
            AttentionBackendType.SDPA,
            False,
            id="packed-keys",
        ),
        pytest.param(
            {"valid_kv_len": _REAL_LEN, "thw": (3, 3, 5)}, AttentionBackendType.AITER_SPARGE, False, id="sparge"
        ),
        pytest.param({"valid_kv_len": _REAL_LEN, "thw": (3, 3, 5)}, AttentionBackendType.AITER_VSA, False, id="vsa"),
        pytest.param(
            {"valid_kv_len": _REAL_LEN, "thw": (3, 3, 5)}, AttentionBackendType.FLEX_BLOCK_ATTN, False, id="ssta"
        ),
    ],
)
def test_usp_refuses_a_declared_pad_with_joint_tensors_where_the_backend_would_keep_it(
    monkeypatch, usp_kv_lens, attention_kwargs, backend, is_causal
):
    monkeypatch.setattr(usp, "get_ulysses_parallel_rank", lambda: 0)
    with pytest.raises(NotImplementedError, match="valid_kv_len cannot be combined with joint tensors"):
        usp.USP(
            *_padded_qkv(),
            is_causal=is_causal,
            joint_strategy="front",
            **_joint(),
            backend=backend,
            attention_kwargs=attention_kwargs,
        )
