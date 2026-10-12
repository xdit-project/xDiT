from unittest import mock

import pytest
import torch

from xfuser.core.attention.spec import AttentionBackendType
from xfuser.model_executor.layers import usp
from xfuser.model_executor.layers.attention_mask import make_attn_mask_with_meta


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
