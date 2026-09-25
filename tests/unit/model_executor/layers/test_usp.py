from unittest import mock

import pytest
import torch

from xfuser.model_executor.layers import usp


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
