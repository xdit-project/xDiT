import torch

from xfuser.core.cache_manager import cache_manager


def test_module_cache_uses_a_nonpersistent_buffer(monkeypatch):
    monkeypatch.setattr(
        cache_manager,
        "runtime_state_is_initialized",
        lambda: False,
    )
    manager = cache_manager.CacheManager()
    layer = torch.nn.Linear(2, 2)
    key_value = torch.randn(1, 3, 8)

    manager.register_cache_entry(layer, "attn")
    result = manager.update_and_get_kv_cache(key_value, layer)

    assert result is key_value
    assert layer._xdit_kv_cache is key_value
    assert "_xdit_kv_cache" not in layer.state_dict()


def test_compiled_update_writes_the_module_cache_buffer(monkeypatch):
    monkeypatch.setattr(
        cache_manager,
        "runtime_state_is_initialized",
        lambda: False,
    )
    manager = cache_manager.CacheManager()
    layer = torch.nn.Linear(2, 2)
    key_value = torch.randn(1, 3, 8)
    manager.register_cache_entry(layer, "attn")
    update = torch.compile(
        lambda value: manager.update_and_get_kv_cache(value, layer),
        backend="eager",
        fullgraph=True,
    )

    assert update(key_value) is key_value
    assert layer._xdit_kv_cache is key_value


def test_sequence_parallel_patch_update_uses_the_module_cache_buffer(
    monkeypatch,
):
    from xfuser.core import distributed

    state = type(
        "RuntimeState",
        (),
        {
            "num_pipeline_patch": 2,
            "patch_mode": False,
            "pipeline_patch_idx": 0,
            "pp_patches_token_num": [2, 2],
            "pp_patches_token_start_idx_local": [0, 2, 4],
        },
    )()
    monkeypatch.setattr(
        cache_manager,
        "runtime_state_is_initialized",
        lambda: True,
    )
    monkeypatch.setattr(
        distributed,
        "get_ulysses_parallel_world_size",
        lambda: 2,
    )
    monkeypatch.setattr(distributed, "get_runtime_state", lambda: state)
    manager = cache_manager.CacheManager()
    layer = torch.nn.Linear(2, 2)
    initial = torch.arange(8).reshape(1, 8, 1)

    manager.register_cache_entry(layer, "attn", "sequence_parallel_attn_cache")
    assert torch.equal(
        manager.update_and_get_kv_cache(initial, layer),
        initial,
    )

    state.patch_mode = True
    state.pipeline_patch_idx = 1
    patch = torch.full((1, 4, 1), 99)
    updated = manager.update_and_get_kv_cache(patch, layer)

    assert torch.equal(updated[:, :4], initial[:, :4])
    assert torch.equal(updated[:, 4:], patch)
    assert layer._xdit_kv_cache is updated


def test_clear_releases_module_and_manager_cache_tensors(monkeypatch):
    monkeypatch.setattr(
        cache_manager,
        "runtime_state_is_initialized",
        lambda: False,
    )
    manager = cache_manager.CacheManager()
    module_layer = torch.nn.Linear(2, 2)
    object_layer = object()
    key_value = torch.randn(1, 3, 8)

    manager.register_cache_entry(module_layer, "attn")
    manager.register_cache_entry(object_layer, "attn")
    manager.update_and_get_kv_cache(key_value, module_layer)
    manager.update_and_get_kv_cache(key_value, object_layer)

    manager.clear()

    assert module_layer._xdit_kv_cache is None
    assert manager.cache["attn", module_layer].tensors == [None]
    assert manager.cache["attn", object_layer].tensors == [None]
