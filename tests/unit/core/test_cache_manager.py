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


def test_compiled_update_reuses_graph_after_cache_initialization(monkeypatch):
    monkeypatch.setattr(
        cache_manager,
        "runtime_state_is_initialized",
        lambda: False,
    )
    manager = cache_manager.CacheManager()
    layer = torch.nn.Linear(2, 2)
    manager.register_cache_entry(layer, "attn")
    update = torch.compile(
        lambda value: manager.update_and_get_kv_cache(value, layer),
        backend="eager",
        fullgraph=True,
    )

    first = torch.randn(1, 3, 8)
    second = torch.randn_like(first)
    third = torch.randn_like(first)
    assert update(first) is first
    with torch._dynamo.config.patch(error_on_recompile=True):
        assert update(second) is second
        manager.clear()
        assert update(third) is third
    assert layer._xdit_kv_cache is third


def test_compiled_sequence_parallel_update_reuses_graph(monkeypatch):
    from xfuser.core import distributed

    state = type("RuntimeState", (), {"num_pipeline_patch": 1})()
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
    manager.register_cache_entry(layer, "attn", "sequence_parallel_attn_cache")
    update = torch.compile(
        lambda key, value: manager.update_and_get_kv_cache([key, value], layer),
        backend="eager",
        fullgraph=True,
    )

    first = (torch.randn(1, 8, 4), torch.randn(1, 8, 4))
    second = (torch.randn(1, 8, 4), torch.randn(1, 8, 4))
    first_result = update(*first)
    with torch._dynamo.config.patch(error_on_recompile=True):
        second_result = update(*second)

    assert all(torch.equal(result, expected) for result, expected in zip(first_result, first))
    assert all(torch.equal(result, expected) for result, expected in zip(second_result, second))
    assert torch.equal(layer._xdit_kv_cache, torch.cat(second, dim=-1))


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


def test_compiled_sequence_parallel_patch_update_reuses_graph(monkeypatch):
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
    manager.update_and_get_kv_cache(initial, layer)

    state.patch_mode = True
    state.pipeline_patch_idx = 1
    update = torch.compile(
        lambda patch: manager.update_and_get_kv_cache(patch, layer),
        backend="eager",
        fullgraph=True,
    )
    first_patch = torch.full((1, 4, 1), 99)
    second_patch = torch.full((1, 4, 1), 77)
    update(first_patch)
    with torch._dynamo.config.patch(error_on_recompile=True):
        updated = update(second_patch)

    assert torch.equal(updated[:, :4], initial[:, :4])
    assert torch.equal(updated[:, 4:], second_patch)
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
