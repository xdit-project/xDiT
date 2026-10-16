import pytest
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


def _sequence_parallel_state(monkeypatch, ulysses_degree, patch_tokens):
    from xfuser.core import distributed

    starts = [0]
    for count in patch_tokens:
        starts.append(starts[-1] + count)
    state = type(
        "RuntimeState",
        (),
        {
            "num_pipeline_patch": len(patch_tokens),
            "patch_mode": False,
            "pipeline_patch_idx": 0,
            "pp_patches_token_num": list(patch_tokens),
            "pp_patches_token_start_idx_local": starts,
        },
    )()
    monkeypatch.setattr(cache_manager, "runtime_state_is_initialized", lambda: True)
    monkeypatch.setattr(distributed, "get_ulysses_parallel_world_size", lambda: ulysses_degree)
    monkeypatch.setattr(distributed, "get_runtime_state", lambda: state)
    return state


def _ulysses_gathered(tokens, ulysses_degree, patch_indices):
    """K/V as Ulysses all-to-all returns it: rank-major over the given patches.

    tokens[rank][patch] is that rank's local slice of the patch.
    """
    return torch.cat(
        [tokens[rank][patch] for rank in range(ulysses_degree) for patch in patch_indices],
        dim=1,
    )


# With as many Ulysses ranks as patches (e.g. 2 and 2) the patch-major and
# rank-major layouts are transposes of each other, so the inverse reordering
# would pass; ranks != patches pins the actual offsets.
@pytest.mark.parametrize(
    ("ulysses_degree", "patch_tokens", "updated_patch"),
    [(2, [2, 3], 1), (3, [2, 3], 1), (2, [1, 2, 1], 1), (2, [1, 2, 1], 2)],
    ids=["ulysses2-patches2", "ulysses3-patches2", "ulysses2-patches3-mid", "ulysses2-patches3-last"],
)
def test_sequence_parallel_patch_update_replaces_only_that_patch(
    monkeypatch, ulysses_degree, patch_tokens, updated_patch
):
    state = _sequence_parallel_state(monkeypatch, ulysses_degree, patch_tokens)
    ranks = range(ulysses_degree)
    old = [[torch.full((1, n, 1), 10 * rank + patch) for patch, n in enumerate(patch_tokens)] for rank in ranks]
    new = [[torch.full((1, n, 1), 100 + 10 * rank + patch) for patch, n in enumerate(patch_tokens)] for rank in ranks]
    manager = cache_manager.CacheManager()
    layer = torch.nn.Linear(2, 2)
    manager.register_cache_entry(layer, "attn", "sequence_parallel_attn_cache")

    manager.update_and_get_kv_cache(_ulysses_gathered(old, ulysses_degree, range(len(patch_tokens))), layer)
    state.patch_mode = True
    state.pipeline_patch_idx = updated_patch
    updated = manager.update_and_get_kv_cache(_ulysses_gathered(new, ulysses_degree, [updated_patch]), layer)

    expected = torch.cat(
        [(new if patch == updated_patch else old)[rank][patch] for patch in range(len(patch_tokens)) for rank in ranks],
        dim=1,
    )
    assert torch.equal(updated, expected)
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
