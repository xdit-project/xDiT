from unittest import mock

import torch

from xfuser.model_executor.layers.attention_mask import (
    MaskMetaCache,
    make_attn_mask_with_meta,
)


def _counting_build():
    return mock.Mock(side_effect=make_attn_mask_with_meta)


def test_a_mask_reused_across_steps_is_built_once():
    cache = MaskMetaCache()
    build = _counting_build()
    mask = torch.tensor([[1, 1, 0, 0]])

    first = cache.get(mask, build)
    for _ in range(3):
        assert cache.get(mask, build) is first
    assert build.call_count == 1


def test_guidance_alternating_two_masks_keeps_both():
    cache = MaskMetaCache()
    build = _counting_build()
    cond, uncond = torch.tensor([[1, 1, 1, 0]]), torch.tensor([[1, 0, 0, 0]])

    for _ in range(3):
        cache.get(cond, build)
        cache.get(uncond, build)
    assert build.call_count == 2


def test_a_different_mask_at_a_cached_masks_address_gets_its_own_metadata():
    """The caching allocator commonly hands the next request's mask the address of the
    previous request's; a new tensor over the same storage reproduces that."""
    cache = MaskMetaCache()
    mask_a = torch.tensor([[1, 1, 0, 0, 0]], dtype=torch.bool)
    cache.get(mask_a, make_attn_mask_with_meta)

    mask_b = torch.empty(0, dtype=torch.bool).set_(mask_a.untyped_storage(), 0, (1, 5))
    mask_b.copy_(torch.tensor([[1, 1, 1, 1, 0]], dtype=torch.bool))
    assert mask_b.data_ptr() == mask_a.data_ptr()

    meta = cache.get(mask_b, make_attn_mask_with_meta)
    assert meta.indices_k.tolist() == [0, 1, 2, 3]
    assert meta.max_seqlen_k == 4


def test_extra_build_arguments_are_part_of_the_key():
    cache = MaskMetaCache()
    mask = torch.tensor([[1, 0]])

    def padded(mask, pad):
        return make_attn_mask_with_meta(torch.cat([mask, mask.new_zeros(1, pad)], dim=1))

    assert cache.get(mask, padded, 1).attn_mask.shape[-1] == 3
    assert cache.get(mask, padded, 2).attn_mask.shape[-1] == 4


def test_the_least_recently_used_mask_is_evicted():
    cache = MaskMetaCache(capacity=2)
    build = _counting_build()
    first, second, third = (torch.tensor([[1, 1, 0]]) for _ in range(3))

    for mask in (first, second, first, third):
        cache.get(mask, build)
    assert build.call_count == 3
    cache.get(first, build)  # still cached: used more recently than second
    assert build.call_count == 3
    cache.get(second, build)
    assert build.call_count == 4


def test_the_lookup_runs_inside_a_compiled_forward():
    """Krea-2's whole transformer forward is compiled by default."""
    cache = MaskMetaCache()
    build = _counting_build()

    @torch.compile(backend="eager")
    def forward(x, mask):
        meta = cache.get(mask, build)
        return x.masked_fill(~meta.attn_mask.flatten(), 0.0)

    x = torch.ones(4)
    mask_a = torch.tensor([[1, 1, 0, 0]], dtype=torch.bool)
    forward(x, mask_a)
    forward(x, mask_a)
    assert build.call_count == 1

    mask_b = torch.tensor([[1, 1, 1, 0]], dtype=torch.bool)
    assert forward(x, mask_b).tolist() == [1.0, 1.0, 1.0, 0.0]
