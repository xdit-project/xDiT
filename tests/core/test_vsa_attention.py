import inspect
import math

import pytest
import torch

from xfuser.config.args import xFuserArgs
from xfuser.config.config import RuntimeConfig
from xfuser.core.distributed.attention_backend import AttentionBackendType
from xfuser.core.distributed.runtime_state import DiTRuntimeState
from xfuser.core.vsa_attention import (
    _first_frame_block_count,
    aiter_vsa_attention,
    block_mask_to_delta_lut,
    build_jenga_block_mask,
    jenga_scheduled_drop_rate,
)
from xfuser.core.sparge_attention.sparge import (
    get_sliced_gilbert_perm,
    setup_sparge,
)


def _reference_jenga_mask(query, key, block_size, top_k, threshold):
    batch, heads, sequence, head_dim = query.shape
    blocks = sequence // block_size
    query_pool = query.reshape(
        batch, heads, blocks, block_size, head_dim
    ).mean(dim=-2)
    key_pool = key.reshape(
        batch, heads, blocks, block_size, head_dim
    ).mean(dim=-2)
    scores = torch.matmul(query_pool, key_pool.transpose(-1, -2))
    probabilities = torch.softmax(scores * head_dim**-0.5, dim=-1)
    sorted_probabilities, indices = probabilities.sort(
        dim=-1, descending=True
    )
    needed = (sorted_probabilities.cumsum(-1) <= threshold).sum(-1) + 1
    needed = needed.clamp(min=top_k, max=blocks)
    result = torch.zeros(
        batch, heads, blocks, blocks, dtype=torch.bool
    )
    for b in range(batch):
        for h in range(heads):
            for q in range(blocks):
                result[b, h, q, indices[b, h, q, : needed[b, h, q]]] = True
    return result


def test_build_jenga_block_mask_matches_reference():
    generator = torch.Generator().manual_seed(17)
    query = torch.randn(2, 3, 32, 8, generator=generator)
    key = torch.randn(2, 3, 32, 8, generator=generator)

    actual = build_jenga_block_mask(
        query,
        key,
        block_size=8,
        top_k=2,
        prob_threshold=0.7,
    )
    expected = _reference_jenga_mask(query, key, 8, 2, 0.7)
    torch.testing.assert_close(actual, expected)


def test_jenga_mask_unions_static_and_first_frame_relations():
    query = torch.zeros(1, 1, 32, 4)
    key = torch.zeros_like(query)
    static = torch.eye(4, dtype=torch.bool)

    mask = build_jenga_block_mask(
        query,
        key,
        block_size=8,
        top_k=1,
        prob_threshold=0.0,
        static_block_mask=static,
        first_frame_blocks=2,
    )

    assert mask[0, 0, 0, :2].all()
    assert mask[0, 0, 1, :2].all()
    assert mask[0, 0, 2, 2]
    assert mask[0, 0, 3, 3]


def test_block_mask_to_delta_lut():
    mask = torch.tensor(
        [[[[True, False, True, False, False, True], [False, True, False, True, False, False]]]]
    )
    lut, counts = block_mask_to_delta_lut(mask)

    assert lut.dtype == torch.int32
    assert counts.dtype == torch.int32
    torch.testing.assert_close(counts, torch.tensor([[[3, 2]]], dtype=torch.int32))
    torch.testing.assert_close(
        lut[0, 0, 0, :3], torch.tensor([0, 2, 3], dtype=torch.int32)
    )
    torch.testing.assert_close(
        lut[0, 0, 1, :2], torch.tensor([1, 2], dtype=torch.int32)
    )


def test_jenga_scheduled_drop_rate_matches_reference():
    rates = [0.75, 0.85]
    assert jenga_scheduled_drop_rate(0, 50, rates) == 0.0
    assert math.isclose(
        jenga_scheduled_drop_rate(1, 50, rates), 0.75 * 10 / 49
    )
    assert math.isclose(
        jenga_scheduled_drop_rate(2, 50, rates), 0.75 * 20 / 49
    )
    assert jenga_scheduled_drop_rate(25, 50, rates) == 0.75
    assert jenga_scheduled_drop_rate(26, 50, rates) == 0.85
    assert jenga_scheduled_drop_rate(10, 20, rates) == 0.75
    assert jenga_scheduled_drop_rate(11, 20, rates) == 0.85
    assert jenga_scheduled_drop_rate(15, 30, rates) == 0.75
    assert jenga_scheduled_drop_rate(16, 30, rates) == 0.85


def test_vsa_probability_defaults_match_benchmark():
    assert (
        xFuserArgs.__dataclass_fields__["vsa_prob_threshold"].default
        == 0.9
    )
    assert (
        RuntimeConfig.__dataclass_fields__["vsa_prob_threshold"].default
        == 0.9
    )
    assert (
        RuntimeConfig.__dataclass_fields__["vsa_collect_density"].default
        is False
    )
    assert (
        inspect.signature(build_jenga_block_mask)
        .parameters["prob_threshold"]
        .default
        == 0.9
    )


def test_vsa_rejects_hybrid_attention_schedule():
    state = DiTRuntimeState.__new__(DiTRuntimeState)
    state.runtime_config = RuntimeConfig(use_hybrid_attn_schedule=True)

    with pytest.raises(RuntimeError, match="hybrid attention schedule"):
        state._check_if_backend_compatible_with_current_configuration(
            AttentionBackendType.AITER_VSA
        )


def test_runtime_state_tracks_vsa_schedule_per_timestep():
    state = DiTRuntimeState.__new__(DiTRuntimeState)
    state.reset_vsa_schedule_state(2)

    assert state.advance_vsa_schedule(1000.0) == (0, 2)
    assert state.advance_vsa_schedule(1000.0) == (0, 2)
    assert state.advance_vsa_schedule(500.0) == (1, 2)
    assert state.advance_vsa_schedule(0.0) == (0, 2)


def test_first_frame_blocks_ignore_cross_frame_partial_block():
    # Wan 480x832 has a 21x30x52 post-patch grid. Twelve 128-token
    # blocks fit wholly in the first frame; the remaining 24 tokens share
    # a boundary block with the next frame.
    assert _first_frame_block_count((21, 30, 52), 128) == 12
    with pytest.raises(ValueError, match="positive"):
        _first_frame_block_count((0, 30, 52), 128)


def test_padded_static_mask_cache_uses_layout_identity():
    query = torch.randn(1, 1, 24, 4)
    kwargs = {
        "thw": (2, 3, 4),
        "sp_size": 1,
        "reorder_sequence": True,
        "use_static_block_mask": True,
        "block_m": 16,
        "block_n": 16,
        "pad_block_divisible": True,
        "use_sliced_gilbert": True,
    }
    first = setup_sparge(query, query, query, **kwargs)[-1]
    second = setup_sparge(query.clone(), query, query, **kwargs)[-1]
    assert first is second


@pytest.mark.parametrize("reorder_sequence", [False, True])
def test_existing_sparge_paths_keep_padded_mask_allocation(reorder_sequence):
    query = torch.randn(1, 1, 24, 4)
    kwargs = {
        "thw": (2, 3, 4),
        "sp_size": 1,
        "reorder_sequence": reorder_sequence,
        "use_static_block_mask": True,
        "block_m": 16,
        "block_n": 16,
        "pad_block_divisible": True,
        "use_sliced_gilbert": False,
    }
    first = setup_sparge(query, query, query, **kwargs)[-1]
    second = setup_sparge(query.clone(), query, query, **kwargs)[-1]

    assert first is not second
    torch.testing.assert_close(first, second)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires AITER GPU")
def test_vsa_density_collection_is_opt_in():
    query = torch.randn(
        1, 4, 256, 128, device="cuda", dtype=torch.bfloat16
    )
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    kwargs = {
        "thw": (1, 16, 16),
        "sp_size": 1,
        "drop_rate": 0.5,
        "reorder_sequence": False,
        "use_static_block_mask": False,
        "use_first_frame_mask": False,
    }
    _, disabled_density = aiter_vsa_attention(
        query, key, value, collect_density=False, **kwargs
    )
    _, enabled_density = aiter_vsa_attention(
        query, key, value, collect_density=True, **kwargs
    )
    assert disabled_density is None
    assert enabled_density is not None
    assert 0.0 < float(enabled_density) <= 1.0


def test_sliced_gilbert_permutation_matches_jenga_reference():
    forward, inverse = get_sliced_gilbert_perm(
        (2, 3, 4), torch.device("cpu")
    )
    assert inverse.tolist() == [
        0, 1, 10, 11, 3, 2, 9, 8, 4, 5, 6, 7,
        23, 22, 13, 12, 20, 21, 14, 15, 19, 18, 17, 16,
    ]
    assert forward.tolist() == [
        0, 1, 5, 4, 8, 9, 10, 11, 7, 6, 2, 3,
        15, 14, 18, 19, 23, 22, 21, 20, 16, 17, 13, 12,
    ]
    torch.testing.assert_close(inverse[forward], torch.arange(24))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires AITER GPU")
def test_ck_vsa_matches_masked_dense():
    try:
        from aiter.ops.jenga_sparse_attention import vsa_sparse_attention
    except ImportError:
        pytest.skip("AITER VSA kernel is unavailable")

    torch.manual_seed(7)
    query = torch.randn(
        1, 4, 256, 128, device="cuda", dtype=torch.bfloat16
    )
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    block_mask = torch.tensor(
        [[[[True, False], [True, True]]]], device="cuda"
    ).expand(1, 4, 2, 2).contiguous()
    lut, counts = block_mask_to_delta_lut(block_mask)
    seqstart = torch.tensor([0, 256], device="cuda", dtype=torch.int32)
    output = torch.empty_like(query)
    output = vsa_sparse_attention(
        query, key, value, lut, counts, output, None, None,
        seqstart, seqstart, 0, 1, 4, 4, 256, 256, 128, 128,
    )

    token_mask = block_mask.repeat_interleave(
        128, dim=-2
    ).repeat_interleave(128, dim=-1)
    scores = torch.matmul(query.float(), key.float().transpose(-1, -2))
    scores.mul_(128 ** -0.5).masked_fill_(~token_mask, -torch.inf)
    reference = torch.matmul(torch.softmax(scores, dim=-1), value.float())
    relative_l2 = (
        torch.linalg.vector_norm(output.float() - reference)
        / torch.linalg.vector_norm(reference)
    )
    cosine = torch.nn.functional.cosine_similarity(
        output.float().flatten(), reference.flatten(), dim=0
    )
    assert float(relative_l2) <= 1e-2
    assert float(cosine) >= 0.999
