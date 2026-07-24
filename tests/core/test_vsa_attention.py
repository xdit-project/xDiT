import math

import torch
import pytest

from xfuser.core.vsa_attention import (
    block_mask_to_delta_lut,
    build_jenga_block_mask,
    jenga_scheduled_drop_rate,
)
from xfuser.core.sparge_attention.sparge import get_sliced_gilbert_perm


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
    assert jenga_scheduled_drop_rate(19, 20, rates) == 0.75
    assert jenga_scheduled_drop_rate(26, 30, rates) == 0.85


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
