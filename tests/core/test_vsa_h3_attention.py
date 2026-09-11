import pytest
import torch

from xfuser.core.vsa_h3_attention import (
    FASTH3_VSA_TILE_ELEMENTS,
    build_h3_vsa_block_mask,
    build_h3_vsa_metadata,
    compute_h3_vsa_topk,
    tile_h3_vsa_tensor,
    untile_h3_vsa_tensor,
)


def test_h3_vsa_metadata_keeps_prefix_segments_in_separate_tiles():
    metadata = build_h3_vsa_metadata(
        prefix_segments=(65, 0, 3),
        video_shape=(2, 3, 5),
        device=torch.device("cpu"),
    )

    assert metadata.num_prefix_tiles == 3
    assert metadata.num_video_tiles == 2
    assert metadata.total_seq_length == 98
    assert metadata.variable_block_sizes.tolist() == [64, 1, 3, 24, 6]
    assert metadata.padded_seq_length == 5 * FASTH3_VSA_TILE_ELEMENTS


def test_h3_vsa_tile_round_trip_with_partial_video_edges():
    metadata = build_h3_vsa_metadata(
        prefix_segments=(3, 2),
        video_shape=(5, 5, 5),
        device=torch.device("cpu"),
    )
    tensor = torch.arange(
        metadata.total_seq_length * 2,
        dtype=torch.float32,
    ).reshape(1, metadata.total_seq_length, 1, 2)

    tiled = tile_h3_vsa_tensor(tensor, metadata)
    restored = untile_h3_vsa_tensor(tiled, metadata)

    assert metadata.variable_block_sizes.tolist() == [
        3,
        2,
        64,
        16,
        16,
        4,
        16,
        4,
        4,
        1,
    ]
    assert torch.count_nonzero(
        tiled[
            :,
            torch.tensor(
                [
                    index
                    for index in range(metadata.padded_seq_length)
                    if index not in set(metadata.packed_to_tiled_index.tolist())
                ]
            ),
        ]
    ) == 0
    torch.testing.assert_close(restored, tensor)


def test_h3_vsa_exempt_mask_keeps_prefix_and_top_video_keys():
    scores = torch.zeros(1, 1, 4, 4)
    scores[0, 0, :, 2] = torch.tensor([4.0, 1.0, 3.0, 0.0])
    scores[0, 0, :, 3] = torch.tensor([1.0, 5.0, 2.0, 6.0])

    mask = build_h3_vsa_block_mask(
        scores,
        num_prefix_tiles=2,
        num_video_tiles=2,
        sparsity=0.9,
    )

    assert mask[..., :2].all()
    assert mask.sum(dim=-1).tolist() == [[[3, 3, 3, 3]]]
    assert mask[0, 0, 0, 2]
    assert mask[0, 0, 1, 3]
    assert mask[0, 0, 2, 2]
    assert mask[0, 0, 3, 3]


def test_h3_vsa_dense_mask_at_zero_sparsity():
    scores = torch.randn(2, 3, 5, 5)

    mask = build_h3_vsa_block_mask(
        scores,
        num_prefix_tiles=2,
        num_video_tiles=3,
        sparsity=0.0,
    )

    assert mask.all()


@pytest.mark.parametrize(
    ("sparsity", "tiles", "expected"),
    [(0.0, 10, 10), (0.5, 10, 5), (0.9, 10, 1), (1.0, 10, 1)],
)
def test_h3_vsa_topk(sparsity, tiles, expected):
    assert compute_h3_vsa_topk(sparsity, tiles) == expected


def test_h3_vsa_rejects_non_64_token_geometry():
    with pytest.raises(ValueError, match="64-token"):
        build_h3_vsa_metadata(
            prefix_segments=(4,),
            video_shape=(4, 4, 4),
            device=torch.device("cpu"),
            tile_shape=(2, 2, 2),
        )
