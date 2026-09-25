import pytest
import torch

from xfuser.core.attention.backends.vsa_h3.attention import (
    FASTH3_VSA_KV_LIST_ALIGNMENT,
    FASTH3_VSA_SPARSITY,
    FASTH3_VSA_TILE_ELEMENTS,
    build_h3_vsa_block_mask,
    build_h3_vsa_kv_blocks,
    build_h3_vsa_kv_list,
    build_h3_vsa_metadata,
    compute_h3_vsa_topk,
    flex_h3_vsa_attention,
    h3_vsa_attention,
    pool_h3_vsa_tiles,
    tile_h3_vsa_bhsd,
    tile_h3_vsa_tensor,
    untile_h3_vsa_bhsd,
    untile_h3_vsa_tensor,
)


CPU = torch.device("cpu")


def test_h3_vsa_metadata_keeps_prefix_segments_in_separate_tiles():
    metadata = build_h3_vsa_metadata(
        prefix_segments=(65, 0, 3),
        video_shape=(2, 3, 5),
        device=CPU,
    )

    assert metadata.num_prefix_tiles == 3
    assert metadata.num_video_tiles == 2
    assert metadata.total_seq_length == 98
    assert metadata.variable_block_sizes.tolist() == [64, 1, 3, 24, 6]
    assert metadata.padded_seq_length == 5 * FASTH3_VSA_TILE_ELEMENTS
    assert metadata.num_prefix_partial_tiles == 2


def test_h3_vsa_metadata_orders_full_video_tiles_first():
    metadata = build_h3_vsa_metadata(
        prefix_segments=(3, 2),
        video_shape=(5, 5, 5),
        device=CPU,
    )

    sizes = metadata.variable_block_sizes[metadata.num_prefix_tiles :]
    full = sizes == FASTH3_VSA_TILE_ELEMENTS
    assert metadata.num_full_video_tiles == int(full.sum())
    # Full tiles form a prefix of the video range, so "is padded" is a compare.
    assert torch.equal(full, torch.arange(sizes.numel()) < full.sum())
    assert metadata.first_partial_video_tile == (
        metadata.num_prefix_tiles + metadata.num_full_video_tiles
    )


def test_h3_vsa_tile_round_trip_with_partial_video_edges():
    metadata = build_h3_vsa_metadata(
        prefix_segments=(3, 2),
        video_shape=(5, 5, 5),
        device=CPU,
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


@pytest.mark.parametrize(
    "prefix_segments,video_shape",
    [((3, 2), (5, 5, 5)), ((65, 0, 3), (2, 3, 5)), ((128, 16), (4, 8, 12))],
)
def test_h3_vsa_bhsd_tile_round_trip(prefix_segments, video_shape):
    metadata = build_h3_vsa_metadata(
        prefix_segments=prefix_segments, video_shape=video_shape, device=CPU
    )
    tensor = torch.randn(1, 3, metadata.total_seq_length, 2)

    tiled = tile_h3_vsa_bhsd(tensor, metadata)
    restored = untile_h3_vsa_bhsd(tiled, metadata)

    torch.testing.assert_close(restored, tensor)
    # Padded slots must be zero: pooling divides by the real token count.
    assert torch.count_nonzero(tiled[:, :, metadata.pad_slot_index]) == 0
    assert tiled.shape[2] == metadata.padded_seq_length


def test_h3_vsa_bhsd_tiling_matches_the_bshd_scatter():
    metadata = build_h3_vsa_metadata(
        prefix_segments=(3, 2), video_shape=(5, 5, 5), device=CPU
    )
    tensor = torch.randn(1, metadata.total_seq_length, 3, 2)

    scattered = tile_h3_vsa_tensor(tensor, metadata).transpose(1, 2)
    gathered = tile_h3_vsa_bhsd(tensor.transpose(1, 2), metadata)

    torch.testing.assert_close(gathered, scattered)
    torch.testing.assert_close(
        untile_h3_vsa_tensor(scattered.transpose(1, 2), metadata).transpose(1, 2),
        untile_h3_vsa_bhsd(gathered, metadata),
    )


def test_h3_vsa_packed_token_tile_maps_every_row_to_its_tile():
    metadata = build_h3_vsa_metadata(
        prefix_segments=(3, 2), video_shape=(5, 5, 5), device=CPU
    )

    assert metadata.packed_token_tile.shape == (metadata.total_seq_length,)
    torch.testing.assert_close(
        metadata.packed_token_tile,
        metadata.packed_to_tiled_index // metadata.tile_elements,
    )
    counts = torch.bincount(
        metadata.packed_token_tile, minlength=metadata.num_tiles
    )
    torch.testing.assert_close(counts, metadata.variable_block_sizes)


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
            device=CPU,
            tile_shape=(2, 2, 2),
        )


@pytest.mark.parametrize(
    "prefix_segments,video_shape,sparsity",
    [
        ((3, 2), (5, 5, 5), FASTH3_VSA_SPARSITY),
        ((65, 3), (5, 6, 7), 0.75),
        ((128, 16), (4, 8, 12), 0.5),
    ],
)
def test_h3_vsa_kv_blocks_select_the_reference_tile_set(
    prefix_segments, video_shape, sparsity
):
    """The fast index builder must pick exactly the reference policy's tiles."""
    torch.manual_seed(0)
    metadata = build_h3_vsa_metadata(
        prefix_segments=prefix_segments, video_shape=video_shape, device=CPU
    )
    num_tiles = metadata.num_tiles
    pooled_query = torch.randn(2, 3, num_tiles, 16)
    pooled_key = torch.randn(2, 3, num_tiles, 16)

    scores = torch.matmul(pooled_query, pooled_key.transpose(-2, -1)) * (16**-0.5)
    reference = build_h3_vsa_block_mask(
        scores,
        metadata.num_prefix_tiles,
        metadata.num_video_tiles,
        sparsity,
    )

    kv_num_blocks, kv_indices, full_kv_num_blocks, full_kv_indices = (
        build_h3_vsa_kv_blocks(pooled_query, pooled_key, metadata, sparsity)
    )

    width = metadata.num_prefix_tiles + compute_h3_vsa_topk(
        sparsity, metadata.num_video_tiles
    )
    ranks = torch.arange(width).view(1, 1, 1, -1)
    # One sentinel column absorbs the unused tail of each row's index buffer.
    rebuilt = reference.new_zeros(*reference.shape[:-1], num_tiles + 1)
    for counts, indices in (
        (kv_num_blocks, kv_indices),
        (full_kv_num_blocks, full_kv_indices),
    ):
        valid = ranks < counts.unsqueeze(-1)
        rebuilt.scatter_(
            -1, torch.where(valid, indices, num_tiles).long(), valid
        )
    rebuilt = rebuilt[..., :num_tiles]

    assert torch.equal(rebuilt, reference)
    # Every block on the partial list is padded and every full one is not, so
    # FlexAttention only runs mask_mod where padding actually exists.
    for counts, indices, expect_partial in (
        (kv_num_blocks, kv_indices, True),
        (full_kv_num_blocks, full_kv_indices, False),
    ):
        valid = ranks < counts.unsqueeze(-1)
        sizes = metadata.variable_block_sizes[indices.long()]
        padded = sizes != metadata.tile_elements
        assert torch.equal(
            padded[valid], torch.full_like(padded[valid], expect_partial)
        )
    assert torch.equal(
        kv_num_blocks + full_kv_num_blocks,
        torch.full_like(kv_num_blocks, width),
    )


def test_h3_vsa_kv_blocks_reject_mismatched_pooled_shape():
    metadata = build_h3_vsa_metadata(
        prefix_segments=(3, 2), video_shape=(5, 5, 5), device=CPU
    )
    with pytest.raises(ValueError, match="one row per tile"):
        build_h3_vsa_kv_blocks(
            torch.randn(1, 1, metadata.num_tiles + 1, 8),
            torch.randn(1, 1, metadata.num_tiles + 1, 8),
            metadata,
        )


def _reference_vsa_h3_attention(query, key, value, metadata, sparsity):
    """Dense masked reference for the tiled sparse branch."""
    pooled_query = pool_h3_vsa_tiles(query, metadata)
    pooled_key = pool_h3_vsa_tiles(key, metadata)
    pooled_value = pool_h3_vsa_tiles(value, metadata)
    scores = torch.matmul(pooled_query, pooled_key.transpose(-2, -1)) * (
        query.shape[-1] ** -0.5
    )
    block_map = build_h3_vsa_block_mask(
        scores, metadata.num_prefix_tiles, metadata.num_video_tiles, sparsity
    )
    token_mask = block_map.repeat_interleave(
        metadata.tile_elements, dim=-1
    ).repeat_interleave(metadata.tile_elements, dim=-2)
    token_mask = token_mask & metadata.tiled_slot_valid.view(1, 1, 1, -1)
    sparse = torch.nn.functional.scaled_dot_product_attention(
        query, key, value, attn_mask=token_mask
    )
    compressed = torch.matmul(torch.softmax(scores, dim=-1), pooled_value)
    return sparse, compressed


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="FlexAttention needs a GPU"
)
def test_flex_h3_vsa_attention_matches_the_dense_reference():
    torch.manual_seed(0)
    device = torch.device("cuda")
    metadata = build_h3_vsa_metadata(
        prefix_segments=(65, 3), video_shape=(5, 6, 7), device=device
    )
    shape = (1, 4, metadata.padded_seq_length, 64)
    query, key, value = (
        torch.randn(shape, device=device, dtype=torch.bfloat16) for _ in range(3)
    )
    for tensor in (query, key, value):
        tensor[:, :, metadata.pad_slot_index] = 0

    sparse, compressed = flex_h3_vsa_attention(query, key, value, metadata)
    sparse_ref, compressed_ref = _reference_vsa_h3_attention(
        query, key, value, metadata, FASTH3_VSA_SPARSITY
    )

    # Rows of padded slots are dropped by untiling, so only compare real tokens.
    valid = metadata.tiled_slot_valid
    torch.testing.assert_close(
        sparse[:, :, valid].float(),
        sparse_ref[:, :, valid].float(),
        rtol=2e-2,
        atol=2e-2,
    )
    torch.testing.assert_close(
        compressed.float(), compressed_ref.float(), rtol=2e-2, atol=2e-2
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_triton_vsa_h3_backend_falls_back_to_flex(monkeypatch):
    """TRITON_VSA_H3 must run the Flex path where the kernel cannot go."""
    from xfuser.core.attention import registry
    from xfuser.core.attention.backends.vsa_h3 import attention as vsa_h3_attention
    from xfuser.core.attention.spec import AttentionBackendType, AttnCall

    torch.manual_seed(0)
    device = torch.device("cuda")
    metadata = build_h3_vsa_metadata(
        prefix_segments=(65, 3), video_shape=(5, 6, 7), device=device
    )
    shape = (1, 4, metadata.total_seq_length, 64)
    query, key, value, gate = (
        torch.randn(shape, device=device, dtype=torch.bfloat16) for _ in range(4)
    )
    kwargs = {"vsa_h3_metadata": metadata, "vsa_h3_gate": gate}

    monkeypatch.setattr(
        vsa_h3_attention, "h3_vsa_triton_is_usable", lambda device: False
    )

    def run(backend):
        spec = registry.get(backend)
        spec.resolved()
        out, _ = spec.run(query, key, value, AttnCall(attention_kwargs=kwargs))
        return out

    fell_back = run(AttentionBackendType.TRITON_VSA_H3)
    flex = run(AttentionBackendType.FLEX_VSA_H3)

    assert torch.equal(fell_back, flex)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("use_triton", [False, True])
def test_h3_vsa_attention_backends_agree_with_the_dense_reference(use_triton):
    """Both kernels must produce the same gated, packed-order output."""
    from xfuser.core.attention.backends.vsa_h3 import (
        triton_kernel as vsa_h3_triton,
    )

    if use_triton and not vsa_h3_triton.is_available():
        pytest.skip("Triton is unavailable")

    torch.manual_seed(0)
    device = torch.device("cuda")
    metadata = build_h3_vsa_metadata(
        prefix_segments=(65, 3), video_shape=(5, 6, 7), device=device
    )
    shape = (1, 4, metadata.total_seq_length, 64)
    query, key, value, gate = (
        torch.randn(shape, device=device, dtype=torch.bfloat16) for _ in range(4)
    )

    actual = h3_vsa_attention(
        query, key, value, gate, metadata, use_triton=use_triton
    )

    sparse_ref, compressed_ref = _reference_vsa_h3_attention(
        tile_h3_vsa_bhsd(query, metadata),
        tile_h3_vsa_bhsd(key, metadata),
        tile_h3_vsa_bhsd(value, metadata),
        metadata,
        FASTH3_VSA_SPARSITY,
    )
    expected = untile_h3_vsa_bhsd(sparse_ref, metadata) + (
        compressed_ref.to(gate.dtype).index_select(
            2, metadata.packed_token_tile
        )
        * gate
    )

    assert actual.shape == expected.shape
    torch.testing.assert_close(
        actual.float(), expected.float(), rtol=2e-2, atol=2e-2
    )


@pytest.mark.parametrize(
    "prefix_segments,video_shape,sparsity",
    [((3, 2), (5, 5, 5), FASTH3_VSA_SPARSITY), ((65, 3), (5, 6, 7), 0.75)],
)
def test_h3_vsa_kv_list_matches_the_block_lists(
    prefix_segments, video_shape, sparsity
):
    """The Triton list and the Flex block lists must retain the same tiles."""
    torch.manual_seed(0)
    metadata = build_h3_vsa_metadata(
        prefix_segments=prefix_segments, video_shape=video_shape, device=CPU
    )
    num_tiles = metadata.num_tiles
    pooled_query = torch.randn(2, 3, num_tiles, 16)
    pooled_key = torch.randn(2, 3, num_tiles, 16)
    width = metadata.num_prefix_tiles + compute_h3_vsa_topk(
        sparsity, metadata.num_video_tiles
    )

    kv_list = build_h3_vsa_kv_list(pooled_query, pooled_key, metadata, sparsity)
    kv_num_blocks, kv_indices, full_kv_num_blocks, full_kv_indices = (
        build_h3_vsa_kv_blocks(pooled_query, pooled_key, metadata, sparsity)
    )

    assert kv_list.shape[-1] % FASTH3_VSA_KV_LIST_ALIGNMENT == 0
    assert kv_list.shape[-1] >= width
    # The alignment tail is the sentinel tile, which reads back as all-invalid.
    assert (kv_list[..., width:] == num_tiles).all()
    # The sentinel tile must be addressable and read back as pure padding.
    sentinel = metadata.tiled_to_packed_row[metadata.padded_seq_length :]
    assert sentinel.numel() == metadata.tile_elements
    assert (sentinel == metadata.total_seq_length).all()

    ranks = torch.arange(width).view(1, 1, 1, -1)
    both = torch.cat(
        (
            torch.where(ranks < kv_num_blocks.unsqueeze(-1), kv_indices, -1),
            torch.where(
                ranks < full_kv_num_blocks.unsqueeze(-1), full_kv_indices, -1
            ),
        ),
        dim=-1,
    )
    from_blocks = both.sort(dim=-1, descending=True).values[..., :width]
    assert torch.equal(
        kv_list[..., :width].sort(dim=-1, descending=True).values, from_blocks
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_h3_vsa_triton_pooling_selects_the_same_tiles_as_torch():
    """The gather-based pool must not move the top-k boundary.

    It sums in a different order than the torch reduction it replaces, so the
    tile means differ by an ULP or so; the contract is that the *selection*
    that reads them is unchanged.
    """
    from xfuser.core.attention.backends.vsa_h3 import (
        triton_kernel as vsa_h3_triton,
    )

    if not vsa_h3_triton.is_available():
        pytest.skip("Triton is unavailable")

    device = torch.device("cuda")
    metadata = build_h3_vsa_metadata(
        prefix_segments=(11, 90), video_shape=(9, 8, 11), device=device
    )
    for seed in range(8):
        torch.manual_seed(seed)
        packed = [
            torch.randn(
                (1, 3, metadata.total_seq_length, 64),
                device=device,
                dtype=torch.bfloat16,
            )
            for _ in range(2)
        ]
        reference = [
            pool_h3_vsa_tiles(tile_h3_vsa_bhsd(tensor, metadata), metadata)
            for tensor in packed
        ]
        gathered = [
            vsa_h3_triton.triton_pool_h3_vsa_tiles(tensor, metadata)
            for tensor in packed
        ]
        torch.testing.assert_close(gathered[0], reference[0], rtol=0, atol=1e-6)
        assert torch.equal(
            build_h3_vsa_kv_list(*gathered, metadata).sort(dim=-1).values,
            build_h3_vsa_kv_list(*reference, metadata).sort(dim=-1).values,
        )


def test_h3_vsa_tiled_to_packed_row_marks_padding_out_of_range():
    metadata = build_h3_vsa_metadata(
        prefix_segments=(3, 2), video_shape=(5, 5, 5), device=CPU
    )
    rows = metadata.tiled_to_packed_row

    assert rows.dtype == torch.int32
    assert rows.shape == (
        metadata.padded_seq_length + metadata.tile_elements,
    )
    rows = rows[: metadata.padded_seq_length]
    # Real slots address their packed row; padded slots are masked off by
    # pointing one past the end of the packed sequence.
    assert (rows[metadata.pad_slot_index] == metadata.total_seq_length).all()
    real = rows[metadata.tiled_slot_valid]
    assert torch.equal(real.sort().values, torch.arange(
        metadata.total_seq_length, dtype=torch.int32
    ))
    assert torch.equal(
        rows[metadata.packed_to_tiled_index],
        torch.arange(metadata.total_seq_length, dtype=torch.int32),
    )


def test_h3_vsa_triton_is_not_usable_on_cpu():
    """The kernel launches on the tensors' device, so CPU rules it out."""
    from xfuser.core.attention.backends.vsa_h3.attention import h3_vsa_triton_is_usable

    assert not h3_vsa_triton_is_usable(torch.device("cpu"))
