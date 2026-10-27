"""The helpers the wrappers share: the pad-row hook fits every output layout, the prompt mask follows the keys."""

import pytest
import torch

from xfuser.model_executor.models.transformers.transformers_utils import (
    sp_pad_spans_ranks,
    text_image_key_mask,
    zero_sp_pad_rows_hook,
)


def test_the_selected_output_is_cleared_and_the_other_streams_are_kept():
    # A double-stream block returns (text, image): only the image is ever padded.
    pad = {"local_rows": 2}
    hook = zero_sp_pad_rows_hook(pad, index=1)
    text, image = torch.ones(1, 3, 4), torch.ones(1, 5, 4)
    for rows in (2, 0, 1):
        pad["local_rows"] = rows
        output = (text.clone(), image.clone())
        hook(None, (), output)
        torch.testing.assert_close(output[0], text)
        torch.testing.assert_close(output[1][:, : 5 - rows], image[:, : 5 - rows])
        assert not output[1][:, 5 - rows :].any()


def test_a_block_returning_one_tensor_is_cleared_whole_when_an_index_is_selected():
    # FLUX.2 single-stream blocks return the joined [text, image] tensor.
    output = torch.ones(1, 6, 4)
    zero_sp_pad_rows_hook({"local_rows": 2}, index=1)(None, (), output)
    assert not output[:, -2:].any()
    assert output[:, :-2].eq(1).all()


def test_the_tokens_are_the_second_to_last_dimension():
    # Cosmos3's generation stream is [tokens, hidden], the others' [batch, tokens, hidden].
    gen = torch.ones(6, 4)
    zero_sp_pad_rows_hook({"local_rows": 1}, index=1)(None, (), (torch.ones(2, 4), gen))
    assert not gen[-1:].any()
    assert gen[:-1].eq(1).all()


@pytest.mark.parametrize(
    "seq_len,sp_size,spans",
    [(8, 4, False), (7, 4, False), (5, 8, True), (9, 8, True), (4225, 8, False), (1, 2, False)],
)
def test_the_pad_spans_ranks_when_it_is_longer_than_one_shard(seq_len, sp_size, spans):
    assert sp_pad_spans_ranks(seq_len, sp_size) is spans


def _mask_rows(mask, keys, text_mask):
    """Check each mask row against the key ids: text ids index the prompt, ids from 100 are image tokens, -1 is pad."""
    assert mask.shape == (text_mask.shape[0], len(keys))
    for sample in range(text_mask.shape[0]):
        for position, key in enumerate(keys):
            assert bool(mask[sample, position]) == (key >= 100 or (key >= 0 and bool(text_mask[sample, key])))


def test_a_sharded_prompt_mask_follows_the_keys_in_gathered_rank_order_and_masks_the_image_padding():
    # Four ranks, a prompt of 8 tokens (2 per rank) and 5 image tokens padded to 8 (2 per rank). Rank r holds
    # [text_r, image_r]; the gathered keys are every rank's chunk in rank order, and the last 3 image rows are pad.
    text_mask = torch.tensor([[1, 1, 1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1]], dtype=torch.bool)
    image = [100 + row if row < 5 else -1 for row in range(8)]
    gathered = [key for rank in range(4) for key in (2 * rank, 2 * rank + 1, image[2 * rank], image[2 * rank + 1])]

    mask = text_image_key_mask(text_mask, image_tokens=5, sp_pad=3, sp_world_size=4)

    _mask_rows(mask, gathered, text_mask)


@pytest.mark.parametrize("joint_strategy, order", [("front", "text-first"), ("rear", "image-first")])
def test_a_replicated_prompt_mask_joins_the_padded_image_keys_where_usp_joins_the_prompt(joint_strategy, order):
    text_mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]], dtype=torch.bool)
    text, image = [0, 1, 2, 3, 4], [100, 101, 102, -1]
    keys = text + image if order == "text-first" else image + text

    # The mask keeps the image padding in K, masked, so no pad is trimmed ahead of the joint keys.
    mask = text_image_key_mask(text_mask, image_tokens=3, sp_pad=1, sp_world_size=2, joint_strategy=joint_strategy)

    _mask_rows(mask, keys, text_mask)
