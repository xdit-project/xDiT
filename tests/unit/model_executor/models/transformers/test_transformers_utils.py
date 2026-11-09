"""The block hook that clears sequence-parallel pad rows serves every wrapper's output layout."""

import pytest
import torch

from xfuser.model_executor.models.transformers.transformers_utils import sp_pad_spans_ranks, zero_sp_pad_rows_hook


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
