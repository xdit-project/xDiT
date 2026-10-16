"""The block hook that clears sequence-parallel pad rows serves every wrapper's output layout."""

import torch

from xfuser.model_executor.models.transformers.transformers_utils import zero_sp_pad_rows_hook


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


def test_the_cleared_dimension_and_the_row_count_key_are_selectable():
    # Cosmos3's generation stream is [tokens, hidden]; HunyuanVideo-1.5 counts image and text pads apart.
    gen = torch.ones(6, 4)
    zero_sp_pad_rows_hook({"local_rows": 1}, index=1, dim=0)(None, (), (torch.ones(2, 4), gen))
    assert not gen[-1:].any()
    assert gen[:-1].eq(1).all()

    pad = {"image_rows": 1, "text_rows": 0}
    image, text = torch.ones(1, 4, 2), torch.ones(1, 3, 2)
    for hook in (
        zero_sp_pad_rows_hook(pad, index=0, rows_key="image_rows"),
        zero_sp_pad_rows_hook(pad, index=1, rows_key="text_rows"),
    ):
        hook(None, (), (image, text))
    assert not image[:, -1:].any()
    assert image[:, :-1].eq(1).all()
    assert text.eq(1).all()
