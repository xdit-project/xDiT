"""Wan and Wan VACE keep the sequence-parallel pad rows of every block's output at zero.

Attention drops the pad keys, but the blocks still write the pad rows, and
those rows enter statistics taken over a whole local tensor (FP8-comms
calibration, a step cache's residual difference, quantization scales shared
across query rows). A forward hook zeroes them on the ranks that hold any.
"""

import pytest
import torch

from xfuser.model_executor.models.transformers.transformer_wan import xFuserWanTransformer3DWrapper
from xfuser.model_executor.models.transformers.transformer_wan_vace import xFuserWanVACETransformer3DWrapper
from xfuser.model_executor.models.transformers.transformers_utils import local_sp_pad_rows, zero_sp_pad_rows_hook

_TINY_CONFIG = dict(
    patch_size=(1, 2, 2),
    num_attention_heads=2,
    attention_head_dim=8,
    in_channels=4,
    out_channels=4,
    text_dim=8,
    freq_dim=8,
    ffn_dim=16,
    num_layers=2,
    rope_max_seq_len=16,
)


@pytest.mark.parametrize(
    "seq_len, sp_size",
    [
        pytest.param(48, 4, id="no-pad"),
        pytest.param(45, 2, id="pad-on-the-last-rank"),
        pytest.param(9, 4, id="a-chunk-of-nothing-but-pad"),
        pytest.param(13, 8, id="pad-across-two-ranks"),
    ],
)
def test_pad_rows_are_counted_on_the_ranks_whose_chunk_holds_them(seq_len, sp_size):
    local_len = -(-seq_len // sp_size)
    for rank in range(sp_size):
        expected = sum(row >= seq_len for row in range(rank * local_len, (rank + 1) * local_len))
        assert local_sp_pad_rows(seq_len, local_len, rank) == expected


def test_the_hook_zeroes_the_pad_rows_and_nothing_else():
    output = torch.ones(1, 6, 4)
    zero_sp_pad_rows_hook({"local_rows": 2})(None, (), output)
    assert not output[:, -2:].any()
    assert output[:, :-2].eq(1).all()


def test_the_hook_leaves_a_chunk_without_pad_alone():
    output = torch.ones(1, 6, 4)
    zero_sp_pad_rows_hook({"local_rows": 0})(None, (), output)
    assert output.eq(1).all()


def test_the_hook_zeroes_both_tensors_a_vace_block_returns():
    conditioning, control = torch.ones(1, 6, 4), torch.ones(1, 6, 4)
    hook = zero_sp_pad_rows_hook({"local_rows": 1})
    hook(None, (), (conditioning, control))
    for tensor in (conditioning, control):
        assert not tensor[:, -1:].any()
        assert tensor[:, :-1].eq(1).all()
    # A block without an output projection returns no conditioning states.
    hook(None, (), (None, control))


@pytest.mark.parametrize(
    "model_cls, extra_config, block_lists",
    [
        pytest.param(xFuserWanTransformer3DWrapper, {}, ("blocks",), id="wan"),
        pytest.param(
            xFuserWanVACETransformer3DWrapper,
            dict(vace_layers=[0, 1], vace_in_channels=4),
            ("blocks", "vace_blocks"),
            id="vace",
        ),
    ],
)
def test_every_block_zeroes_the_pad_rows_the_model_counted(model_cls, extra_config, block_lists):
    model = model_cls(**_TINY_CONFIG, **extra_config)
    model._sp_pad["local_rows"] = 1
    for name in block_lists:
        for block in getattr(model, name):
            output = torch.ones(1, 3, 2)
            for hook in block._forward_hooks.values():
                hook(block, (), output)
            assert not output[:, -1:].any(), f"{name}: pad row kept"
            assert output[:, :-1].eq(1).all(), f"{name}: real rows changed"
