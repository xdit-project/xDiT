"""LingBot's block hooks also run when a cache executes blocks directly."""

import pytest
import torch

from xfuser.model_executor.models.transformers.transformer_lingbot_video import (
    xFuserLingBotVideoTransformer3DWrapper,
)


class _UpstreamBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = torch.nn.Identity()

    def forward(self, hidden_states):
        return hidden_states + 1


class _Wrapper(xFuserLingBotVideoTransformer3DWrapper, torch.nn.Module):
    def __init__(self):
        torch.nn.Module.__init__(self)
        self.blocks = torch.nn.ModuleList([_UpstreamBlock(), _UpstreamBlock()])
        self._install_usp_attention()


@pytest.mark.parametrize("local_pad_rows", [0, 1, 3])
@torch.no_grad()
def test_direct_block_execution_keeps_pad_rows_zero_and_preserves_real_rows(local_pad_rows):
    model = _Wrapper()
    model._sp_pad["local_rows"] = local_pad_rows
    output = torch.zeros(1, 3, 4)
    for block in model.blocks:
        output = block(output)
        if local_pad_rows:
            assert not output[:, -local_pad_rows:].any()
    if local_pad_rows < 3:
        assert output[:, : 3 - local_pad_rows].eq(2).all()

    # Reusing the installed hooks for a forward with no pad must preserve all rows.
    model._sp_pad["local_rows"] = 0
    output = torch.zeros(1, 3, 4)
    for block in model.blocks:
        output = block(output)
    assert output.eq(2).all()
