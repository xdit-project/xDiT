"""HunyuanDiT skip buffers hold every image and optional CFG copy."""

from types import SimpleNamespace

import pytest

from xfuser.config import InputConfig
from xfuser.core.distributed import runtime_state
from xfuser.core.distributed.runtime_state import DiTRuntimeState


@pytest.mark.parametrize(
    "prompts, images_per_prompt, cfg, cfg_degree, batch",
    [
        (1, 1, True, 1, 2),
        (1, 2, True, 1, 4),
        (2, 3, True, 1, 12),
        (2, 3, True, 2, 6),
        (1, 1, False, 1, 1),
        (2, 3, False, 1, 6),
    ],
)
def test_skip_buffer_holds_the_batch_the_stages_send(monkeypatch, prompts, images_per_prompt, cfg, cfg_degree, batch):
    buffers = {}

    def set_skip_tensor_recv_buffer(patches_shape_list, feature_map_shape):
        buffers["patches"] = patches_shape_list
        buffers["full"] = feature_map_shape

    monkeypatch.setattr(
        runtime_state,
        "get_pp_group",
        lambda: SimpleNamespace(set_skip_tensor_recv_buffer=set_skip_tensor_recv_buffer),
    )
    state = DiTRuntimeState.__new__(DiTRuntimeState)
    state.input_config = InputConfig(batch_size=prompts)
    state.parallel_config = SimpleNamespace(cfg_degree=cfg_degree)
    state.backbone_inner_dim = 8
    state.pp_patches_token_start_end_idx_global = [[0, 16], [16, 24]]

    state._reset_recv_skip_buffer(
        num_blocks_per_stage=3,
        num_images_per_prompt=images_per_prompt,
        classifier_free_guidance=cfg,
    )

    assert buffers["patches"] == [[3, batch, 16, 8], [3, batch, 8, 8]]
    assert buffers["full"] == [3, batch, 24, 8]
