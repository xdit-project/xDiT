"""The FLUX.3 wrapper must build from a checkpoint config exactly as the stock transformer does.

diffusers picks the config keys it passes to ``__init__`` from the class signature, so a
wrapper that hid upstream's parameters would build every checkpoint with defaults; a
flow-map (step-distilled) checkpoint would lose its target-timestep embedder.
"""

import pytest

transformer_flux3 = pytest.importorskip("xfuser.model_executor.models.transformers.transformer_flux3")

TINY = {
    "in_channels": 8,
    "cond_in_channels": 8,
    "num_layers": 1,
    "num_single_layers": 1,
    "attention_head_dim": 16,
    "num_attention_heads": 2,
    "joint_attention_dim": 12,
    "timestep_channels": 32,
    "axes_dims_rope": (4, 4, 4, 4),
    "rope_theta": 2000,
}


@pytest.mark.parametrize("flow_map", [True, False])
def test_from_config_keeps_every_checkpoint_setting(flow_map):
    from diffusers.models.transformers.transformer_flux3 import Flux3Transformer2DModel

    stock = Flux3Transformer2DModel(**TINY, flow_map=flow_map)

    wrapper = transformer_flux3.xFuserFlux3Transformer2DWrapper.from_config(stock.config)

    assert dict(wrapper.config) == dict(stock.config)
    assert (wrapper.target_timestep_embedder is not None) == flow_map
    assert wrapper.state_dict().keys() == stock.state_dict().keys()


@pytest.mark.parametrize(
    ("start", "length", "expected_rows"),
    [
        (0, 4, [0, 0, 0, 0]),  # image only
        (4, 4, [0, 0, 1, 1]),  # straddles image and reference
        (8, 4, [1, 1, 1, 1]),  # reference, then SP pad
        (2, 0, []),  # a rank holding only text
    ],
)
def test_content_modulation_gives_each_position_its_stream_row(start, length, expected_rows):
    import torch

    wrapper = transformer_flux3.xFuserFlux3Transformer2DWrapper(**TINY, flow_map=True)
    rows = [torch.full((1, 1, 3), float(index)) for index in range(2)]

    mod = wrapper._content_modulation(rows, [6, 4], start=start, length=length)

    assert mod.shape == (1, length, 3)
    assert mod[0, :, 0].tolist() == expected_rows


def test_releasing_the_step_constants_drops_the_prompt_they_hold():
    import torch

    wrapper = transformer_flux3.xFuserFlux3Transformer2DWrapper(**TINY, flow_map=True)
    calls = []
    context = torch.ones(3)

    def compute(x):
        calls.append(1)
        return x * 2

    with torch.no_grad():
        wrapper._step_constants("context", context, compute)
        wrapper._step_constants("context", context, compute)
        assert len(calls) == 1
        block_input = wrapper._block_inputs("joint.cos", torch.ones(4, 2))
        wrapper.release_step_constants()
        wrapper._step_constants("context", context, compute)
        assert len(calls) == 2
        # The block inputs keep their addresses across calls.
        assert wrapper._block_inputs("joint.cos", torch.ones(4, 2)).data_ptr() == block_input.data_ptr()
