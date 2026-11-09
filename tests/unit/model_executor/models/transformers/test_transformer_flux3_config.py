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
