"""The FLUX.3 wrapper must build from a checkpoint config exactly as the stock transformer does.

diffusers picks the config keys it passes to ``__init__`` from the class signature, so a
wrapper that hid upstream's parameters would build every checkpoint with defaults; a
flow-map (step-distilled) checkpoint would lose its target-timestep embedder.
"""

from types import SimpleNamespace

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


def test_each_transformer_call_advances_the_step_schedule(monkeypatch):
    # The hybrid GEMM schedule switches precision per step from this counter.
    calls = []
    state = SimpleNamespace(increment_step_counter=lambda: calls.append(1))
    monkeypatch.setattr(transformer_flux3, "get_runtime_state", lambda: state)
    wrapper = transformer_flux3.xFuserFlux3Transformer2DWrapper(**TINY, flow_map=True)

    for hook in wrapper._forward_pre_hooks.values():
        hook(wrapper, ())

    assert len(calls) == 1


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


def test_block_inputs_keep_one_address_per_shape():
    import torch

    wrapper = transformer_flux3.xFuserFlux3Transformer2DWrapper(**TINY, flow_map=True)
    first = wrapper._block_input("joint.cos", torch.ones(4, 2))
    second = wrapper._block_input("joint.cos", torch.zeros(4, 2))
    other_shape = wrapper._block_input("joint.cos", torch.zeros(5, 2))

    assert second.data_ptr() == first.data_ptr() and torch.equal(second, torch.zeros(4, 2))
    assert other_shape.data_ptr() != first.data_ptr()
    assert getattr(first, "_dynamo_static_input_type", None) == "unguarded"


def test_step_constants_recompute_for_a_new_or_modified_source():
    import torch

    wrapper = transformer_flux3.xFuserFlux3Transformer2DWrapper(**TINY, flow_map=True)
    calls = []

    def compute(x):
        calls.append(1)
        return x * 2

    ids = torch.ones(3)
    with torch.no_grad():
        first = wrapper._step_constant("ids", ids, compute)
        assert wrapper._step_constant("ids", ids, compute) is first and len(calls) == 1
        ids.add_(1)
        assert torch.equal(wrapper._step_constant("ids", ids, compute), torch.full((3,), 4.0)) and len(calls) == 2
        wrapper._step_constant("ids", torch.ones(3), compute)
        assert len(calls) == 3
    wrapper._step_constant("ids", ids, compute)
    wrapper._step_constant("ids", ids, compute)
    assert len(calls) == 5
    with torch.inference_mode():
        inference_ids = torch.ones(3)
        first = wrapper._step_constant("ids", inference_ids, compute)
        assert wrapper._step_constant("ids", inference_ids, compute) is first and len(calls) == 6


def test_step_constants_serve_both_cfg_branches():
    import torch

    wrapper = transformer_flux3.xFuserFlux3Transformer2DWrapper(**TINY, flow_map=True)
    calls = []

    def compute(x):
        calls.append(1)
        return x * 2

    conditional, unconditional, third = torch.ones(3), torch.zeros(3), torch.ones(4)
    with torch.no_grad():
        results = [wrapper._step_constant("context", source, compute) for source in (conditional, unconditional) * 3]
        assert len(calls) == 2
        assert results[4] is results[0] and results[5] is results[1]
        wrapper._step_constant("context", third, compute)
        wrapper._step_constant("context", unconditional, compute)
        assert len(calls) == 3  # the third source evicted the oldest entry only
        wrapper._step_constant("context", conditional, compute)
        assert len(calls) == 4
