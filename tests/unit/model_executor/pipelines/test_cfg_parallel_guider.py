"""xFuserCFGParallelGuidance on one CFG rank, with the CFG group stubbed."""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("diffusers.guiders", reason="installed diffusers has no modular guiders")

from diffusers.guiders import ClassifierFreeGuidance

from xfuser.model_executor.pipelines import guiders

COND = torch.tensor([1.0, 2.0])
UNCOND = torch.tensor([0.5, 0.5])


def _inputs():
    return {"encoder_hidden_states": (torch.zeros(1), torch.ones(1))}


@pytest.fixture
def unconditional_rank(monkeypatch):
    """This process is CFG rank 1 of 2; the gather hands back rank 0's conditional prediction."""
    monkeypatch.setattr(guiders, "get_classifier_free_guidance_world_size", lambda: 2)
    monkeypatch.setattr(guiders, "get_classifier_free_guidance_rank", lambda: 1)
    gather = SimpleNamespace(all_gather=lambda tensor, separate_tensors: [COND, tensor])
    monkeypatch.setattr(guiders, "get_cfg_group", lambda: gather)


def _step(guider):
    guider.set_state(step=0, num_inference_steps=2, timestep=torch.tensor(700.0))
    return guider.prepare_inputs(_inputs())


def test_a_rank_gets_only_its_branch_and_the_combine_uses_both(unconditional_rank):
    guider = guiders.xFuserCFGParallelGuidance(guidance_scale=3.5)

    (batch,) = _step(guider)
    guider.prepare_models(torch.nn.Identity())
    assert torch.equal(batch.encoder_hidden_states, torch.ones(1))  # the negative branch
    assert not guider.is_conditional
    guider.cleanup_models(torch.nn.Identity())
    batch.noise_pred = UNCOND

    expected = ClassifierFreeGuidance(guidance_scale=3.5).forward(COND, UNCOND).pred
    assert torch.equal(guider([batch]).pred, expected)


def test_a_disabled_guidance_cannot_be_split(unconditional_rank):
    guider = guiders.xFuserCFGParallelGuidance(guidance_scale=1.0)

    with pytest.raises(ValueError, match="guidance scale of 1 disables"):
        _step(guider)


def test_without_cfg_parallel_it_is_the_stock_guider(monkeypatch):
    monkeypatch.setattr(guiders, "get_classifier_free_guidance_world_size", lambda: 1)
    guider = guiders.xFuserCFGParallelGuidance(guidance_scale=3.5)
    batches = _step(guider)
    for batch, pred in zip(batches, (COND, UNCOND)):
        batch.noise_pred = pred

    assert len(batches) == 2
    assert torch.equal(guider(batches).pred, ClassifierFreeGuidance(guidance_scale=3.5).forward(COND, UNCOND).pred)
