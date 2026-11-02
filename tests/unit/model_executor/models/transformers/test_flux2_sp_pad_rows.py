"""FLUX.2 shards the prompt only where it splits evenly, and refuses padding it cannot keep out of attention."""

from types import SimpleNamespace

import pytest
import torch

from xfuser.model_executor.models.transformers import transformer_flux2 as flux2

SP_DEGREE = 3


def _model():
    return flux2.xFuserFlux2Transformer2DWrapper(
        patch_size=1,
        in_channels=8,
        num_layers=1,
        num_single_layers=1,
        attention_head_dim=8,
        num_attention_heads=4,
        joint_attention_dim=16,
        timestep_guidance_channels=16,
        axes_dims_rope=(2, 2, 2, 2),
        guidance_embeds=False,
    )


def _forward(monkeypatch, image_tokens, prompt_tokens, *, fp8_comms=None, pipeline_patches=1, degree=SP_DEGREE):
    """Run the wrapper as the last of `degree` ranks; return what its backbone was handed."""
    seen = {}

    def backbone(self, hidden_states, encoder_hidden_states, **kwargs):
        seen.update(image=hidden_states, prompt=encoder_hidden_states, txt_ids=kwargs["txt_ids"])
        seen["kwargs"] = dict(model._self_attn_kwargs)
        return (hidden_states,)

    model = _model()
    monkeypatch.setattr(flux2.Flux2Transformer2DModel, "forward", backbone)
    monkeypatch.setattr(flux2, "get_sequence_parallel_world_size", lambda: degree)
    monkeypatch.setattr(flux2, "get_sequence_parallel_rank", lambda: degree - 1)
    monkeypatch.setattr(flux2, "get_classifier_free_guidance_world_size", lambda: 1)
    monkeypatch.setattr(flux2, "get_classifier_free_guidance_rank", lambda: 0)
    monkeypatch.setattr(flux2, "get_sp_group", lambda: SimpleNamespace(all_gather=lambda x, dim: x))
    monkeypatch.setattr(flux2, "get_cfg_group", lambda: SimpleNamespace(all_gather=lambda x, dim: x))
    monkeypatch.setattr(
        flux2, "get_runtime_state", lambda: SimpleNamespace(fp8_comms=fp8_comms, num_pipeline_patch=pipeline_patches)
    )
    model(
        torch.ones(1, image_tokens, 8),
        encoder_hidden_states=torch.ones(1, prompt_tokens, 16),
        timestep=torch.tensor([0.5]),
        img_ids=torch.zeros(1, image_tokens, 4),
        txt_ids=torch.zeros(1, prompt_tokens, 4),
    )
    return seen


def test_a_prompt_that_splits_evenly_is_sharded_and_counted_with_the_image(monkeypatch):
    # 8 image tokens pad to 9; the 9 prompt tokens split in threes.
    seen = _forward(monkeypatch, image_tokens=8, prompt_tokens=9)
    assert seen["prompt"].shape[1] == 3 and seen["txt_ids"].shape[1] == 3
    assert seen["kwargs"] == {"valid_kv_len": 8 + 9, "replicated_prompt_tokens": 0}


def test_a_prompt_that_does_not_split_evenly_stays_whole_on_every_rank_and_out_of_the_declared_length(monkeypatch):
    seen = _forward(monkeypatch, image_tokens=8, prompt_tokens=8)
    assert seen["prompt"].shape[1] == 8 and seen["txt_ids"].shape[1] == 8
    assert seen["kwargs"] == {"valid_kv_len": 8, "replicated_prompt_tokens": 8}


def test_without_padding_nothing_is_declared(monkeypatch):
    seen = _forward(monkeypatch, image_tokens=9, prompt_tokens=8)
    assert seen["kwargs"] == {"valid_kv_len": None, "replicated_prompt_tokens": 8}


def test_a_replicated_prompt_is_refused_when_fp8_comms_are_on(monkeypatch):
    with pytest.raises(ValueError, match=r"--use_fp8_comms needs the prompt length \(8 tokens\).*\(3\)"):
        _forward(monkeypatch, image_tokens=9, prompt_tokens=8, fp8_comms=object())


def test_a_sharded_prompt_runs_with_fp8_comms_on(monkeypatch):
    _forward(monkeypatch, image_tokens=9, prompt_tokens=9, fp8_comms=object())


def test_padding_under_pipefusion_is_refused(monkeypatch):
    with pytest.raises(NotImplementedError, match="under PipeFusion"):
        _forward(monkeypatch, image_tokens=8, prompt_tokens=9, pipeline_patches=2)


def test_padding_across_image_shards_is_rejected_before_attention(monkeypatch):
    with pytest.raises(ValueError, match="padding must fit in the last image shard"):
        _forward(monkeypatch, image_tokens=1, prompt_tokens=4, degree=4)
