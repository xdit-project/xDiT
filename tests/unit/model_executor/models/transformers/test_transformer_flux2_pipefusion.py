"""The FLUX.2 PipeFusion backbone must match diffusers' own forward.

The PipeFusion wrapper re-implements Flux2Transformer2DModel.forward so it can split
blocks across stages, which means it calls the blocks directly and has to follow their
signature. With one stage that holds every block, it must reproduce the model exactly.
"""

import pytest
import torch

pytest.importorskip("diffusers.models.transformers.transformer_flux2")

from diffusers.models.transformers.transformer_flux2 import Flux2Transformer2DModel  # noqa: E402

from xfuser.model_executor.models.transformers import transformer_flux2  # noqa: E402


def _tiny_model():
    torch.manual_seed(0)
    return Flux2Transformer2DModel(
        in_channels=8,
        num_layers=1,
        num_single_layers=1,
        attention_head_dim=16,
        num_attention_heads=2,
        joint_attention_dim=12,
        timestep_guidance_channels=32,
        axes_dims_rope=(4, 4, 4, 4),
    ).eval()


def _inputs(image_tokens=6, text_tokens=3):
    generator = torch.Generator().manual_seed(0)
    img_ids = torch.zeros(image_tokens, 4)
    img_ids[:, 1] = torch.arange(image_tokens)
    txt_ids = torch.zeros(text_tokens, 4)
    txt_ids[:, 3] = torch.arange(text_tokens)
    return {
        "hidden_states": torch.randn(1, image_tokens, 8, generator=generator),
        "encoder_hidden_states": torch.randn(1, text_tokens, 12, generator=generator),
        "timestep": torch.tensor([0.4]),
        "guidance": torch.tensor([3.5]),
        "img_ids": img_ids,
        "txt_ids": txt_ids,
    }


def test_single_stage_pipefusion_forward_matches_diffusers(monkeypatch):
    monkeypatch.setattr(transformer_flux2, "is_pipeline_first_stage", lambda: True)
    monkeypatch.setattr(transformer_flux2, "is_pipeline_last_stage", lambda: True)
    model = _tiny_model()
    inputs = _inputs()

    with torch.no_grad():
        expected = model(**inputs, return_dict=False)[0]
        # The wrapper delegates attribute access to the wrapped model, so its forward
        # can run against the bare model; this keeps distributed setup out of the test.
        ((noise_pred, encoder_out),) = transformer_flux2.xFuserFlux2Transformer2DModelWrapper.forward(
            model, **inputs, return_dict=False
        )

    assert encoder_out is None
    torch.testing.assert_close(noise_pred, expected)
