"""The FLUX.2 transformer wrapper must build on every diffusers release with FLUX.2.

``Flux2Transformer2DModel`` takes ``guidance_embeds`` only from diffusers 0.37. FLUX.2-dev
supports 0.36, where the model always has a guidance embedder and rejects the argument.
"""

import pytest

transformer_flux2 = pytest.importorskip("xfuser.model_executor.models.transformers.transformer_flux2")

from diffusers.models.transformers.transformer_flux2 import Flux2Transformer2DModel  # noqa: E402

TINY = {
    "in_channels": 8,
    "num_layers": 1,
    "num_single_layers": 1,
    "attention_head_dim": 16,
    "num_attention_heads": 2,
    "joint_attention_dim": 12,
    "timestep_guidance_channels": 32,
    "axes_dims_rope": (4, 4, 4, 4),
}


@pytest.fixture
def diffusers_without_guidance_embeds(monkeypatch):
    """Make ``Flux2Transformer2DModel.__init__`` reject ``guidance_embeds``, as before 0.37."""
    init = Flux2Transformer2DModel.__init__

    def init_before_0_37(self, **kwargs):
        if "guidance_embeds" in kwargs:
            raise TypeError("__init__() got an unexpected keyword argument 'guidance_embeds'")
        init(self, **kwargs)

    monkeypatch.setattr(Flux2Transformer2DModel, "__init__", init_before_0_37)


def _has_guidance_embedder(model):
    return getattr(model.time_guidance_embed, "guidance_embedder", None) is not None


def test_wrapper_builds_guidance_embedder_with_installed_diffusers():
    model = transformer_flux2.xFuserFlux2Transformer2DWrapper(**TINY)

    assert _has_guidance_embedder(model)


def test_wrapper_builds_guidance_embedder_on_diffusers_without_guidance_embeds(
    diffusers_without_guidance_embeds,
):
    model = transformer_flux2.xFuserFlux2Transformer2DWrapper(**TINY)

    assert _has_guidance_embedder(model)


def test_wrapper_does_not_drop_guidance_embeds_false_silently(diffusers_without_guidance_embeds):
    with pytest.raises(TypeError, match="guidance_embeds"):
        transformer_flux2.xFuserFlux2Transformer2DWrapper(**TINY, guidance_embeds=False)
