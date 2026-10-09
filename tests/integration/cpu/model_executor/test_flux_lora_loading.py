import pytest


@pytest.mark.parametrize("scale", [0.0, 0.5])
def test_flux_runner_fuses_local_lora_into_native_weights(tmp_path, monkeypatch, scale):
    pytest.importorskip("peft")
    import torch
    from diffusers import FluxPipeline, FluxTransformer2DModel, FlowMatchEulerDiscreteScheduler
    from safetensors.torch import save_file
    from types import SimpleNamespace

    from xfuser.config import xFuserArgs
    from xfuser.model_executor.models.runner_models.flux import xFuserFluxModel

    transformer = FluxTransformer2DModel(
        in_channels=16,
        num_layers=1,
        num_single_layers=1,
        attention_head_dim=8,
        num_attention_heads=2,
        joint_attention_dim=16,
        pooled_projection_dim=8,
        axes_dims_rope=(2, 2, 4),
    )
    checkpoint = tmp_path / "base"
    transformer.save_pretrained(checkpoint / "transformer")

    def load_pipeline(pretrained_model_name_or_path, transformer, **kwargs):
        # Replace sibling-component filesystem loading; retain the actual wrapped transformer and PEFT API.
        return FluxPipeline(
            transformer=transformer,
            scheduler=FlowMatchEulerDiscreteScheduler(),
            vae=None,
            text_encoder=None,
            text_encoder_2=None,
            tokenizer=None,
            tokenizer_2=None,
        )

    monkeypatch.setattr(FluxPipeline, "from_pretrained", load_pipeline)
    original = transformer.transformer_blocks[0].attn.to_q.weight.detach().to(torch.bfloat16)
    adapter = tmp_path / "adapter.safetensors"
    a = torch.full((4, original.shape[1]), 0.02)
    b = torch.full((original.shape[0], 4), 0.05)
    save_file(
        {
            "transformer.transformer_blocks.0.attn.to_q.lora_A.weight": a,
            "transformer.transformer_blocks.0.attn.to_q.lora_B.weight": b,
        },
        str(adapter),
    )

    model = xFuserFluxModel(xFuserArgs(lora_path=str(adapter), lora_scale=scale))
    model.settings.model_name = str(checkpoint)
    # Only replace the distributed process boundary; all loading and PEFT work is real CPU code.
    monkeypatch.setattr(
        "xfuser.model_executor.models.runner_models.loading.meta_load.get_world_group",
        lambda: SimpleNamespace(world_size=1),
    )
    model.loader.preflight(world_size=1)
    loaded = model._load_model()
    projection = loaded.transformer.transformer_blocks[0].attn.to_q
    expected = original + scale * (b.to(torch.bfloat16) @ a.to(torch.bfloat16))
    torch.testing.assert_close(projection.weight, expected)
    sample = torch.ones(1, original.shape[1], dtype=torch.bfloat16)
    torch.testing.assert_close(projection(sample), torch.nn.functional.linear(sample, expected, projection.bias))
    assert isinstance(projection, torch.nn.Linear)
    assert not any("lora_" in name for name, _ in loaded.transformer.named_parameters())
