"""Z-Image refreshes its SP length and pad rows for each of its three stages."""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("diffusers.models.transformers.transformer_z_image")

from xfuser.model_executor.models.transformers import transformer_z_image as z_image  # noqa: E402
from xfuser.model_executor.models.transformers import transformers_utils  # noqa: E402


@torch.no_grad()
def test_all_three_stages_publish_their_own_lengths_and_clear_each_blocks_pad_rows(monkeypatch):
    model = z_image.xFuserZImageTransformer2DWrapper(
        in_channels=2,
        dim=16,
        n_layers=2,
        n_refiner_layers=2,
        n_heads=2,
        n_kv_heads=2,
        cap_feat_dim=8,
        axes_dims=[2, 2, 4],
        axes_lens=[128, 16, 16],
    ).eval()
    monkeypatch.setattr(z_image, "get_sequence_parallel_rank", lambda: 4)
    monkeypatch.setattr(z_image, "get_sequence_parallel_world_size", lambda: 5)
    monkeypatch.setattr(z_image, "get_classifier_free_guidance_world_size", lambda: 1)
    monkeypatch.setattr(z_image, "get_classifier_free_guidance_rank", lambda: 0)
    monkeypatch.setattr(
        transformers_utils,
        "get_sp_group",
        lambda: SimpleNamespace(all_gather=lambda x, dim: torch.cat([x] * 5, dim=dim)),
    )
    observed = []
    for stage, layers in (
        ("noise", model.noise_refiner),
        ("context", model.context_refiner),
        ("unified", model.layers),
    ):
        for layer in layers:
            raw = []

            def forward(hidden_states, *args, _raw=raw):
                output = hidden_states + 1
                _raw[:] = [output.clone()]
                return output

            monkeypatch.setattr(layer, "forward", forward)

            def observe(module, args, output, _stage=stage, _raw=raw):
                observed.append(
                    (
                        _stage,
                        model._self_attn_kwargs["valid_kv_len"],
                        model._sp_pad["local_rows"],
                        output.clone(),
                        _raw[0],
                    )
                )

            layer.register_forward_hook(observe)

    model([torch.ones(2, 1, 2, 2)], torch.ones(1), [torch.ones(40, 8)])

    assert len(observed) == 6
    expected = {"noise": (32, 3), "context": (64, 1), "unified": (96, 4)}
    for stage, valid_kv_len, rows, output, raw in observed:
        assert (valid_kv_len, rows) == expected[stage]
        assert not output[:, -rows:].any()
        assert raw[:, -rows:].abs().sum() > 0
        torch.testing.assert_close(output[:, :-rows], raw[:, :-rows])
