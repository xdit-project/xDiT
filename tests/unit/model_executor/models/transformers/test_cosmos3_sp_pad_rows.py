"""Cosmos3 clears SP padding on generation rows and preserves understanding rows."""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("diffusers.models.transformers.transformer_cosmos3")

from xfuser.model_executor.models.transformers import transformer_cosmos3 as cosmos3  # noqa: E402
from xfuser.model_executor.models.transformers import transformers_utils  # noqa: E402


@torch.no_grad()
def test_decoder_outputs_clear_only_gen_padding_and_refresh_the_next_forward(monkeypatch):
    model = cosmos3.get_cosmos3_transformer_wrapper_class()(
        head_dim=6,
        hidden_act="relu2",
        hidden_size=12,
        intermediate_size=24,
        latent_channel=2,
        latent_patch_size=1,
        num_attention_heads=2,
        num_hidden_layers=2,
        num_key_value_heads=1,
        patch_latent_dim=2,
        qk_norm_for_text=False,
        rope_axes_dim=[1, 1, 1],
        vocab_size=32,
    ).eval()
    model._install_xfuser_processors()
    monkeypatch.setattr(cosmos3, "get_runtime_state", lambda: SimpleNamespace(increment_step_counter=lambda: None))
    monkeypatch.setattr(cosmos3, "get_sequence_parallel_rank", lambda: 1)
    monkeypatch.setattr(cosmos3, "get_sequence_parallel_world_size", lambda: 2)
    monkeypatch.setattr(
        transformers_utils,
        "get_sp_group",
        lambda: SimpleNamespace(all_gather=lambda x, dim: torch.cat([x, x], dim=dim)),
    )
    observed = []
    for layer in model.layers:
        raw = []

        def forward(und_seq, gen_seq, rotary_emb, _raw=raw):
            output = und_seq + 1, gen_seq + 2
            _raw[:] = [x.clone() for x in output]
            return output

        monkeypatch.setattr(layer, "forward", forward)

        def observe(module, args, output, _raw=raw):
            observed.append((model._self_attn_kwargs["valid_kv_len"], [x.clone() for x in output], list(_raw)))

        layer.register_forward_hook(observe)

    for gen_len in (3, 4):
        observed.clear()
        indexes = torch.arange(2, 2 + gen_len)
        model(
            input_ids=torch.tensor([1, 2]),
            text_indexes=torch.tensor([0, 1]),
            position_ids=torch.zeros(3, 2 + gen_len, dtype=torch.long),
            und_len=2,
            sequence_length=2 + gen_len,
            vision_tokens=[torch.ones(1, 2, 1, 1, gen_len)],
            vision_token_shapes=[(1, 1, gen_len)],
            vision_sequence_indexes=indexes,
            vision_mse_loss_indexes=indexes,
            vision_timesteps=torch.ones(gen_len),
            vision_noisy_frame_indexes=[torch.tensor([0])],
            return_dict=False,
        )
        assert len(observed) == 2
        for valid_kv_len, (und, gen), (raw_und, raw_gen) in observed:
            torch.testing.assert_close(und, raw_und)
            if gen_len == 3:
                assert valid_kv_len == 3
                assert not gen[-1:].any()
                assert raw_gen[-1:].abs().sum() > 0
                torch.testing.assert_close(gen[:-1], raw_gen[:-1])
            else:
                assert valid_kv_len is None
                torch.testing.assert_close(gen, raw_gen)
