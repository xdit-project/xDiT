"""LTX-2 clears video SP padding while preserving its replicated audio stream."""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("diffusers.models.transformers.transformer_ltx2")

from xfuser.core.attention.spec import AttentionBackendType  # noqa: E402
from xfuser.model_executor.models.transformers import transformer_ltx2 as ltx2  # noqa: E402
from xfuser.model_executor.models.transformers import transformers_utils  # noqa: E402


def _tiny_model(perturbed_attn=False):
    return ltx2.xFuserLTX2VideoTransformer3DWrapper(
        in_channels=4,
        out_channels=4,
        num_attention_heads=2,
        attention_head_dim=8,
        cross_attention_dim=16,
        audio_in_channels=4,
        audio_out_channels=4,
        audio_num_attention_heads=2,
        audio_attention_head_dim=8,
        audio_cross_attention_dim=16,
        num_layers=2,
        caption_channels=16,
        perturbed_attn=perturbed_attn,
    ).eval()


@pytest.mark.parametrize("perturbed_attn", [False, True])
@torch.no_grad()
def test_each_block_clears_only_video_padding_and_resets_on_a_divisible_forward(monkeypatch, perturbed_attn):
    model = _tiny_model(perturbed_attn)
    monkeypatch.setattr(ltx2, "get_sequence_parallel_rank", lambda: 1)
    monkeypatch.setattr(ltx2, "get_sequence_parallel_world_size", lambda: 2)
    monkeypatch.setattr(
        transformers_utils,
        "get_sp_group",
        lambda: SimpleNamespace(all_gather=lambda x, dim: torch.cat([x, x], dim=dim)),
    )
    observed = []
    for block in model.transformer_blocks:
        raw = []

        def forward(*args, _raw=raw, **kwargs):
            # Model the upstream block's biased outputs, including the pad rows.
            output = kwargs["hidden_states"] + 1, kwargs["audio_hidden_states"] + 2
            _raw[:] = [x.clone() for x in output]
            return output

        monkeypatch.setattr(block, "forward", forward)

        def observe(module, args, output, _raw=raw):
            observed.append((model._self_attn_kwargs["valid_kv_len"], [x.clone() for x in output], list(_raw)))

        block.register_forward_hook(observe)

    for length in (3, 4):
        observed.clear()
        model(
            hidden_states=torch.ones(1, length, 4),
            audio_hidden_states=torch.ones(1, 2, 4),
            encoder_hidden_states=torch.ones(1, 1, 16),
            audio_encoder_hidden_states=torch.ones(1, 1, 16),
            timestep=torch.ones(1),
            num_frames=1,
            height=1,
            width=length,
            audio_num_frames=2,
        )
        assert len(observed) == 2
        for valid_kv_len, (video, audio), (raw_video, raw_audio) in observed:
            if length == 3:
                assert valid_kv_len == 3
                assert not video[:, -1:].any()
                assert raw_video[:, -1:].abs().sum() > 0
                torch.testing.assert_close(video[:, :-1], raw_video[:, :-1])
            else:
                assert valid_kv_len is None
                torch.testing.assert_close(video, raw_video)
            torch.testing.assert_close(audio, raw_audio)


@pytest.mark.parametrize(
    "processor_cls", [ltx2.xFuserLTX2AudioVideoAttnProcessor, ltx2.xFuserLTX2PerturbedAttnProcessor]
)
@pytest.mark.parametrize("rope_type", ["interleaved", "split"])
@torch.no_grad()
def test_gathered_video_attention_excludes_pad_keys_and_uses_the_next_real_length(
    monkeypatch, processor_cls, rope_type
):
    generator = torch.Generator().manual_seed(3)
    queries = torch.randn(1, 2, 16, generator=generator)
    full_video = torch.randn(1, 4, 16, generator=generator)
    full_video[:, -1] = 50
    shape = (1, 4, 16) if rope_type == "interleaved" else (1, 2, 4, 4)
    cos, sin = torch.ones(shape), torch.zeros(shape)

    def gather(x, dim):
        if dim == 1:
            return full_video
        return cos if x.any() else sin

    monkeypatch.setattr(ltx2, "get_sp_group", lambda: SimpleNamespace(all_gather=gather))
    monkeypatch.setattr(
        "xfuser.model_executor.layers.usp.get_runtime_state",
        lambda: SimpleNamespace(attention_backend=AttentionBackendType.SDPA_MATH),
    )
    kwargs = {"valid_kv_len": 3}
    processor = processor_cls(use_parallel_attention=False, gather_kv=True, attention_kwargs=kwargs)
    unpadded = processor_cls(use_parallel_attention=False)
    attn = SimpleNamespace(
        heads=2,
        rope_type=rope_type,
        to_gate_logits=None,
        to_q=torch.nn.Identity(),
        to_k=torch.nn.Identity(),
        to_v=torch.nn.Identity(),
        norm_q=torch.nn.Identity(),
        norm_k=torch.nn.Identity(),
        to_out=[torch.nn.Identity(), torch.nn.Identity()],
    )
    for valid_len in (3, 4):
        kwargs["valid_kv_len"] = valid_len if valid_len < 4 else None
        out = processor(
            attn,
            queries,
            encoder_hidden_states=full_video[:, 2:],
            query_rotary_emb=(cos[..., :2, :], sin[..., :2, :]),
            key_rotary_emb=(cos[..., 2:, :], sin[..., 2:, :]),
        )
        expected = unpadded(
            attn,
            queries,
            encoder_hidden_states=full_video[:, :valid_len],
            query_rotary_emb=(cos[..., :2, :], sin[..., :2, :]),
            key_rotary_emb=(cos[..., :valid_len, :], sin[..., :valid_len, :]),
        )
        torch.testing.assert_close(out, expected)
