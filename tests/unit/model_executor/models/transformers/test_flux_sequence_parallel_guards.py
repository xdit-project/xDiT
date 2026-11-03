"""FLUX refuses unsupported combinations of padding, prompt replication, and attention."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

pytest.importorskip("diffusers.models.transformers.transformer_flux")

from xfuser.core.distributed.fp8_comms import Fp8CommsState  # noqa: E402
from xfuser.core.attention.spec import AttentionBackendType  # noqa: E402
from xfuser.model_executor.models.transformers import transformer_flux as flux  # noqa: E402


@pytest.fixture
def forward_context(monkeypatch):
    """Stub distributed state and the third-party transformer at the wrapper boundary."""

    def make(*, ulysses, ring, fp8_comms, attention_backend=AttentionBackendType.SDPA):
        runtime = SimpleNamespace(fp8_comms=fp8_comms, split_text_embed_in_sp=True, attention_backend=attention_backend)
        monkeypatch.setattr(flux, "get_runtime_state", lambda: runtime)
        monkeypatch.setattr(flux, "get_sequence_parallel_world_size", lambda: ulysses * ring)
        monkeypatch.setattr(flux, "get_sequence_parallel_rank", lambda: 1)
        monkeypatch.setattr(flux, "get_ulysses_parallel_world_size", lambda: ulysses)
        monkeypatch.setattr(flux, "get_ring_parallel_world_size", lambda: ring)
        monkeypatch.setattr(flux, "get_classifier_free_guidance_world_size", lambda: 1)
        monkeypatch.setattr(flux, "get_classifier_free_guidance_rank", lambda: 0)
        sp_gather = Mock(side_effect=lambda sample, dim: sample)
        monkeypatch.setattr(flux, "get_sp_group", lambda: SimpleNamespace(all_gather=sp_gather))
        monkeypatch.setattr(flux, "get_cfg_group", lambda: SimpleNamespace(all_gather=lambda sample, dim: sample))
        transformer_forward = Mock(side_effect=lambda hidden_states, encoder_hidden_states, **kwargs: (hidden_states,))
        monkeypatch.setattr(flux.FluxTransformer2DModel, "forward", transformer_forward)
        model = flux.xFuserFlux1Transformer2DWrapper(
            in_channels=8,
            num_layers=0,
            num_single_layers=0,
            attention_head_dim=16,
            num_attention_heads=2,
            joint_attention_dim=12,
            pooled_projection_dim=8,
            axes_dims_rope=(4, 6, 6),
        )
        return model, runtime, transformer_forward, sp_gather

    return make


def _inputs(text_tokens, image_tokens=16):
    return {
        "hidden_states": torch.arange(image_tokens * 8, dtype=torch.float32).reshape(1, image_tokens, 8),
        "encoder_hidden_states": torch.arange(text_tokens * 12, dtype=torch.float32).reshape(1, text_tokens, 12),
        "img_ids": torch.arange(image_tokens * 3, dtype=torch.float32).reshape(image_tokens, 3),
        "txt_ids": torch.arange(text_tokens * 3, dtype=torch.float32).reshape(text_tokens, 3),
        "return_dict": False,
    }


@pytest.mark.parametrize("fixed_scale", [None, 0.5], ids=["automatic-calibration", "fixed-scale"])
@pytest.mark.parametrize(
    "ulysses, ring, text_tokens",
    [pytest.param(2, 1, 7, id="ulysses"), pytest.param(2, 2, 6, id="full-hybrid-degree")],
)
def test_fp8_comms_rejects_replicated_prompt_before_transformer_or_collectives(
    forward_context, fixed_scale, ulysses, ring, text_tokens
):
    model, runtime, transformer_forward, sp_gather = forward_context(
        ulysses=ulysses, ring=ring, fp8_comms=Fp8CommsState(fixed_scale=fixed_scale)
    )

    with pytest.raises(ValueError, match="--use_fp8_comms requires the prompt token count") as caught:
        model(**_inputs(text_tokens))

    message = str(caught.value)
    assert f"({text_tokens})" in message
    assert f"sequence parallel degree ({ulysses * ring};" in message
    assert f"ulysses_degree={ulysses}, ring_degree={ring}" in message
    assert "--max_sequence_length" in message
    assert "parallel degrees" in message
    assert "disable --use_fp8_comms" in message
    transformer_forward.assert_not_called()
    sp_gather.assert_not_called()
    assert runtime.split_text_embed_in_sp is True


@pytest.mark.parametrize("fixed_scale", [None, 0.5], ids=["automatic-calibration", "fixed-scale"])
def test_fp8_comms_accepts_divisible_prompt_and_forwards_its_local_shard(forward_context, fixed_scale):
    model, runtime, transformer_forward, _ = forward_context(
        ulysses=2, ring=2, fp8_comms=Fp8CommsState(fixed_scale=fixed_scale)
    )
    inputs = _inputs(8)

    model(**inputs)

    forwarded_prompt = transformer_forward.call_args.args[1]
    forwarded_kwargs = transformer_forward.call_args.kwargs
    torch.testing.assert_close(forwarded_prompt, inputs["encoder_hidden_states"][:, 2:4])
    torch.testing.assert_close(forwarded_kwargs["txt_ids"], inputs["txt_ids"][2:4])
    assert forwarded_kwargs["joint_attention_kwargs"]["sp_text_tokens"] == 2
    assert runtime.split_text_embed_in_sp is True


@pytest.mark.parametrize("backend", [AttentionBackendType.SDPA, AttentionBackendType.AITER_VSA])
def test_unpadded_replicated_prompt_reaches_transformer_in_full(forward_context, backend):
    model, runtime, transformer_forward, _ = forward_context(
        ulysses=2, ring=2, fp8_comms=None, attention_backend=backend
    )
    inputs = _inputs(6)

    model(**inputs)

    forwarded_prompt = transformer_forward.call_args.args[1]
    forwarded_kwargs = transformer_forward.call_args.kwargs
    torch.testing.assert_close(forwarded_prompt, inputs["encoder_hidden_states"])
    torch.testing.assert_close(forwarded_kwargs["txt_ids"], inputs["txt_ids"])
    assert forwarded_kwargs["joint_attention_kwargs"]["sp_text_tokens"] == 6
    assert runtime.split_text_embed_in_sp is False


def test_vsa_rejects_padded_images_with_replicated_prompt_before_attention(forward_context):
    model, runtime, transformer_forward, sp_gather = forward_context(
        ulysses=2, ring=1, fp8_comms=None, attention_backend=AttentionBackendType.AITER_VSA
    )

    with pytest.raises(NotImplementedError, match="cannot pad 15 image tokens for 2 sequence parallel ranks") as caught:
        model(**_inputs(7, image_tokens=15))

    assert AttentionBackendType.AITER_VSA.value in str(caught.value)
    assert "while the 7-token prompt is replicated" in str(caught.value)
    assert "image token count or the prompt length divisible by the sequence parallel degree" in str(caught.value)
    transformer_forward.assert_not_called()
    sp_gather.assert_not_called()
    assert runtime.split_text_embed_in_sp is True


@pytest.mark.parametrize(
    "backend, text_tokens, prompt_slice",
    [
        pytest.param(AttentionBackendType.AITER_VSA, 8, slice(4, 8), id="vsa-sharded-prompt"),
        pytest.param(AttentionBackendType.SDPA, 7, slice(None), id="dense-replicated-prompt"),
    ],
)
def test_supported_backends_forward_padded_images_and_ids_with_the_prompt(
    forward_context, backend, text_tokens, prompt_slice
):
    model, _, transformer_forward, _ = forward_context(ulysses=2, ring=1, fp8_comms=None, attention_backend=backend)
    inputs = _inputs(text_tokens, image_tokens=15)

    model(**inputs)

    forwarded_image, forwarded_prompt = transformer_forward.call_args.args
    forwarded_kwargs = transformer_forward.call_args.kwargs
    expected_image = torch.cat([inputs["hidden_states"][:, 8:], torch.zeros(1, 1, 8)], dim=1)
    expected_ids = torch.cat([inputs["img_ids"][8:], torch.zeros(1, 3)], dim=0)
    torch.testing.assert_close(forwarded_image, expected_image)
    torch.testing.assert_close(forwarded_kwargs["img_ids"], expected_ids)
    torch.testing.assert_close(forwarded_prompt, inputs["encoder_hidden_states"][:, prompt_slice])
    torch.testing.assert_close(forwarded_kwargs["txt_ids"], inputs["txt_ids"][prompt_slice])
    assert forwarded_kwargs["joint_attention_kwargs"]["sp_padding"] == 1


@pytest.mark.parametrize(
    "ulysses, ring, image_tokens",
    [pytest.param(2, 1, 1, id="ulysses"), pytest.param(1, 4, 9, id="ring"), pytest.param(2, 2, 3, id="hybrid")],
)
def test_image_whose_last_rank_would_hold_only_padding_is_rejected(forward_context, ulysses, ring, image_tokens):
    model, _, transformer_forward, sp_gather = forward_context(ulysses=ulysses, ring=ring, fp8_comms=None)

    with pytest.raises(ValueError, match=f"{image_tokens} image tokens are too few to shard across {ulysses * ring}"):
        model(**_inputs(8, image_tokens=image_tokens))

    transformer_forward.assert_not_called()
    sp_gather.assert_not_called()


def test_pipefusion_rejects_padded_image_tokens(monkeypatch):
    monkeypatch.setattr(flux, "get_runtime_state", lambda: SimpleNamespace(num_pipeline_patch=2))

    with pytest.raises(NotImplementedError, match="PipeFusion does not support image tokens padded"):
        flux.xFuserFluxAttnProcessor()(Mock(), torch.zeros(1, 4, 8), sp_padding=1)
