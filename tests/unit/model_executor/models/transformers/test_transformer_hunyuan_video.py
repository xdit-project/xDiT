"""HunyuanVideo publishes its real video length to USP and keeps the sequence-parallel pad rows at zero.

When the video token count does not divide the sequence-parallel degree, the wrapper zero-pads the
video tokens. Attention drops the pad keys, but every block still writes the pad rows, and those rows
enter statistics taken over a whole local tensor (a step cache's residual difference, quantization
scales shared across query rows). A forward hook on every double and single
block zeroes them; the prompt rows and the real video rows must not change.
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from diffusers.models.transformers.transformer_hunyuan_video import HunyuanVideoTransformer3DModel

from xfuser.model_executor.models.transformers import transformer_hunyuan_video as hunyuan
from xfuser.model_executor.models.transformers import transformers_utils

SP_DEGREE = 2
_CONFIG = dict(
    in_channels=4,
    out_channels=4,
    num_attention_heads=6,
    attention_head_dim=8,
    num_layers=2,
    num_single_layers=2,
    num_refiner_layers=1,
    patch_size=1,
    patch_size_t=1,
    text_embed_dim=16,
    pooled_projection_dim=8,
    rope_axes_dim=(2, 2, 4),
)
_TEXT_TOKENS = 5  # of 7, so the prompt never divides two ranks


def _inputs(frames, text_tokens=_TEXT_TOKENS):
    generator = torch.Generator().manual_seed(1)
    mask = torch.ones(1, 7, dtype=torch.int64)
    mask[:, text_tokens:] = 0
    return dict(
        hidden_states=torch.randn(1, 4, frames, 3, 3, generator=generator),
        timestep=torch.tensor([700]),
        encoder_hidden_states=torch.randn(1, 7, 16, generator=generator),
        encoder_attention_mask=mask,
        pooled_projections=torch.randn(1, 8, generator=generator),
        guidance=torch.tensor([6000.0]),
        return_dict=False,
    )


def _build(monkeypatch, usp):
    """The wrapper as the last of SP_DEGREE ranks, with the collectives and USP stubbed at the boundary."""
    runtime = SimpleNamespace(
        increment_step_counter=lambda: None,
        split_text_embed_in_sp=False,
        max_condition_sequence_length=_TEXT_TOKENS,
    )
    monkeypatch.setattr(hunyuan, "USP", usp)
    monkeypatch.setattr(hunyuan, "get_runtime_state", lambda: runtime)
    monkeypatch.setattr(hunyuan, "get_sequence_parallel_world_size", lambda: SP_DEGREE)
    monkeypatch.setattr(hunyuan, "get_sequence_parallel_rank", lambda: SP_DEGREE - 1)
    monkeypatch.setattr(hunyuan, "get_pipeline_parallel_world_size", lambda: 1)
    monkeypatch.setattr(hunyuan, "get_classifier_free_guidance_world_size", lambda: 1)
    monkeypatch.setattr(hunyuan, "get_classifier_free_guidance_rank", lambda: 0)
    monkeypatch.setattr(hunyuan, "get_cfg_group", lambda: SimpleNamespace(all_gather=lambda x, dim: x))
    # The other ranks' shards gather as zeros, so the sample holds only this rank's real rows.
    monkeypatch.setattr(
        transformers_utils,
        "get_sp_group",
        lambda: SimpleNamespace(
            all_gather=lambda x, dim: torch.cat([torch.zeros_like(x)] * (SP_DEGREE - 1) + [x], dim=dim)
        ),
    )
    torch.manual_seed(0)
    model = hunyuan.xFuserHunyuanVideoTransformer3DWrapper.from_config(
        HunyuanVideoTransformer3DModel(**_CONFIG).config
    ).eval()
    return model, runtime


@pytest.mark.parametrize(
    ("frames", "text_tokens", "valid_kv_len", "split_text"),
    [(3, 5, 27, False), (3, 6, 27, False), (2, 5, None, False), (2, 6, None, True)],
    ids=["padded", "padded-divisible-prompt", "divisible-replicated-prompt", "divisible"],
)
def test_padding_is_published_to_the_usp_calls_through_the_shared_dict(
    monkeypatch, frames, text_tokens, valid_kv_len, split_text
):
    """The wrapper pads the video tokens, publishes their real count and hands USP the shared dict.

    A padded video keeps the prompt whole and joins it behind the video keys, whatever the prompt length.
    """
    calls = []

    def usp_recorder(query, key, value, **kwargs):
        shared = kwargs.get("attention_kwargs")
        calls.append((shared, dict(shared or {}), kwargs.get("joint_query") is not None))
        joint_query = kwargs.get("joint_query")
        return query if joint_query is None else torch.cat([query, joint_query], dim=2)

    wrapper, runtime = _build(monkeypatch, usp_recorder)
    with torch.no_grad():
        wrapper(**_inputs(frames, text_tokens))

    assert len(calls) == _CONFIG["num_layers"] + _CONFIG["num_single_layers"]
    assert runtime.split_text_embed_in_sp is split_text
    for shared, published, joint in calls:
        assert joint is not split_text
        if not split_text:
            assert shared is wrapper._self_attn_kwargs
            assert published == {"valid_kv_len": valid_kv_len}


def _local_usp(query, key, value, *, joint_query=None, joint_key=None, joint_value=None, attention_kwargs=None, **_):
    """This rank's share of USP: attention to its own keys minus the declared pad, then the joint prompt."""
    valid_kv_len = (attention_kwargs or {}).get("valid_kv_len")
    keep = key.shape[2] - (0 if valid_kv_len is None else SP_DEGREE * key.shape[2] - valid_kv_len)
    key, value = key[:, :, :keep], value[:, :, :keep]
    if joint_query is not None:
        query = torch.cat([query, joint_query], dim=2)
        key = torch.cat([key, joint_key], dim=2)
        value = torch.cat([value, joint_value], dim=2)
    return F.scaled_dot_product_attention(query, key, value)


@pytest.fixture
def last_rank(monkeypatch):
    model, _ = _build(monkeypatch, _local_usp)
    return model


def _blocks(model):
    return [*model.transformer_blocks, *model.single_transformer_blocks]


def _forward(model, frames):
    """Run a forward; return the sample and every block's (video, text) output, cloned as the block left it."""
    outputs = []
    handles = [
        block.register_forward_hook(lambda module, args, out: outputs.append(tuple(t.clone() for t in out)))
        for block in _blocks(model)
    ]
    try:
        with torch.no_grad():
            (sample,) = model(**_inputs(frames))
    finally:
        for handle in handles:
            handle.remove()
    return sample, outputs


def test_every_double_and_single_block_carries_one_pad_row_hook(tmp_path):
    reference = HunyuanVideoTransformer3DModel(**_CONFIG)
    reference.save_pretrained(tmp_path)
    for model in (
        hunyuan.xFuserHunyuanVideoTransformer3DWrapper.from_config(reference.config),
        hunyuan.xFuserHunyuanVideoTransformer3DWrapper.from_pretrained(tmp_path),
    ):
        blocks = _blocks(model)
        assert len(blocks) == _CONFIG["num_layers"] + _CONFIG["num_single_layers"]
        for block in blocks:
            assert len(block._forward_hooks) == 1


def test_a_padded_forward_zeroes_only_the_video_pad_rows(last_rank):
    # 27 video tokens pad to 28: the last rank holds 13 real rows and 1 pad row; the prompt stays whole.
    sample, hooked = _forward(last_rank, frames=3)
    assert last_rank._sp_pad["local_rows"] == 1

    for block in _blocks(last_rank):
        block._forward_hooks.clear()
    reference_sample, unhooked = _forward(last_rank, frames=3)

    assert len(hooked) == len(unhooked) == 4
    for index, ((video, text), (reference_video, reference_text)) in enumerate(zip(hooked, unhooked)):
        assert video.shape[1] == 14, f"block {index}: unexpected local video length"
        assert not video[:, -1:].any(), f"block {index}: pad row kept"
        assert reference_video[:, -1:].any(), f"block {index}: the reference pad row is already zero"
        assert torch.equal(video[:, :-1], reference_video[:, :-1]), f"block {index}: real video rows changed"
        assert torch.equal(text, reference_text), f"block {index}: prompt rows changed"
    assert torch.equal(sample, reference_sample)


def test_an_unpadded_forward_declares_no_pad_rows(last_rank):
    last_rank._sp_pad["local_rows"] = 3  # left over from an earlier padded request
    _, hooked = _forward(last_rank, frames=2)
    assert last_rank._sp_pad["local_rows"] == 0
    assert all(video[:, -1:].any() for video, _ in hooked)
