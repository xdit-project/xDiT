from types import SimpleNamespace

import torch

from xfuser.model_executor.models.transformers import transformer_wan
from xfuser.model_executor.models.transformers.transformer_wan import (
    xFuserWanPipeFusionAttnProcessor,
    xFuserWanPipeFusionTransformerWrapper,
    wan_pipefusion_patch_major,
)


def test_wan_pipefusion_reorders_full_video_tokens_by_height_stripe():
    # Frame-major input: frame 0 rows 0..3, followed by frame 1 rows 0..3.
    tensor = torch.arange(8, dtype=torch.float32).reshape(1, 8, 1, 1)

    result = wan_pipefusion_patch_major(
        tensor,
        grid=(2, 4, 1),
        patch_rows=(2, 2),
    )

    assert result.flatten().tolist() == [0, 1, 4, 5, 2, 3, 6, 7]


def test_wan_pipefusion_attention_populates_patch_major_stale_kv(monkeypatch):
    processor = xFuserWanPipeFusionAttnProcessor()
    query = torch.arange(8, dtype=torch.float32).reshape(1, 8, 1, 1)
    key = query + 10
    value = query + 20
    seen = {}

    processor._get_qkv_projections = lambda *_args: (query, key, value)
    processor._qk_norm_rope_reference = lambda _attn, q, k, v, _rope: (q, k, v)

    def attention(q, k, v, **_kwargs):
        seen["attention_kv"] = (k.clone(), v.clone())
        return q

    processor.attention_function = attention

    class Cache:
        def update_and_get_kv_cache(self, *, new_kv, **kwargs):
            seen["cache_input"] = tuple(tensor.clone() for tensor in new_kv)
            seen["cache_kwargs"] = kwargs
            return new_kv[0] + 100, new_kv[1] + 200

    state = SimpleNamespace(patch_mode=False, pp_patches_height=(2, 2))
    monkeypatch.setattr(transformer_wan, "_HAS_FLYDSL", False)
    monkeypatch.setattr(transformer_wan, "get_runtime_state", lambda: state)
    monkeypatch.setattr(transformer_wan, "get_cache_manager", Cache)
    attn = SimpleNamespace(
        heads=1,
        _xfuser_wan_grid=(2, 4, 1),
        _xfuser_wan_spatial_patch_size=1,
        to_out=[torch.nn.Identity(), torch.nn.Identity()],
    )

    output = processor(attn, torch.zeros(1, 8, 1))

    cached_key, cached_value = seen["cache_input"]
    assert cached_key.flatten().tolist() == [10, 11, 14, 15, 12, 13, 16, 17]
    assert cached_value.flatten().tolist() == [20, 21, 24, 25, 22, 23, 26, 27]
    assert seen["cache_kwargs"] == {
        "layer": attn,
        "slice_dim": 1,
        "layer_type": "attn",
    }
    attention_key, attention_value = seen["attention_kv"]
    assert torch.equal(attention_key, (cached_key + 100).transpose(1, 2))
    assert torch.equal(attention_value, (cached_value + 200).transpose(1, 2))
    assert output.shape == (1, 8, 1)


def test_wan_patch_rotary_embedding_uses_global_height_rows():
    positions = torch.arange(8, dtype=torch.float32)
    transformer = SimpleNamespace(
        config=SimpleNamespace(patch_size=(1, 2, 2)),
        rope=SimpleNamespace(
            t_dim=1,
            h_dim=1,
            w_dim=1,
            freqs_cos=torch.stack((positions, positions + 100, positions + 200), dim=1),
            freqs_sin=torch.stack((positions + 300, positions + 400, positions + 500), dim=1),
        ),
    )

    cos, sin = xFuserWanPipeFusionTransformerWrapper._rotary_emb(
        transformer,
        num_frames=2,
        patch_height=4,
        width=4,
        patch_start_height=4,
    )

    assert cos.shape == (1, 8, 1, 3)
    assert cos[0, 0, 0].tolist() == [0, 102, 200]
    assert cos[0, 4, 0].tolist() == [1, 102, 200]
    assert sin[0, 0, 0].tolist() == [300, 402, 500]


def test_wan_compile_capture_propagates_stage_shapes(monkeypatch):
    transformer = SimpleNamespace(
        config=SimpleNamespace(
            patch_size=(1, 2, 2),
            num_attention_heads=24,
            attention_head_dim=128,
            out_channels=48,
        )
    )
    hidden_states = torch.zeros(1, 48, 2, 4, 4)
    timestep = torch.zeros(1, 8)
    condition = torch.zeros(1, 512, 4096)

    monkeypatch.setattr(transformer_wan, "is_pipeline_first_stage", lambda: True)
    monkeypatch.setattr(transformer_wan, "is_pipeline_last_stage", lambda: False)
    first_output = xFuserWanPipeFusionTransformerWrapper.pipefusion_compile_capture_forward(
        transformer,
        hidden_states,
        timestep,
        condition,
        return_dict=False,
    )[0]

    monkeypatch.setattr(transformer_wan, "is_pipeline_first_stage", lambda: False)
    monkeypatch.setattr(transformer_wan, "is_pipeline_last_stage", lambda: True)
    last_output = xFuserWanPipeFusionTransformerWrapper.pipefusion_compile_capture_forward(
        transformer,
        hidden_states,
        timestep,
        condition,
        pipeline_hidden_states=torch.zeros(2, 8, 3072),
        return_dict=False,
    )[0]

    assert first_output.shape == (1, 8, 3072)
    assert last_output.shape == (2, 48, 2, 4, 4)
