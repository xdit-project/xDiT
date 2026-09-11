from types import SimpleNamespace

import inspect
import numpy as np
import pytest
import torch
from PIL import Image


def _tiny_config():
    return {
        "num_attention_heads": 2,
        "attention_head_dim": 16,
        "hidden_size": 24,
        "num_layers": 2,
        "num_refiner_layers": 2,
        "ffn_dim": 32,
        "in_channels": 4,
        "audio_in_channels": 6,
        "patch_size": (1, 2, 2),
        "text_dim": 8,
        "freq_dim": 8,
        "time_embed_hidden_dim": 24,
        "time_embed_dim": 16,
        "rope_freq_dim": 2,
    }


def _tiny_inputs(device):
    text_tokens = 4
    audio_tokens = 12
    video_tokens = 48
    sequence_length = text_tokens + audio_tokens + video_tokens
    text_indices = torch.arange(text_tokens, device=device)
    audio_indices = torch.arange(
        text_tokens,
        text_tokens + audio_tokens,
        device=device,
    )
    video_indices = torch.arange(
        text_tokens + audio_tokens,
        sequence_length,
        device=device,
    )

    token_tags = torch.empty(sequence_length, dtype=torch.long, device=device)
    token_tags[text_indices] = 1
    token_tags[audio_indices] = 2
    token_tags[video_indices] = 0

    timestep_indices = torch.zeros(
        sequence_length,
        dtype=torch.long,
        device=device,
    )
    timestep_indices[audio_indices] = 1

    position_ids = torch.zeros(
        sequence_length,
        3,
        dtype=torch.float32,
        device=device,
    )
    position_ids[:, 0] = torch.arange(
        sequence_length,
        dtype=torch.float32,
        device=device,
    )

    generator = torch.Generator(device="cpu").manual_seed(0)
    return {
        "hidden_states": torch.randn(
            1,
            video_tokens,
            16,
            generator=generator,
            device=device,
        ),
        "audio_hidden_states": torch.randn(
            1,
            audio_tokens,
            6,
            generator=generator,
            device=device,
        ),
        "encoder_hidden_states": torch.randn(
            1,
            text_tokens,
            8,
            generator=generator,
            device=device,
        ),
        "timestep": torch.tensor([0.7, 0.3], device=device),
        "timestep_indices": timestep_indices,
        "token_tags": token_tags,
        "position_ids": position_ids,
        "video_indices": video_indices,
        "audio_indices": audio_indices,
        "text_indices": text_indices,
    }


def _patch_minimax_runtime_state(monkeypatch, *, track_steps=False):
    from xfuser.core.distributed.attention_backend import AttentionBackendType
    from xfuser.model_executor.models.runner_models import minimax_h3 as minimax_h3_runner
    from xfuser.model_executor.models.transformers import transformer_minimax_h3

    calls = []

    class _RuntimeState:
        attention_backend = AttentionBackendType.SDPA

        def has_attention_schedule(self):
            return False

        def increment_step_counter(self):
            if track_steps:
                calls.append(True)

    runtime_state = _RuntimeState()
    getter = lambda: runtime_state
    monkeypatch.setattr(transformer_minimax_h3, "get_runtime_state", getter)
    monkeypatch.setattr(minimax_h3_runner, "get_runtime_state", getter)
    monkeypatch.setattr("xfuser.core.distributed.get_runtime_state", getter)
    monkeypatch.setattr("xfuser.model_executor.layers.usp.get_runtime_state", getter)
    return calls


def test_minimax_h3_wrapper_matches_diffusers_u1(monkeypatch):
    from diffusers import MiniMaxH3Transformer3DModel

    from xfuser.core.distributed.attention_backend import AttentionBackendType
    from xfuser.model_executor.models.transformers import transformer_minimax_h3
    from xfuser.model_executor.models.transformers.transformer_minimax_h3 import (
        xFuserMiniMaxH3Transformer3DWrapper,
    )

    monkeypatch.setattr(
        transformer_minimax_h3,
        "get_ulysses_parallel_world_size",
        lambda: 1,
    )
    monkeypatch.setattr(
        transformer_minimax_h3,
        "get_ulysses_parallel_rank",
        lambda: 0,
    )
    _patch_minimax_runtime_state(monkeypatch)

    config = _tiny_config()
    base = MiniMaxH3Transformer3DModel(**config).eval()
    wrapped = xFuserMiniMaxH3Transformer3DWrapper(
        **config,
        attention_backend=AttentionBackendType.SDPA,
    ).eval()
    wrapped.load_state_dict(base.state_dict())
    inputs = _tiny_inputs(torch.device("cpu"))

    with torch.no_grad():
        expected = base(**inputs)
        actual = wrapped(**inputs)
        wrapped.fuse_qkv_projections()
        fused_actual = wrapped(**inputs)

    torch.testing.assert_close(actual.sample, expected.sample)
    torch.testing.assert_close(actual.audio_sample, expected.audio_sample)
    torch.testing.assert_close(fused_actual.sample, expected.sample)
    torch.testing.assert_close(fused_actual.audio_sample, expected.audio_sample)


def test_minimax_h3_padding_alignment():
    from xfuser.model_executor.models.transformers.transformer_minimax_h3 import (
        xFuserMiniMaxH3Transformer3DWrapper,
    )

    hidden_states = torch.randn(1, 65, 8)
    timestep_indices = torch.zeros(65, dtype=torch.long)
    token_tags = torch.zeros(65, dtype=torch.long)
    position_ids = torch.zeros(65, 3)

    padded = xFuserMiniMaxH3Transformer3DWrapper._pad_rows(
        hidden_states,
        timestep_indices,
        token_tags,
        position_ids,
    )

    assert padded[-1] == 63
    assert padded[0].shape[1] == 128
    assert padded[1].shape == (128,)
    assert padded[2].shape == (128,)
    assert padded[3].shape == (128, 3)
    assert torch.all(padded[2][65:] == -1)


def test_minimax_h3_wrapper_exposes_diffusers_config_signature():
    from xfuser.model_executor.models.transformers.transformer_minimax_h3 import (
        xFuserMiniMaxH3Transformer3DWrapper,
    )

    init_keys = xFuserMiniMaxH3Transformer3DWrapper._get_init_keys(
        xFuserMiniMaxH3Transformer3DWrapper
    )

    assert "hidden_size" in init_keys
    assert "num_layers" in init_keys
    assert "enable_fasth3_vsa" in init_keys


def test_minimax_h3_runner_registration():
    import xfuser.model_executor.models.runner_models
    from xfuser.model_executor.models.runner_models.base_model import MODEL_REGISTRY

    assert "MiniMaxAI/MiniMax-H3" in MODEL_REGISTRY
    assert "MiniMax-H3" in MODEL_REGISTRY
    assert "MiniMax-H3-Ref2VA" in MODEL_REGISTRY
    assert "FastH3" in MODEL_REGISTRY
    assert (
        "FastVideo/FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree"
        in MODEL_REGISTRY
    )


def test_fasth3_defaults_match_inference_contract():
    from xfuser.model_executor.models.runner_models.minimax_h3 import (
        FASTH3_MODEL_ID,
        xFuserFastH3Model,
    )

    assert xFuserFastH3Model.settings.model_name == FASTH3_MODEL_ID
    assert xFuserFastH3Model.settings.valid_tasks == ["t2va"]
    assert xFuserFastH3Model.default_input_values.num_inference_steps == 5
    assert xFuserFastH3Model._warmup_num_inference_steps == 5
    assert xFuserFastH3Model._enable_fasth3_vsa


def test_fasth3_wrapper_defines_checkpoint_compression_gates(monkeypatch):
    from xfuser.model_executor.models.transformers.transformer_minimax_h3 import (
        xFuserMiniMaxH3Transformer3DWrapper,
    )

    _patch_minimax_runtime_state(monkeypatch)
    model = xFuserMiniMaxH3Transformer3DWrapper(
        **_tiny_config(),
        enable_fasth3_vsa=True,
    )

    for block in model.transformer_blocks:
        gate = block.attn.to_gate_compress
        assert gate.in_features == block.attn.to_q.in_features
        assert gate.out_features == block.attn.to_q.out_features
        assert gate.bias is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA/HIP")
def test_fasth3_wrapper_runs_vsa_attention(monkeypatch):
    from xfuser.model_executor.layers import usp
    from xfuser.model_executor.models.transformers import transformer_minimax_h3
    from xfuser.model_executor.models.transformers.transformer_minimax_h3 import (
        xFuserMiniMaxH3Transformer3DWrapper,
    )

    monkeypatch.setattr(
        transformer_minimax_h3,
        "get_ulysses_parallel_world_size",
        lambda: 1,
    )
    monkeypatch.setattr(
        transformer_minimax_h3,
        "get_ulysses_parallel_rank",
        lambda: 0,
    )
    monkeypatch.setattr(usp, "get_ulysses_parallel_world_size", lambda: 1)
    _patch_minimax_runtime_state(monkeypatch)
    model = (
        xFuserMiniMaxH3Transformer3DWrapper(
            **_tiny_config(),
            enable_fasth3_vsa=True,
        )
        .to(device="cuda", dtype=torch.bfloat16)
        .eval()
    )

    inputs = {
        name: value.to("cuda") if isinstance(value, torch.Tensor) else value
        for name, value in _tiny_inputs("cpu").items()
    }
    with torch.no_grad():
        output = model(**inputs)

    assert output.sample.shape == (1, 48, 16)
    assert output.audio_sample.shape == (1, 12, 6)
    assert torch.isfinite(output.sample).all()
    assert torch.isfinite(output.audio_sample).all()


def test_fasth3_accepts_vsa_runtime_configuration():
    from xfuser import xFuserArgs
    from xfuser.model_executor.models.runner_models.minimax_h3 import (
        xFuserFastH3Model,
    )

    config = xFuserArgs(
        model="FastH3",
        task="t2va",
        attention_backend="AITER",
    )

    xFuserFastH3Model(config)


@pytest.mark.parametrize(
    "unsupported",
    [
        {"use_torch_compile": True},
        {
            "use_hybrid_attn_schedule": True,
            "hybrid_attn_high_precision_backend": "AITER",
            "hybrid_attn_low_precision_backend": "AITER_FP8",
        },
    ],
)
def test_fasth3_rejects_unsupported_compile_modes(unsupported):
    from xfuser import xFuserArgs
    from xfuser.model_executor.models.runner_models.minimax_h3 import (
        xFuserFastH3Model,
    )

    config = xFuserArgs(
        model="FastH3",
        task="t2va",
        attention_backend="AITER",
        **unsupported,
    )

    with pytest.raises(ValueError, match="FastH3"):
        xFuserFastH3Model(config)


def test_minimax_h3_fp8_quantization_policy():
    from xfuser.model_executor.models.runner_models.minimax_h3 import (
        xFuserMiniMaxH3Model,
        xFuserMiniMaxH3Ref2VAModel,
    )

    expected = ("attn.to_qkv", "ff.net.0.proj")
    assert xFuserMiniMaxH3Model.settings.fp8_gemm_include_suffixes == expected
    assert xFuserMiniMaxH3Ref2VAModel.settings.fp8_gemm_include_suffixes == expected


def test_minimax_h3_fp4_quantization_policy():
    from xfuser.model_executor.models.runner_models.minimax_h3 import (
        xFuserMiniMaxH3Model,
        xFuserMiniMaxH3Ref2VAModel,
    )

    expected = ("attn.to_out.0", "ff.net.2", "adaln_proj.linear")
    assert xFuserMiniMaxH3Model.settings.fp8_precision_override_suffixes == expected
    assert (
        xFuserMiniMaxH3Ref2VAModel.settings.fp8_precision_override_suffixes
        == expected
    )


def test_minimax_h3_text_encoder_tp_plan():
    from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel

    from xfuser.model_executor.models.runner_models.minimax_h3 import (
        xFuserMiniMaxH3Model,
    )

    plan = xFuserMiniMaxH3Model._build_text_encoder_tp_plan(2)

    assert len(plan) == 14
    assert isinstance(plan["layers.0.self_attn.q_proj"], ColwiseParallel)
    assert isinstance(plan["layers.0.self_attn.o_proj"], RowwiseParallel)
    assert isinstance(plan["layers.1.mlp.gate_proj"], ColwiseParallel)
    assert isinstance(plan["layers.1.mlp.down_proj"], RowwiseParallel)


def test_text_encoder_tp_requires_model_capability():
    from xfuser.config import xFuserArgs
    from xfuser.model_executor.models.runner_models.base_model import (
        DiffusionOutput,
        ModelSettings,
        xFuserModel,
    )

    class UnsupportedTextEncoderTPModel(xFuserModel):
        settings = ModelSettings(model_name="unsupported-text-encoder-tp")

        def _load_model(self):
            raise NotImplementedError

        def _run_pipe(self, input_args: dict) -> DiffusionOutput:
            raise NotImplementedError

    config = xFuserArgs(
        model="unsupported-text-encoder-tp",
        text_encoder_tp_degree=2,
    )

    with pytest.raises(
        ValueError,
        match="does not support text_encoder_tp_degree",
    ):
        UnsupportedTextEncoderTPModel(config)

    from xfuser.model_executor.models.runner_models.minimax_h3 import (
        xFuserMiniMaxH3Model,
    )

    assert xFuserMiniMaxH3Model.capabilities.text_encoder_tp_degree


def test_minimax_h3_parallel_vae_uses_native_tile_split(monkeypatch):
    from xfuser.config import xFuserArgs
    from xfuser.model_executor.models.runner_models.minimax_h3 import (
        xFuserMiniMaxH3Model,
    )

    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    model = xFuserMiniMaxH3Model(
        xFuserArgs(
            model="MiniMax-H3",
            task="t2va",
            use_parallel_vae=True,
        )
    )

    assert model.capabilities.use_parallel_vae

    with pytest.raises(ValueError, match="does not support dedicated VAE-only ranks"):
        xFuserMiniMaxH3Model(
            xFuserArgs(
                model="MiniMax-H3",
                task="t2va",
                use_parallel_vae=True,
                vae_parallel_size=1,
            )
        )


def test_minimax_h3_parallel_vae_single_rank_falls_back(monkeypatch):
    from xfuser.model_executor.models.runner_models import minimax_h3

    class FakeVae:
        use_tiling = True

        def __init__(self):
            self._decode_clip = lambda z: z + 1

    monkeypatch.setattr(
        minimax_h3,
        "get_vae_parallel_group",
        lambda: SimpleNamespace(world_size=1),
    )
    monkeypatch.setattr(minimax_h3, "log", lambda message: None)

    vae = FakeVae()
    minimax_h3.install_minimax_h3_vae_tile_parallel(vae)
    actual = vae._decode_clip(torch.zeros(1))

    torch.testing.assert_close(actual, torch.ones(1))
    assert vae._xfuser_tile_parallel


class _FakeTileDecoder:
    def __init__(self, calls):
        self.out_channels = 3
        self.patch_size_t = 1
        self.calls = calls

    def __call__(self, tile):
        self.calls.append(tile.detach().clone())
        batch, _, frames, latent_h, latent_w = tile.shape
        value = tile.flatten()[0]
        return torch.full(
            (
                batch,
                self.out_channels,
                frames * self.patch_size_t,
                latent_h * 2,
                latent_w * 2,
            ),
            fill_value=float(value),
            dtype=torch.float32,
        )


class _FakeTiledVae:
    use_tiling = True
    spatial_compression_ratio = 2
    tile_sample_min_height = 4
    tile_sample_min_width = 4
    tile_sample_min_overlap_height = 0
    tile_sample_min_overlap_width = 0

    def __init__(self):
        self.decode_calls = []
        self.decoder = _FakeTileDecoder(self.decode_calls)
        self.post_quant_conv = lambda tile: tile
        self._decode_clip = lambda z: z

    def _split_tiles(self, size, min_size, overlap):
        count = size // min_size
        indices = [index * min_size for index in range(count)]
        lengths = [min_size] * count
        overlaps = [0] * count
        return indices, lengths, overlaps

    def _stitch_tiles(self, rows, y_overlaps, x_overlaps):
        return torch.cat([torch.cat(row, dim=-1) for row in rows], dim=-2)


class _FakeVaeGroup:
    def __init__(self, world_size, rank, peer_tiles, output_dtype):
        self.world_size = world_size
        self.rank_in_group = rank
        self._peer_tiles = peer_tiles
        self._output_dtype = output_dtype
        self._broadcast_index = 0
        self.object_broadcasts = 0

    def broadcast_object_list(self, object_list, src=0):
        self.object_broadcasts += 1
        if self.rank_in_group != src:
            object_list[0] = self._output_dtype

    def broadcast(self, tensor, src):
        tile_index = self._broadcast_index
        self._broadcast_index += 1
        if self.rank_in_group == src:
            return tensor
        return self._peer_tiles[tile_index % len(self._peer_tiles)]


def test_minimax_h3_parallel_vae_shards_complete_tiles(monkeypatch):
    from xfuser.model_executor.models.runner_models import minimax_h3

    world_size = 2
    latent = torch.zeros(1, 4, 2, 2, 4)
    for y in range(latent.shape[-2]):
        for x in range(latent.shape[-1]):
            latent[..., y, x] = y * 10 + x

    sequential = _FakeTiledVae()
    expected_inputs = {}
    expected_tiles = {}
    ratio = sequential.spatial_compression_ratio
    y_indices, y_lengths, _ = sequential._split_tiles(
        latent.shape[-2] * ratio,
        sequential.tile_sample_min_height,
        sequential.tile_sample_min_overlap_height,
    )
    x_indices, x_lengths, _ = sequential._split_tiles(
        latent.shape[-1] * ratio,
        sequential.tile_sample_min_width,
        sequential.tile_sample_min_overlap_width,
    )
    tile_index = 0
    for y, tile_height in zip(y_indices, y_lengths):
        for x, tile_width in zip(x_indices, x_lengths):
            tile = latent[
                ...,
                y // ratio : y // ratio + tile_height // ratio,
                x // ratio : x // ratio + tile_width // ratio,
            ]
            expected_inputs[tile_index] = tile.clone()
            expected_tiles[tile_index] = sequential.decoder(
                sequential.post_quant_conv(tile)
            )
            tile_index += 1
    expected = sequential._stitch_tiles(
        [[expected_tiles[0], expected_tiles[1]]],
        [0],
        [0, 0],
    )

    monkeypatch.setattr(minimax_h3, "log", lambda message: None)

    for rank in range(world_size):
        vae = _FakeTiledVae()
        group = _FakeVaeGroup(
            world_size,
            rank,
            expected_tiles,
            expected_tiles[0].dtype,
        )
        monkeypatch.setattr(
            minimax_h3,
            "get_vae_parallel_group",
            lambda group=group: group,
        )
        minimax_h3.install_minimax_h3_vae_tile_parallel(vae)
        actual = vae._decode_clip(latent)
        second = vae._decode_clip(latent)

        owned = [index for index in expected_tiles if index % world_size == rank]
        assert group.object_broadcasts == 0
        assert len(vae.decode_calls) == 2 * len(owned)
        for call, index in zip(vae.decode_calls[: len(owned)], owned):
            torch.testing.assert_close(call, expected_inputs[index])
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(second, expected)


def test_minimax_h3_parallel_vae_caches_dtype_broadcast(monkeypatch):
    from xfuser.model_executor.models.runner_models import minimax_h3

    latent = torch.zeros(1, 4, 2, 2, 4)
    sequential = _FakeTiledVae()
    expected_tiles = {}
    ratio = sequential.spatial_compression_ratio
    y_indices, y_lengths, _ = sequential._split_tiles(
        latent.shape[-2] * ratio,
        sequential.tile_sample_min_height,
        sequential.tile_sample_min_overlap_height,
    )
    x_indices, x_lengths, _ = sequential._split_tiles(
        latent.shape[-1] * ratio,
        sequential.tile_sample_min_width,
        sequential.tile_sample_min_overlap_width,
    )
    tile_index = 0
    for y, tile_height in zip(y_indices, y_lengths):
        for x, tile_width in zip(x_indices, x_lengths):
            tile = latent[
                ...,
                y // ratio : y // ratio + tile_height // ratio,
                x // ratio : x // ratio + tile_width // ratio,
            ]
            expected_tiles[tile_index] = sequential.decoder(
                sequential.post_quant_conv(tile)
            )
            tile_index += 1

    monkeypatch.setattr(minimax_h3, "log", lambda message: None)
    vae = _FakeTiledVae()
    group = _FakeVaeGroup(3, 2, expected_tiles, expected_tiles[0].dtype)
    monkeypatch.setattr(minimax_h3, "get_vae_parallel_group", lambda: group)
    minimax_h3.install_minimax_h3_vae_tile_parallel(vae)

    vae._decode_clip(latent)
    assert group.object_broadcasts == 1
    vae._decode_clip(latent)
    assert group.object_broadcasts == 1


@pytest.mark.parametrize(
    "offload_flag",
    [
        "enable_model_cpu_offload",
        "enable_sequential_cpu_offload",
        "enable_group_cpu_offload",
    ],
)
def test_minimax_h3_text_encoder_tp_rejects_cpu_offload(offload_flag):
    from xfuser.config import xFuserArgs
    from xfuser.model_executor.models.runner_models.minimax_h3 import (
        xFuserMiniMaxH3Model,
    )

    config = xFuserArgs(
        model="MiniMax-H3",
        task="t2va",
        ulysses_degree=2,
        text_encoder_tp_degree=2,
        **{offload_flag: True},
    )

    with pytest.raises(ValueError, match="incompatible with CPU offloading"):
        xFuserMiniMaxH3Model(config)


def test_minimax_h3_patches_shared_qwen_encoder_helper(monkeypatch):
    from diffusers.modular_pipelines.minimax_h3 import encoders

    from xfuser.model_executor.models.runner_models import minimax_h3

    expected = torch.randn(1, 4, 8)
    calls = []

    def fake_get_prompt_embeds(*args, **kwargs):
        calls.append((args, kwargs))
        return expected

    monkeypatch.setattr(encoders, "get_qwen3vl_prompt_embeds", fake_get_prompt_embeds)
    monkeypatch.setattr(encoders, "_xfuser_broadcast_patched", False, raising=False)
    monkeypatch.setattr(
        minimax_h3,
        "get_world_group",
        lambda: SimpleNamespace(world_size=1),
    )

    minimax_h3._patch_minimax_h3_text_encoder_broadcast()
    actual = encoders.get_qwen3vl_prompt_embeds(
        SimpleNamespace(device=torch.device("cpu"), dtype=torch.bfloat16),
        object(),
        [1, 2, 3],
    )

    assert actual is expected
    assert len(calls) == 1
    assert encoders._xfuser_broadcast_patched


class _FakeMiniMaxPipe:
    def __init__(self):
        self.text_encoder = SimpleNamespace(lm_head=object())
        self.loaded_dtype = None

    def update_components(self, **components):
        for name, component in components.items():
            setattr(self, name, component)

    def load_components(self, dtype):
        self.loaded_dtype = dtype


class _FakeTransformer:
    def __init__(self):
        self.qkv_fused = False

    def fuse_qkv_projections(self):
        self.qkv_fused = True


@pytest.mark.parametrize(
    ("task", "expected_workflow"),
    [("t2va", "t2va"), ("i2va", "fl2va"), ("fl2va", "fl2va")],
)
def test_minimax_h3_loads_task_workflow(
    monkeypatch,
    task,
    expected_workflow,
):
    from diffusers import ModularPipeline

    from xfuser.model_executor.models.runner_models import minimax_h3
    from xfuser.model_executor.models.runner_models.minimax_h3 import (
        xFuserMiniMaxH3Model,
    )
    from xfuser.model_executor.models.transformers.transformer_minimax_h3 import (
        xFuserMiniMaxH3Transformer3DWrapper,
    )

    pipe = _FakeMiniMaxPipe()
    transformer = _FakeTransformer()
    workflows = []

    def fake_from_pretrained(model_name, workflow):
        workflows.append((model_name, workflow))
        return pipe

    monkeypatch.setattr(ModularPipeline, "from_pretrained", fake_from_pretrained)
    monkeypatch.setattr(
        xFuserMiniMaxH3Transformer3DWrapper,
        "from_pretrained",
        lambda *args, **kwargs: transformer,
    )
    monkeypatch.setattr(
        minimax_h3,
        "_patch_minimax_h3_text_encoder_broadcast",
        lambda: None,
    )
    monkeypatch.setattr(minimax_h3, "log", lambda message: None)

    model = object.__new__(xFuserMiniMaxH3Model)
    model.config = SimpleNamespace(task=task, text_encoder_tp_degree=1)
    model._parallelize_text_encoder = lambda text_encoder: None

    actual = model._load_model()

    assert actual is pipe
    assert workflows == [(model.settings.model_name, expected_workflow)]
    assert pipe.transformer is transformer
    assert transformer.qkv_fused
    assert pipe.loaded_dtype == torch.bfloat16
    assert pipe.text_encoder.lm_head is None


def test_fasth3_loads_published_checkpoint(monkeypatch):
    from diffusers import ModularPipeline

    from xfuser.model_executor.models.runner_models import minimax_h3
    from xfuser.model_executor.models.runner_models.minimax_h3 import (
        FASTH3_MODEL_ID,
        xFuserFastH3Model,
    )
    from xfuser.model_executor.models.transformers.transformer_minimax_h3 import (
        xFuserMiniMaxH3Transformer3DWrapper,
    )

    pipe = _FakeMiniMaxPipe()
    transformer = _FakeTransformer()
    pipeline_loads = []
    transformer_loads = []

    def fake_pipeline_from_pretrained(model_name, workflow):
        pipeline_loads.append((model_name, workflow))
        return pipe

    def fake_transformer_from_pretrained(model_name, **kwargs):
        transformer_loads.append((model_name, kwargs))
        return transformer

    monkeypatch.setattr(
        ModularPipeline,
        "from_pretrained",
        fake_pipeline_from_pretrained,
    )
    monkeypatch.setattr(
        xFuserMiniMaxH3Transformer3DWrapper,
        "from_pretrained",
        fake_transformer_from_pretrained,
    )
    monkeypatch.setattr(
        minimax_h3,
        "_patch_minimax_h3_text_encoder_broadcast",
        lambda: None,
    )
    monkeypatch.setattr(minimax_h3, "log", lambda message: None)

    model = object.__new__(xFuserFastH3Model)
    model.config = SimpleNamespace(task="t2va", text_encoder_tp_degree=1)
    model._parallelize_text_encoder = lambda text_encoder: None

    actual = model._load_model()

    assert actual is pipe
    assert pipeline_loads == [(FASTH3_MODEL_ID, "t2va")]
    assert transformer_loads == [
        (
            FASTH3_MODEL_ID,
            {
                "subfolder": "transformer",
                "dtype": torch.bfloat16,
                "enable_fasth3_vsa": True,
            },
        )
    ]
    assert pipe.transformer is transformer


def test_minimax_h3_ref2va_loads_workflow(monkeypatch):
    from diffusers import ModularPipeline

    from xfuser.model_executor.models.runner_models import minimax_h3
    from xfuser.model_executor.models.runner_models.minimax_h3 import (
        xFuserMiniMaxH3Ref2VAModel,
    )
    from xfuser.model_executor.models.transformers.transformer_minimax_h3 import (
        xFuserMiniMaxH3Transformer3DWrapper,
    )

    pipe = _FakeMiniMaxPipe()
    transformer = _FakeTransformer()
    workflows = []

    def fake_from_pretrained(model_name, workflow):
        workflows.append((model_name, workflow))
        return pipe

    monkeypatch.setattr(ModularPipeline, "from_pretrained", fake_from_pretrained)
    monkeypatch.setattr(
        xFuserMiniMaxH3Transformer3DWrapper,
        "from_pretrained",
        lambda *args, **kwargs: transformer,
    )
    monkeypatch.setattr(
        minimax_h3,
        "_patch_minimax_h3_text_encoder_broadcast",
        lambda: None,
    )
    monkeypatch.setattr(minimax_h3, "log", lambda message: None)

    model = object.__new__(xFuserMiniMaxH3Ref2VAModel)
    model.config = SimpleNamespace(text_encoder_tp_degree=1)
    model._parallelize_text_encoder = lambda text_encoder: None

    actual = model._load_model()

    assert actual is pipe
    assert workflows == [(model.settings.model_name, "ref2va")]
    assert pipe.transformer_ref is transformer
    assert transformer.qkv_fused
    assert not hasattr(pipe, "transformer")


def test_minimax_h3_ref2va_runtime_state_uses_ref_transformer():
    from xfuser.model_executor.models.runner_models.minimax_h3 import (
        xFuserMiniMaxH3Ref2VAModel,
    )

    transformer = object()
    model = object.__new__(xFuserMiniMaxH3Ref2VAModel)
    model.pipe = SimpleNamespace(transformer_ref=transformer)

    runtime_pipeline = model._get_runtime_state_pipeline()

    assert runtime_pipeline.transformer is transformer
    assert not hasattr(model.pipe, "transformer")


def test_minimax_h3_compile_preserves_forward_signature(monkeypatch):
    from xfuser.model_executor.models.runner_models.minimax_h3 import (
        xFuserMiniMaxH3Model,
    )
    from xfuser.model_executor.models.transformers.transformer_minimax_h3 import (
        xFuserMiniMaxH3Transformer3DWrapper,
    )

    _patch_minimax_runtime_state(monkeypatch)
    transformer = xFuserMiniMaxH3Transformer3DWrapper(**_tiny_config()).eval()
    model = object.__new__(xFuserMiniMaxH3Model)
    model.config = SimpleNamespace(fully_shard_degree=1)
    compile_calls = []
    vae = SimpleNamespace(
        compile_repeated_blocks=lambda **kwargs: compile_calls.append(kwargs)
    )
    model.pipe = SimpleNamespace(transformer=transformer, vae=vae)
    model._enable_compute_comm_overlap = lambda: None
    model._get_compile_mode = lambda: "default"
    model._run_timed_pipe = lambda input_args: None
    monkeypatch.setattr(
        "xfuser.model_executor.models.runner_models.minimax_h3.log",
        lambda message: None,
    )

    model._compile_model({"num_inference_steps": 50})

    parameters = inspect.signature(model.pipe.transformer.forward).parameters
    assert "token_tags" in parameters
    assert "position_ids" in parameters
    assert compile_calls == [{"mode": "default", "fullgraph": False}]


def test_fasth3_warmup_uses_four_forward_schedule(monkeypatch):
    from xfuser.model_executor.models.runner_models.minimax_h3 import (
        xFuserFastH3Model,
    )

    model = object.__new__(xFuserFastH3Model)
    model.config = SimpleNamespace(warmup_calls=2)
    calls = []
    model._run_timed_pipe = lambda input_args: calls.append(input_args)
    monkeypatch.setattr(
        "xfuser.model_executor.models.runner_models.minimax_h3.log",
        lambda message: None,
    )
    input_args = {"num_inference_steps": 50, "prompt": "test"}

    model._run_warmup_calls(input_args)

    assert input_args["num_inference_steps"] == 50
    assert [call["num_inference_steps"] for call in calls] == [5, 5]


def test_minimax_h3_ref2va_uses_typed_image_references():
    from diffusers.modular_pipelines.minimax_h3 import MiniMaxH3ImageReference

    from xfuser.model_executor.models.runner_models.minimax_h3 import (
        xFuserMiniMaxH3Ref2VAModel,
    )

    captured = {}

    def fake_pipe(**kwargs):
        captured.update(kwargs)
        return {
            "videos": np.zeros((1, 1, 1, 1, 3), dtype=np.float32),
            "audio": torch.zeros(1, 2, 1),
            "sampling_rate": 24_000,
        }

    model = object.__new__(xFuserMiniMaxH3Ref2VAModel)
    model.pipe = fake_pipe
    image = Image.new("RGB", (32, 32))

    model._run_pipe(
        {
            "prompt": "Animate this reference.",
            "input_images": [image],
            "height": 32,
            "width": 32,
            "num_frames": 5,
            "num_inference_steps": 1,
            "seed": 0,
        }
    )

    assert len(captured["references"]) == 1
    assert isinstance(captured["references"][0], MiniMaxH3ImageReference)
    assert captured["references"][0].image is image


def test_minimax_h3_supports_hybrid_attention_capability():
    from xfuser.model_executor.models.runner_models.minimax_h3 import (
        xFuserMiniMaxH3Model,
        xFuserMiniMaxH3Ref2VAModel,
    )

    assert xFuserMiniMaxH3Model.capabilities.use_hybrid_attn_schedule
    assert xFuserMiniMaxH3Ref2VAModel.capabilities.use_hybrid_attn_schedule
    assert (
        xFuserMiniMaxH3Model.default_input_values.num_hybrid_attn_high_precision_steps
        == 5
    )


def test_minimax_h3_accepts_hybrid_attention_backends():
    from xfuser.config import xFuserArgs
    from xfuser.model_executor.models.runner_models.minimax_h3 import (
        xFuserMiniMaxH3Model,
    )

    config = xFuserArgs(
        model="MiniMax-H3",
        task="t2va",
        use_hybrid_attn_schedule=True,
        hybrid_attn_high_precision_backend="cudnn",
        hybrid_attn_low_precision_backend="nvte_fp8",
        num_hybrid_attn_high_precision_steps=5,
    )

    xFuserMiniMaxH3Model(config)


def test_minimax_h3_rejects_unsupported_hybrid_backend():
    from xfuser.config import xFuserArgs
    from xfuser.model_executor.models.runner_models.minimax_h3 import (
        xFuserMiniMaxH3Model,
    )

    config = xFuserArgs(
        model="MiniMax-H3",
        task="t2va",
        use_hybrid_attn_schedule=True,
        hybrid_attn_high_precision_backend="cudnn",
        hybrid_attn_low_precision_backend="flash_3_fp8",
        num_hybrid_attn_high_precision_steps=5,
    )

    with pytest.raises(ValueError, match="does not support attention backend"):
        xFuserMiniMaxH3Model(config)


def test_minimax_h3_forward_increments_hybrid_step_counter(monkeypatch):
    from xfuser.model_executor.models.transformers import transformer_minimax_h3
    from xfuser.model_executor.models.transformers.transformer_minimax_h3 import (
        xFuserMiniMaxH3Transformer3DWrapper,
    )

    monkeypatch.setattr(
        transformer_minimax_h3,
        "get_ulysses_parallel_world_size",
        lambda: 1,
    )
    monkeypatch.setattr(
        transformer_minimax_h3,
        "get_ulysses_parallel_rank",
        lambda: 0,
    )
    calls = _patch_minimax_runtime_state(monkeypatch, track_steps=True)

    wrapper = xFuserMiniMaxH3Transformer3DWrapper(**_tiny_config()).eval()
    inputs = _tiny_inputs(torch.device("cpu"))

    with torch.no_grad():
        wrapper(**inputs)

    assert len(calls) == 1
