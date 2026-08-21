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


def test_minimax_h3_runner_registration():
    import xfuser.model_executor.models.runner_models
    from xfuser.model_executor.models.runner_models.base_model import MODEL_REGISTRY

    assert "MiniMaxAI/MiniMax-H3" in MODEL_REGISTRY
    assert "MiniMax-H3" in MODEL_REGISTRY
    assert "MiniMax-H3-Ref2VA" in MODEL_REGISTRY


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


def test_minimax_h3_compile_preserves_forward_signature():
    from xfuser.model_executor.models.runner_models.minimax_h3 import (
        xFuserMiniMaxH3Model,
    )
    from xfuser.model_executor.models.transformers.transformer_minimax_h3 import (
        xFuserMiniMaxH3Transformer3DWrapper,
    )

    transformer = xFuserMiniMaxH3Transformer3DWrapper(**_tiny_config()).eval()
    model = object.__new__(xFuserMiniMaxH3Model)
    model.config = SimpleNamespace(fully_shard_degree=1)
    model.pipe = SimpleNamespace(transformer=transformer)
    model._enable_compute_comm_overlap = lambda: None
    model._get_compile_mode = lambda: "default"
    model._run_timed_pipe = lambda input_args: None

    model._compile_model({"num_inference_steps": 50})

    parameters = inspect.signature(model.pipe.transformer.forward).parameters
    assert "token_tags" in parameters
    assert "position_ids" in parameters


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
