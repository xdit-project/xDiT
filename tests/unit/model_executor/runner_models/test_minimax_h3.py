from types import SimpleNamespace

import inspect
import sys
import types

import pytest
import torch
from PIL import Image

pytest.importorskip(
    "diffusers.models.transformers.transformer_minimax_h3",
    reason="installed diffusers does not include MiniMax-H3",
)


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


def _tiny_inputs(device, text_tokens=4, audio_tokens=12, video_tokens=48):
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


def _patch_minimax_runtime_state(monkeypatch, *, track_steps=False, attention_backend=None):
    from xfuser.core.attention.spec import AttentionBackendType
    from xfuser.model_executor.models.runner_models import minimax_h3 as minimax_h3_runner
    from xfuser.model_executor.models.transformers import transformer_minimax_h3

    calls = []
    selected_backend = AttentionBackendType.SDPA if attention_backend is None else attention_backend

    class _RuntimeState:
        attention_backend = selected_backend

        def has_attention_schedule(self):
            return False

        def increment_step_counter(self):
            if track_steps:
                calls.append(True)

    runtime_state = _RuntimeState()

    def getter():
        return runtime_state

    monkeypatch.setattr(transformer_minimax_h3, "get_runtime_state", getter)
    monkeypatch.setattr(minimax_h3_runner, "get_runtime_state", getter)
    monkeypatch.setattr("xfuser.core.distributed.get_runtime_state", getter)
    monkeypatch.setattr("xfuser.model_executor.layers.usp.get_runtime_state", getter)
    return calls


def test_minimax_h3_wrapper_matches_diffusers_u1(monkeypatch):
    diffusers = pytest.importorskip("diffusers")
    MiniMaxH3Transformer3DModel = getattr(diffusers, "MiniMaxH3Transformer3DModel", None)
    if MiniMaxH3Transformer3DModel is None:
        pytest.skip("installed diffusers does not include MiniMax-H3")

    from xfuser.core.attention.spec import AttentionBackendType
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


def test_minimax_h3_publishes_the_trailing_pad_length(monkeypatch):
    """Backends without a key-padding mask slice K/V by valid_kv_len instead.

    The key is published on every forward, carrying None when the sequence
    already aligns, because torch.compile guards on this dict's key set.
    """
    from xfuser.core.attention.spec import AttentionBackendType
    from xfuser.model_executor.models.transformers import transformer_minimax_h3
    from xfuser.model_executor.models.transformers.transformer_minimax_h3 import (
        xFuserMiniMaxH3Transformer3DWrapper,
    )

    monkeypatch.setattr(transformer_minimax_h3, "get_ulysses_parallel_world_size", lambda: 1)
    monkeypatch.setattr(transformer_minimax_h3, "get_ulysses_parallel_rank", lambda: 0)
    _patch_minimax_runtime_state(monkeypatch)

    wrapper = xFuserMiniMaxH3Transformer3DWrapper(
        **_tiny_config(),
        attention_backend=AttentionBackendType.SDPA,
    ).eval()

    with torch.no_grad():
        wrapper(**_tiny_inputs(torch.device("cpu")))
    aligned_keys = set(wrapper._usp_attention_kwargs)
    assert wrapper._usp_attention_kwargs["valid_kv_len"] is None

    with torch.no_grad():
        wrapper(**_tiny_inputs(torch.device("cpu"), text_tokens=5))
    assert wrapper._usp_attention_kwargs["valid_kv_len"] == 65
    assert wrapper._usp_attention_kwargs["max_seqlen_k"] == 65
    assert set(wrapper._usp_attention_kwargs) == aligned_keys


def test_minimax_h3_wrapper_exposes_diffusers_config_signature():
    from xfuser.model_executor.models.transformers.transformer_minimax_h3 import (
        xFuserMiniMaxH3Transformer3DWrapper,
    )

    init_keys = xFuserMiniMaxH3Transformer3DWrapper._get_init_keys(xFuserMiniMaxH3Transformer3DWrapper)

    assert "hidden_size" in init_keys
    assert "num_layers" in init_keys
    assert "enable_fasth3_vsa" in init_keys


def test_minimax_h3_runner_registration():
    from xfuser.model_executor.models.runner_models.base_model import MODEL_REGISTRY

    assert "MiniMaxAI/MiniMax-H3" in MODEL_REGISTRY
    assert "MiniMax-H3" in MODEL_REGISTRY
    assert "MiniMax-H3-Ref2VA" in MODEL_REGISTRY
    assert "FastH3" in MODEL_REGISTRY
    assert "FastVideo/FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree" in MODEL_REGISTRY
    assert "FastH3-Dense" in MODEL_REGISTRY
    assert "FastVideo/FastVideo-FastH3-4-step-Preview-v1-Dense-DataFree" in MODEL_REGISTRY
    assert "MiniMax-H3-Turbo" in MODEL_REGISTRY


def test_fasth3_defaults_match_inference_contract():
    from xfuser.core.attention import registry as attention_registry
    from xfuser.core.attention.spec import AttentionBackendType, Sparsity

    VSA_H3_BACKENDS = attention_registry.types_where(sparsity=Sparsity.H3)
    from xfuser.model_executor.models.runner_models.minimax_h3 import (
        FASTH3_V1_DATAFREE_MODEL_ID,
        xFuserFastH3Model,
        xFuserMiniMaxH3Model,
    )

    assert xFuserFastH3Model.settings.model_name == FASTH3_V1_DATAFREE_MODEL_ID
    assert xFuserFastH3Model.settings.valid_tasks == ["t2va"]
    assert xFuserFastH3Model.settings.default_attention_backend == AttentionBackendType.TRITON_VSA_H3.name
    assert xFuserFastH3Model.default_input_values.num_inference_steps == 4
    assert xFuserFastH3Model._warmup_num_inference_steps == 4
    assert xFuserFastH3Model._enable_fasth3_vsa
    assert xFuserFastH3Model.supported_attn_backends == (xFuserMiniMaxH3Model.supported_attn_backends | VSA_H3_BACKENDS)


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


def test_fasth3_dense_defaults_match_inference_contract():
    from xfuser.core.attention import registry as attention_registry
    from xfuser.core.attention.spec import Sparsity

    VSA_H3_BACKENDS = attention_registry.types_where(sparsity=Sparsity.H3)
    from xfuser.model_executor.models.runner_models.minimax_h3 import (
        FASTH3_V1_DENSE_DATAFREE_MODEL_ID,
        xFuserFastH3DenseModel,
        xFuserMiniMaxH3Model,
    )

    assert xFuserFastH3DenseModel.settings.model_name == FASTH3_V1_DENSE_DATAFREE_MODEL_ID
    assert xFuserFastH3DenseModel.settings.output_name == "fasth3_dense"
    assert xFuserFastH3DenseModel.settings.valid_tasks == ["t2va"]
    # The dense ablation has no preferred backend, so --attention_backend decides.
    assert xFuserFastH3DenseModel.settings.default_attention_backend is None
    assert xFuserFastH3DenseModel.default_input_values.num_inference_steps == 4
    assert xFuserFastH3DenseModel._warmup_num_inference_steps == 4
    assert not xFuserFastH3DenseModel._enable_fasth3_vsa
    assert xFuserFastH3DenseModel.supported_attn_backends == xFuserMiniMaxH3Model.supported_attn_backends
    assert not (xFuserFastH3DenseModel.supported_attn_backends & VSA_H3_BACKENDS)


def test_fasth3_dense_accepts_dense_backends_and_rejects_vsa(monkeypatch):
    from xfuser import xFuserArgs
    from xfuser.model_executor.models.runner_models.minimax_h3 import (
        xFuserFastH3DenseModel,
    )

    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")

    for backend in ("AITER", "CUDNN"):
        config = xFuserArgs(
            model="FastH3-Dense",
            task="t2va",
            attention_backend=backend,
        )
        xFuserFastH3DenseModel(config)
        assert config.attention_backend == backend

    defaulted = xFuserArgs(model="FastH3-Dense", task="t2va")
    xFuserFastH3DenseModel(defaulted)
    assert defaulted.attention_backend is None

    with pytest.raises(ValueError, match="does not support attention backend"):
        xFuserFastH3DenseModel(
            xFuserArgs(
                model="FastH3-Dense",
                task="t2va",
                attention_backend="TRITON_VSA_H3",
            )
        )


def test_fasth3_dense_allows_the_hybrid_attention_schedule(monkeypatch):
    """Dense attention has no VSA step-coverage constraint, unlike VSA FastH3."""
    from xfuser import xFuserArgs
    from xfuser.model_executor.models.runner_models.minimax_h3 import (
        xFuserFastH3DenseModel,
    )

    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    aiter = types.ModuleType("aiter")
    aiter.flash_attn_varlen_fp8_pertensor_func = object()
    monkeypatch.setitem(sys.modules, "aiter", aiter)

    config = xFuserArgs(
        model="FastH3-Dense",
        task="t2va",
        use_hybrid_attn_schedule=True,
        hybrid_attn_high_precision_backend="AITER",
        hybrid_attn_low_precision_backend="AITER_FP8",
    )
    xFuserFastH3DenseModel(config)


def test_fasth3_dense_wrapper_omits_compression_gates(monkeypatch):
    from xfuser.model_executor.models.transformers.transformer_minimax_h3 import (
        xFuserMiniMaxH3Transformer3DWrapper,
    )

    _patch_minimax_runtime_state(monkeypatch)
    model = xFuserMiniMaxH3Transformer3DWrapper(
        **_tiny_config(),
        enable_fasth3_vsa=False,
    )

    # The dense checkpoint ships no to_gate_compress keys, so defining the
    # modules would leave them uninitialized after load.
    for block in model.transformer_blocks:
        assert not hasattr(block.attn, "to_gate_compress")
    assert not model.use_vsa_h3


@pytest.mark.accelerator
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
    from xfuser.core.attention.spec import AttentionBackendType

    _patch_minimax_runtime_state(
        monkeypatch,
        attention_backend=AttentionBackendType.FLEX_VSA_H3,
    )
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


def test_fasth3_accepts_vsa_and_dense_attention_backends(monkeypatch):
    from xfuser import xFuserArgs
    from xfuser.model_executor.models.runner_models.minimax_h3 import (
        xFuserFastH3Model,
    )

    # Announcing the default backend goes through runner_utils.log, which reads
    # RANK/WORLD_SIZE straight from the env.
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")

    for backend in ("AITER", "FLEX_VSA_H3", "TRITON_VSA_H3"):
        config = xFuserArgs(
            model="FastH3",
            task="t2va",
            attention_backend=backend,
        )
        xFuserFastH3Model(config)
        assert config.attention_backend == backend

    defaulted = xFuserArgs(model="FastH3", task="t2va")
    xFuserFastH3Model(defaulted)
    assert defaulted.attention_backend == "TRITON_VSA_H3"


def test_default_attention_backend_is_reusable_by_any_model(monkeypatch):
    """The default backend lives in ModelSettings, so it is not a FastH3-only feature."""
    from xfuser import xFuserArgs
    from xfuser.model_executor.models.runner_models.minimax_h3 import (
        xFuserMiniMaxH3Model,
    )

    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")

    model = xFuserMiniMaxH3Model(xFuserArgs(model="MiniMax-H3", task="t2va", attention_backend="AITER"))
    model.settings.default_attention_backend = "AITER_FP8"

    explicit = xFuserArgs(model="MiniMax-H3", task="t2va", attention_backend="AITER")
    model._apply_default_attention_backend(explicit)
    assert explicit.attention_backend == "AITER"

    defaulted = xFuserArgs(model="MiniMax-H3", task="t2va")
    model._apply_default_attention_backend(defaulted)
    assert defaulted.attention_backend == "AITER_FP8"

    model.settings.default_attention_backend = "NOT_A_BACKEND"
    with pytest.raises(ValueError, match="default attention backend"):
        model._apply_default_attention_backend(xFuserArgs(model="MiniMax-H3", task="t2va"))


def test_fasth3_rejects_unsupported_attention_backend():
    from xfuser import xFuserArgs
    from xfuser.model_executor.models.runner_models.minimax_h3 import (
        xFuserFastH3Model,
    )

    config = xFuserArgs(
        model="FastH3",
        task="t2va",
        attention_backend="FLASH_3",
    )

    with pytest.raises(ValueError, match="does not support attention backend"):
        xFuserFastH3Model(config)


def test_minimax_h3_accepts_dense_mha_v4_backends(monkeypatch):
    """Dense MHA v4 serves the 64-row pad by slicing K/V, so those rows are in.

    Their Sparge counterparts are not: the sorted-sparse launch requires the key
    length to stay padded to its KV tile.
    """
    from xfuser import xFuserArgs
    from xfuser.core.attention import registry as attention_registry
    from xfuser.core.attention.backends.aiter_mha_v4.spec import (
        DENSE_BACKENDS as AITER_MHA_V4_ONLY_BACKENDS,
    )
    from xfuser.core.attention.spec import Sparsity

    AITER_MHA_V4_SPARGE_BACKEND_SET = attention_registry.types_where(sparsity=Sparsity.SPARGE, head_balanced=True)
    from xfuser.model_executor.models.runner_models.minimax_h3 import (
        xFuserFastH3DenseModel,
        xFuserMiniMaxH3Model,
    )

    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")

    from xfuser.model_executor.models.runner_models.minimax_h3 import (
        _UNSERVED_MHA_V4_BACKENDS,
    )

    for backend in AITER_MHA_V4_ONLY_BACKENDS:
        if backend in _UNSERVED_MHA_V4_BACKENDS:
            # f4f4 has no manifest row of its own; mxfp4 covers it, so this is
            # about availability rather than the alignment bug it used to be.
            assert backend not in xFuserMiniMaxH3Model.supported_attn_backends
            continue
        assert backend in xFuserMiniMaxH3Model.supported_attn_backends
        config = xFuserArgs(model="MiniMax-H3", task="t2va", attention_backend=backend.name)
        xFuserMiniMaxH3Model(config)
        assert config.attention_backend == backend.name

    assert not (xFuserFastH3DenseModel.supported_attn_backends & AITER_MHA_V4_SPARGE_BACKEND_SET)
    # The model's own supported list refuses it before the generic Sparge gate
    # is reached, so the message names what the model does support.
    with pytest.raises(ValueError, match="does not support attention backend"):
        xFuserFastH3DenseModel(
            xFuserArgs(
                model="FastH3-Dense",
                task="t2va",
                attention_backend="AITER_I8FP8_SPARGE",
            )
        )


@pytest.mark.parametrize(
    "unsupported",
    [
        {
            "use_hybrid_attn_schedule": True,
            "hybrid_attn_high_precision_backend": "AITER",
            "hybrid_attn_low_precision_backend": "AITER_FP8",
        },
    ],
)
@pytest.mark.parametrize("backend", ["FLEX_VSA_H3", "TRITON_VSA_H3"])
def test_fasth3_rejects_unsupported_compile_modes(unsupported, backend):
    from xfuser import xFuserArgs
    from xfuser.model_executor.models.runner_models.minimax_h3 import (
        xFuserFastH3Model,
    )

    config = xFuserArgs(
        model="FastH3",
        task="t2va",
        attention_backend=backend,
        **unsupported,
    )

    with pytest.raises(ValueError, match="VSA-H3"):
        xFuserFastH3Model(config)


def test_minimax_h3_fp8_quantization_policy():
    from xfuser.model_executor.models.runner_models.minimax_h3 import (
        xFuserMiniMaxH3Model,
        xFuserMiniMaxH3Ref2VAModel,
    )

    expected = ("attn.to_qkv", "ff.net.0.proj")
    assert xFuserMiniMaxH3Model.settings.fp8_gemm_include_suffixes == expected
    assert xFuserMiniMaxH3Ref2VAModel.settings.fp8_gemm_include_suffixes == expected


def _tiny_lora_target():
    block = torch.nn.Module()
    block.attn = torch.nn.Module()
    block.attn.to_q = torch.nn.Linear(4, 6, bias=False)
    transformer = torch.nn.Module()
    transformer.transformer_blocks = torch.nn.ModuleList([block])
    return transformer


def _save_lora(tmp_path, tensors, metadata=None, name="turbo.safetensors"):
    from safetensors.torch import save_file

    path = tmp_path / name
    save_file(tensors, str(path), metadata=metadata)
    return str(path)


@pytest.mark.parametrize("metadata, alpha", [({"alpha": "16"}, 16.0), (None, 8.0)])
def test_turbo_lora_merges_with_peft_scaling_and_recorded_alpha(tmp_path, monkeypatch, metadata, alpha):
    from xfuser.model_executor.models.runner_models.minimax_h3 import _merge_lora_into_transformer

    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    transformer = _tiny_lora_target()
    base = transformer.transformer_blocks[0].attn.to_q.weight.detach().clone()
    lora_a, lora_b = torch.randn(2, 4), torch.randn(6, 2)
    path = _save_lora(
        tmp_path,
        {
            "transformer_blocks.0.attn.to_q.lora_A.default.weight": lora_a,
            "transformer_blocks.0.attn.to_q.lora_B.default.weight": lora_b,
        },
        metadata=metadata,
    )

    _merge_lora_into_transformer(transformer, path, scale=0.5)

    expected = base + (lora_b @ lora_a) * (alpha / 2 * 0.5)
    torch.testing.assert_close(transformer.transformer_blocks[0].attn.to_q.weight.detach(), expected)


@pytest.mark.parametrize(
    "tensors, match",
    [
        ({"transformer_blocks.0.attn.to_q.lora_A.default.weight": torch.randn(2, 4)}, "incomplete"),
        (
            {
                "transformer_blocks.0.attn.to_k.lora_A.default.weight": torch.randn(2, 4),
                "transformer_blocks.0.attn.to_k.lora_B.default.weight": torch.randn(6, 2),
            },
            "unresolved",
        ),
        (
            {
                "transformer_blocks.0.attn.to_q.lora_A.default.weight": torch.randn(2, 5),
                "transformer_blocks.0.attn.to_q.lora_B.default.weight": torch.randn(6, 2),
            },
            "does not match",
        ),
        ({"transformer_blocks.0.attn.to_q.alpha": torch.tensor(8.0)}, "Unexpected"),
    ],
)
def test_turbo_lora_rejects_adapters_that_do_not_fit(tmp_path, tensors, match):
    from xfuser.model_executor.models.runner_models.minimax_h3 import _merge_lora_into_transformer

    with pytest.raises(ValueError, match=match):
        _merge_lora_into_transformer(_tiny_lora_target(), _save_lora(tmp_path, tensors), scale=1.0)


def test_lora_file_resolves_from_a_file_a_directory_or_a_repository(tmp_path, monkeypatch):
    import huggingface_hub

    from xfuser.model_executor.models.runner_models.minimax_h3 import _resolve_lora_file

    path = _save_lora(tmp_path, {"x.lora_A.default.weight": torch.zeros(1, 1)}, name="adapter.safetensors")
    downloads = []
    monkeypatch.setattr(
        huggingface_hub,
        "hf_hub_download",
        lambda repo_id, filename: downloads.append((repo_id, filename)) or f"/cache/{filename}",
    )

    assert _resolve_lora_file(path, None) == path
    assert _resolve_lora_file(str(tmp_path), "adapter.safetensors") == path
    assert _resolve_lora_file("org/repo", "adapter.safetensors") == "/cache/adapter.safetensors"
    assert downloads == [("org/repo", "adapter.safetensors")]
    with pytest.raises(ValueError, match="lora_weight_name is required"):
        _resolve_lora_file("org/repo", None)


@pytest.mark.parametrize(
    "steps, shift, expected",
    [
        # The worked example in the Minimax-H3-Turbo model specs.
        (4, 12.0, [1.0, 0.9730, 0.9231, 0.8000, 0.0]),
        (4, None, [1.0, 0.9730, 0.9231, 0.8000, 0.0]),
        (4, 6.0, [1.0, 0.9474, 0.8571, 0.6667, 0.0]),
    ],
)
def test_turbo_video_schedule_matches_the_published_grid(steps, shift, expected):
    from diffusers import MiniMaxH3Scheduler

    from xfuser.model_executor.models.runner_models.minimax_h3 import _count_steps_as_forwards, _set_video_shift

    scheduler = MiniMaxH3Scheduler(shift=12.0)
    _count_steps_as_forwards(scheduler)
    scheduler.set_shift(1.0)
    _set_video_shift(scheduler, shift)
    scheduler.set_timesteps(steps)

    torch.testing.assert_close(scheduler.sigmas.float(), torch.tensor(expected), atol=1e-4, rtol=0)


TURBO_TEST_WEIGHT = "minimax_h3_fl2v_turbo_8step_v1.0_768p_bf16.safetensors"


def _turbo_model(tmp_path, **options):
    from xfuser.config import xFuserArgs
    from xfuser.model_executor.models.runner_models.minimax_h3 import xFuserMiniMaxH3TurboModel

    return xFuserMiniMaxH3TurboModel(
        xFuserArgs(model="MiniMax-H3-Turbo", input_images=[], output_directory=str(tmp_path), **options)
    )


def _turbo_args(steps, flow_shift):
    return {
        "prompt": "a cat",
        "dataset_path": None,
        "negative_prompt": None,
        "height": 768,
        "width": 1344,
        "num_inference_steps": steps,
        "flow_shift": flow_shift,
    }


def test_turbo_defaults_to_the_studio_adapter(tmp_path):
    from xfuser.model_executor.models.runner_models.minimax_h3 import (
        TURBO_DEFAULT_WEIGHT_NAME,
        TURBO_LORA_REPO,
        xFuserMiniMaxH3Model,
    )

    model = _turbo_model(tmp_path)

    assert (model.config.lora_path, model.config.lora_weight_name) == (TURBO_LORA_REPO, TURBO_DEFAULT_WEIGHT_NAME)
    assert model.settings.model_name == xFuserMiniMaxH3Model.settings.model_name
    assert model.settings.output_name == "minimax_h3_turbo"
    assert (model.default_input_values.num_inference_steps, model.default_input_values.flow_shift) == (8, 6.0)


@pytest.mark.parametrize(
    "weight_name, steps, shift, refused",
    [
        ("minimax_h3_fl2v_turbo_4step_v0.1.safetensors", 4, 12.0, 8),
        ("minimax_h3_fl2v_turbo_8step_v1.0_bf16.safetensors", 8, 12.0, 6),
        ("minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors", 4, 6.0, 8),
        ("minimax_h3_fl2v_turbo_8step_v1.0_768p_bf16.safetensors", 8, 6.0, 4),
    ],
)
def test_turbo_defaults_and_step_counts_follow_the_adapter(tmp_path, weight_name, steps, shift, refused):
    model = _turbo_model(tmp_path, lora_path="lightx2v/Minimax-h3-Turbo", lora_weight_name=weight_name)

    assert (model.default_input_values.num_inference_steps, model.default_input_values.flow_shift) == (steps, shift)
    model._validate_args(_turbo_args(steps, shift))
    with pytest.raises(ValueError, match="distilled for"):
        model._validate_args(_turbo_args(refused, shift))


def test_turbo_8step_544p_adapter_also_samples_in_4_steps(tmp_path):
    model = _turbo_model(
        tmp_path,
        lora_path="lightx2v/Minimax-h3-Turbo",
        lora_weight_name="minimax_h3_fl2v_turbo_8step_v1.0_bf16.safetensors",
    )

    model._validate_args(_turbo_args(4, 12.0))


def test_turbo_unknown_adapter_needs_explicit_steps_and_shift(tmp_path):
    path = _save_lora(tmp_path, {"x.lora_A.default.weight": torch.zeros(1, 1)}, name="my_adapter.safetensors")
    model = _turbo_model(tmp_path, lora_path=path)

    assert model.default_input_values.num_inference_steps is None
    with pytest.raises(ValueError, match="no published Turbo settings"):
        model._validate_args(_turbo_args(None, None))
    model._validate_args(_turbo_args(6, 9.0))


@pytest.mark.parametrize(
    "options, match",
    [
        ({"lora_weight_name": "minimax_h3_ref2v_turbo_4step_v0.1_bf16.safetensors"}, "Ref2VA adapter"),
        ({"lora_weight_name": None}, "lora_weight_name is required"),
        ({"lora_weight_name": TURBO_TEST_WEIGHT, "fully_shard_degree": 2}, "without FSDP"),
    ],
)
def test_turbo_refuses_adapters_and_layouts_it_cannot_run(tmp_path, options, match):
    with pytest.raises(ValueError, match=match):
        _turbo_model(tmp_path, lora_path="lightx2v/Minimax-h3-Turbo", **options)


def test_turbo_accepts_fp8_gemms_but_the_base_runners_refuse_lora(tmp_path, monkeypatch):
    from xfuser.config import xFuserArgs
    from xfuser.model_executor.models.runner_models.minimax_h3 import xFuserFastH3V2Model, xFuserMiniMaxH3Model

    # The base runners log their default attention backend, which reads RANK/WORLD_SIZE.
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    model = _turbo_model(tmp_path, gemm_quantization="fp8")
    assert not model.config.gemm_quantization_spec.is_pure("none")
    for runner in (xFuserMiniMaxH3Model, xFuserFastH3V2Model):
        with pytest.raises(ValueError, match="does not support startup LoRA"):
            runner(xFuserArgs(lora_path="adapter", input_images=[], output_directory=str(tmp_path)))


def test_minimax_h3_fp4_quantization_policy():
    from xfuser.model_executor.models.runner_models.minimax_h3 import (
        xFuserMiniMaxH3Model,
        xFuserMiniMaxH3Ref2VAModel,
    )

    expected = ("attn.to_out.0", "ff.net.2", "adaln_proj.linear")
    assert xFuserMiniMaxH3Model.settings.fp8_precision_override_suffixes == expected
    assert xFuserMiniMaxH3Ref2VAModel.settings.fp8_precision_override_suffixes == expected


def test_vdn_supported_attn_backends_exclude_unservable_paths():
    from xfuser.core.attention.spec import AttentionBackendType
    from xfuser.model_executor.models.runner_models.minimax_h3 import (
        xFuserMiniMaxH3Model,
        xFuserVDNMiniMaxH3Model,
    )

    supported = xFuserVDNMiniMaxH3Model.supported_attn_backends
    # VDN only serves the three window kernels, a strict subset of base H3's set.
    assert supported == frozenset(
        {AttentionBackendType.SDPA, AttentionBackendType.AITER, AttentionBackendType.AITER_FP8}
    )
    assert supported < xFuserMiniMaxH3Model.supported_attn_backends
    assert AttentionBackendType.CUDNN not in supported
    assert AttentionBackendType.NVTE_FP8 not in supported


def test_vdn_window_backend_derived_from_attention_backend():
    from types import SimpleNamespace

    from xfuser.model_executor.models.runner_models.minimax_h3 import (
        xFuserVDNMiniMaxH3Model,
    )

    resolve = xFuserVDNMiniMaxH3Model._resolve_window_backend

    def model(attention_backend=None, vdn_window_backend=None):
        return SimpleNamespace(
            config=SimpleNamespace(attention_backend=attention_backend, vdn_window_backend=vdn_window_backend)
        )

    assert resolve(model(attention_backend="SDPA")) == "sdpa"
    assert resolve(model(attention_backend="AITER")) == "aiter_ck"
    assert resolve(model(attention_backend="AITER_FP8")) == "aiter_fp8"
    # an explicit --vdn_window_backend wins over the derived default
    assert resolve(model(attention_backend="SDPA", vdn_window_backend="aiter")) == "aiter"
    # no backend set -> SDPA default
    assert resolve(model()) == "sdpa"


def test_vdn_step_count_runs_the_distilled_9_point_grid(monkeypatch):
    from types import SimpleNamespace

    from xfuser.model_executor.models.runner_models.minimax_h3 import (
        xFuserMiniMaxH3Model,
        xFuserVDNMiniMaxH3Model,
        _count_steps_as_forwards,
    )

    assert xFuserVDNMiniMaxH3Model.default_input_values.num_inference_steps == 8

    monkeypatch.setattr(xFuserMiniMaxH3Model, "_validate_args", lambda self, input_args: None)
    model = SimpleNamespace()
    xFuserVDNMiniMaxH3Model._validate_args(model, {"num_inference_steps": 8})
    with pytest.raises(ValueError, match="8 transformer forwards"):
        xFuserVDNMiniMaxH3Model._validate_args(model, {"num_inference_steps": 9})

    grid_points = []
    scheduler = SimpleNamespace(set_timesteps=lambda n, device=None: grid_points.append(n))
    _count_steps_as_forwards(scheduler)
    scheduler.set_timesteps(xFuserVDNMiniMaxH3Model.default_input_values.num_inference_steps)
    assert grid_points == [9]


def test_vdn_load_leaves_the_fused_transformer_in_eval_mode(monkeypatch):
    import diffusers

    from xfuser.model_executor.models.runner_models import minimax_h3
    from xfuser.model_executor.models.runner_models.minimax_h3 import xFuserVDNMiniMaxH3Model
    from xfuser.model_executor.models.transformers.transformer_minimax_h3 import (
        xFuserMiniMaxH3Transformer3DWrapper,
    )
    from xfuser.model_executor.models.transformers.vdn_minimax_h3 import transformer as vdn_transformer
    from xfuser.model_executor.models.transformers.vdn_minimax_h3 import window_softmax

    class FakePipe:
        def __init__(self):
            self.text_encoder = SimpleNamespace(lm_head=object())
            self.scheduler = SimpleNamespace(set_timesteps=lambda n, device=None: None)
            self.audio_scheduler = SimpleNamespace(set_timesteps=lambda n, device=None: None)

        def update_components(self, transformer):
            self.transformer = transformer

        def load_components(self, dtype):
            pass

    # from_pretrained returns modules in eval mode; the tiny base wrapper fuses QKV like the VDN one.
    loaded = xFuserMiniMaxH3Transformer3DWrapper(**_tiny_config()).eval()
    monkeypatch.setattr(diffusers.ModularPipeline, "from_pretrained", lambda *args, **kwargs: FakePipe())
    monkeypatch.setattr(
        vdn_transformer.xFuserMiniMaxH3VDNTransformer3DWrapper,
        "from_pretrained",
        lambda *args, **kwargs: loaded,
    )
    monkeypatch.setattr(window_softmax, "set_window_backend", lambda name: None)
    monkeypatch.setattr(minimax_h3, "log", lambda message: None)
    model = SimpleNamespace(
        settings=SimpleNamespace(model_name="OpenVDN/vdn-minimax-h3"),
        config=SimpleNamespace(task="t2va", text_encoder_tp_degree=2, attention_backend="SDPA"),
        _vdn_transformer_dir=lambda: "/unused",
        _resolve_window_backend=lambda: "sdpa",
        _parallelize_text_encoder=lambda text_encoder: None,
    )

    pipe = xFuserVDNMiniMaxH3Model._load_model(model)

    assert pipe.transformer is loaded
    assert any(name.endswith("to_qkv") for name, _ in loaded.named_modules())
    assert [name for name, module in loaded.named_modules() if module.training] == []


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
            expected_tiles[tile_index] = sequential.decoder(sequential.post_quant_conv(tile))
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
            expected_tiles[tile_index] = sequential.decoder(sequential.post_quant_conv(tile))
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


def _attach_online_loader(model, monkeypatch):
    """Give a bare runner the checkpoint identity its loader would build online."""
    from xfuser.model_executor.models.runner_models.loading.meta_load import (
        ModelLoader,
    )

    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_OFFLINE", False)
    model.loader = object.__new__(ModelLoader)
    model.loader.model = model


class _FakeMiniMaxPipe:
    def __init__(self):
        self.text_encoder = SimpleNamespace(lm_head=object())
        self.loaded_dtype = None
        self.scheduler = SimpleNamespace(set_timesteps=lambda *args, **kwargs: None)
        self.audio_scheduler = SimpleNamespace(set_timesteps=lambda *args, **kwargs: None)

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
    model.config = SimpleNamespace(task=task, text_encoder_tp_degree=1, lora_path=None)
    model._parallelize_text_encoder = lambda text_encoder: None

    _attach_online_loader(model, monkeypatch)

    actual = model._load_model()

    assert actual is pipe
    assert workflows == [(model.settings.model_name, expected_workflow)]
    assert pipe.transformer is transformer
    assert transformer.qkv_fused
    assert pipe.loaded_dtype == torch.bfloat16
    assert pipe.text_encoder.lm_head is None


def test_minimax_h3_turbo_merges_the_lora_before_fusing_qkv(monkeypatch):
    from diffusers import ModularPipeline

    from xfuser.model_executor.models.runner_models import minimax_h3
    from xfuser.model_executor.models.runner_models.minimax_h3 import xFuserMiniMaxH3TurboModel
    from xfuser.model_executor.models.transformers.transformer_minimax_h3 import (
        xFuserMiniMaxH3Transformer3DWrapper,
    )

    pipe = _FakeMiniMaxPipe()
    transformer = _FakeTransformer()
    merges = []
    monkeypatch.setattr(ModularPipeline, "from_pretrained", lambda model_name, workflow: pipe)
    monkeypatch.setattr(xFuserMiniMaxH3Transformer3DWrapper, "from_pretrained", lambda *args, **kwargs: transformer)
    monkeypatch.setattr(minimax_h3, "_patch_minimax_h3_text_encoder_broadcast", lambda: None)
    monkeypatch.setattr(minimax_h3, "_resolve_lora_file", lambda path, name: f"/cache/{name}")
    monkeypatch.setattr(
        minimax_h3,
        "_merge_lora_into_transformer",
        lambda target, path, scale: merges.append((target, path, scale, target.qkv_fused)),
    )
    monkeypatch.setattr(minimax_h3, "log", lambda message: None)

    model = object.__new__(xFuserMiniMaxH3TurboModel)
    model.config = SimpleNamespace(
        task="i2va",
        text_encoder_tp_degree=1,
        lora_path="lightx2v/Minimax-h3-Turbo",
        lora_weight_name="adapter.safetensors",
        lora_scale=0.5,
    )
    model._parallelize_text_encoder = lambda text_encoder: None
    _attach_online_loader(model, monkeypatch)

    model._load_model()

    assert merges == [(transformer, "/cache/adapter.safetensors", 0.5, False)]
    assert transformer.qkv_fused


def test_fasth3_loads_published_checkpoint(monkeypatch):
    from diffusers import ModularPipeline

    from xfuser.model_executor.models.runner_models import minimax_h3
    from xfuser.model_executor.models.runner_models.minimax_h3 import (
        FASTH3_V1_DATAFREE_MODEL_ID,
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
    model.config = SimpleNamespace(task="t2va", text_encoder_tp_degree=1, lora_path=None)
    model._parallelize_text_encoder = lambda text_encoder: None

    _attach_online_loader(model, monkeypatch)

    actual = model._load_model()

    assert actual is pipe
    assert pipeline_loads == [(FASTH3_V1_DATAFREE_MODEL_ID, "t2va")]
    assert transformer_loads == [
        (
            FASTH3_V1_DATAFREE_MODEL_ID,
            {
                "subfolder": "transformer",
                "dtype": torch.bfloat16,
                "local_files_only": False,
                "enable_fasth3_vsa": True,
                "attention_backend": None,
            },
        )
    ]
    assert pipe.transformer is transformer


def test_fasth3_dense_loads_published_checkpoint(monkeypatch):
    from diffusers import ModularPipeline

    from xfuser.core.attention.spec import AttentionBackendType
    from xfuser.model_executor.models.runner_models import minimax_h3
    from xfuser.model_executor.models.runner_models.minimax_h3 import (
        FASTH3_V1_DENSE_DATAFREE_MODEL_ID,
        xFuserFastH3DenseModel,
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

    model = object.__new__(xFuserFastH3DenseModel)
    model.config = SimpleNamespace(
        task="t2va",
        text_encoder_tp_degree=1,
        attention_backend="AITER",
        lora_path=None,
    )
    model._parallelize_text_encoder = lambda text_encoder: None

    _attach_online_loader(model, monkeypatch)

    actual = model._load_model()

    assert actual is pipe
    assert pipeline_loads == [(FASTH3_V1_DENSE_DATAFREE_MODEL_ID, "t2va")]
    assert transformer_loads == [
        (
            FASTH3_V1_DENSE_DATAFREE_MODEL_ID,
            {
                "subfolder": "transformer",
                "dtype": torch.bfloat16,
                "local_files_only": False,
                "enable_fasth3_vsa": False,
                "attention_backend": AttentionBackendType.AITER,
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

    _attach_online_loader(model, monkeypatch)

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
    vae = SimpleNamespace(compile_repeated_blocks=lambda **kwargs: compile_calls.append(kwargs))
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
    assert [call["num_inference_steps"] for call in calls] == [4, 4]


def test_minimax_h3_ref2va_uses_typed_image_references(monkeypatch):
    from diffusers.modular_pipelines.minimax_h3 import MiniMaxH3ImageReference

    from xfuser.model_executor.models.runner_models.minimax_h3 import (
        xFuserMiniMaxH3Ref2VAModel,
    )
    from xfuser.model_executor.models.runner_models import minimax_h3

    generator = torch.Generator
    monkeypatch.setattr(
        minimax_h3.torch,
        "Generator",
        lambda device=None: generator(),
    )

    captured = {}

    def fake_pipe(**kwargs):
        captured.update(kwargs)
        return {
            "videos": torch.zeros((1, 1, 3, 1, 1), dtype=torch.float32),
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
    assert captured["output_type"] == "pt"


def test_minimax_h3_decoded_videos_convert_to_uint8_frames_on_device():
    from xfuser.model_executor.models.runner_models.minimax_h3 import (
        xFuserMiniMaxH3Model,
    )

    # [B, F, C, H, W] denormalized to [0, 1], as postprocess_video leaves it.
    videos = torch.tensor([0.0, 0.5, 1.0]).view(1, 1, 3, 1, 1).expand(1, 2, 3, 4, 5)

    frames = xFuserMiniMaxH3Model._decoded_videos_to_frames(videos)

    assert isinstance(frames, list) and len(frames) == 1
    frame = frames[0]
    assert frame.shape == (2, 4, 5, 3)
    assert frame.dtype == torch.uint8
    assert frame.is_contiguous()
    assert frame[..., 0].unique().tolist() == [0]
    assert frame[..., 1].unique().tolist() == [128]
    assert frame[..., 2].unique().tolist() == [255]


def test_minimax_h3_decoded_videos_do_not_rescale_uint8():
    from xfuser.model_executor.models.runner_models.minimax_h3 import (
        xFuserMiniMaxH3Model,
    )

    videos = torch.full((1, 2, 3, 4, 5), 7, dtype=torch.uint8)

    frames = xFuserMiniMaxH3Model._decoded_videos_to_frames(videos)

    assert frames[0].shape == (2, 4, 5, 3)
    assert frames[0].unique().tolist() == [7]


def test_minimax_h3_supports_hybrid_attention_capability():
    from xfuser.model_executor.models.runner_models.minimax_h3 import (
        xFuserMiniMaxH3Model,
        xFuserMiniMaxH3Ref2VAModel,
    )

    assert xFuserMiniMaxH3Model.capabilities.use_hybrid_attn_schedule
    assert xFuserMiniMaxH3Ref2VAModel.capabilities.use_hybrid_attn_schedule
    assert xFuserMiniMaxH3Model.default_input_values.num_hybrid_attn_high_precision_steps == 5


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


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="VSA-H3 needs a GPU")
@pytest.mark.parametrize("vsa_backend", ["FLEX_VSA_H3", "TRITON_VSA_H3"])
def test_fasth3_vsa_transformer_compiles_fullgraph(monkeypatch, vsa_backend):
    """Both VSA-H3 kernels must trace without graph breaks under fullgraph."""
    import torch._dynamo

    from xfuser.core.attention.backends.vsa_h3 import (
        triton_kernel as vsa_h3_triton,
    )

    if vsa_backend == "TRITON_VSA_H3" and not vsa_h3_triton.is_available():
        pytest.skip("Triton is unavailable")

    from xfuser.core.attention.spec import AttentionBackendType
    from xfuser.core.attention.backends.vsa_h3.attention import (
        build_h3_vsa_metadata,
    )
    from xfuser.model_executor.models.transformers import transformer_minimax_h3
    from xfuser.model_executor.models.transformers.transformer_minimax_h3 import (
        xFuserMiniMaxH3Transformer3DWrapper,
    )

    monkeypatch.setattr(transformer_minimax_h3, "get_ulysses_parallel_world_size", lambda: 1)
    monkeypatch.setattr(transformer_minimax_h3, "get_ulysses_parallel_rank", lambda: 0)
    _patch_minimax_runtime_state(monkeypatch)

    device = torch.device("cuda")
    # The token refiner falls back to the dense AITER kernel, which has no
    # head_dim=16 variant, so this case needs a wider head than _tiny_config.
    config = dict(
        _tiny_config(),
        attention_head_dim=64,
        hidden_size=128,
        time_embed_hidden_dim=128,
    )
    wrapper = (
        xFuserMiniMaxH3Transformer3DWrapper(
            **config,
            attention_backend=AttentionBackendType[vsa_backend],
            enable_fasth3_vsa=True,
        )
        .eval()
        .to(device=device, dtype=torch.bfloat16)
    )
    assert wrapper.use_vsa_h3
    # _tiny_inputs draws from a CPU generator, so build on CPU and move. The
    # dense fallback kernel is bf16/fp16 only, hence the cast.
    inputs = {name: tensor.to(device) for name, tensor in _tiny_inputs(torch.device("cpu")).items()}
    for name in ("hidden_states", "audio_hidden_states", "encoder_hidden_states"):
        inputs[name] = inputs[name].to(torch.bfloat16)

    with torch.no_grad():
        eager = wrapper(**inputs)

    # The runner primes the geometry outside the compiled region; do the same.
    wrapper.prepare_vsa_h3_metadata(
        inputs["position_ids"],
        inputs["video_indices"],
        inputs["audio_indices"],
        inputs["text_indices"],
    )
    torch._dynamo.reset()
    torch._dynamo.utils.counters.clear()
    misses = build_h3_vsa_metadata.cache_info().misses
    compiled = torch.compile(wrapper.forward, fullgraph=True)
    with torch.no_grad():
        compiled_out = compiled(**inputs)
        graphs = torch._dynamo.utils.counters["stats"]["unique_graphs"]
        # A second call must not recompile: the attention_kwargs key set is
        # stable and the geometry cache stays warm.
        compiled(**inputs)
    assert torch._dynamo.utils.counters["stats"]["unique_graphs"] == graphs
    # The untraceable geometry recovery must stay out of the graph entirely.
    assert build_h3_vsa_metadata.cache_info().misses == misses

    # A smoke check that the graph computes the model, not an equivalence test:
    # Inductor fuses reductions across the whole transformer in a different
    # order than eager, so a few bf16 elements land a handful of ULP apart. What
    # the sparse branch actually computes is pinned against a dense reference in
    # tests/unit/core/test_vsa_h3_attention.py.
    torch.testing.assert_close(compiled_out.sample, eager.sample, rtol=2e-2, atol=6e-2)
    torch.testing.assert_close(compiled_out.audio_sample, eager.audio_sample, rtol=2e-2, atol=6e-2)


def _vsa_geometry_inputs(text_tokens, audio_tokens, video_shape):
    """Packed T2VA indices and position_ids for one video grid."""
    frames, height, width = video_shape
    prefix = text_tokens + audio_tokens
    sequence_length = prefix + frames * height * width

    position_ids = torch.zeros(sequence_length, 3, dtype=torch.float32)
    grid = torch.cartesian_prod(
        torch.arange(frames, dtype=torch.float32),
        torch.arange(height, dtype=torch.float32),
        torch.arange(width, dtype=torch.float32),
    )
    position_ids[prefix:] = grid
    return dict(
        position_ids=position_ids,
        text_indices=torch.arange(text_tokens),
        audio_indices=torch.arange(text_tokens, prefix),
        video_indices=torch.arange(prefix, sequence_length),
    )


def _vsa_geometry_transformer(monkeypatch):
    from xfuser.core.attention.spec import AttentionBackendType
    from xfuser.model_executor.models.transformers import transformer_minimax_h3
    from xfuser.model_executor.models.transformers.transformer_minimax_h3 import (
        xFuserMiniMaxH3Transformer3DWrapper,
    )

    monkeypatch.setattr(transformer_minimax_h3, "get_ulysses_parallel_world_size", lambda: 1)
    monkeypatch.setattr(transformer_minimax_h3, "get_ulysses_parallel_rank", lambda: 0)
    _patch_minimax_runtime_state(monkeypatch)
    return xFuserMiniMaxH3Transformer3DWrapper(
        **_tiny_config(),
        attention_backend=AttentionBackendType.FLEX_VSA_H3,
        enable_fasth3_vsa=True,
    ).eval()


def test_fasth3_vsa_geometry_key_separates_transposed_video_grids(monkeypatch):
    """Two grids with the same token count must not share a cached geometry."""
    from xfuser.core.attention.backends.vsa_h3.attention import (
        build_h3_vsa_metadata,
    )

    wrapper = _vsa_geometry_transformer(monkeypatch)
    # Both grids split into two video tiles of the same two sizes; what differs
    # is which token lands in which, so a sizes-only key cannot tell them apart.
    upright = _vsa_geometry_inputs(4, 8, (2, 5, 4))
    rotated = _vsa_geometry_inputs(4, 8, (2, 4, 5))
    assert upright["position_ids"].shape == rotated["position_ids"].shape, (
        "the two grids must pack the same token count for this to test anything"
    )

    wrapper.prepare_vsa_h3_metadata(**upright)
    first = wrapper._vsa_h3_metadata
    misses = build_h3_vsa_metadata.cache_info().misses

    wrapper.prepare_vsa_h3_metadata(**rotated)
    second = wrapper._vsa_h3_metadata
    assert build_h3_vsa_metadata.cache_info().misses == misses + 1
    assert torch.equal(first.variable_block_sizes, second.variable_block_sizes)
    assert not torch.equal(first.packed_token_tile, second.packed_token_tile)


def test_fasth3_vsa_tracing_rejects_a_stale_geometry(monkeypatch):
    """A geometry left over from another resolution must not be traced against."""
    wrapper = _vsa_geometry_transformer(monkeypatch)
    wrapper.prepare_vsa_h3_metadata(**_vsa_geometry_inputs(4, 8, (2, 3, 4)))
    longer = _vsa_geometry_inputs(4, 8, (2, 3, 8))

    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)

    def _no_device_reads(*args, **kwargs):
        raise AssertionError("tracing must not read tensor values")

    # Same geometry: tracing is fine, and the guard must stay off device values.
    with monkeypatch.context() as traced:
        traced.setattr(torch.Tensor, "tolist", _no_device_reads)
        traced.setattr(torch, "equal", _no_device_reads)
        wrapper.prepare_vsa_h3_metadata(**_vsa_geometry_inputs(4, 8, (2, 3, 4)))

    with pytest.raises(RuntimeError, match="was not primed"):
        wrapper.prepare_vsa_h3_metadata(**longer)

    wrapper._vsa_h3_metadata = None
    with pytest.raises(RuntimeError, match="was not primed"):
        wrapper.prepare_vsa_h3_metadata(**longer)


class _WrapperTransformer:
    """Stand-in with the real forward's parameter names and order."""

    def __init__(self, use_vsa_h3):
        self.use_vsa_h3 = use_vsa_h3
        self.primed = []

    def prepare_vsa_h3_metadata(self, position_ids, video, audio, text):
        self.primed.append(position_ids)

    def forward(
        self,
        hidden_states,
        timestep,
        position_ids,
        video_indices,
        audio_indices,
        text_indices,
    ):
        return hidden_states


def test_fasth3_compile_wrapper_primes_geometry_and_marks_the_timestep(
    monkeypatch,
):
    """Both jobs the compiled region cannot do itself, in one wrapper."""
    from xfuser.model_executor.models.runner_models import minimax_h3

    marked = []
    monkeypatch.setattr(minimax_h3, "_mark_dynamic_timestep", marked.append)

    transformer = _WrapperTransformer(use_vsa_h3=True)
    compiled_calls = []
    forward = minimax_h3._wrap_compiled_forward(
        transformer,
        transformer.forward,
        lambda *args, **kwargs: compiled_calls.append((args, kwargs)),
    )

    # Timestep positionally, the rest by keyword: the wrapper binds the
    # signature, so where an argument came from does not matter.
    timestep = torch.tensor([0.7, 0.3])
    forward(
        1,
        timestep,
        position_ids=2,
        video_indices=3,
        audio_indices=4,
        text_indices=5,
    )

    assert transformer.primed == [2]
    assert marked == [timestep]
    assert len(compiled_calls) == 1
    assert inspect.signature(forward) == inspect.signature(transformer.forward)


def test_fasth3_compile_wrapper_marks_the_timestep_for_other_backends(
    monkeypatch,
):
    """Non-VSA-H3 backends skip the priming but keep the timestep marking."""
    from xfuser.model_executor.models.runner_models import minimax_h3

    marked = []
    monkeypatch.setattr(minimax_h3, "_mark_dynamic_timestep", marked.append)

    transformer = _WrapperTransformer(use_vsa_h3=False)
    forward = minimax_h3._wrap_compiled_forward(transformer, transformer.forward, lambda *args, **kwargs: None)

    timestep = torch.tensor([0.7])
    forward(1, timestep, 2, 3, 4, 5)

    assert transformer.primed == []
    assert marked == [timestep]


def test_fasth3_compile_wrapper_ignores_a_scalar_timestep(monkeypatch):
    """Nothing to mark when the timestep has no length to vary."""
    from xfuser.model_executor.models.runner_models import minimax_h3

    marked = []
    monkeypatch.setattr(minimax_h3, "_mark_dynamic_timestep", marked.append)

    transformer = _WrapperTransformer(use_vsa_h3=False)
    forward = minimax_h3._wrap_compiled_forward(transformer, transformer.forward, lambda *args, **kwargs: None)

    forward(1, torch.tensor(0.7), 2, 3, 4, 5)

    assert marked == []


def test_fasth3_accepts_torch_compile(monkeypatch):
    """--use_torch_compile is no longer rejected for the VSA-H3 backends."""
    from xfuser.model_executor.models.runner_models.minimax_h3 import (
        xFuserFastH3Model,
    )

    config = SimpleNamespace(
        attention_backend="TRITON_VSA_H3",
        use_hybrid_attn_schedule=False,
        use_torch_compile=True,
    )
    monkeypatch.setattr(xFuserFastH3Model.__mro__[1], "_validate_config", lambda self, config: None)
    xFuserFastH3Model._validate_config(object.__new__(xFuserFastH3Model), config)
