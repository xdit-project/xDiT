import pytest
import torch
from diffusers.models.transformers.transformer_wan import WanTransformer3DModel

from xfuser.model_executor.models.runner_models.loading.contracts import (
    QuantizationBackend,
    QuantizationFormat,
)
from xfuser.model_executor.models.runner_models.loading.format_backends import (
    AiterInt8BackendAdapter,
    _probe_aiter_int8_apis,
)


def _aiter_int8_available() -> tuple[bool, str | None]:
    if not torch.cuda.is_available() or torch.version.hip is None:
        return False, "requires ROCm"
    return _probe_aiter_int8_apis()


_AVAILABLE, _REASON = _aiter_int8_available()
pytestmark = pytest.mark.skipif(not _AVAILABLE, reason=f"AITER INT8 GEMMs unavailable: {_REASON}")


@torch.no_grad()
def test_tiny_wan_transformer_runs_with_aiter_int8_blocks():
    config = {
        "patch_size": (1, 2, 2),
        "num_attention_heads": 4,
        "attention_head_dim": 128,
        "in_channels": 4,
        "out_channels": 4,
        "text_dim": 512,
        "freq_dim": 256,
        "ffn_dim": 512,
        "num_layers": 1,
        "cross_attn_norm": True,
        "qk_norm": "rms_norm_across_heads",
        "eps": 1e-6,
        "rope_max_seq_len": 32,
    }
    torch.manual_seed(3)
    reference = WanTransformer3DModel(**config).eval().to(device="cuda", dtype=torch.bfloat16)
    quantized = WanTransformer3DModel(**config).eval().to(device="cuda", dtype=torch.bfloat16)
    quantized.load_state_dict(reference.state_dict())
    adapter = AiterInt8BackendAdapter(
        backend=QuantizationBackend.AITER,
        format_=QuantizationFormat.INT8,
    )
    assert adapter.convert_module(quantized.blocks, device="cuda") == 10

    hidden_states = torch.randn(
        (1, 4, 3, 4, 6),
        device="cuda",
        dtype=torch.bfloat16,
    )
    encoder_hidden_states = torch.randn(
        (1, 7, 512),
        device="cuda",
        dtype=torch.bfloat16,
    )
    timestep = torch.tensor([500], device="cuda")
    expected = reference(
        hidden_states,
        timestep,
        encoder_hidden_states,
        return_dict=False,
    )[0]
    actual = quantized(
        hidden_states,
        timestep,
        encoder_hidden_states,
        return_dict=False,
    )[0]

    assert actual.shape == expected.shape
    assert torch.isfinite(actual.float()).all()
    cosine = torch.nn.functional.cosine_similarity(
        actual.float().flatten(),
        expected.float().flatten(),
        dim=0,
    )
    assert cosine > 0.999
