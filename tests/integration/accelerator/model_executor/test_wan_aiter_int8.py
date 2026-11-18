import copy

import pytest
import torch
from diffusers.models.transformers.transformer_wan import WanTransformer3DModel

from xfuser.model_executor.models.runner_models.loading.contracts import QuantizationBackend, QuantizationFormat
from xfuser.model_executor.models.runner_models.loading.format_backends import (
    AiterInt8BackendAdapter,
    _probe_aiter_int8_apis,
)

_AVAILABLE, _REASON = (
    _probe_aiter_int8_apis() if torch.cuda.is_available() and torch.version.hip else (False, "requires ROCm")
)
pytestmark = pytest.mark.skipif(not _AVAILABLE, reason=f"AITER INT8 GEMMs unavailable: {_REASON}")


@torch.no_grad()
def test_tiny_wan_transformer_runs_with_aiter_int8_blocks():
    torch.manual_seed(3)
    reference = WanTransformer3DModel(
        patch_size=(1, 2, 2),
        num_attention_heads=4,
        attention_head_dim=128,
        in_channels=4,
        out_channels=4,
        text_dim=512,
        freq_dim=256,
        ffn_dim=512,
        num_layers=1,
        cross_attn_norm=True,
        qk_norm="rms_norm_across_heads",
        eps=1e-6,
        rope_max_seq_len=32,
    )
    reference = reference.eval().to(device="cuda", dtype=torch.bfloat16)
    quantized = copy.deepcopy(reference)
    adapter = AiterInt8BackendAdapter(backend=QuantizationBackend.AITER, format_=QuantizationFormat.INT8)
    assert adapter.convert_module(quantized.blocks, device="cuda") == 10

    inputs = (
        torch.randn((1, 4, 3, 4, 6), device="cuda", dtype=torch.bfloat16),
        torch.tensor([500], device="cuda"),
        torch.randn((1, 7, 512), device="cuda", dtype=torch.bfloat16),
    )
    expected = reference(*inputs, return_dict=False)[0]
    actual = quantized(*inputs, return_dict=False)[0]
    cosine = torch.nn.functional.cosine_similarity(actual.float().flatten(), expected.float().flatten(), dim=0)
    assert cosine > 0.999
