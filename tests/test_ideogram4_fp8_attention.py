import inspect
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F


def _require_gfx950_aiter():
    if not torch.cuda.is_available() or torch.version.hip is None:
        pytest.skip("AITER FP8 attention requires a ROCm GPU.")

    arch_name = getattr(torch.cuda.get_device_properties(0), "gcnArchName", "")
    if "gfx950" not in arch_name:
        pytest.skip(
            f"Ideogram 4 HD256 FP8 validation requires gfx950, got {arch_name}."
        )

    try:
        import aiter
    except ImportError:
        pytest.skip("AITER is not installed.")

    fp8_attention = getattr(aiter, "flash_attn_fp8_pertensor_func", None)
    if fp8_attention is None:
        pytest.skip("AITER does not expose flash_attn_fp8_pertensor_func.")

    parameters = inspect.signature(fp8_attention).parameters
    required_descales = {"q_descale", "k_descale", "v_descale"}
    if not required_descales.issubset(parameters):
        pytest.skip("AITER FP8 attention does not expose the per-tensor descale ABI.")

    aiter_root = Path(aiter.__file__).resolve().parent.parent
    hd256_kernel = (
        aiter_root
        / "hsa"
        / "gfx950"
        / "fmha_v3_fwd"
        / "fwd_hd256_fp8.co"
    )
    if not hd256_kernel.exists():
        pytest.skip(
            "AITER does not include the gfx950 HD256 FP8 FMHA kernel from PR 3732."
        )


@pytest.mark.parametrize("sequence_length", [256, 1024])
def test_ideogram4_aiter_fp8_attention_hd256(sequence_length):
    _require_gfx950_aiter()

    from xfuser.core.distributed.attention_backend import _aiter_fp8_attn_call

    torch.manual_seed(1234)
    device = torch.device("cuda")
    shape = (1, 18, sequence_length, 256)
    query = torch.randn(shape, device=device, dtype=torch.bfloat16)
    key = torch.randn(shape, device=device, dtype=torch.bfloat16)
    value = torch.randn(shape, device=device, dtype=torch.bfloat16)

    with torch.no_grad():
        reference = F.scaled_dot_product_attention(query, key, value)
        output, _ = _aiter_fp8_attn_call(
            query,
            key,
            value,
            dropout_p=0.0,
            is_causal=False,
        )

    output_float = output.float()
    reference_float = reference.float()
    relative_l2 = (
        torch.linalg.vector_norm(output_float - reference_float)
        / torch.linalg.vector_norm(reference_float)
    ).item()
    cosine_similarity = F.cosine_similarity(
        output_float.flatten(),
        reference_float.flatten(),
        dim=0,
    ).item()
    print(
        f"sequence_length={sequence_length} "
        f"relative_l2={relative_l2:.6f} "
        f"cosine_similarity={cosine_similarity:.6f}"
    )

    assert output.shape == reference.shape
    assert torch.isfinite(output).all()
    assert relative_l2 < 0.08
    assert cosine_similarity > 0.995


def test_ideogram4_aiter_fp8_attention_hd256_compiles_fullgraph():
    _require_gfx950_aiter()

    from xfuser.core.distributed.attention_backend import _aiter_fp8_attn_call

    torch.manual_seed(1234)
    shape = (1, 18, 256, 256)
    query = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    key = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    value = torch.randn(shape, device="cuda", dtype=torch.bfloat16)

    def attention(query, key, value):
        return _aiter_fp8_attn_call(
            query,
            key,
            value,
            dropout_p=0.0,
            is_causal=False,
        )[0]

    compiled_attention = torch.compile(attention, fullgraph=True)
    output = compiled_attention(query, key, value)
    assert output.shape == query.shape
    assert torch.isfinite(output).all()
