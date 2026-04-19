"""Runtime FP4 quantization helpers for the FLASH_4_FP4 attention backend.

Uses flashinfer's nvfp4_quantize to convert BF16/FP16 Q/K tensors to NVFP4
(Float4E2M1FN) with block-scaled scale factors in the MMA tile layout
expected by the hao-ai-lab/flash-attention-fp4 kernel.

NOTE: cutlass-dsl's cute_tensor_like + convert_cute_tensor also works for
quantization and produces slightly faster kernel execution (~1-2%), but it
relies on @cute.jit which JIT-compiles a new CUDA kernel for every distinct
tensor shape. This makes it unsuitable for E2E inference where attention is
called hundreds of times during warmup/compilation. flashinfer's nvfp4_quantize
is a pre-compiled kernel with no JIT overhead.

Requires: flashinfer-python (pip install flashinfer-python)
Requires: CUTE_DSL_ENABLE_TVM_FFI=1 (set automatically in envs.py)
"""

import math

import torch
from flashinfer.quantization import nvfp4_quantize, SfLayout


def quantize_qk_to_fp4(
    tensor: torch.Tensor,
    sf_vec_size: int = 16,
):
    """Quantize a BF16/FP16 Q or K tensor to NVFP4 for the Flash Attention FP4 kernel.

    Uses flashinfer's nvfp4_quantize with layout_128x4 scale factor layout,
    then reshapes the packed FP4 data and permutes scale factors into the
    (32, 4, rest_m, 4, rest_k, nheads, batch) MMA tile layout expected by
    the FlashAttentionForwardSm100FP4 kernel.

    Args:
        tensor: Input tensor of shape (batch, seqlen, nheads, headdim) in BF16/FP16.
        sf_vec_size: Number of elements per scale factor block (default 16).

    Returns:
        (fp4_tensor, sf_tensor) -- fp4_tensor is torch.float4_e2m1fn_x2
        with shape (batch, seqlen, nheads, headdim//2), and sf_tensor is
        the scale factors in MMA tile layout.
    """
    batch, seqlen, nheads, headdim = tensor.shape

    global_sf = torch.ones(1, device=tensor.device, dtype=torch.float32)
    fp4_data, sf_data = nvfp4_quantize(
        tensor.reshape(batch * seqlen, nheads * headdim),
        global_sf,
        sfLayout=SfLayout.layout_128x4,
        do_shuffle=False,
    )

    fp4 = (
        fp4_data
        .reshape(batch, seqlen, nheads, headdim // 2)
        .view(torch.int8)
        .view(torch.float4_e2m1fn_x2)
    )

    # nvfp4_quantize pads M to next multiple of 128 internally
    rest_m = math.ceil(seqlen / 128)
    sf_kph = headdim // sf_vec_size
    rest_k = sf_kph // 4
    total_m = batch * rest_m
    total_k = (nheads * sf_kph) // 4

    sf = (
        sf_data
        .reshape(total_m, total_k, 32, 4, 4)
        .reshape(batch, rest_m, nheads, rest_k, 32, 4, 4)
        .permute(0, 2, 1, 3, 4, 5, 6)
        .contiguous()
        .permute(4, 5, 2, 6, 3, 1, 0)
    )

    return fp4, sf
