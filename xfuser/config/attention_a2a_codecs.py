"""Activation codecs of the Attention A2A output hop.

Each codec names the AITER attention-output A2A layout that an output-projection
GEMM consumes. The transport, the startup validation, and the quantized linears
share these names.
"""

import torch

E4M3_EXTERNAL_SCALE = "e4m3_external_scale"
MXFP4 = "mxfp4"
MXFP6 = "mxfp6"

# The codec that the output projection of each --gemm_quantization format consumes.
GEMM_FORMAT_OUTPUT_CODECS = {
    "fp8": E4M3_EXTERNAL_SCALE,
    "fp4": MXFP4,
    "fp6": MXFP6,
    "a6w4": MXFP6,
}
OUTPUT_CODECS = tuple(dict.fromkeys(GEMM_FORMAT_OUTPUT_CODECS.values()))


def e4m3_external_scale_from_amax(amax: torch.Tensor) -> torch.Tensor:
    """FP32 descales that TorchAO's dynamic per-tensor FP8 quantization derives from ``amax``.

    ``amax`` must keep the activation dtype: TorchAO clamps and divides in that dtype
    before converting the scale to FP32.
    """
    from xfuser.core.utils.runner_utils import FP8_ACTIVATION_SCALE_FLOOR

    scale = torch.clamp(amax, min=FP8_ACTIVATION_SCALE_FLOOR) / torch.finfo(torch.float8_e4m3fn).max
    return scale.to(torch.float32)
