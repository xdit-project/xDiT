"""A linear that switches per diffusion step between two precisions."""

import torch
from torch import nn

from xfuser.core.distributed import get_runtime_state


class xFuserHybridLinear(nn.Module):
    """Holds two linears and picks one per step.

    Nothing here is specific to either precision: the caller that knows the
    run's formats builds both layers and hands them over. It was named for
    MXFP4 only because MXFP4 was the first low-precision half it was given.
    """

    def __init__(
        self,
        high_precision_linear: nn.Module,
        low_precision_linear: nn.Module,
    ) -> None:
        super().__init__()
        self.high_precision_linear = high_precision_linear
        self.low_precision_linear = low_precision_linear

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        runtime_state = get_runtime_state()
        use_high_precision = getattr(runtime_state, "use_high_precision_gemm", True)
        if use_high_precision:
            return self.high_precision_linear(input)
        return self.low_precision_linear(input)

    def extra_repr(self):
        return "hybrid_gemm_schedule=True"
