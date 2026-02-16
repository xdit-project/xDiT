import torch
import torch.nn as nn
import math
import aiter
from aiter.ops.shuffle import shuffle_weight
from typing import Optional


@torch.library.custom_op("mylib::mxfp4_gemm", mutates_args=())
def _mxfp4_gemm(a: torch.Tensor, w_quant: torch.Tensor, w_scale: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    quant_func = aiter.get_hip_quant(aiter.QuantType.per_1x32)
    a_quant, a_scale = quant_func(a, shuffle=True)
    output = aiter.gemm_a4w4(a_quant, w_quant, a_scale, w_scale, bpreshuffle=True, bias=bias)
    return output

@_mxfp4_gemm.register_fake
def _(a: torch.Tensor, w_quant: torch.Tensor, w_scale: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Fake implementation for torch. compile shape inference
    """
    M, _ = a.shape
    N, _ = w_quant.shape
    
    # Return fake tensor with correct shape
    return torch.empty(M, N, dtype=a.dtype, device=a.device)

class xFuserMXFP4Linear(nn.Module):
    """
    Custom Linear layer using MXFP4 GEMM operation
    
    Drop-in replacement for nn.Linear.
    """
    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features

        self.weight_shuffle = None
        self.weight_scale = None
        
        self.weight = nn.Parameter(
            torch.empty((out_features, in_features), **factory_kwargs)
        )
        
        if bias:
            self.bias = nn.Parameter(
                torch.empty(out_features, **factory_kwargs)
            )
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()
        self.mm = self._run_mxfp4_gemm
    
    def reset_parameters(self) -> None:
        """Initialize weights using Kaiming uniform (same as nn.Linear)"""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
    
    def load_and_quantize_weights(self, weights: torch.Tensor, bias: Optional[torch.Tensor] = None) -> None:
        with torch.no_grad():
            self.weight.data.copy_(weights.data)
            if bias is not None:
                self.bias.data.copy_(bias.data)
        self._quantize_weights()

    def _quantize_weights(self) -> None:
        quant_func = aiter.get_hip_quant(aiter.QuantType.per_1x32)
        weight_quant, self.weight_scale = quant_func(self.weight, shuffle=True)
        self.weight_shuffle = shuffle_weight(weight_quant, layout=(16, 16))

        del self.weight
        self.weight = None

    def _run_mxfp4_gemm(self, a: torch.Tensor, w_quant: torch.Tensor, w_scale: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        return torch.ops.mylib.mxfp4_gemm(a, w_quant, w_scale, bias)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass using MXFP4 GEMM
        """

        if self.weight_shuffle is None:
            self._quantize_weights()

        # Save original shape
        original_shape = input.shape
        
        # Flatten all batch dimensions: [..., in_features] -> [M, in_features]
        input_2d = input.view(-1, self.in_features)
        
        output = self.mm(
            input_2d,
            self.weight_shuffle,
            self.weight_scale,
            None
        )
        if self.bias is not None:
            output = output + self.bias
        
        # Reshape back to original batch dimensions
        # [M, N] -> [..., out_features]
        output = output.view(*original_shape[:-1], self.out_features)
        
        return output
    
    def extra_repr(self):
        """String representation (for print(model))"""
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}'