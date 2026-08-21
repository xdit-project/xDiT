import torch
import torch.nn as nn
import math
try:
    import aiter
    from aiter.ops.shuffle import shuffle_weight
except ImportError:
    pass # Error will be thrown in base_model.py, if mxfp4 gemms are enabled but AITER is not available.
from typing import Optional
from xfuser.core.distributed.runtime_state import get_runtime_state


@torch.library.custom_op("xfuser::mxfp4_gemm", mutates_args=())
def _mxfp4_gemm(a: torch.Tensor, w_quant: torch.Tensor, w_scale: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    quant_func = aiter.get_hip_quant(aiter.QuantType.per_1x32)
    a_quant, a_scale = quant_func(a, shuffle=True)
    output = aiter.gemm_a4w4(a_quant, w_quant, a_scale, w_scale, bpreshuffle=True, bias=bias)
    return output

@_mxfp4_gemm.register_fake
def _(a: torch.Tensor, w_quant: torch.Tensor, w_scale: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Fake implementation for torch.compile shape inference
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
    
    def load_and_quantize_weights(
        self, 
        weights: torch.Tensor, 
        bias: Optional[torch.Tensor] = None
    ) -> None:
        """
        Load pre-trained weights and quantize them.
        
        Args:
            weights: Full-precision weight tensor [out_features, in_features]
            bias: Optional bias tensor [out_features]
        """
        with torch.no_grad():
            # Temporarily restore weight parameter if it was deleted
            if self.weight is None:
                self.weight = nn.Parameter(
                    torch.empty_like(weights, device=weights.device, dtype=weights.dtype)
                )
            
            self.weight.data.copy_(weights.data)
            if bias is not None and self.bias is not None:
                self.bias.data.copy_(bias.data)
        
        self._quantize_weights()
    
    def _quantize_weights(self) -> None:
        """
        Quantize weights to FP4 and register the shardable weight and scale.
        
        This ensures proper device movement with .to(), .cuda(), CPU offload,
        and distributed training frameworks (FSDP, DDP).
        """
        if self.weight is None:
            raise RuntimeError(
                "Cannot quantize: weight parameter is None."
                "Call load_and_quantize_weights() or reset_parameters() first."
            )
        
        quant_func = aiter.get_hip_quant(aiter.QuantType.per_1x32)
        weight_quant, weight_scale = quant_func(self.weight, shuffle=True)
        weight_shuffle = shuffle_weight(weight_quant, layout=(16, 16))
        
        # FSDP only shards parameters. Keep the large packed weight non-trainable but registered
        # as a parameter; the much smaller scale remains a persistent buffer.
        if hasattr(self, 'weight_shuffle'):
            delattr(self, 'weight_shuffle')
        if hasattr(self, 'weight_scale'):
            delattr(self, 'weight_scale')
        self.register_parameter(
            'weight_shuffle', nn.Parameter(weight_shuffle, requires_grad=False)
        )
        self.register_buffer('weight_scale', weight_scale, persistent=True)
        
        # Properly remove the original weight parameter to save memory
        # This maintains module structure while freeing memory
        delattr(self, 'weight')
        self.register_parameter('weight', None)

    def _is_fsdp_managed_parameter(self, parameter) -> bool:
        """Whether replacing ``parameter`` would break composable FSDP ownership."""
        try:
            from torch.distributed.tensor import DTensor
        except ImportError:
            return False
        return isinstance(parameter, DTensor)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        """Match this layer's registered layout to full-precision or packed checkpoints."""
        weight_key = prefix + "weight"
        packed_key = prefix + "weight_shuffle"
        scale_key = prefix + "weight_scale"
        has_quantized_state = packed_key in state_dict and scale_key in state_dict

        if has_quantized_state:
            current_parameter = (
                self.weight_shuffle
                if hasattr(self, "weight_shuffle")
                else self.weight
            )
            if self._is_fsdp_managed_parameter(current_parameter):
                raise RuntimeError(
                    "MXFP4 packed state cannot be loaded after FSDP wrapping; "
                    "load the packed checkpoint before fully_shard."
                )
            current_device = (
                self.weight_shuffle.device
                if hasattr(self, "weight_shuffle")
                else self.weight.device
            )
            incoming_device = state_dict[packed_key].device
            destination_device = (
                incoming_device if current_device.type == "meta" else current_device
            )
            if hasattr(self, "weight_shuffle"):
                delattr(self, "weight_shuffle")
            if hasattr(self, "weight_scale"):
                delattr(self, "weight_scale")
            if self.weight is not None:
                delattr(self, "weight")
                self.register_parameter("weight", None)
            self.register_parameter(
                "weight_shuffle",
                nn.Parameter(
                    torch.empty(
                        state_dict[packed_key].shape,
                        dtype=state_dict[packed_key].dtype,
                        device=destination_device,
                    ),
                    requires_grad=False,
                ),
            )
            self.register_buffer(
                "weight_scale",
                torch.empty(
                    state_dict[scale_key].shape,
                    dtype=state_dict[scale_key].dtype,
                    device=destination_device,
                ),
                persistent=True,
            )
        elif weight_key in state_dict and self.weight is None:
            if self._is_fsdp_managed_parameter(self.weight_shuffle):
                raise RuntimeError(
                    "MXFP4 full-precision state cannot replace an FSDP-managed packed parameter; "
                    "load the full-precision checkpoint before fully_shard."
                )
            current_device = self.weight_shuffle.device
            incoming_device = state_dict[weight_key].device
            destination_device = (
                incoming_device if current_device.type == "meta" else current_device
            )
            if hasattr(self, "weight_shuffle"):
                delattr(self, "weight_shuffle")
            if hasattr(self, "weight_scale"):
                delattr(self, "weight_scale")
            delattr(self, "weight")
            self.register_parameter(
                "weight",
                nn.Parameter(
                    torch.empty(
                        state_dict[weight_key].shape,
                        dtype=state_dict[weight_key].dtype,
                        device=destination_device,
                    )
                ),
            )

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def _run_mxfp4_gemm(self, a: torch.Tensor, w_quant: torch.Tensor, w_scale: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        return torch.ops.xfuser.mxfp4_gemm(a, w_quant, w_scale, bias)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass using MXFP4 GEMM
        """

        if not hasattr(self, "weight_shuffle"):
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


class xFuserHybridMXFP4Linear(nn.Module):
    """
    Hybrid linear layer that switches per diffusion step between
    high precision (FP8-quantized nn.Linear path) and low precision (MXFP4 GEMM path).
    """

    def __init__(
        self,
        high_precision_linear: nn.Module,
        low_precision_linear: xFuserMXFP4Linear,
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
