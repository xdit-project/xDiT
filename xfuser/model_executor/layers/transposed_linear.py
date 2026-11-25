"""Linears whose weight is stored transposed, for GPUs whose GEMM is slower when handed a transposed operand."""

import torch
import torch.nn.functional as F

from xfuser.core.attention.requirements import device_arch
from xfuser.model_executor.layers.plain_linear import plain_linear

# Measured on gfx942 and gfx950; gfx950 is faster with the standard layout.
_ARCHS_PREFERRING_UNTRANSPOSED_WEIGHTS = ("gfx942",)


def prefers_untransposed_weights() -> bool:
    """Whether this device's GEMM is faster when a linear's weight is stored as [in, out]."""
    return device_arch().split(":")[0] in _ARCHS_PREFERRING_UNTRANSPOSED_WEIGHTS


class TransposedLinear(torch.nn.Linear):
    """A linear whose `weight` is stored as [in_features, out_features]; it computes what it did before.

    Its state dict keeps the standard [out_features, in_features] layout, as a view on save.
    """

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.linear(input, self.weight.t(), self.bias)

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        super()._save_to_state_dict(destination, prefix, keep_vars)
        destination[prefix + "weight"] = destination[prefix + "weight"].t()

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        if prefix + "weight" in state_dict:
            state_dict[prefix + "weight"] = state_dict[prefix + "weight"].t()
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)


def _owns_its_weight(linear: torch.nn.Linear) -> bool:
    return linear.weight.untyped_storage().nbytes() == linear.weight.nbytes


def transpose_plain_linears(model: torch.nn.Module) -> int:
    """Store the weight of each plain linear of `model` transposed, in place of the weight it had.

    `F.linear` hands the GEMM its weight as a transposed operand; with the weight stored as
    [in, out] neither operand is transposed. A linear that is a view of a fused weight (see
    `share_fused_weights`) is no longer called: it keeps its shape and follows the weight it is
    a view of, so that nothing keeps the old one alive. Returns the number of linears transposed.
    """
    linears = [module for module in model.modules() if plain_linear(module)]
    views = {}
    for linear in linears:
        if not _owns_its_weight(linear):
            views.setdefault(linear.weight.untyped_storage().data_ptr(), []).append(linear)
    count = 0
    for linear in linears:
        if not _owns_its_weight(linear):
            continue
        old = linear.weight.untyped_storage().data_ptr()
        linear.weight.data = linear.weight.data.t().contiguous()
        linear.__class__ = TransposedLinear
        for view in views.pop(old, ()):
            row = view.weight.storage_offset() // view.in_features
            view.weight.data = linear.weight.data.t()[row : row + view.out_features]
        count += 1
    return count
