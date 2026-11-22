"""Linears whose weight is stored transposed, for GPUs whose GEMM is slower when handed a transposed operand."""

import torch
import torch.nn.functional as F

from xfuser.model_executor.layers.plain_linear import plain_linear


class TransposedLinear(torch.nn.Linear):
    """A linear whose `weight` is stored as [in_features, out_features]; it computes what it did before."""

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.linear(input, self.weight.t(), self.bias)


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
