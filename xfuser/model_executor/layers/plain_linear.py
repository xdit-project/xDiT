"""What the layers that stack or share out linear weights take for a plain linear."""

import torch

from xfuser.core.utils.runner_utils import _weight_is_torchao_quantized


def plain_linear(module) -> bool:
    """Whether ``module`` is exactly an ``nn.Linear`` whose weight is an ordinary tensor, whole on this rank.

    Quantized GEMMs and LoRA replace the module, torchao quantizes the weight in place, and
    sharded (FSDP) weights are DTensors with placements.
    """
    return (
        type(module) is torch.nn.Linear
        and not _weight_is_torchao_quantized(module)
        and not hasattr(module.weight, "placements")
    )
