from __future__ import annotations

import torch

try:
    import aiter as _aiter

    # Wrapping aiter.rms_norm as a non-mutating custom op fixes graph capture
    @torch.library.custom_op("xfuser::aiter_rms_norm", mutates_args=())
    def _aiter_rms_norm(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
        return _aiter.rms_norm(x, w, eps)

    @_aiter_rms_norm.register_fake
    def _(x, w, eps):
        return torch.empty_like(x)

    @_aiter_rms_norm.register_autograd
    def _(ctx, grad_output):
        # Inference only, backward is never called; return identity gradient.
        return grad_output, None, None

    _HAS_AITER = True
except ImportError:
    _HAS_AITER = False


class _AITERRMSNorm(torch.nn.Module):
    """Drop-in for diffusers RMSNorm using AITER's CK-Tile kernel.
    Diffusers RMSNorm up-casts to float32 internally; AITER's kernel works
    natively in bfloat16 and is faster.  Handles elementwise_affine=False
    (no learned weight) via a registered ones buffer so AITER always receives a
    weight tensor.
    """

    def __init__(self, weight: torch.nn.Parameter | None, eps: float, dim: int) -> None:
        super().__init__()
        if weight is not None:
            self.weight = weight
            self._use_ones = False
        else:
            self.register_buffer("_ones_weight", torch.ones(dim))
            self._use_ones = True
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        w = self._ones_weight if self._use_ones else self.weight
        out = _aiter_rms_norm(x.view(-1, shape[-1]), w, self.eps)
        return out.view(shape)


def _replace_rms_norms_with_aiter(model: torch.nn.Module) -> None:
    """Walk ``model`` and replace diffusers RMSNorm modules with :class:`_AITERRMSNorm`.
    Only replaces ``diffusers.models.normalization.RMSNorm`` (Python impl,
    float32 up-cast), leaving ``torch.nn.RMSNorm`` (C++ impl) untouched.
    Must be called BEFORE ``model.to(device)`` so buffers are placed correctly.
    """
    if not _HAS_AITER:
        return
    try:
        from diffusers.models.normalization import RMSNorm as _DiffusersRMSNorm
    except ImportError:
        return

    replacements: list[tuple[torch.nn.Module, str, _AITERRMSNorm]] = []
    for _parent_name, parent_module in model.named_modules():
        for child_name, child_module in parent_module.named_children():
            if type(child_module) is _DiffusersRMSNorm and hasattr(child_module, "eps"):
                if child_module.weight is not None:
                    dim = child_module.weight.shape[0]
                elif hasattr(child_module, "dim"):
                    dim = (
                        int(child_module.dim[0])
                        if hasattr(child_module.dim, "__len__")
                        else int(child_module.dim)
                    )
                else:
                    continue
                replacements.append(
                    (
                        parent_module,
                        child_name,
                        _AITERRMSNorm(child_module.weight, child_module.eps, dim),
                    )
                )
    for parent, name, replacement in replacements:
        setattr(parent, name, replacement)