"""Casting a model to a compute dtype without dropping the precision its own loader would keep.

Model libraries name the sub-modules whose fp32 precision matters -- normalisations, scale-shift
tables, timestep embedders -- and their loaders honour that by casting each loaded tensor
individually rather than the whole module. A blanket ``module.to(dtype)`` does not, so anywhere
xFuser applies a compute dtype itself it has to reproduce the policy or the same checkpoint ends up
lower precision than an ordinary load of it.

The two libraries xFuser loads from disagree on when the policy applies, so the rule has to be
resolved per model rather than read off one attribute:

  * diffusers honours ``_keep_in_fp32_modules`` at any compute dtype.
  * transformers added ``_keep_in_fp32_modules`` only to avoid fp16 overflow, so it applies at fp16
    alone, and added ``_keep_in_fp32_modules_strict`` for the modules that need fp32 at bf16 too.

Non-persistent buffers are protected for a different reason: they are absent from the state dict, so
no loader ever casts them and they keep whatever dtype ``__init__`` computed. Wan's RoPE frequency
tables are the case that matters.
"""

import torch


def _is_transformers_model(model) -> bool:
    """Whether ``model`` is a transformers PreTrainedModel, without importing transformers."""

    return any(
        cls.__name__ == "PreTrainedModel" and cls.__module__.startswith("transformers.")
        for cls in type(model).__mro__
    )


def fp32_modules_for(model, dtype) -> tuple:
    """The sub-modules whose fp32 precision ``model``'s own loader would keep at ``dtype``."""

    pinned = tuple(getattr(model, "_keep_in_fp32_modules", None) or ())
    if not _is_transformers_model(model):
        return pinned
    strict = tuple(getattr(model, "_keep_in_fp32_modules_strict", None) or ())
    applicable = pinned if dtype is torch.float16 else ()
    if dtype in (torch.float16, torch.bfloat16):
        applicable += strict
    return tuple(dict.fromkeys(applicable))


def keeps_fp32(name: str, fp32_modules) -> bool:
    """Whether a tensor's dotted name sits under one of ``fp32_modules``.

    Matches whole path segments, as the loaders do, so ``norm2`` pins ``blocks.0.norm2.bias``
    without also pinning a hypothetical ``blocks.0.norm2_extra.bias``.
    """

    segments = name.split(".")
    return any(module in segments for module in fp32_modules)


def persistent_named_buffers(module):
    """Named buffers saved by state_dict, using each buffer owner's persistence set."""

    persistent = []
    for name, buffer in module.named_buffers(recurse=True, remove_duplicate=False):
        parent_name, _, local_name = name.rpartition(".")
        owner = module.get_submodule(parent_name) if parent_name else module
        if local_name not in owner._non_persistent_buffers_set:
            persistent.append((name, buffer))
    return persistent


def pinned_fp32_parameters(module, fp32_modules) -> set:
    """The parameters under ``module`` that ``fp32_modules`` pins, by identity.

    FSDP groups parameters into one flat allocation per unit and rejects a unit whose parameters do
    not share a dtype, so a caller that shards a model has to hand these to FSDP as ignored
    parameters rather than let them into a group.

    Takes the resolved module names rather than a dtype so a caller can ask about a sub-module, whose
    own class carries no policy, and so it can be re-asked after a load: filling a parameter through
    ``set_module_tensor_to_device`` rebinds the slot to a new object, which leaves any set collected
    beforehand pointing at parameters the module no longer holds.
    """

    if not fp32_modules:
        return set()
    return {
        parameter
        for name, parameter in module.named_parameters(
            recurse=True, remove_duplicate=False
        )
        if torch.is_floating_point(parameter) and keeps_fp32(name, fp32_modules)
    }


def cast_preserving_fp32_modules(model, dtype):
    """Cast ``model``'s floating-point tensors to ``dtype``, minus the ones pinned to fp32.

    Casts per tensor rather than through ``module.to`` so a pinned tensor is never rounded on the
    way: for a model whose weights are already real, an upcast afterwards could not recover the
    discarded bits. Each cast is the same ``tensor.data.to(dtype)`` that ``module.to`` applies, so
    tensor subclasses such as a quantized weight are handled exactly as before.
    """

    fp32_modules = fp32_modules_for(model, dtype)
    persistent = {name for name, _ in persistent_named_buffers(model)}
    tensors = list(model.named_parameters(recurse=True, remove_duplicate=False))
    tensors += [
        (name, buffer)
        for name, buffer in model.named_buffers(recurse=True, remove_duplicate=False)
        if name in persistent
    ]
    for name, tensor in tensors:
        if not torch.is_floating_point(tensor):
            continue
        if keeps_fp32(name, fp32_modules):
            continue
        tensor.data = tensor.data.to(dtype)
    return model
