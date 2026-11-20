from typing import Any, Callable

import torch
from torch._dynamo import mark_static_address

from xfuser.core.distributed import (
    get_sp_group,
)


class StaticBlockInputs:
    """Persistent buffers for the inputs every compiled block of a step reads.

    Compiled blocks that replay as CUDA/HIP graphs copy each input of unknown address into
    their own memory before every replay, so 28 blocks reading the same RoPE table copy it
    28 times. A buffer marked static keeps one address and is filled once per step instead;
    stream order keeps a later fill behind earlier readers.

    Each name keeps its ``shapes_per_name`` most recent shapes (two: CFG's branches may
    differ in text length). A graph replayed with a static input at a new address is
    recorded again, so a name that cycled through more shapes would re-record every call.
    """

    def __init__(self, shapes_per_name: int = 2):
        self.shapes_per_name = shapes_per_name
        self._buffers: dict[str, list[torch.Tensor]] = {}

    def __call__(self, name: str, tensor: torch.Tensor) -> torch.Tensor:
        """``tensor``, copied into the buffer that ``name`` keeps for its shape."""
        buffers = self._buffers.setdefault(name, [])
        for index, buffer in enumerate(buffers):
            if buffer.shape == tensor.shape and buffer.dtype == tensor.dtype and buffer.device == tensor.device:
                buffers.append(buffers.pop(index))
                break
        else:
            buffer = torch.empty(tensor.shape, dtype=tensor.dtype, device=tensor.device)
            mark_static_address(buffer)
            buffers[:] = [*buffers[max(0, len(buffers) + 1 - self.shapes_per_name) :], buffer]
        buffer.copy_(tensor)
        return buffer

    def clear(self) -> None:
        self._buffers.clear()


class PerCallCache:
    """``compute(source)``, reused while the steps of a call pass the same, unmodified ``source``.

    A denoise loop hands every step the same prompt embeddings and position ids, so what is
    derived from them (a text projection, RoPE tables) needs computing once per call, or once
    per branch under CFG: each name keeps its ``entries_per_name`` most recent sources. An
    entry holds its ``source``, so a later tensor cannot reuse its identity; in-place writes
    bump its version. Inference tensors have no version counter, so for them identity alone
    decides, which needs a pipeline that never writes its conditioning in place. Autograd
    always recomputes. :meth:`clear` drops the entries, and with them the sources they hold.
    """

    def __init__(self, entries_per_name: int = 2):
        self.entries_per_name = entries_per_name
        self._entries: dict[str, list[tuple[torch.Tensor, Any, Any]]] = {}

    def __call__(self, name: str, source: torch.Tensor, compute: Callable[[torch.Tensor], Any]) -> Any:
        if torch.is_grad_enabled():
            return compute(source)
        version = None if source.is_inference() else source._version
        entries = self._entries.setdefault(name, [])
        for cached_source, cached_version, result in entries:
            if cached_source is source and cached_version == version:
                return result
        result = compute(source)
        kept = [entry for entry in entries if entry[0] is not source]
        entries[:] = [*kept[max(0, len(kept) + 1 - self.entries_per_name) :], (source, version, result)]
        return result

    def clear(self) -> None:
        self._entries.clear()


def chunk_and_pad_sequence(
    x: torch.Tensor, sp_world_rank: int, sp_world_size: int, pad_amount: int, dim: int
) -> torch.Tensor:
    if pad_amount > 0:
        if dim < 0:
            dim = x.ndim + dim
        pad_shape = list(x.shape)
        pad_shape[dim] = pad_amount
        x = torch.cat(
            [
                x,
                torch.zeros(
                    pad_shape,
                    dtype=x.dtype,
                    device=x.device,
                ),
            ],
            dim=dim,
        )
    x = torch.chunk(x, sp_world_size, dim=dim)[sp_world_rank]
    return x


def gather_and_unpad(x: torch.Tensor, pad_amount: int, dim: int) -> torch.Tensor:
    x = get_sp_group().all_gather(x, dim=dim)
    size = x.size(dim)
    return x.narrow(dim=dim, start=0, length=size - pad_amount)


def prefix_key_mask(_key, lengths: tuple[int, ...], keys: int, device: torch.device) -> torch.Tensor:
    """The ``[batch, keys]`` bool mask of the first ``lengths[i]`` keys of each row.

    Filled in place rather than compared against a tensor of the lengths, whose
    copy to the device would wait for the queued work. ``_key`` is the mask cache's
    key, None for lengths known on the host (see ``usp.key_padding_mask``).
    """
    mask = torch.zeros((len(lengths), keys), dtype=torch.bool, device=device)
    for row, length in enumerate(lengths):
        mask[row, :length] = True
    return mask


def text_image_key_mask(
    text_mask: torch.Tensor,
    image_tokens: int,
    sp_pad: int = 0,
    sp_world_size: int = 1,
    joint_strategy: str | None = None,
) -> torch.Tensor:
    """The ``[batch, keys]`` mask a prompt next to the image gives USP, in the order the backend sees the keys.

    ``text_mask`` is the prompt's ``[batch, text]`` mask; the image tokens are
    attended and the ``sp_pad`` rows that pad the image to a multiple of
    ``sp_world_size`` are not. USP keeps those rows in K when the call has a
    mask (``usp.key_padding_attention_kwargs``), so the mask covers them.

    - ``joint_strategy=None``: the prompt is sharded (text divisible by the degree). Each rank holds
      ``[text_r, image_r]``, so the gathered keys are ``[text_0, image_0, text_1, image_1, ...]``,
      with the padding at the end of the padded image.
    - ``"front"`` / ``"rear"``: every rank holds the whole prompt and USP joins it as joint tokens,
      ahead of or behind the gathered image keys and their padding.
    """
    text_mask = text_mask.to(torch.bool)
    batch_size = text_mask.shape[0]
    image_mask = torch.cat(
        [text_mask.new_ones((batch_size, image_tokens)), text_mask.new_zeros((batch_size, sp_pad))], dim=1
    )
    if joint_strategy is not None:
        return torch.cat([text_mask, image_mask] if joint_strategy == "front" else [image_mask, text_mask], dim=1)
    return torch.cat(
        [text_mask.view(batch_size, sp_world_size, -1), image_mask.view(batch_size, sp_world_size, -1)], dim=2
    ).flatten(1)


def local_sp_pad_rows(seq_len: int, local_len: int, sp_rank: int) -> int:
    """How many of this rank's `local_len` rows are SP pad, for a real length `seq_len`.

    The pad trails the global sequence, so only the last ranks' chunks hold any.
    """
    return min(local_len, max(0, (sp_rank + 1) * local_len - seq_len))


def sp_pad_spans_ranks(seq_len: int, sp_size: int) -> bool:
    """Whether padding `seq_len` to a multiple of `sp_size` takes more than one shard.

    Then the pad sits on several ranks, and a layout that places something after each rank's
    image shard would put it between real tokens. It needs fewer tokens than about `sp_size`
    squared, so no real image reaches it.
    """
    return (-seq_len) % sp_size > -(-seq_len // sp_size)


def share_fused_weights(fused: torch.nn.Linear, parts) -> None:
    """Turn `parts` into views of `fused`, the linear that stacks their weights, so that fusing costs no memory.

    The parts compute what they did before; only their own copies of the weights are freed.
    """
    offset = 0
    for part in parts:
        rows = slice(offset, offset + part.out_features)
        part.weight.data = fused.weight.data[rows]
        if part.bias is not None:
            part.bias.data = fused.bias.data[rows]
        offset += part.out_features


# The fused projections an attention layer may hold, with the projections each one stacks.
_FUSED_PROJECTIONS = (
    ("to_qkv", ("to_q", "to_k", "to_v")),
    ("to_kv", ("to_k", "to_v")),
    ("to_added_qkv", ("add_q_proj", "add_k_proj", "add_v_proj")),
)


def fuse_plain_qkv_projections(model: torch.nn.Module) -> int:
    """Fuse Q, K and V in the attention layers of `model` whose projections are all plain linears.

    A layer with a quantized, wrapped or sharded projection keeps its separate projections. The
    separate projections of a fused layer become views of the fused weights. Returns the number
    of layers fused.
    """
    from diffusers.models.attention import AttentionModuleMixin
    from xfuser.model_executor.layers.usp_sdma import plain_linear

    fused = 0
    for module in model.modules():
        if not isinstance(module, AttentionModuleMixin):
            continue
        projections = [getattr(module, name, None) for names in _FUSED_PROJECTIONS for name in names[1]]
        plain = all(projection is None or plain_linear(projection) for projection in projections)
        if not plain or not plain_linear(getattr(module, "to_q", None)):
            continue
        module.fuse_projections()
        if not module.fused_projections:
            continue
        fused += 1
        for fused_name, names in _FUSED_PROJECTIONS:
            if hasattr(module, fused_name):
                share_fused_weights(getattr(module, fused_name), [getattr(module, name) for name in names])
    return fused


def zero_sp_pad_rows(tensor: torch.Tensor, rows: int) -> None:
    """Clear a tensor's trailing `rows` SP pad rows in place; a no-op when `rows` is 0.

    The pad trails the token dimension, which is the second-to-last one of a
    [batch, tokens, hidden] or [tokens, hidden] stream.

    Shared by :func:`zero_sp_pad_rows_hook` and by callers that bypass the hook
    because they drive a block's internals rather than its forward -- Z-Image's
    pipelined stack runner is the one such caller today. Keeping the definition
    in one place means what "zeroed" means cannot drift between them.
    """
    if rows:
        tensor.narrow(-2, tensor.shape[-2] - rows, rows).zero_()


def zero_sp_pad_rows_hook(sp_pad: dict, index: int | None = None):
    """Forward hook keeping the SP pad rows of a block's output at zero.

    The pad rows never reach real tokens' attention once USP drops their keys,
    but what a block writes there still enters statistics taken across the
    rows of whole local tensors, which then apply to the real rows:
    - FP8-comms calibration: pad rows set the per-tensor V scale in some layers
      (a larger scale makes V coarser for every real token);
    - the step cache's residual difference;
    - cross-attention kernels that share a quantization scale across query
      rows: Sage (256-row Q blocks) and AITER FP8 (one Q scale per tensor).
    The list spans features and backends, so the zeroing is not gated on them;
    missing one would fail silently. It only runs on a rank holding pad rows,
    and costs nothing measurable: under torch.compile it fuses into the block's
    last pointwise op, eagerly it is ~5 us per block.

    A hook rather than a line in forward's loop, so it also runs when a step
    cache iterates the blocks itself. ``sp_pad["local_rows"]`` is the number of
    trailing rows to clear along the second-to-last dimension (the tokens of a
    [batch, tokens, hidden] or [tokens, hidden] stream), set by the wrapper on
    every forward.
    ``index`` selects the one output tensor that carries the padded stream; by
    default every tensor a block returns is cleared (a VACE block returns its
    conditioning states with its control states, and both are added to the main
    stream), and a tensor that is None is skipped.

    A hook only fires for callers that go through ``Module.__call__``. A runner
    that drives a block's internals directly steps around it and has to clear
    the pad rows itself with :func:`zero_sp_pad_rows` -- see
    ``_run_modulated_stack`` in the Z-Image transformer.
    """

    def hook(module, args, output):
        rows = sp_pad["local_rows"]
        if not rows:
            return
        if index is not None:
            tensors = (output[index] if isinstance(output, tuple) else output,)
        else:
            tensors = output if isinstance(output, tuple) else (output,)
        for tensor in tensors:
            if tensor is not None:
                zero_sp_pad_rows(tensor, rows)

    return hook
