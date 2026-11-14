import torch
from xfuser.core.distributed import (
    get_sp_group,
)


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
                tensor.narrow(-2, tensor.shape[-2] - rows, rows).zero_()

    return hook
