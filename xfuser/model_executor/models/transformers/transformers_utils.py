import torch
from xfuser.core.distributed import (
    get_sp_group,
)

def chunk_and_pad_sequence(x: torch.Tensor, sp_world_rank: int, sp_world_size: int, pad_amount: int, dim: int) -> torch.Tensor:
    if pad_amount > 0:
        if dim < 0:
            dim = x.ndim + dim
        pad_shape = list(x.shape)
        pad_shape[dim] = pad_amount
        x = torch.cat([x,
                    torch.zeros(
                        pad_shape,
                        dtype=x.dtype,
                        device=x.device,
                    )], dim=dim)
    x = torch.chunk(x,
                    sp_world_size,
                    dim=dim)[sp_world_rank]
    return x

def gather_and_unpad(x: torch.Tensor, pad_amount: int, dim: int) -> torch.Tensor:
    x = get_sp_group().all_gather(x, dim=dim)
    size = x.size(dim)
    return x.narrow(dim=dim, start=0, length=size - pad_amount)


def local_sp_pad_rows(seq_len: int, local_len: int, sp_rank: int) -> int:
    """How many of this rank's `local_len` rows are SP pad, for a real length `seq_len`.

    The pad trails the global sequence, so only the last ranks' chunks hold any.
    """
    return min(local_len, max(0, (sp_rank + 1) * local_len - seq_len))


def zero_sp_pad_rows_hook(sp_pad: dict, index: int | None = None, rows_key: str = "local_rows", dim: int = 1):
    """Forward hook keeping the SP pad rows of a block's output at zero.

    The pad rows never reach real tokens' attention once USP drops their keys,
    but what a block writes there still enters statistics taken across the
    rows of whole local tensors, which then apply to the real rows:
    - FP8-comms calibration: pad rows set the per-tensor V scale in some layers
      (measured up to 1.5x coarser for every real token);
    - the step cache's residual difference;
    - cross-attention kernels that share a quantization scale across query
      rows: Sage (256-row Q blocks) and AITER FP8 (one Q scale per tensor).
    The list spans features and backends, so the zeroing is not gated on them;
    missing one would fail silently. It only runs on a rank holding pad rows,
    and costs nothing measurable: under torch.compile it fuses into the block's
    last pointwise op, eagerly it is ~5 us per block.

    A hook rather than a line in forward's loop, so it also runs when a step
    cache iterates the blocks itself. ``sp_pad[rows_key]`` is the number of
    trailing rows to clear along ``dim``, set by the wrapper on every forward.
    ``index`` selects the one output tensor that carries the padded stream; by
    default every tensor a block returns is cleared (a VACE block returns its
    conditioning states with its control states, and both are added to the main
    stream), and a tensor that is None is skipped.
    """

    def hook(module, args, output):
        rows = sp_pad[rows_key]
        if not rows:
            return
        if index is not None:
            tensors = (output[index] if isinstance(output, tuple) else output,)
        else:
            tensors = output if isinstance(output, tuple) else (output,)
        for tensor in tensors:
            if tensor is not None:
                tensor.narrow(dim, tensor.shape[dim] - rows, rows).zero_()

    return hook
