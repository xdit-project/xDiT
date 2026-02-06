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
