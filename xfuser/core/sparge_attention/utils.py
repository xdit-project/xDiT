import math
from typing import Optional

import numpy
import torch
from numba import njit


@njit
def _neighbors(i: int, j: int, k: int, F: int, H: int, W: int) -> numpy.ndarray:
    directions = numpy.array(
        [
            [1, 0, 0], [-1, 0, 0],
            [0, 1, 0], [0, -1, 0],
            [0, 0, 1], [0, 0, -1]
        ]
    )

    result = numpy.empty((6, 3), dtype=numpy.int64)
    count = 0
    for di, dj, dk in directions:
        ni, nj, nk = i + di, j + dj, k + dk
        if (0 <= ni < F) and (0 <= nj < H) and (0 <= nk < W):
            result[count, 0] = numpy.int64(ni)
            result[count, 1] = numpy.int64(nj)
            result[count, 2] = numpy.int64(nk)
            count += 1

    return result[:count]


@njit
def _dot(x: numpy.ndarray, y: numpy.ndarray, stride: int = 1) -> float:
    s = 0.0
    for i in range(0, x.shape[0], stride):
        s += x[i] * y[i]
    return s


@njit
def _curve(t: numpy.ndarray) -> numpy.ndarray:
    _, F, H, W = t.shape

    indices = numpy.arange(F * H * W).reshape(F, H, W)

    stack = numpy.empty((F * H * W, 3), dtype=numpy.int64)
    stack_ptr = 0
    stack[stack_ptr, :] = numpy.array([0, 0, 0], dtype=numpy.int64)

    visited = numpy.zeros((F, H, W), dtype=numpy.bool_)
    visited[0, 0, 0] = True

    order = numpy.empty(F * H * W, dtype=numpy.int64)
    order_ptr = 0
    order[order_ptr] = indices[0, 0, 0]
    order_ptr += 1

    while stack_ptr >= 0:
        i = stack[stack_ptr, 0]
        j = stack[stack_ptr, 1]
        k = stack[stack_ptr, 2]

        nbrs = _neighbors(i, j, k, F, H, W)

        unvisited_ptr = 0
        unvisited = numpy.empty((6, 3), dtype=numpy.int64)
        for idx in range(nbrs.shape[0]):
            ni = nbrs[idx, 0]
            nj = nbrs[idx, 1]
            nk = nbrs[idx, 2]
            if not visited[ni, nj, nk]:
                unvisited[unvisited_ptr, 0] = ni
                unvisited[unvisited_ptr, 1] = nj
                unvisited[unvisited_ptr, 2] = nk
                unvisited_ptr += 1

        if unvisited_ptr == 0:
            stack_ptr -= 1
            continue

        current_voxel = t[:, i, j, k]
        best_similarity = numpy.float32(-2.0)
        best_neighbor = -1

        for ptr in range(unvisited_ptr):
            ni = unvisited[ptr, 0]
            nj = unvisited[ptr, 1]
            nk = unvisited[ptr, 2]
            neighbor_voxel = t[:, ni, nj, nk]
            similarity = _dot(current_voxel, neighbor_voxel)
            if similarity > best_similarity:
                best_similarity = similarity
                best_neighbor = ptr

        if best_neighbor < 0:
            stack_ptr -= 1
            continue

        ni = unvisited[best_neighbor, 0]
        nj = unvisited[best_neighbor, 1]
        nk = unvisited[best_neighbor, 2]
        visited[ni, nj, nk] = True

        if stack_ptr + 1 >= stack.shape[0]:
            break

        stack_ptr += 1
        stack[stack_ptr, 0] = ni
        stack[stack_ptr, 1] = nj
        stack[stack_ptr, 2] = nk
        order[order_ptr] = indices[ni, nj, nk]
        order_ptr += 1

    return order[:order_ptr]


def curve(t: torch.Tensor, proj_dim: int = 16, seed: Optional[int] = None) -> torch.Tensor:
    B, C, F, H, W = t.shape
    device = t.device
    # Use float32 for projection and norm to avoid overflow when C is large or input is fp16
    t = t.float()
    # Random projection (C, k) with JL scaling; fixed seed for reproducibility
    generator = torch.Generator(device=device).manual_seed(seed) if seed is not None else None
    proj = torch.randn(C, proj_dim, device=device, dtype=torch.float32, generator=generator) * (
        1.0 / math.sqrt(proj_dim)
    )
    t_flat = t.reshape(B, C, -1)
    t_reduced = (t_flat.transpose(1, 2) @ proj).transpose(1, 2).reshape(B, proj_dim, F, H, W)
    t = t_reduced / (t_reduced.norm(dim=1, keepdim=True) + 1.0e-9)
    t = t.cpu().numpy()
    order = []
    for i in range(t.shape[0]):
        order.append(_curve(t[i]))
    order = torch.from_numpy(numpy.stack(order)).to(device)

    return order


def reorder_sequence(t: torch.Tensor, order: torch.Tensor) -> torch.Tensor:
    nt = []
    for i in range(t.shape[0]):
        nt.append(
            t[i].reshape(t.shape[1], -1).transpose(0, 1)[order[i]]
        )
    nt = torch.stack(nt).contiguous()

    return nt


def restore_sequence_order(t: torch.Tensor, order: torch.Tensor) -> torch.Tensor:
    nt = []
    for i in range(t.shape[0]):
        indices = torch.argsort(order[i])
        nt.append(t[i, indices])
    nt = torch.stack(nt).contiguous()

    return nt
