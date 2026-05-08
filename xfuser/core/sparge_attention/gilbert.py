# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2018 Jakub Červený
# Copyright (c) 2024 abetusk
# All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
# 
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from typing import List, Optional, Tuple

import torch


def _sgn(x: int) -> int:
    return -1 if x < 0 else (1 if x > 0 else 0)


def _generate3d_impl(
    out: List[int],
    x: int, y: int, z: int,
    ax: int, ay: int, az: int,
    bx: int, by: int, bz: int,
    cx: int, cy: int, cz: int,
) -> None:
    """Append flattened (x, y, z) triples in Gilbert order to out."""
    w = abs(ax + ay + az)
    h = abs(bx + by + bz)
    d = abs(cx + cy + cz)

    dax, day, daz = _sgn(ax), _sgn(ay), _sgn(az)
    dbx, dby, dbz = _sgn(bx), _sgn(by), _sgn(bz)
    dcx, dcy, dcz = _sgn(cx), _sgn(cy), _sgn(cz)

    if h == 1 and d == 1:
        for _ in range(w):
            out.append(x); out.append(y); out.append(z)
            x, y, z = x + dax, y + day, z + daz
        return

    if w == 1 and d == 1:
        for _ in range(h):
            out.append(x); out.append(y); out.append(z)
            x, y, z = x + dbx, y + dby, z + dbz
        return

    if w == 1 and h == 1:
        for _ in range(d):
            out.append(x); out.append(y); out.append(z)
            x, y, z = x + dcx, y + dcy, z + dcz
        return

    ax2, ay2, az2 = ax // 2, ay // 2, az // 2
    bx2, by2, bz2 = bx // 2, by // 2, bz // 2
    cx2, cy2, cz2 = cx // 2, cy // 2, cz // 2

    w2 = abs(ax2 + ay2 + az2)
    h2 = abs(bx2 + by2 + bz2)
    d2 = abs(cx2 + cy2 + cz2)

    if (w2 % 2) and (w > 2):
        ax2, ay2, az2 = ax2 + dax, ay2 + day, az2 + daz
    if (h2 % 2) and (h > 2):
        bx2, by2, bz2 = bx2 + dbx, by2 + dby, bz2 + dbz
    if (d2 % 2) and (d > 2):
        cx2, cy2, cz2 = cx2 + dcx, cy2 + dcy, cz2 + dcz

    if (2 * w > 3 * h) and (2 * w > 3 * d):
        _generate3d_impl(out, x, y, z,
                         ax2, ay2, az2,
                         bx, by, bz,
                         cx, cy, cz)
        _generate3d_impl(out, x + ax2, y + ay2, z + az2,
                         ax - ax2, ay - ay2, az - az2,
                         bx, by, bz,
                         cx, cy, cz)
        return

    if 3 * h > 4 * d:
        _generate3d_impl(out, x, y, z,
                         bx2, by2, bz2,
                         cx, cy, cz,
                         ax2, ay2, az2)
        _generate3d_impl(out, x + bx2, y + by2, z + bz2,
                         ax, ay, az,
                         bx - bx2, by - by2, bz - bz2,
                         cx, cy, cz)
        _generate3d_impl(out,
                         x + (ax - dax) + (bx2 - dbx),
                         y + (ay - day) + (by2 - dby),
                         z + (az - daz) + (bz2 - dbz),
                         -bx2, -by2, -bz2,
                         cx, cy, cz,
                         -(ax - ax2), -(ay - ay2), -(az - az2))
        return

    if 3 * d > 4 * h:
        _generate3d_impl(out, x, y, z,
                         cx2, cy2, cz2,
                         ax2, ay2, az2,
                         bx, by, bz)
        _generate3d_impl(out, x + cx2, y + cy2, z + cz2,
                         ax, ay, az,
                         bx, by, bz,
                         cx - cx2, cy - cy2, cz - cz2)
        _generate3d_impl(out,
                         x + (ax - dax) + (cx2 - dcx),
                         y + (ay - day) + (cy2 - dcy),
                         z + (az - daz) + (cz2 - dcz),
                         -cx2, -cy2, -cz2,
                         -(ax - ax2), -(ay - ay2), -(az - az2),
                         bx, by, bz)
        return

    _generate3d_impl(out, x, y, z,
                     bx2, by2, bz2,
                     cx2, cy2, cz2,
                     ax2, ay2, az2)
    _generate3d_impl(out, x + bx2, y + by2, z + bz2,
                     cx, cy, cz,
                     ax2, ay2, az2,
                     bx - bx2, by - by2, bz - bz2)
    _generate3d_impl(out,
                     x + (bx2 - dbx) + (cx - dcx),
                     y + (by2 - dby) + (cy - dcy),
                     z + (bz2 - dbz) + (cz - dcz),
                     ax, ay, az,
                     -bx2, -by2, -bz2,
                     -(cx - cx2), -(cy - cy2), -(cz - cz2))
    _generate3d_impl(out,
                     x + (ax - dax) + bx2 + (cx - dcx),
                     y + (ay - day) + by2 + (cy - dcy),
                     z + (az - daz) + bz2 + (cz - dcz),
                     -cx, -cy, -cz,
                     -(ax - ax2), -(ay - ay2), -(az - az2),
                     bx - bx2, by - by2, bz - bz2)
    _generate3d_impl(out,
                     x + (ax - dax) + (bx2 - dbx),
                     y + (ay - day) + (by2 - dby),
                     z + (az - daz) + (bz2 - dbz),
                     -bx2, -by2, -bz2,
                     cx2, cy2, cz2,
                     -(ax - ax2), -(ay - ay2), -(az - az2))


# Module-local cache of the raw flat int list for a (width, height, depth) cuboid.
# Keyed by Python ints, populated once per shape, reused across devices.
_GILBERT_COORDS_CACHE: dict = {}


def _gilbert3d_coords(width: int, height: int, depth: int) -> List[int]:
    """Return a flat list[int] of length 3 * width * height * depth
    containing [x0, y0, z0, x1, y1, z1, ...] in Gilbert curve order."""
    key = (int(width), int(height), int(depth))
    cached = _GILBERT_COORDS_CACHE.get(key)
    if cached is not None:
        return cached

    out: List[int] = []
    if width >= height and width >= depth:
        _generate3d_impl(out, 0, 0, 0,
                         width, 0, 0,
                         0, height, 0,
                         0, 0, depth)
    elif height >= width and height >= depth:
        _generate3d_impl(out, 0, 0, 0,
                         0, height, 0,
                         width, 0, 0,
                         0, 0, depth)
    else:  # depth >= width and depth >= height
        _generate3d_impl(out, 0, 0, 0,
                         0, 0, depth,
                         width, 0, 0,
                         0, height, 0)

    _GILBERT_COORDS_CACHE[key] = out
    return out


def curve(depth: int, height: int, width: int, device: torch.device
          ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute Gilbert-curve permutations for a (depth, height, width) cuboid.

    Returns (order, inverse_order) as int64 tensors on device:
        - order[linear]  -> step index along the Gilbert curve
        - inverse_order[step] -> linear index z*h*w + y*w + x

    The recursion runs in pure Python over Python ints; the result is
    materialized via a single torch.tensor(list, ...) call so there are
    no per-cell torch ops baked into the compiled graph.
    """
    n = width * height * depth
    coords_flat = _gilbert3d_coords(width, height, depth)
    coords = torch.tensor(coords_flat, dtype=torch.int64, device=device).view(n, 3)
    inverse_order = (
        coords[:, 2] * (height * width)
        + coords[:, 1] * width
        + coords[:, 0]
    )
    order = torch.argsort(inverse_order)
    return order, inverse_order

def _curve_index_array(width: int, height: int, depth: int) -> List[int]:
    """Return a Python list of length width*height*depth such that
    arr[z*height*width + y*width + x] is the Gilbert curve index of the
    point (x, y, z) in a (width, height, depth) cuboid."""
    coords_flat = _gilbert3d_coords(width, height, depth)
    n = width * height * depth
    arr = [0] * n
    hw = height * width
    for step in range(n):
        x = coords_flat[3 * step]
        y = coords_flat[3 * step + 1]
        z = coords_flat[3 * step + 2]
        arr[z * hw + y * width + x] = step
    return arr

def gilbert_xyz2d(x: int, y: int, z: int,
                  width: int, height: int, depth: int) -> int:
    """Return the Gilbert curve index of point (x, y, z) in a
    (width, height, depth) cuboid. Pure-Python lookup against the cached
    curve."""
    arr = _curve_index_array(width, height, depth)
    return arr[z * (height * width) + y * width + x]

def transpose_gilbert_mapping(dims, order=None):
    """Build (linear_to_hilbert, hilbert_to_linear) for a 3D box with the
    given axis order permutation. Returns Python lists of length
    prod(dims)."""
    if len(dims) != 3:
        raise ValueError("Dimensions must be three-dimensional")
    if order is None:
        order = [0, 1, 2]
    if len(order) != 3 or set(order) != {0, 1, 2}:
        raise ValueError("order must be a permutation of 0,1,2")

    # Box dimensions in transposed (curve) frame.
    t_box = dims[order[0]]
    h_box = dims[order[1]]
    w_box = dims[order[2]]
    arr = _curve_index_array(w_box, h_box, t_box)

    d0, d1, d2 = dims[0], dims[1], dims[2]
    total = d0 * d1 * d2
    linear_to_hilbert = [0] * total
    hilbert_to_linear = [0] * total

    inv = [0, 0, 0]
    for i, o in enumerate(order):
        inv[o] = i

    # For each original-frame coordinate (i0, i1, i2) -> linear_idx,
    # compute its (x, y, z) in the transposed (curve) frame and look up its
    # curve step from `arr`.
    coords = (0, 0, 0)
    linear_idx = 0
    for i0 in range(d0):
        for i1 in range(d1):
            for i2 in range(d2):
                coords = (i0, i1, i2)
                # coords in the curve frame: x = coords[order[2]], y = coords[order[1]], z = coords[order[0]]
                x = coords[order[2]]
                y = coords[order[1]]
                z = coords[order[0]]
                hilbert_idx = arr[z * (h_box * w_box) + y * w_box + x]
                linear_to_hilbert[linear_idx] = hilbert_idx
                hilbert_to_linear[hilbert_idx] = linear_idx
                linear_idx += 1

    return linear_to_hilbert, hilbert_to_linear

def sliced_gilbert_mapping(t: int, h: int, w: int,
                           transpose_order: Optional[List[int]] = None
                           ) -> Tuple[List[int], List[int]]:
    """Build a sliced Gilbert mapping (2D Gilbert per z-slice with
    directional flipping for slice continuity), or fall back to the
    transposed 3D mapping when transpose_order is supplied.

    Returns (linear_to_hilbert, hilbert_to_linear) as Python int lists.
    """
    dims = [t, h, w]

    if transpose_order is not None:
        return transpose_gilbert_mapping(dims, transpose_order)

    total_points = t * h * w
    linear_to_hilbert = [0] * total_points
    hilbert_to_linear = [0] * total_points

    # Precompute the (unflipped) 2D Gilbert curve over (w, h, 1) once.
    # `slice_coords[i] = (x_i, y_i)` is the i-th step of the curve.
    coords_flat_2d = _gilbert3d_coords(w, h, 1)
    slice_points = h * w
    slice_xy: List[Tuple[int, int]] = [
        (coords_flat_2d[3 * i], coords_flat_2d[3 * i + 1])
        for i in range(slice_points)
    ]

    current_hilbert_idx = 0
    last_end_pos: Optional[Tuple[int, int]] = None

    for z in range(t):
        if last_end_pos is None:
            flip_x, flip_y = False, False
        else:
            end_x, end_y = last_end_pos
            if end_x < w / 2 and end_y < h / 2:
                flip_x, flip_y = False, False
            elif end_x >= w / 2 and end_y < h / 2:
                flip_x, flip_y = True, False
            elif end_x < w / 2 and end_y >= h / 2:
                flip_x, flip_y = False, True
            else:
                flip_x, flip_y = True, True

        # Apply flips: the curve at step i lands on (cx, cy) where
        # cx = w-1-x_i if flip_x else x_i (and similarly for y).
        slice_h2l = [0] * slice_points
        for step, (xi, yi) in enumerate(slice_xy):
            cx = (w - 1 - xi) if flip_x else xi
            cy = (h - 1 - yi) if flip_y else yi
            slice_h2l[step] = cy * w + cx

        slice_l2h = [0] * slice_points
        for step, lin in enumerate(slice_h2l):
            slice_l2h[lin] = step

        last_end_lin = slice_h2l[slice_points - 1]
        last_end_pos = (last_end_lin % w, last_end_lin // w)

        # Stitch into the global mapping.
        base = z * h * w
        for lin in range(slice_points):
            local_h = slice_l2h[lin]
            global_lin = base + lin
            global_h = current_hilbert_idx + local_h
            linear_to_hilbert[global_lin] = global_h
            hilbert_to_linear[global_h] = global_lin

        current_hilbert_idx += slice_points

    return linear_to_hilbert, hilbert_to_linear

def sliced_curve(linear_to_hilbert: List[int], hilbert_to_linear: List[int],
                 device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """Materialize (linear_to_hilbert, hilbert_to_linear) lists as int64
    tensors on device."""
    order = torch.tensor(linear_to_hilbert, device=device, dtype=torch.int64)
    inverse_order = torch.tensor(hilbert_to_linear, device=device, dtype=torch.int64)
    return order, inverse_order

def _sliced_gilbert_block_neighbor_mapping(
    t: int, h: int, w: int,
    block_m: int, block_n: int,
    linear_to_hilbert: torch.Tensor,
) -> torch.Tensor:
    """Build the (qblocks, kblocks) boolean block-neighbor mask.

    linear_to_hilbert is an int64 1D tensor of length t*h*w indexed
    by linear position z*h*w + y*w + x. The returned tensor lives on the
    same device.
    """
    if linear_to_hilbert.dtype != torch.int64:
        linear_to_hilbert = linear_to_hilbert.to(torch.int64)
    if linear_to_hilbert.numel() != t * h * w:
        raise ValueError(
            f"linear_to_hilbert length {linear_to_hilbert.numel()} does not "
            f"match t*h*w = {t*h*w}"
        )

    device = linear_to_hilbert.device
    total_points = t * h * w
    qblocks = (total_points + block_m - 1) // block_m
    kblocks = (total_points + block_n - 1) // block_n

    qb_thw = (linear_to_hilbert // int(block_m)).view(t, h, w)
    kb_thw = (linear_to_hilbert // int(block_n)).view(t, h, w)

    mask = torch.zeros(qblocks, kblocks, dtype=torch.bool, device=device)

    # Walk all 27 (dz, dy, dx) deltas in [-1, 0, 1]. The (0,0,0) case sets
    # the self-block entry. Non-zero deltas set both the (curr_qb, nbr_kb)
    # and (nbr_qb, curr_kb) entries — matching the original numba semantics.
    for dz in (-1, 0, 1):
        z_lo = max(0, -dz)
        z_hi = t - max(0, dz)
        if z_lo >= z_hi:
            continue
        for dy in (-1, 0, 1):
            y_lo = max(0, -dy)
            y_hi = h - max(0, dy)
            if y_lo >= y_hi:
                continue
            for dx in (-1, 0, 1):
                x_lo = max(0, -dx)
                x_hi = w - max(0, dx)
                if x_lo >= x_hi:
                    continue

                curr_qb = qb_thw[z_lo:z_hi, y_lo:y_hi, x_lo:x_hi].reshape(-1)
                curr_kb = kb_thw[z_lo:z_hi, y_lo:y_hi, x_lo:x_hi].reshape(-1)
                nbr_qb = qb_thw[z_lo + dz:z_hi + dz,
                                y_lo + dy:y_hi + dy,
                                x_lo + dx:x_hi + dx].reshape(-1)
                nbr_kb = kb_thw[z_lo + dz:z_hi + dz,
                                y_lo + dy:y_hi + dy,
                                x_lo + dx:x_hi + dx].reshape(-1)

                mask[curr_qb, nbr_kb] = True
                mask[nbr_qb, curr_kb] = True

    return mask


def sliced_gilbert_block_neighbor_mapping(
    t: int, h: int, w: int,
    block_m: int, block_n: int,
    device: torch.device,
    gilbert_mapping: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    transpose_order: Optional[List[int]] = None,
) -> torch.Tensor:
    """Build the block-neighbor mask from the (sliced) Gilbert mapping.

    Returns a boolean tensor of shape (qblocks, kblocks) on device.
    """
    if gilbert_mapping is None:
        l2h, _ = sliced_gilbert_mapping(t, h, w, transpose_order)
        linear_to_hilbert = torch.tensor(l2h, dtype=torch.int64, device=device)
    else:
        lth = gilbert_mapping[0]
        if isinstance(lth, torch.Tensor):
            linear_to_hilbert = lth.to(device=device, dtype=torch.int64)
        else:
            linear_to_hilbert = torch.tensor(list(lth), dtype=torch.int64, device=device)
    return _sliced_gilbert_block_neighbor_mapping(t, h, w, block_m, block_n,
                                                  linear_to_hilbert)
