# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2018 Jakub Červený
# Copyright (c) 2024 abetusk

from re import L
import numpy
from numba import njit
import torch
from typing import Optional, Tuple, List, Set



@njit
def sgn(x):
    return -1 if x < 0 else (1 if x > 0 else 0)


@njit
def _generate3d_impl(out, idx, x, y, z,
                    ax, ay, az,
                    bx, by, bz,
                    cx, cy, cz):
    w = abs(ax + ay + az)
    h = abs(bx + by + bz)
    d = abs(cx + cy + cz)

    (dax, day, daz) = (sgn(ax), sgn(ay), sgn(az))
    (dbx, dby, dbz) = (sgn(bx), sgn(by), sgn(bz))
    (dcx, dcy, dcz) = (sgn(cx), sgn(cy), sgn(cz))

    # trivial row/column fills
    if h == 1 and d == 1:
        for i in range(0, w):
            out[idx, 0] = x
            out[idx, 1] = y
            out[idx, 2] = z
            idx += 1
            (x, y, z) = (x + dax, y + day, z + daz)
        return idx

    if w == 1 and d == 1:
        for i in range(0, h):
            out[idx, 0] = x
            out[idx, 1] = y
            out[idx, 2] = z
            idx += 1
            (x, y, z) = (x + dbx, y + dby, z + dbz)
        return idx

    if w == 1 and h == 1:
        for i in range(0, d):
            out[idx, 0] = x
            out[idx, 1] = y
            out[idx, 2] = z
            idx += 1
            (x, y, z) = (x + dcx, y + dcy, z + dcz)
        return idx

    (ax2, ay2, az2) = (ax//2, ay//2, az//2)
    (bx2, by2, bz2) = (bx//2, by//2, bz//2)
    (cx2, cy2, cz2) = (cx//2, cy//2, cz//2)

    w2 = abs(ax2 + ay2 + az2)
    h2 = abs(bx2 + by2 + bz2)
    d2 = abs(cx2 + cy2 + cz2)

    # prefer even steps
    if (w2 % 2) and (w > 2):
       (ax2, ay2, az2) = (ax2 + dax, ay2 + day, az2 + daz)

    if (h2 % 2) and (h > 2):
       (bx2, by2, bz2) = (bx2 + dbx, by2 + dby, bz2 + dbz)

    if (d2 % 2) and (d > 2):
       (cx2, cy2, cz2) = (cx2 + dcx, cy2 + dcy, cz2 + dcz)

    # wide case, split in w only
    if (2*w > 3*h) and (2*w > 3*d):
       idx = _generate3d_impl(out, idx, x, y, z,
                              ax2, ay2, az2,
                              bx, by, bz,
                              cx, cy, cz)

       idx = _generate3d_impl(out, idx, x+ax2, y+ay2, z+az2,
                              ax-ax2, ay-ay2, az-az2,
                              bx, by, bz,
                              cx, cy, cz)
       return idx

    # do not split in d
    elif 3*h > 4*d:
       idx = _generate3d_impl(out, idx, x, y, z,
                              bx2, by2, bz2,
                              cx, cy, cz,
                              ax2, ay2, az2)

       idx = _generate3d_impl(out, idx, x+bx2, y+by2, z+bz2,
                              ax, ay, az,
                              bx-bx2, by-by2, bz-bz2,
                              cx, cy, cz)

       idx = _generate3d_impl(out, idx,
                              x+(ax-dax)+(bx2-dbx),
                              y+(ay-day)+(by2-dby),
                              z+(az-daz)+(bz2-dbz),
                              -bx2, -by2, -bz2,
                              cx, cy, cz,
                              -(ax-ax2), -(ay-ay2), -(az-az2))
       return idx

    # do not split in h
    elif 3*d > 4*h:
       idx = _generate3d_impl(out, idx, x, y, z,
                              cx2, cy2, cz2,
                              ax2, ay2, az2,
                              bx, by, bz)

       idx = _generate3d_impl(out, idx, x+cx2, y+cy2, z+cz2,
                              ax, ay, az,
                              bx, by, bz,
                              cx-cx2, cy-cy2, cz-cz2)

       idx = _generate3d_impl(out, idx,
                              x+(ax-dax)+(cx2-dcx),
                              y+(ay-day)+(cy2-dcy),
                              z+(az-daz)+(cz2-dcz),
                              -cx2, -cy2, -cz2,
                              -(ax-ax2), -(ay-ay2), -(az-az2),
                              bx, by, bz)
       return idx

    # regular case, split in all w/h/d
    else:
       idx = _generate3d_impl(out, idx, x, y, z,
                              bx2, by2, bz2,
                              cx2, cy2, cz2,
                              ax2, ay2, az2)

       idx = _generate3d_impl(out, idx, x+bx2, y+by2, z+bz2,
                              cx, cy, cz,
                              ax2, ay2, az2,
                              bx-bx2, by-by2, bz-bz2)

       idx = _generate3d_impl(out, idx,
                              x+(bx2-dbx)+(cx-dcx),
                              y+(by2-dby)+(cy-dcy),
                              z+(bz2-dbz)+(cz-dcz),
                              ax, ay, az,
                              -bx2, -by2, -bz2,
                              -(cx-cx2), -(cy-cy2), -(cz-cz2))

       idx = _generate3d_impl(out, idx,
                              x+(ax-dax)+bx2+(cx-dcx),
                              y+(ay-day)+by2+(cy-dcy),
                              z+(az-daz)+bz2+(cz-dcz),
                              -cx, -cy, -cz,
                              -(ax-ax2), -(ay-ay2), -(az-az2),
                              bx-bx2, by-by2, bz-bz2)

       idx = _generate3d_impl(out, idx,
                              x+(ax-dax)+(bx2-dbx),
                              y+(ay-day)+(by2-dby),
                              z+(az-daz)+(bz2-dbz),
                              -bx2, -by2, -bz2,
                              cx2, cy2, cz2,
                              -(ax-ax2), -(ay-ay2), -(az-az2))
       return idx


@njit
def gilbert3d(width, height, depth):
    """
    Generalized Hilbert ('Gilbert') space-filling curve for arbitrary-sized
    3D rectangular grids. Returns discrete 3D coordinates filling a cuboid
    of size (width x height x depth). Even sizes are recommended in 3D.

    Returns:
        out: array of shape (width*height*depth, 3), dtype int64, rows are (x,y,z).
    """
    n = width * height * depth
    out = numpy.empty((n, 3), dtype=numpy.int64)

    if width >= height and width >= depth:
        _generate3d_impl(out, 0, 0, 0, 0,
                         width, 0, 0,
                         0, height, 0,
                         0, 0, depth)
    elif height >= width and height >= depth:
        _generate3d_impl(out, 0, 0, 0, 0,
                         0, height, 0,
                         width, 0, 0,
                         0, 0, depth)
    else:  # depth >= width and depth >= height
        _generate3d_impl(out, 0, 0, 0, 0,
                         0, 0, depth,
                         width, 0, 0,
                         0, height, 0)

    return out


def curve(depth: int, height: int, width: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    coords = gilbert3d(width, height, depth)
    inverse_order = coords[:, 2] * height * width + coords[:, 1] * width + coords[:, 0]
    inverse_order = torch.from_numpy(inverse_order).to(device=device)
    order = torch.argsort(inverse_order)
    return order, inverse_order


@njit
def gilbert_xyz2d(x, y, z, width, height, depth):
    """
    Generalized Hilbert ('Gilbert') space-filling curve for arbitrary-sized
    3D rectangular grids. Generates discrete 3D coordinates to fill a cuboid
    of size (width x height x depth). Even sizes are recommended in 3D.
    """

    if width >= height and width >= depth:
       return gilbert_xyz2d_r(0,x,y,z,
                              0, 0, 0,
                              width, 0, 0,
                              0, height, 0,
                              0, 0, depth)

    elif height >= width and height >= depth:
       return gilbert_xyz2d_r(0,x,y,z,
                              0, 0, 0,
                              0, height, 0,
                              width, 0, 0,
                              0, 0, depth)

    else: # depth >= width and depth >= height
       return gilbert_xyz2d_r(0,x,y,z,
                              0, 0, 0,
                              0, 0, depth,
                              width, 0, 0,
                              0, height, 0)


@njit
def in_bounds(x, y, z, x_s, y_s, z_s, ax, ay, az, bx, by, bz, cx, cy, cz):

    dx = ax + bx + cx
    dy = ay + by + cy
    dz = az + bz + cz

    if dx < 0:
        if (x > x_s) or (x <= (x_s + dx)): return False
    else:
        if (x < x_s) or (x >= (x_s + dx)): return False

    if dy < 0:
        if (y > y_s) or (y <= (y_s + dy)): return False
    else:
        if (y < y_s) or (y >= (y_s + dy)): return False

    if dz <0:
        if (z > z_s) or (z <= (z_s + dz)): return False
    else:
        if (z < z_s) or (z >= (z_s + dz)): return False

    return True


@njit
def gilbert_xyz2d_r(cur_idx,
                    x_dst,y_dst,z_dst,
                    x, y, z,
                    ax, ay, az,
                    bx, by, bz,
                    cx, cy, cz):

    w = abs(ax + ay + az)
    h = abs(bx + by + bz)
    d = abs(cx + cy + cz)

    (dax, day, daz) = (sgn(ax), sgn(ay), sgn(az)) # unit major direction ("right")
    (dbx, dby, dbz) = (sgn(bx), sgn(by), sgn(bz)) # unit ortho direction ("forward")
    (dcx, dcy, dcz) = (sgn(cx), sgn(cy), sgn(cz)) # unit ortho direction ("up")

    # trivial row/column fills
    if h == 1 and d == 1:
        return cur_idx + (dax*(x_dst - x)) + (day*(y_dst - y)) + (daz*(z_dst - z))

    if w == 1 and d == 1:
        return cur_idx + (dbx*(x_dst - x)) + (dby*(y_dst - y)) + (dbz*(z_dst - z))

    if w == 1 and h == 1:
        return cur_idx + (dcx*(x_dst - x)) + (dcy*(y_dst - y)) + (dcz*(z_dst - z))

    (ax2, ay2, az2) = (ax//2, ay//2, az//2)
    (bx2, by2, bz2) = (bx//2, by//2, bz//2)
    (cx2, cy2, cz2) = (cx//2, cy//2, cz//2)

    w2 = abs(ax2 + ay2 + az2)
    h2 = abs(bx2 + by2 + bz2)
    d2 = abs(cx2 + cy2 + cz2)

    # prefer even steps
    if (w2 % 2) and (w > 2):
       (ax2, ay2, az2) = (ax2 + dax, ay2 + day, az2 + daz)

    if (h2 % 2) and (h > 2):
       (bx2, by2, bz2) = (bx2 + dbx, by2 + dby, bz2 + dbz)

    if (d2 % 2) and (d > 2):
       (cx2, cy2, cz2) = (cx2 + dcx, cy2 + dcy, cz2 + dcz)

    # wide case, split in w only
    if (2*w > 3*h) and (2*w > 3*d):
        if in_bounds(x_dst,y_dst,z_dst,
                     x,y,z,
                     ax2,ay2,az2,
                     bx,by,bz,
                     cx,cy,cz):
            return gilbert_xyz2d_r(cur_idx,
                                   x_dst,y_dst,z_dst,
                                   x, y, z,
                                   ax2, ay2, az2,
                                   bx, by, bz,
                                   cx, cy, cz)
        cur_idx += abs( (ax2 + ay2 + az2)*(bx + by + bz)*(cx + cy + cz) )

        return gilbert_xyz2d_r(cur_idx,
                               x_dst,y_dst,z_dst,
                               x+ax2, y+ay2, z+az2,
                               ax-ax2, ay-ay2, az-az2,
                               bx, by, bz,
                               cx, cy, cz)

    # do not split in d
    elif 3*h > 4*d:
        if in_bounds(x_dst,y_dst,z_dst,
                     x,y,z,
                     bx2,by2,bz2,
                     cx,cy,cz,
                     ax2,ay2,az2):
            return gilbert_xyz2d_r(cur_idx,
                                   x_dst,y_dst,z_dst,
                                   x, y, z,
                                   bx2, by2, bz2,
                                   cx, cy, cz,
                                   ax2, ay2, az2)
        cur_idx += abs( (bx2 + by2 + bz2)*(cx + cy + cz)*(ax2 + ay2 + az2) )

        if in_bounds(x_dst,y_dst,z_dst,
                     x+bx2,y+by2,z+bz2,
                     ax,ay,az,
                     bx-bx2,by-by2,bz-bz2,
                     cx,cy,cz):
            return gilbert_xyz2d_r(cur_idx,
                                   x_dst,y_dst,z_dst,
                                   x+bx2, y+by2, z+bz2,
                                   ax, ay, az,
                                   bx-bx2, by-by2, bz-bz2,
                                   cx, cy, cz)
        cur_idx += abs( (ax + ay + az)*((bx - bx2) + (by - by2) + (bz - bz2))*(cx + cy + cz) )

        return gilbert_xyz2d_r(cur_idx,
                               x_dst,y_dst,z_dst,
                               x+(ax-dax)+(bx2-dbx),
                               y+(ay-day)+(by2-dby),
                               z+(az-daz)+(bz2-dbz),
                               -bx2, -by2, -bz2,
                               cx, cy, cz,
                               -(ax-ax2), -(ay-ay2), -(az-az2))

    # do not split in h
    elif 3*d > 4*h:
        if in_bounds(x_dst,y_dst,z_dst,
                     x,y,z,
                     cx2,cy2,cz2,
                     ax2,ay2,az2, bx,by,bz):
            return gilbert_xyz2d_r(cur_idx,
                                   x_dst,y_dst,z_dst,
                                   x, y, z,
                                   cx2, cy2, cz2,
                                   ax2, ay2, az2,
                                   bx, by, bz)
        cur_idx += abs( (cx2 + cy2 + cz2)*(ax2 + ay2 + az2)*(bx + by + bz) )

        if in_bounds(x_dst,y_dst,z_dst,
                     x+cx2,y+cy2,z+cz2,
                     ax,ay,az, bx,by,bz,
                     cx-cx2,cy-cy2,cz-cz2):
            return gilbert_xyz2d_r(cur_idx,
                                   x_dst,y_dst,z_dst,
                                   x+cx2, y+cy2, z+cz2,
                                   ax, ay, az,
                                   bx, by, bz,
                                   cx-cx2, cy-cy2, cz-cz2)
        cur_idx += abs( (ax + ay + az)*(bx + by + bz)*((cx - cx2) + (cy - cy2) + (cz - cz2)) )

        return gilbert_xyz2d_r(cur_idx,
                               x_dst,y_dst,z_dst,
                               x+(ax-dax)+(cx2-dcx),
                               y+(ay-day)+(cy2-dcy),
                               z+(az-daz)+(cz2-dcz),
                               -cx2, -cy2, -cz2,
                               -(ax-ax2), -(ay-ay2), -(az-az2),
                               bx, by, bz)

    # regular case, split in all w/h/d
    if in_bounds(x_dst,y_dst,z_dst,
                 x,y,z,
                 bx2,by2,bz2,
                 cx2,cy2,cz2,
                 ax2,ay2,az2):
        return gilbert_xyz2d_r(cur_idx,x_dst,y_dst,z_dst,
                              x, y, z,
                              bx2, by2, bz2,
                              cx2, cy2, cz2,
                              ax2, ay2, az2)
    cur_idx += abs( (bx2 + by2 + bz2)*(cx2 + cy2 + cz2)*(ax2 + ay2 + az2) )

    if in_bounds(x_dst,y_dst,z_dst,
                 x+bx2, y+by2, z+bz2,
                 cx, cy, cz,
                 ax2, ay2, az2,
                 bx-bx2, by-by2, bz-bz2):
        return gilbert_xyz2d_r(cur_idx,
                              x_dst,y_dst,z_dst,
                              x+bx2, y+by2, z+bz2,
                              cx, cy, cz,
                              ax2, ay2, az2,
                              bx-bx2, by-by2, bz-bz2)
    cur_idx += abs( (cx + cy + cz)*(ax2 + ay2 + az2)*((bx - bx2) + (by - by2) + (bz - bz2)) )

    if in_bounds(x_dst,y_dst,z_dst,
                 x+(bx2-dbx)+(cx-dcx),
                 y+(by2-dby)+(cy-dcy),
                 z+(bz2-dbz)+(cz-dcz),
                 ax, ay, az,
                 -bx2, -by2, -bz2,
                 -(cx-cx2), -(cy-cy2), -(cz-cz2)):
        return gilbert_xyz2d_r(cur_idx,
                               x_dst,y_dst,z_dst,
                               x+(bx2-dbx)+(cx-dcx),
                               y+(by2-dby)+(cy-dcy),
                               z+(bz2-dbz)+(cz-dcz),
                               ax, ay, az,
                               -bx2, -by2, -bz2,
                               -(cx-cx2), -(cy-cy2), -(cz-cz2))
    cur_idx += abs( (ax + ay + az)*(-bx2 - by2 - bz2)*(-(cx - cx2) - (cy - cy2) - (cz - cz2)) )

    if in_bounds(x_dst,y_dst,z_dst,
                 x+(ax-dax)+bx2+(cx-dcx),
                 y+(ay-day)+by2+(cy-dcy),
                 z+(az-daz)+bz2+(cz-dcz),
                 -cx, -cy, -cz,
                 -(ax-ax2), -(ay-ay2), -(az-az2),
                 bx-bx2, by-by2, bz-bz2):
        return gilbert_xyz2d_r(cur_idx,
                               x_dst,y_dst,z_dst,
                               x+(ax-dax)+bx2+(cx-dcx),
                               y+(ay-day)+by2+(cy-dcy),
                               z+(az-daz)+bz2+(cz-dcz),
                               -cx, -cy, -cz,
                               -(ax-ax2), -(ay-ay2), -(az-az2),
                               bx-bx2, by-by2, bz-bz2)
    cur_idx += abs( (-cx - cy - cz)*(-(ax - ax2) - (ay - ay2) - (az - az2))*((bx - bx2) + (by - by2) + (bz - bz2)) )

    return gilbert_xyz2d_r(cur_idx,
                           x_dst,y_dst,z_dst,
                           x+(ax-dax)+(bx2-dbx),
                           y+(ay-day)+(by2-dby),
                           z+(az-daz)+(bz2-dbz),
                           -bx2, -by2, -bz2,
                           cx2, cy2, cz2,
                           -(ax-ax2), -(ay-ay2), -(az-az2))


@njit
def transpose_gilbert_mapping(dims, order=None):
    """
    Create mapping between linear indices and Gilbert curve indices, supporting different axis orders
    
    Parameters:
        dims: List or tuple of three dimensions, e.g. [t, h, w]
        order: Order of axes, default is [0,1,2], representing [t,h,w]
               Can be specified as [2,1,0] to represent [w,h,t] or other orders
        
    Returns:
        linear_to_hilbert: List of length dims[0]*dims[1]*dims[2], storing Gilbert curve indices corresponding to linear indices
        hilbert_to_linear: List of length dims[0]*dims[1]*dims[2], storing linear indices corresponding to Gilbert curve indices
    """
    if len(dims) != 3:
        raise ValueError("Dimensions must be three-dimensional")
    
    # If no order specified, use default [0,1,2]
    if order is None:
        order = [0, 1, 2]
    
    if len(order) != 3 or set(order) != {0, 1, 2}:
        raise ValueError("order must be a permutation of 0,1,2")
    
    # Extract original dimensions
    dims_array = numpy.array(dims)
    
    # Rearrange dimensions according to order
    t, h, w = dims_array[order]
    
    # Calculate total number of points
    total_points = numpy.prod(dims)
    
    # Initialize mapping arrays
    linear_to_hilbert = [0] * total_points
    hilbert_to_linear = [0] * total_points
        
    # Calculate Gilbert indices for all points
    # Create iterator for all coordinates
    coords_iter = numpy.ndindex(*dims)
    
    for linear_idx, coords in enumerate(coords_iter):
        # Rearrange coordinates according to order
        # For example, if order=[2,1,0], then x corresponds to coords[2], y to coords[1], z to coords[0]
        transposed_coords = [coords[order[2]], coords[order[1]], coords[order[0]]]
        
        # Calculate Gilbert curve index
        x, y, z = transposed_coords
        hilbert_idx = gilbert_xyz2d(x, y, z, w, h, t)
        
        # Set mapping
        linear_to_hilbert[linear_idx] = hilbert_idx
        hilbert_to_linear[hilbert_idx] = linear_idx
    
    return linear_to_hilbert, hilbert_to_linear


@njit
def sliced_gilbert_mapping(t: int, h: int, w: int, transpose_order: Optional[List[int]] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create a sliced Gilbert curve mapping, prioritizing scanning in spatial dimensions (h,w),
    then continuous in time dimension (t).
    Ensures continuous connection between adjacent time slices.
    
    Parameters:
        t: Size of the first dimension
        h: Size of the second dimension
        w: Size of the third dimension
        
    Returns:
        linear_to_hilbert: List of length t*h*w, storing Gilbert curve indices corresponding to linear indices
        hilbert_order: List of length t*h*w, storing linear indices corresponding to Gilbert curve indices
    """
    dims = [t, h, w]

    if transpose_order is None:
        # Standard Gilbert mapping, no transposition
        total_points = t * h * w
        
        # Initialize mapping arrays
        linear_to_hilbert = [0] * total_points
        hilbert_to_linear = [0] * total_points
        
        # Calculate Gilbert curve for each time slice
        current_hilbert_idx = 0
        last_end_pos = None  # Record end position of previous slice
        
        for z in range(t):
            # Calculate Gilbert curve for current slice
            slice_points = h * w
            slice_linear_to_hilbert = [0] * slice_points
            slice_hilbert_to_linear = [0] * slice_points
            
            # Determine starting position and direction for current slice
            if last_end_pos is not None:
                # Based on end position of previous slice, determine starting position and direction
                end_x, end_y = last_end_pos
                # Choose closest corner point as starting point
                if end_x < w/2 and end_y < h/2:
                    start_x, start_y = 0, 0
                    flip_x, flip_y = False, False
                elif end_x >= w/2 and end_y < h/2:
                    start_x, start_y = w-1, 0
                    flip_x, flip_y = True, False
                elif end_x < w/2 and end_y >= h/2:
                    start_x, start_y = 0, h-1
                    flip_x, flip_y = False, True
                else:
                    start_x, start_y = w-1, h-1
                    flip_x, flip_y = True, True
            else:
                # First slice starts from (0,0)
                start_x, start_y = 0, 0
                flip_x, flip_y = False, False
            
            # Calculate Gilbert curve for current slice
            for y in range(h):
                for x in range(w):
                    # Calculate actual coordinates (considering flipping)
                    actual_x = w-1-x if flip_x else x
                    actual_y = h-1-y if flip_y else y
                    
                    # Calculate linear index (row-major order: y*w + x)
                    linear_idx = y * w + x
                    
                    # Calculate Gilbert curve index
                    hilbert_idx = gilbert_xyz2d(actual_x, actual_y, 0, w, h, 1)
                    
                    # Set mapping
                    slice_linear_to_hilbert[linear_idx] = hilbert_idx
                    slice_hilbert_to_linear[hilbert_idx] = linear_idx
            
            # Record end position of current slice
            last_end_idx = slice_hilbert_to_linear[slice_points-1]
            last_end_y = last_end_idx // w
            last_end_x = last_end_idx % w
            last_end_pos = (last_end_x, last_end_y)
            
            # Add current slice mapping to overall mapping
            for y in range(h):
                for x in range(w):
                    # Calculate global linear index
                    global_linear_idx = z * h * w + y * w + x
                    
                    # Calculate local linear index within current slice
                    local_linear_idx = y * w + x
                    
                    # Get Gilbert index within current slice
                    local_hilbert_idx = slice_linear_to_hilbert[local_linear_idx]
                    
                    # Set global mapping
                    linear_to_hilbert[global_linear_idx] = current_hilbert_idx + local_hilbert_idx
                    hilbert_to_linear[current_hilbert_idx + local_hilbert_idx] = global_linear_idx
            
            # Update starting index for next slice
            current_hilbert_idx += slice_points
    else:
        # Use transposed mapping
        linear_to_hilbert, hilbert_to_linear = transpose_gilbert_mapping(dims, transpose_order)
            
    return linear_to_hilbert, hilbert_to_linear


def sliced_curve(linear_to_hilbert: List[int], hilbert_to_linear: List[int], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create a sliced curve mapping, prioritizing scanning in spatial dimensions (h,w),
    then continuous in time dimension (t).
    Ensures continuous connection between adjacent time slices.
    """
    order = torch.tensor(linear_to_hilbert, device=device, dtype=torch.int64)
    inverse_order = torch.tensor(hilbert_to_linear, device=device, dtype=torch.int64)
    return order, inverse_order


@njit
def _sliced_gilbert_block_neighbor_mapping(
    t: int, h: int, w: int,
    block_m: int, block_n: int,
    linear_to_hilbert: numpy.ndarray,
) -> numpy.ndarray:
    """
    Numba nopython core: build (qblocks, kblocks) boolean mask of block neighbors.
    linear_to_hilbert: 1D int64 array of length t*h*w (Gilbert curve index per linear index).
    """
    total_points = t * h * w
    qblocks = (total_points + block_m - 1) // block_m
    kblocks = (total_points + block_n - 1) // block_n

    block_color_map = numpy.zeros((w, h, t, 2), dtype=numpy.int64)
    for z in range(t):
        for y in range(h):
            for x in range(w):
                linear_idx = z * h * w + y * w + x
                hilbert_idx = linear_to_hilbert[linear_idx]
                block_color_map[x, y, z, 0] = hilbert_idx // block_m
                block_color_map[x, y, z, 1] = hilbert_idx // block_n

    mask = numpy.zeros((qblocks, kblocks), dtype=numpy.bool_)
    for x in range(w):
        for y in range(h):
            for z in range(t):
                qb = block_color_map[x, y, z, 0]
                kb = block_color_map[x, y, z, 1]
                mask[qb, kb] = True
                for dx in range(-1, 2):
                    nx = x + dx
                    if nx < 0 or nx >= w:
                        continue
                    for dy in range(-1, 2):
                        ny = y + dy
                        if ny < 0 or ny >= h:
                            continue
                        for dz in range(-1, 2):
                            nz = z + dz
                            if nz < 0 or nz >= t:
                                continue
                            if dx == 0 and dy == 0 and dz == 0:
                                continue
                            neighbor_qb = block_color_map[nx, ny, nz, 0]
                            neighbor_kb = block_color_map[nx, ny, nz, 1]
                            mask[neighbor_qb, neighbor_kb] = True
    return mask


def sliced_gilbert_block_neighbor_mapping(
    t: int, h: int, w: int,
    block_m: int, block_n: int,
    gilbert_mapping: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    transpose_order: Optional[List[int]] = None
) -> numpy.ndarray:
    """
    Build block-neighbor mask from Gilbert curve mapping.
    Resolves optional mapping; returns boolean mask of shape (qblocks, kblocks).
    """
    if gilbert_mapping is None:
        linear_to_hilbert, _ = sliced_gilbert_mapping(t, h, w, transpose_order)
        arr = numpy.array(linear_to_hilbert, dtype=numpy.int64)
    else:
        lth = gilbert_mapping[0]
        if isinstance(lth, torch.Tensor):
            arr = lth.cpu().numpy().astype(numpy.int64)
        else:
            arr = numpy.asarray(lth, dtype=numpy.int64)
    return _sliced_gilbert_block_neighbor_mapping(t, h, w, block_m, block_n, arr)
