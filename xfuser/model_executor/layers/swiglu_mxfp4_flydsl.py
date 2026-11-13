"""SwiGLU -> MXFP4 quantization (FlyDSL), the input of FLUX.3's feed-forward output GEMM.

A dense-row port of AITER's MoE stage-1 epilogue ``silu_and_mul_fq``
(``aiter/ops/flydsl/kernels/silu_and_mul_fq.py``), reusing its tiling, activation and
MX helpers. Differences from that kernel:

- rows are addressed directly (no sorted token ids) with a runtime row stride, so the
  input can be a column slice of a wider GEMM output;
- the packed values and scales go to columns ``col_offset`` onwards of rows of
  ``out_cols`` columns, so the SwiGLU can fill part of a wider GEMM input;
- the activation is rounded to bf16 before quantizing, and the block amax is floored at
  1e-10, so the result matches ``aiter.get_hip_quant(per_1x32)(y, shuffle=True)`` of the
  bf16 SwiGLU ``y``.

The scale layout is the HIP quantization's shuffled one.
"""

# NOTE: no ``from __future__ import annotations``: flydsl detects runtime Int32 kernel
# parameters from their annotations.

from functools import lru_cache

import torch

from xfuser.model_executor.layers.flux3_block_flydsl import VEC, WAVE_SIZE, mxfp4_pack, mxfp4_scale_offset
from xfuser.model_executor.layers.flydsl_utils import get_device_wave_size

try:
    import flydsl.compiler as flyc
    import flydsl.expr as fx
    from flydsl.expr import range_constexpr
    from flydsl.expr.typing import Int32, T

    from aiter.ops.flydsl.kernels.act import sigmoid_f32
    from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled, ptr_arg

    _HAS_FLYDSL = True
except Exception:  # pragma: no cover - only where FlyDSL/AITER are absent
    _HAS_FLYDSL = False

BLOCK_THREADS = 256


if _HAS_FLYDSL:
    # TODO(aiter): upstream the differences listed above (dense rows with a runtime row stride,
    # a column offset into wider rows, bf16 rounding before quantizing, the 1e-10 amax floor) as
    # options of AITER's silu_and_mul_fq, and call that kernel once it lands.
    @lru_cache(maxsize=16)
    def _build_kernel(cols: int, out_cols: int, col_offset: int, scale_cols: int):
        """Launcher for ``[rows, 2 * cols]`` bf16 input rows into ``[rows, out_cols / 2]``
        FP4 values and ``[*, scale_cols]`` shuffled E8M0 scales, from column ``col_offset``."""
        cols_per_iter = BLOCK_THREADS * VEC
        n_iters = (cols + cols_per_iter - 1) // cols_per_iter
        pad_cols = n_iters * cols_per_iter
        n32 = scale_cols * 32
        group_offset = col_offset // 32
        out_row_bytes = out_cols // 2

        @flyc.kernel(name=f"swiglu_mxfp4_c{cols}_o{out_cols}_off{col_offset}_s{scale_cols}_flydsl")
        def kernel(x: fx.Pointer, fp4: fx.Pointer, scale: fx.Pointer, x_rs: Int32):
            row = fx.Int32(fx.block_idx.x)
            tid = fx.Int32(fx.thread_idx.x)
            global_space = fx.AddressSpace.Global
            scale_i8 = fx.inttoptr(
                fx.PointerType.get(T.i8, address_space=global_space, alignment=1),
                fx.Int64(fx.ptrtoint(scale)),
            )

            def buffer_view(base_i64, elem_ir, shape, strides, nbytes):
                pointer_type = fx.PointerType.get(elem_ir, address_space=global_space, alignment=2)
                view = fx.make_view(fx.inttoptr(pointer_type, base_i64), fx.make_layout(shape, strides))
                return fx.rocdl.make_buffer_tensor(view, num_records_bytes=fx.Int64(nbytes))

            copy_ld = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.BFloat16)
            tile_ld, tv_ld = fx.make_layout_tv(
                fx.make_layout((1, BLOCK_THREADS), (BLOCK_THREADS, 1)),
                fx.make_layout((1, VEC), (VEC, 1)),
            )
            thr_ld = fx.make_tiled_copy(copy_ld, tv_ld, tile_ld).get_slice(tid)
            store_bytes = VEC // 2
            copy_st = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Int8)
            tile_st, tv_st = fx.make_layout_tv(
                fx.make_layout((1, BLOCK_THREADS), (BLOCK_THREADS, 1)),
                fx.make_layout((1, store_bytes), (store_bytes, 1)),
            )
            thr_st = fx.make_tiled_copy(copy_st, tv_st, tile_st).get_slice(tid)

            x_row = fx.Int64(fx.ptrtoint(x)) + fx.Int64(row) * fx.Int64(x_rs) * fx.Int64(2)
            gate_bt = buffer_view(x_row, fx.BFloat16.ir_type, (1, pad_cols), (pad_cols, 1), cols * 2)
            up_bt = buffer_view(x_row + fx.Int64(cols * 2), fx.BFloat16.ir_type, (1, pad_cols), (pad_cols, 1), cols * 2)
            out_row = fx.Int64(fx.ptrtoint(fp4)) + fx.Int64(row) * fx.Int64(out_row_bytes) + fx.Int64(col_offset // 2)
            out_bt = buffer_view(out_row, fx.Int8.ir_type, (1, pad_cols // 2), (pad_cols // 2, 1), cols // 2)

            vec_bf16 = T.vec(VEC, T.bf16)
            vec_f32 = T.vec(VEC, T.f32)

            for it in range_constexpr(n_iters):
                col0 = tid * VEC + it * cols_per_iter

                def part_ld(buf, it=it):
                    return thr_ld.partition_S(fx.slice(fx.zipped_divide(buf, tile_ld), (None, (0, it))))

                def part_st(buf, it=it):
                    return thr_st.partition_D(fx.slice(fx.zipped_divide(buf, tile_st), (None, (0, it))))

                if col0 < cols:
                    p_gate = part_ld(gate_bt)
                    p_up = part_ld(up_bt)
                    gate_frag = fx.make_fragment_like(p_gate)
                    up_frag = fx.make_fragment_like(p_up)
                    fx.copy(copy_ld, p_gate, gate_frag)
                    fx.copy(copy_ld, p_up, up_frag)
                    gate = fx.Vector(fx.memref_load_vec(gate_frag)).reshape((VEC,)).extf(vec_f32)
                    up = fx.Vector(fx.memref_load_vec(up_frag)).reshape((VEC,)).extf(vec_f32)

                    act = [gate[i] * sigmoid_f32(gate[i]) * up[i] for i in range_constexpr(VEC)]
                    rounded = fx.Vector.from_elements([a.ir_value() for a in act], dtype=fx.Float32)
                    rounded = fx.Vector(rounded.truncf(vec_bf16)).extf(vec_f32)
                    exponent, packed = mxfp4_pack([rounded[i] for i in range_constexpr(VEC)])
                    p_out = part_st(out_bt)
                    out_frag = fx.make_fragment_like(p_out)
                    fx.memref_store_vec(fx.Vector.from_elements([packed], dtype=fx.Int32).bitcast(fx.Int8), out_frag)
                    fx.copy(copy_st, out_frag, p_out)

                    if (col0 & 31) == 0:
                        offset = mxfp4_scale_offset(row, fx.Int32(group_offset) + (col0 >> 5), n32)
                        scale_i8[fx.Int32(offset)] = exponent.to(fx.Int8)

        @flyc.jit
        def launch(
            x: fx.Pointer, fp4: fx.Pointer, scale: fx.Pointer, x_rs: fx.Int32, rows: fx.Int32, stream: fx.Stream
        ):
            kernel(x, fp4, scale, x_rs).launch(grid=(fx.Int64(rows), 1, 1), block=(BLOCK_THREADS, 1, 1), stream=stream)

        return launch


def supported(rows: torch.Tensor, fp4: torch.Tensor, col_offset: int) -> bool:
    """Whether ``launch_swiglu_mxfp4`` takes these ``[M, 2n]`` rows into ``fp4`` at ``col_offset``."""
    if not _HAS_FLYDSL or not rows.is_cuda or rows.dtype != torch.bfloat16 or rows.dim() != 2:
        return False
    if get_device_wave_size(rows) != WAVE_SIZE:
        return False
    cols = rows.shape[1] // 2
    if cols % 32 or col_offset % 32 or rows.stride(1) != 1:
        return False
    # 128-bit loads need 16-byte aligned rows.
    return rows.stride(0) % VEC == 0 and rows.storage_offset() % VEC == 0 and fp4.is_contiguous()


def launch_swiglu_mxfp4(rows: torch.Tensor, fp4: torch.Tensor, quant_scale: torch.Tensor, col_offset: int) -> None:
    """Quantize ``silu(rows[:, :n]) * rows[:, n:]`` into columns ``col_offset`` onwards of
    ``fp4`` ``[M, K / 2]`` and ``quant_scale``, the buffers of ``mxfp4_fused_quant.mxfp4_buffers``."""
    if not rows.shape[0]:
        return
    launcher = _build_kernel(rows.shape[1] // 2, fp4.shape[1] * 2, col_offset, quant_scale.shape[1])
    _run_compiled(
        launcher,
        ptr_arg(rows),
        ptr_arg(fp4),
        ptr_arg(quant_scale),
        rows.stride(0),
        rows.shape[0],
        torch.cuda.current_stream(rows.device),
    )
