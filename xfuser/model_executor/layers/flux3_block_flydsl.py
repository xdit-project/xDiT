"""FlyDSL kernels for FLUX.3's fused block ops (``fused_block_ops``, ``mxfp4_fused_quant``).

- ``launch_qk_norm_rope``: the q/k RMSNorm and interleaved RoPE of a block's projected
  ``[b, s, h * d]`` q, k (fp32 math, one rounding), into the Ulysses send buffer
  ``[P, s, b, 3, h/P, d]`` with v copied alongside, or into ``[b, s, 2, h, d]`` q and k.
  With FP8 scales, q and k are also Hadamard-rotated and q, k, v quantized per tensor.
- ``launch_block_rows``: one row of the joint ``[b, s0 + s1]`` sequence per wave: the gated
  residual ``x + gate * (branch0 [+ branch1])`` (optional), then
  ``(1 + scale) * layer_norm(.) + shift``, in bf16 or as MXFP4.
- ``launch_seq_major_output_mxfp4``: the ``[b * s, P * h * d]`` rows of a Ulysses output
  receive buffer ``[P, s, b, h, d]`` (bf16, or FP8 with a descale) as MXFP4.

MXFP4 follows AITER's HIP per-1x32 quantization: the scale is
``ceil_pow2(max(amax, 1e-10) / 6)`` (E8M0, round up), the values round to E2M1 after a
multiply by its exact reciprocal, and the scales go to the HIP kernel's shuffled layout.

One q/k block holds one token (``h * d / 8`` lanes, 8 channels each); one row block is
one wave. Both keep their reductions inside a wave, as ``shuffle_xor`` butterflies.
"""

# NOTE: no ``from __future__ import annotations``: flydsl detects runtime Int32 kernel
# parameters from their annotations.

from functools import lru_cache
from typing import Optional, Sequence, Tuple

import torch

from xfuser.model_executor.layers.flydsl_utils import get_device_wave_size

try:
    import flydsl.compiler as flyc
    import flydsl.expr as fx
    from flydsl.expr import const_expr, range_constexpr
    from flydsl.expr import math as fmath
    from flydsl.expr.rocdl import cvt_pk_f32_fp8, cvt_pk_fp8_f32
    from flydsl.expr.typing import Int32, T

    from aiter.ops.flydsl.kernels.quant_utils import emit_f32_to_e2m1, emit_mx_e8m0_scale
    from aiter.ops.flydsl.kernels.tensor_shim import GTensor, _run_compiled, ptr_arg
    from aiter.utility.mx_types import MxDtypeInt, MxScaleRoundModeInt

    _HAS_FLYDSL = True
except Exception:  # pragma: no cover - only where FlyDSL/AITER are absent
    _HAS_FLYDSL = False

WAVE_SIZE = 64
# Channels per lane: one 128-bit bf16 access.
VEC = 8
ROW_COLS = WAVE_SIZE * VEC
MAX_BLOCK_THREADS = 1024
# Rotation block of the FP8 send (hadamard.rotate_qk).
HADAMARD_BLOCK = 128


# TODO(aiter): have AITER export mxfp4_scale_cols, mxfp4_pack and mxfp4_scale_offset, and use
# its versions once they land. They re-derive the scale padding, E8M0/E2M1 packing and shuffled
# scale layout of get_hip_quant(per_1x32)(..., shuffle=True), so a layout change in AITER
# diverges from xDiT silently wherever the AMD accelerator suite does not run.
def mxfp4_scale_cols(cols: int) -> int:
    """Columns of the shuffled E8M0 scales of ``cols``-column rows, as the HIP quantization pads them."""
    return (cols // 32 + 7) // 8 * 8


# Trace-time helpers, called from the kernel bodies; plain Python loops unroll.


def _addr(ptr, elems, elem_bytes):
    """Address of element ``elems`` (Int64) of the buffer at ``ptr``."""
    return fx.Int64(fx.ptrtoint(ptr)) + elems * fx.Int64(elem_bytes)


def _load_f32(tensor, offset, count, f32_source):
    """``count`` consecutive elements of a bf16 or fp32 ``GTensor`` as fp32 scalars."""
    width = 4 if f32_source else VEC
    values = []
    for start in range(0, count, width):
        part = fx.Vector(tensor.load(offset + start, vec_size=min(width, count - start))).to(fx.Float32)
        values += [part[i] for i in range(min(width, count - start))]
    return values


def _bf16_vector(values):
    return fx.Vector.from_elements([v.ir_value() for v in values], dtype=fx.Float32).truncf(T.vec(len(values), T.bf16))


def _round_bf16(values):
    rounded = fx.Vector(_bf16_vector(values)).to(fx.Float32)
    return [rounded[i] for i in range(len(values))]


def _group_sum(value, width):
    """Sum over aligned groups of ``width`` lanes, every lane holding the result."""
    offset = width // 2
    while offset:
        value = value + value.shuffle_xor(fx.Int32(offset), fx.Int32(width))
        offset //= 2
    return value


def mxfp4_pack(values):
    """The E8M0 exponent and packed E2M1 word of a lane's ``VEC`` fp32 values holding bf16
    ones, the ``32 // VEC`` lanes of each 32-column group adjacent in the wave."""
    amax = fx.Float32(0.0)
    for v in values:
        amax = amax.maximumf(fmath.absf(v))
    dist = 1
    while dist < 32 // VEC:
        amax = amax.maximumf(amax.shuffle_xor(fx.Int32(dist), fx.Int32(WAVE_SIZE)))
        dist *= 2
    amax = amax.maximumf(fx.Float32(1e-10))
    exponent = fx.Int32(
        emit_mx_e8m0_scale(amax.ir_value(), mode=MxScaleRoundModeInt.RoundUp, dtype=MxDtypeInt.FP4_E2M1)
    )
    inverse = ((fx.Int32(254) - exponent) << fx.Int32(23)).bitcast(fx.Float32)
    nibbles = [fx.Int32(emit_f32_to_e2m1((v * inverse).ir_value())) for v in values]
    packed = nibbles[0] | (nibbles[1] << 4)
    for k in range(1, VEC // 2):
        packed = packed | ((nibbles[2 * k] | (nibbles[2 * k + 1] << 4)) << (k * 8))
    return exponent, packed


def mxfp4_scale_offset(row, group, n32):
    """Byte offset of the scale of 32-column ``group`` of ``row`` in the HIP quantization's
    shuffled layout, ``n32`` its scale columns times 32."""
    return (
        (row >> 5) * n32
        + (group >> 3) * 256
        + (group & 3) * 64
        + (row & 15) * 4
        + ((group >> 2) & 1) * 2
        + ((row >> 4) & 1)
    )


def _store_mxfp4(values, row, col, fp4_row, scale_i8, n32, group_offset):
    """Quantize 8 fp32 values holding bf16 ones, columns ``col`` onwards of ``row``; all 4
    lanes of each 32-column group write its (identical) scale."""
    exponent, packed = mxfp4_pack(values)
    fp4_row.store(col >> 3, packed)
    scale_i8[mxfp4_scale_offset(row, fx.Int32(group_offset) + (col >> 5), n32)] = exponent.to(fx.Int8)


def _scale_pointer(scale):
    return fx.inttoptr(
        fx.PointerType.get(T.i8, address_space=fx.AddressSpace.Global, alignment=1), fx.Int64(fx.ptrtoint(scale))
    )


def _pack_fp8(values):
    """fp32 values (a multiple of 4) as the words of the device's packed FP8 format."""
    words = []
    for i in range(0, len(values), 4):
        low = cvt_pk_fp8_f32(res=T.i32, src_a=values[i], src_b=values[i + 1], old=fx.Int32(0), word_sel=False)
        word = cvt_pk_fp8_f32(res=T.i32, src_a=values[i + 2], src_b=values[i + 3], old=low, word_sel=True)
        words.append(fx.Int32(word).ir_value())
    return fx.Vector.from_elements(words, dtype=fx.Int32)


def _unpack_fp8(words):
    """The fp32 values of packed FP8 words."""
    values = []
    for word in words:
        for high in (False, True):
            pair = fx.Vector(cvt_pk_f32_fp8(res=T.vec(2, T.f32), src=word, word_sel=high))
            values += [pair[0], pair[1]]
    return values


def _hardware_fp8_dtype(device: torch.device) -> Optional[torch.dtype]:
    """The FP8 format of the device's conversion instructions."""
    arch = torch.cuda.get_device_properties(device).gcnArchName
    return {"gfx950": torch.float8_e4m3fn, "gfx942": torch.float8_e4m3fnuz}.get(arch.split(":")[0])


if _HAS_FLYDSL:

    @lru_cache(maxsize=32)
    def _build_qk_kernel(
        heads: int,
        head_dim: int,
        heads_per_rank: int,
        eps: float,
        weight_f32: bool,
        rope_f32: bool,
        write_v: bool,
        fp8_dtype: Optional[torch.dtype],
    ):
        lanes_per_head = head_dim // VEC
        threads = heads * lanes_per_head
        inv_head_dim = 1.0 / head_dim
        fp8 = fp8_dtype is not None
        fp8_max = float(torch.finfo(fp8_dtype).max) if fp8 else 0.0
        block_r = head_dim
        lane_stages = [[(i, i | h) for i in range(VEC) if not i & h] for h in (1, 2, 4)]
        cross_masks = [1 << j for j in range((block_r // VEC).bit_length() - 1)]
        rope_bytes = 4 if rope_f32 else 2
        out_bytes = 1 if fp8 else 2
        name = (
            f"flux3_qk_norm_rope_h{heads}_d{head_dim}_p{heads_per_rank}_w{int(weight_f32)}{int(rope_f32)}"
            f"_v{int(write_v)}_{str(fp8_dtype).rsplit('_', 1)[-1] if fp8 else 'bf16'}_flydsl"
        )

        @flyc.kernel(name=name)
        def kernel(
            q: fx.Pointer,
            k: fx.Pointer,
            v: fx.Pointer,
            q_weight: fx.Pointer,
            k_weight: fx.Pointer,
            cos: fx.Pointer,
            sin: fx.Pointer,
            rotation: fx.Pointer,
            q_scale: fx.Pointer,
            k_scale: fx.Pointer,
            v_scale: fx.Pointer,
            out: fx.Pointer,
            seq_len: Int32,
            in_stride_b: Int32,
            in_stride_s: Int32,
            rope_stride_s: Int32,
            out_stride_rank: Int32,
            out_stride_s: Int32,
            out_stride_b: Int32,
            out_stride_which: Int32,
        ):
            token = fx.Int32(fx.block_idx.x)
            tid = fx.Int32(fx.thread_idx.x)
            b = token // seq_len
            s = token % seq_len
            head = tid // lanes_per_head
            lane = tid % lanes_per_head
            dim = lane * VEC

            in_row = fx.Int64(b) * fx.Int64(in_stride_b) + fx.Int64(s) * fx.Int64(in_stride_s)
            in_offset = head * head_dim + dim
            rope_row = fx.Int64(s) * fx.Int64(rope_stride_s)
            rope_dtype = T.f32 if rope_f32 else T.bf16
            weight_dtype = T.f32 if weight_f32 else T.bf16
            cos_v = _load_f32(GTensor(_addr(cos, rope_row, rope_bytes), rope_dtype, (-1,)), dim, VEC, rope_f32)
            sin_v = _load_f32(GTensor(_addr(sin, rope_row, rope_bytes), rope_dtype, (-1,)), dim, VEC, rope_f32)

            out_row = fx.Int64(s) * fx.Int64(out_stride_s) + fx.Int64(b) * fx.Int64(out_stride_b)
            out_offset = (head // heads_per_rank) * out_stride_rank + (head % heads_per_rank) * head_dim + dim
            # FP8 rows are stored as words of 4 values.
            out_t = GTensor(_addr(out, out_row, out_bytes), T.i32 if fp8 else T.bf16, (-1,))

            def norm_rope(x_ptr, w_ptr):
                x = _load_f32(GTensor(_addr(x_ptr, in_row, 2), T.bf16, (-1,)), in_offset, VEC, False)
                w = _load_f32(GTensor(fx.Int64(fx.ptrtoint(w_ptr)), weight_dtype, (-1,)), dim, VEC, weight_f32)
                squares = fx.Float32(0.0)
                for i in range_constexpr(VEC):
                    squares = fmath.fma(x[i], x[i], squares)
                rstd = fmath.rsqrt(_group_sum(squares, lanes_per_head) * inv_head_dim + eps)
                y = [x[i] * rstd * w[i] for i in range_constexpr(VEC)]
                outs = []
                for p in range_constexpr(VEC // 2):
                    even, odd = y[2 * p], y[2 * p + 1]
                    outs.append(fmath.fma(even, cos_v[2 * p], -odd * sin_v[2 * p]))
                    outs.append(fmath.fma(odd, cos_v[2 * p + 1], even * sin_v[2 * p + 1]))
                return outs

            def store_fp8(values, scale_ptr, offset):
                scale = fx.Float32(GTensor(fx.Int64(fx.ptrtoint(scale_ptr)), T.f32, (-1,)).load(0))
                quantized = [(x / scale).maximumf(fx.Float32(-fp8_max)).minimumf(fx.Float32(fp8_max)) for x in values]
                out_t.store(offset >> 2, _pack_fp8(quantized))

            def hadamard(values):
                """``values @ H`` for this lane's part of a head, H the Sylvester Hadamard
                matrix scaled by ``rotation[0, 0]``: 3 butterfly stages in the lane, the rest
                across its lanes."""
                x = list(values)
                for stage in range_constexpr(len(lane_stages)):
                    for pair in range_constexpr(len(lane_stages[stage])):
                        lo, hi = lane_stages[stage][pair]
                        x[lo], x[hi] = x[lo] + x[hi], x[lo] - x[hi]
                for stage in range_constexpr(len(cross_masks)):
                    upper = (lane & cross_masks[stage]) != 0
                    for i in range_constexpr(VEC):
                        partner = x[i].shuffle_xor(fx.Int32(cross_masks[stage]), fx.Int32(lanes_per_head))
                        x[i] = upper.select(partner - x[i], x[i] + partner)
                entry = fx.Vector(GTensor(fx.Int64(fx.ptrtoint(rotation)), T.bf16, (-1,)).load(0, vec_size=2))
                entry = entry.to(fx.Float32)[0]
                return [xi * entry for xi in x]

            q_out = norm_rope(q, q_weight)
            k_out = norm_rope(k, k_weight)
            if const_expr(fp8):
                q_out = _round_bf16(hadamard(_round_bf16(q_out)))
                k_out = _round_bf16(hadamard(_round_bf16(k_out)))
                store_fp8(q_out, q_scale, out_offset)
                store_fp8(k_out, k_scale, out_offset + out_stride_which)
                v_in = _load_f32(GTensor(_addr(v, in_row, 2), T.bf16, (-1,)), in_offset, VEC, False)
                store_fp8(v_in, v_scale, out_offset + 2 * out_stride_which)
            else:
                out_t.store(out_offset, _bf16_vector(q_out))
                out_t.store(out_offset + out_stride_which, _bf16_vector(k_out))
                if const_expr(write_v):
                    v_t = GTensor(_addr(v, in_row, 2), T.bf16, (-1,))
                    out_t.store(out_offset + 2 * out_stride_which, v_t.load(in_offset, vec_size=VEC))

        @flyc.jit
        def launch(
            q: fx.Pointer,
            k: fx.Pointer,
            v: fx.Pointer,
            q_weight: fx.Pointer,
            k_weight: fx.Pointer,
            cos: fx.Pointer,
            sin: fx.Pointer,
            rotation: fx.Pointer,
            q_scale: fx.Pointer,
            k_scale: fx.Pointer,
            v_scale: fx.Pointer,
            out: fx.Pointer,
            seq_len: fx.Int32,
            in_stride_b: fx.Int32,
            in_stride_s: fx.Int32,
            rope_stride_s: fx.Int32,
            out_stride_rank: fx.Int32,
            out_stride_s: fx.Int32,
            out_stride_b: fx.Int32,
            out_stride_which: fx.Int32,
            tokens: fx.Int32,
            stream: fx.Stream,
        ):
            kernel(
                q,
                k,
                v,
                q_weight,
                k_weight,
                cos,
                sin,
                rotation,
                q_scale,
                k_scale,
                v_scale,
                out,
                seq_len,
                in_stride_b,
                in_stride_s,
                rope_stride_s,
                out_stride_rank,
                out_stride_s,
                out_stride_b,
                out_stride_which,
            ).launch(grid=(fx.Int64(tokens), 1, 1), block=(threads, 1, 1), stream=stream)

        return launch

    # TODO(aiter): upstream the quantizing variant as a LayerNorm / adaLN counterpart of AITER's
    # fused_rms_mxfp4_quant (optional gated residual, then (1 + scale) * layer_norm(x) + shift, to
    # MXFP4 with shuffled scales), and launch the AITER kernel once it lands.
    @lru_cache(maxsize=32)
    def _build_row_kernel(cols: int, residual: bool, two_branches: bool, quantize: bool, eps: float):
        n_iters = cols // ROW_COLS
        n32 = mxfp4_scale_cols(cols) * 32
        name = f"flux3_block_rows_c{cols}_r{int(residual)}{int(two_branches)}_q{int(quantize)}_flydsl"

        @flyc.kernel(name=name)
        def kernel(
            x0: fx.Pointer,
            gate0: fx.Pointer,
            shift0: fx.Pointer,
            scale0: fx.Pointer,
            x1: fx.Pointer,
            gate1: fx.Pointer,
            shift1: fx.Pointer,
            scale1: fx.Pointer,
            branch0: fx.Pointer,
            branch1: fx.Pointer,
            hidden: fx.Pointer,
            out: fx.Pointer,
            fp4: fx.Pointer,
            quant_scale: fx.Pointer,
            seq_len0: Int32,
            rows_per_batch: Int32,
            x0_sb: Int32,
            x0_ss: Int32,
            gate0_sb: Int32,
            gate0_ss: Int32,
            shift0_sb: Int32,
            shift0_ss: Int32,
            scale0_sb: Int32,
            scale0_ss: Int32,
            x1_sb: Int32,
            x1_ss: Int32,
            gate1_sb: Int32,
            gate1_ss: Int32,
            shift1_sb: Int32,
            shift1_ss: Int32,
            scale1_sb: Int32,
            scale1_ss: Int32,
            branch0_sb: Int32,
            branch0_ss: Int32,
            branch1_sb: Int32,
            branch1_ss: Int32,
        ):
            row = fx.Int32(fx.block_idx.x)
            lane = fx.Int32(fx.thread_idx.x)
            b = row // rows_per_batch
            joint_s = row % rows_per_batch
            first = joint_s < seq_len0

            def row_tensor(ptr, stride_b, stride_s, s):
                return _addr(ptr, fx.Int64(b) * fx.Int64(stride_b) + fx.Int64(s) * fx.Int64(stride_s), 2)

            def segment_row(p0, sb0, ss0, p1, sb1, ss1):
                address = first.select(row_tensor(p0, sb0, ss0, joint_s), row_tensor(p1, sb1, ss1, joint_s - seq_len0))
                return GTensor(address, T.bf16, (-1,))

            x_t = segment_row(x0, x0_sb, x0_ss, x1, x1_sb, x1_ss)
            shift_t = segment_row(shift0, shift0_sb, shift0_ss, shift1, shift1_sb, shift1_ss)
            scale_t = segment_row(scale0, scale0_sb, scale0_ss, scale1, scale1_sb, scale1_ss)
            row_elems = fx.Int64(row) * fx.Int64(cols)

            values = []
            for it in range_constexpr(n_iters):
                col = lane * VEC + it * ROW_COLS
                x = _load_f32(x_t, col, VEC, False)
                if const_expr(residual):
                    gate = _load_f32(segment_row(gate0, gate0_sb, gate0_ss, gate1, gate1_sb, gate1_ss), col, VEC, False)
                    branch = _load_f32(
                        GTensor(row_tensor(branch0, branch0_sb, branch0_ss, joint_s), T.bf16, (-1,)), col, VEC, False
                    )
                    if const_expr(two_branches):
                        other = _load_f32(
                            GTensor(row_tensor(branch1, branch1_sb, branch1_ss, joint_s), T.bf16, (-1,)),
                            col,
                            VEC,
                            False,
                        )
                        branch = [branch[i] + other[i] for i in range_constexpr(VEC)]
                    summed = _bf16_vector([fmath.fma(gate[i], branch[i], x[i]) for i in range_constexpr(VEC)])
                    GTensor(_addr(hidden, row_elems, 2), T.bf16, (-1,)).store(col, summed)
                    rounded = fx.Vector(summed).to(fx.Float32)
                    x = [rounded[i] for i in range_constexpr(VEC)]
                values.append(x)

            total = fx.Float32(0.0)
            for it in range_constexpr(n_iters):
                for i in range_constexpr(VEC):
                    total = total + values[it][i]
            mean = _group_sum(total, WAVE_SIZE) / fx.Float32(float(cols))
            squares = fx.Float32(0.0)
            for it in range_constexpr(n_iters):
                for i in range_constexpr(VEC):
                    centered = values[it][i] - mean
                    squares = fmath.fma(centered, centered, squares)
            rstd = fmath.rsqrt(_group_sum(squares, WAVE_SIZE) / fx.Float32(float(cols)) + eps)

            out_t = GTensor(_addr(out, row_elems, 2), T.bf16, (-1,))
            fp4_t = GTensor(fx.Int64(fx.ptrtoint(fp4)) + fx.Int64(row) * fx.Int64(cols // 2), T.i32, (-1,))
            scale_i8 = _scale_pointer(quant_scale)
            for it in range_constexpr(n_iters):
                col = lane * VEC + it * ROW_COLS
                shift = _load_f32(shift_t, col, VEC, False)
                scale = _load_f32(scale_t, col, VEC, False)
                y = [fmath.fma(1.0 + scale[i], (values[it][i] - mean) * rstd, shift[i]) for i in range_constexpr(VEC)]
                if const_expr(quantize):
                    _store_mxfp4(_round_bf16(y), row, col, fp4_t, scale_i8, n32, 0)
                else:
                    out_t.store(col, _bf16_vector(y))

        @flyc.jit
        def launch(
            x0: fx.Pointer,
            gate0: fx.Pointer,
            shift0: fx.Pointer,
            scale0: fx.Pointer,
            x1: fx.Pointer,
            gate1: fx.Pointer,
            shift1: fx.Pointer,
            scale1: fx.Pointer,
            branch0: fx.Pointer,
            branch1: fx.Pointer,
            hidden: fx.Pointer,
            out: fx.Pointer,
            fp4: fx.Pointer,
            quant_scale: fx.Pointer,
            seq_len0: fx.Int32,
            rows_per_batch: fx.Int32,
            x0_sb: fx.Int32,
            x0_ss: fx.Int32,
            gate0_sb: fx.Int32,
            gate0_ss: fx.Int32,
            shift0_sb: fx.Int32,
            shift0_ss: fx.Int32,
            scale0_sb: fx.Int32,
            scale0_ss: fx.Int32,
            x1_sb: fx.Int32,
            x1_ss: fx.Int32,
            gate1_sb: fx.Int32,
            gate1_ss: fx.Int32,
            shift1_sb: fx.Int32,
            shift1_ss: fx.Int32,
            scale1_sb: fx.Int32,
            scale1_ss: fx.Int32,
            branch0_sb: fx.Int32,
            branch0_ss: fx.Int32,
            branch1_sb: fx.Int32,
            branch1_ss: fx.Int32,
            rows: fx.Int32,
            stream: fx.Stream,
        ):
            kernel(
                x0,
                gate0,
                shift0,
                scale0,
                x1,
                gate1,
                shift1,
                scale1,
                branch0,
                branch1,
                hidden,
                out,
                fp4,
                quant_scale,
                seq_len0,
                rows_per_batch,
                x0_sb,
                x0_ss,
                gate0_sb,
                gate0_ss,
                shift0_sb,
                shift0_ss,
                scale0_sb,
                scale0_ss,
                x1_sb,
                x1_ss,
                gate1_sb,
                gate1_ss,
                shift1_sb,
                shift1_ss,
                scale1_sb,
                scale1_ss,
                branch0_sb,
                branch0_ss,
                branch1_sb,
                branch1_ss,
            ).launch(grid=(fx.Int64(rows), 1, 1), block=(WAVE_SIZE, 1, 1), stream=stream)

        return launch

    @lru_cache(maxsize=32)
    def _build_seq_major_output_kernel(
        cols: int, chunk_cols: int, out_cols: int, col_offset: int, fp8_dtype: Optional[torch.dtype]
    ):
        n_iters = cols // ROW_COLS
        n32 = mxfp4_scale_cols(out_cols) * 32
        fp8 = fp8_dtype is not None
        in_bytes = 1 if fp8 else 2
        name = (
            f"flux3_seq_major_output_mxfp4_c{cols}_k{chunk_cols}_o{out_cols}_off{col_offset}"
            f"_{str(fp8_dtype).rsplit('_', 1)[-1] if fp8 else 'bf16'}_flydsl"
        )

        @flyc.kernel(name=name)
        def kernel(
            received: fx.Pointer,
            fp4: fx.Pointer,
            quant_scale: fx.Pointer,
            descale: fx.Pointer,
            seq_len: Int32,
            stride_rank: Int32,
            stride_s: Int32,
            stride_b: Int32,
        ):
            row = fx.Int32(fx.block_idx.x)
            lane = fx.Int32(fx.thread_idx.x)
            b = row // seq_len
            s = row % seq_len
            base = _addr(received, fx.Int64(b) * fx.Int64(stride_b) + fx.Int64(s) * fx.Int64(stride_s), in_bytes)
            # FP8 rows are read as words of 4 values.
            in_t = GTensor(base, T.i32 if fp8 else T.bf16, (-1,))
            fp4_t = GTensor(
                fx.Int64(fx.ptrtoint(fp4)) + fx.Int64(row) * fx.Int64(out_cols // 2) + fx.Int64(col_offset // 2),
                T.i32,
                (-1,),
            )
            scale_i8 = _scale_pointer(quant_scale)
            if const_expr(fp8):
                factor = fx.Float32(GTensor(fx.Int64(fx.ptrtoint(descale)), T.f32, (-1,)).load(0))
            for it in range_constexpr(n_iters):
                col = lane * VEC + it * ROW_COLS
                rank = col // chunk_cols
                offset = rank * stride_rank + (col - rank * chunk_cols)
                if const_expr(fp8):
                    words = fx.Vector(in_t.load(offset >> 2, vec_size=VEC // 4))
                    raw = _unpack_fp8([words[i] for i in range_constexpr(VEC // 4)])
                    values = _round_bf16([raw[i] * factor for i in range_constexpr(VEC)])
                else:
                    values = _load_f32(in_t, offset, VEC, False)
                _store_mxfp4(values, row, col, fp4_t, scale_i8, n32, col_offset // 32)

        @flyc.jit
        def launch(
            received: fx.Pointer,
            fp4: fx.Pointer,
            quant_scale: fx.Pointer,
            descale: fx.Pointer,
            seq_len: fx.Int32,
            stride_rank: fx.Int32,
            stride_s: fx.Int32,
            stride_b: fx.Int32,
            rows: fx.Int32,
            stream: fx.Stream,
        ):
            kernel(received, fp4, quant_scale, descale, seq_len, stride_rank, stride_s, stride_b).launch(
                grid=(fx.Int64(rows), 1, 1), block=(WAVE_SIZE, 1, 1), stream=stream
            )

        return launch


def _flydsl_device(tensor: torch.Tensor) -> bool:
    return _HAS_FLYDSL and tensor.is_cuda and get_device_wave_size(tensor) == WAVE_SIZE


def _aligned_rows(tensor: torch.Tensor, dtypes=(torch.bfloat16,)) -> bool:
    """Rows of one of ``dtypes`` whose 16-byte groups start 16-byte aligned."""
    group = 16 // tensor.element_size()
    return (
        tensor.dtype in dtypes
        and tensor.stride(-1) == 1
        and all(stride % group == 0 for stride in tensor.stride()[:-1])
        and tensor.data_ptr() % 16 == 0
    )


def qk_supported(
    query: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    heads: int,
    head_dim: int,
    fp8_dtype: Optional[torch.dtype] = None,
) -> bool:
    """Whether ``launch_qk_norm_rope`` takes ``[b, s, heads * head_dim]`` q, k, v like
    ``query`` with these RoPE rows and weights, into an FP8 send buffer of ``fp8_dtype``."""
    if not _flydsl_device(query) or not _aligned_rows(query):
        return False
    if head_dim % VEC or head_dim & (head_dim - 1):
        return False
    if fp8_dtype is not None and (head_dim != HADAMARD_BLOCK or fp8_dtype != _hardware_fp8_dtype(query.device)):
        return False
    float_dtypes = (torch.bfloat16, torch.float32)
    if cos.dtype != sin.dtype or not all(_aligned_rows(t, float_dtypes) for t in (cos, sin)):
        return False
    if q_weight.dtype != k_weight.dtype or not all(_aligned_rows(t, float_dtypes) for t in (q_weight, k_weight)):
        return False
    lanes_per_head = head_dim // VEC
    return lanes_per_head <= WAVE_SIZE and heads * lanes_per_head <= MAX_BLOCK_THREADS


def rows_supported(cols: int, *tensors: torch.Tensor) -> bool:
    """Whether the row kernels take rows of ``cols`` columns held by ``tensors``."""
    return bool(tensors) and _flydsl_device(tensors[0]) and cols % ROW_COLS == 0 and all(map(_aligned_rows, tensors))


def seq_major_output_supported(received: torch.Tensor, fp4: torch.Tensor) -> bool:
    """Whether ``launch_seq_major_output_mxfp4`` takes a ``[P, s, b, h, d]`` bf16 or FP8
    receive buffer like ``received`` into ``fp4``."""
    world_size, _, _, heads, head_dim = received.shape
    if not _flydsl_device(received) or (world_size * heads * head_dim) % ROW_COLS or (heads * head_dim) % VEC:
        return False
    if not _aligned_rows(fp4, (torch.uint8,)):
        return False
    if received.element_size() == 1:
        return (
            received.dtype == _hardware_fp8_dtype(received.device)
            and all(stride % VEC == 0 for stride in received.stride()[:3])
            and received.data_ptr() % VEC == 0
        )
    return _aligned_rows(received)


def _int32(*values: int) -> Tuple[int, ...]:
    for value in values:
        if not -(2**31) <= value < 2**31:
            raise ValueError(f"Stride {value} does not fit the kernels' 32-bit arguments.")
    return values


def launch_qk_norm_rope(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    eps: float,
    cos: torch.Tensor,
    sin: torch.Tensor,
    heads: int,
    out: torch.Tensor,
    seq_major: bool,
    fp8_scales: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = None,
) -> torch.Tensor:
    """``fused_block_ops``' q/k kernel on ``[b, s, h * d]`` q, k, v into ``out``: the send
    buffer ``[P, s, b, 3, h/P, d]`` if ``seq_major`` (in FP8 with ``fp8_scales``, the
    rotation and the q, k, v scales), else ``[b, s, 2, h, d]`` q and k."""
    batch, seq_len, _ = query.shape
    head_dim = query.shape[-1] // heads
    if seq_major:
        heads_per_rank = out.shape[4]
        strides = (out.stride(0), out.stride(1), out.stride(2), out.stride(3))
    else:
        heads_per_rank = heads
        strides = (0, out.stride(1), out.stride(0), out.stride(2))
    if not batch * seq_len:
        return out
    fp8_dtype = None if fp8_scales is None else out.dtype
    launcher = _build_qk_kernel(
        heads,
        head_dim,
        heads_per_rank,
        float(eps),
        q_weight.dtype == torch.float32,
        cos.dtype == torch.float32,
        seq_major,
        fp8_dtype,
    )
    rotation, q_scale, k_scale, v_scale = fp8_scales if fp8_scales is not None else (out,) * 4
    _run_compiled(
        launcher,
        ptr_arg(query),
        ptr_arg(key),
        ptr_arg(value),
        ptr_arg(q_weight),
        ptr_arg(k_weight),
        ptr_arg(cos),
        ptr_arg(sin),
        ptr_arg(rotation),
        ptr_arg(q_scale),
        ptr_arg(k_scale),
        ptr_arg(v_scale),
        ptr_arg(out),
        seq_len,
        *_int32(query.stride(0), query.stride(1), cos.stride(0), *strides),
        batch * seq_len,
        torch.cuda.current_stream(query.device),
    )
    return out


def launch_block_rows(
    segments: Sequence[Tuple[torch.Tensor, ...]],
    branches: Sequence[torch.Tensor],
    eps: float,
    hidden: torch.Tensor,
    out: torch.Tensor,
    fp4: torch.Tensor,
    quant_scale: torch.Tensor,
    quantize: bool,
) -> None:
    """The row kernel over one or two segments laid end to end along the sequence: each
    ``(x, gate, shift, scale)`` (``gate`` None and no ``branches`` for a plain modulated
    norm), its views broadcast to ``x``'s shape. Writes the residual into ``hidden`` when
    there are ``branches``, and the modulated norm into ``out`` or, with ``quantize``,
    ``fp4`` and ``quant_scale``."""
    residual = bool(branches)
    segments = list(segments) * (3 - len(segments))
    segments = segments[:2]
    x0 = segments[0][0]
    batch, cols = x0.shape[0], x0.shape[-1]
    seq_len0 = x0.shape[1]
    rows_per_batch = seq_len0 + (segments[1][0].shape[1] if segments[1] is not segments[0] else 0)
    if not batch * rows_per_batch:
        return
    launcher = _build_row_kernel(cols, residual, len(branches) == 2, quantize, float(eps))
    pointers, strides = [], []
    for x, gate, shift, scale in segments:
        for tensor in (x, x if gate is None else gate, shift, scale):
            pointers.append(ptr_arg(tensor))
            strides += tensor.stride()[:2]
    branch0 = branches[0] if residual else x0
    branch1 = branches[-1] if residual else x0
    _run_compiled(
        launcher,
        *pointers,
        ptr_arg(branch0),
        ptr_arg(branch1),
        ptr_arg(hidden),
        ptr_arg(out),
        ptr_arg(fp4),
        ptr_arg(quant_scale),
        seq_len0,
        rows_per_batch,
        *_int32(*strides, *branch0.stride()[:2], *branch1.stride()[:2]),
        batch * rows_per_batch,
        torch.cuda.current_stream(x0.device),
    )


def launch_seq_major_output_mxfp4(
    received: torch.Tensor,
    fp4: torch.Tensor,
    quant_scale: torch.Tensor,
    col_offset: int,
    descale: Optional[torch.Tensor] = None,
) -> None:
    """Quantize the rows of a ``[P, s, b, h, d]`` receive buffer into columns ``col_offset``
    onwards of ``fp4`` / ``quant_scale``; an FP8 buffer is dequantized by ``descale`` first."""
    world_size, seq_len, batch, heads, head_dim = received.shape
    cols = world_size * heads * head_dim
    if not batch * seq_len:
        return
    launcher = _build_seq_major_output_kernel(
        cols, heads * head_dim, fp4.shape[1] * 2, col_offset, None if descale is None else received.dtype
    )
    _run_compiled(
        launcher,
        ptr_arg(received),
        ptr_arg(fp4),
        ptr_arg(quant_scale),
        ptr_arg(received if descale is None else descale),
        seq_len,
        *_int32(received.stride(0), received.stride(1), received.stride(2)),
        batch * seq_len,
        torch.cuda.current_stream(received.device),
    )
