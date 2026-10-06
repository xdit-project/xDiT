"""Fused QK-RMSNorm + RoPE (FlyDSL), for FLUX joint attention and MiniMax-H3.

Written in the high-level ``flydsl.expr`` (fx) API on top of aiter's
``GTensor`` buffer-tensor shim (the same style aiter uses in
``aiter/ops/flydsl/kernels/qk_norm_rope_quant.py``).

Two rotations share this kernel, picked by ``rope_style`` and specialised at
build time with ``const_expr`` so neither pays for the other:

``"gptj"`` (default; FLUX)
    Interleaved pairs ``(2k, 2k+1)``, full-width tables, rotate in fp32.

``"neox"`` (MiniMax-H3)
    ``rotate_half`` pairs ``(k, k + rotary_dim/2)`` over the leading
    ``rotary_dim`` channels with the remainder passed through, rotate in bf16.
    The partner channel lives in a *different* lane, and ``shuffle_xor`` cannot
    express that lane shift -- but the RMS butterfly is an all-reduce, so every
    lane holds the same ``rstd`` and simply re-loads the partner's raw value and
    weight to rebuild its normalized value locally, bit-for-bit, with no
    cross-lane traffic.  The three-way channel classification (lower half /
    upper half / pass-through) is folded into the *tables* by
    :func:`prepare_neox_rope_tables`, so the kernel body stays branch-free.

Because the flags are ``const_expr`` and are part of the kernel symbol name,
``NEOX=False`` builds emit exactly the code this file emitted before the neox
variant existed.

Layout / algorithm (per attention layer the baseline runs norm_q/norm_k then
diffusers ``apply_rotary_emb``; this fuses the whole chain):

    one block == one (token, head); the block's ``BLOCK_THREADS`` lanes cover the
    head's ``D`` channels with each lane owning ``VEC`` contiguous channels
    (``VEC`` even so a GPT-J pair ``(2k, 2k+1)`` never straddles two lanes).
    The RMS reduction over ``D`` is a wave-local ``shuffle_xor`` butterfly (no
    LDS) since ``BLOCK_THREADS <= 64``.  cos/sin are loaded once and shared by q
    and k.

Numerics: the kernel reproduces the reference *rounding schedule*, not just the
formula, and the two RMSNorm implementations round in different places --
``diffusers`` RMSNorm rounds to the (bf16) weight dtype after the rstd-multiply
and again after the affine-multiply, while ``torch.nn.RMSNorm`` stays in fp32
throughout and rounds once at the end.  ``ROUND_AFTER_NORM`` /
``ROUND_AFTER_AFFINE`` replay whichever schedule the caller's module uses (see
:func:`_norm_schedule`) so the fused output matches the unfused output to within
a ULP.  ``apply_rotary_emb`` then upcasts to fp32 for the rotate and rounds once
at the end either way.

Inputs may be non-contiguous: real models build q/k as chunks of a fused QKV
projection, so the row stride is passed to the kernel at runtime and rows are
read in place rather than copied.  Only the ``(H, D)`` block within a token has
to be packed.  Outputs are always freshly allocated and contiguous.

Envelope: bf16 activations, 4-D ``[B, S, H, D]`` (sequence_dim == 1), even ``D``
that factors into a power-of-two lane count with an even per-lane width, plain
(affine-or-weightless) RMSNorm, and a full-D cos/sin table whose row count is
``S`` or ``B*S``.  neox additionally needs ``VEC`` to divide ``rotary_dim/2`` and
a pass-through tail no wider than ``rotary_dim/2``.  Anything outside that falls
back to the unfused reference.

Selecting the fused path is bit-exact for GPT-J -- it replays the reference's
rounding schedule, not just its formula.  neox is not: it rotates in fp32 and
rounds once where eager diffusers rounds three times in bf16, so it differs by
at most 1 bf16 ULP (the pass-through tail stays bit-exact).  See
:func:`flydsl_fused_qk_norm_rope` and the note in the kernel body.
"""

# NOTE: no ``from __future__ import annotations`` -- PEP 563 stringifies the
# annotations and defeats flydsl's runtime-arg detection for the Int32 kernel
# params (cos_rows / tok_off / q_rs / k_rs), forcing a fresh JIT per value.

import math
import os
from functools import lru_cache
from typing import Optional, Tuple

import torch

from diffusers.models.embeddings import apply_rotary_emb
from diffusers.models.normalization import RMSNorm as _DiffusersRMSNorm

# The neox fallback has to be diffusers' own rotate, not a copy of it, or the
# out-of-envelope path can silently drift from the model it stands in for.
# Guarded because MiniMax-H3 postdates the ``diffusers>=0.33`` floor in
# pyproject.toml, and this module is also imported by FLUX, FLUX.2, Qwen and
# Z-Image, none of which may be broken by a MiniMax-only import.  ``None`` is
# unreachable from the only caller: xfuser's MiniMax-H3 transformer imports this
# same symbol at its own module scope, so it cannot load without it.
try:
    from diffusers.models.transformers.transformer_minimax_h3 import (
        _apply_rotary_emb as _minimax_h3_apply_rotary_emb,
    )
except ImportError:
    _minimax_h3_apply_rotary_emb = None

from xfuser.logger import init_logger
from xfuser.model_executor.layers.flydsl_utils import get_device_wave_size

logger = init_logger(__name__)

# The FlyDSL + aiter stack is optional; importing this module must never hard
# fail on a box without them (CPU-only CI, non-ROCm).  ``_HAS_FLYDSL`` gates the
# fast path and the wrapper falls back to the diffusers reference when it is
# False.  Callers gate on ``_HAS_FLYDSL`` themselves to skip the fused path
# entirely (and avoid the round-trip through this wrapper) when AITER is absent.
try:
    import flydsl.compiler as flyc
    import flydsl.expr as fx
    from flydsl.expr import const_expr, range_constexpr
    from flydsl.expr import math as fmath
    from flydsl.expr.arith import FastMathFlags
    from flydsl.expr.typing import Int32, ReductionOp, T

    from aiter.ops.flydsl.kernels.tensor_shim import GTensor, _run_compiled

    _HAS_FLYDSL = True
except Exception:  # pragma: no cover - only exercised where flydsl is absent
    _HAS_FLYDSL = False


# HW grid Y is a 16-bit field on AMD HIP; cap blocks/launch and chunk tokens in
# Python (a folded launch would run tail blocks with tok >= T -- flydsl's
# ``if cond: return`` does not early-exit a kernel body -- and fault).
MAX_GRID_Y = 65535


def _pick_block(
    d: int, wave_size: int, half: Optional[int] = None
) -> Optional[Tuple[int, int]]:
    """Return ``(BLOCK_THREADS, VEC)`` for head_dim ``d`` or None if unsupported.

    Prefer the widest wave (few lanes, more work each) while keeping VEC even
    (GPT-J pairs stay lane-local) and BLOCK_THREADS no larger than the hardware
    wave, so the sum-of-squares reduction is a pure shuffle_xor butterfly.

    ``half`` is ``rotary_dim // 2`` for the neox variant and is None for GPT-J
    (which leaves the search below exactly as it was).  Under neox the two
    halves of the rotary span take opposite-signed sin coefficients, so a lane's
    VEC-wide block must not straddle the ``half`` boundary; requiring
    ``half % vec == 0`` guarantees that, and -- since ``vec`` also divides ``d``
    -- that the partner block is VEC-aligned too.
    """
    bt = wave_size
    while bt >= 2:
        if d % bt == 0:
            vec = d // bt
            if vec >= 2 and vec % 2 == 0 and (half is None or half % vec == 0):
                return bt, vec
        bt //= 2
    return None


def prepare_neox_rope_tables(
    cos: torch.Tensor, sin: torch.Tensor, head_dim: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fold the neox sign pattern and the pass-through tail into width-D tables.

    neox (``rotate_half``) RoPE over the leading ``rot`` channels of a ``D``-wide
    head is, with ``half = rot // 2``::

        out[c]        = x[c]        * cos[c] - x[c + half] * sin[c]   c <  half
        out[c]        = x[c]        * cos[c] + x[c - half] * sin[c]   half <= c < rot
        out[c]        = x[c]                                          c >= rot

    Three cases, selected on a channel index the kernel only knows at runtime.
    Folding the sign and the tail into the *tables* collapses all three to one
    branch-free expression over the full width ``D``::

        out[c] = x[c] * cos_pad[c] + x[(c + half) % rot] * sin_fold[c]

    because ``cos_pad`` is 1 and ``sin_fold`` is 0 on the pass-through tail, and
    ``sin_fold`` carries the sign.  ``(c + half) % rot`` lands inside
    ``[half, rot)`` for ``c >= rot`` -- in bounds, and multiplied by zero.  This
    needs no select/mask primitives in the kernel at all.

    Exactness: ``(-x)*s == x*(-s)`` is a sign-bit flip with no rounding, and
    ``own*1 + partner*0 == own``.  The one wart is that a pass-through value of
    exactly ``-0.0`` comes back as ``+0.0``.

    Returns ``(cos_pad, sin_fold)``, both ``[rows, head_dim]`` contiguous bf16 --
    bf16 because the diffusers reference does ``cos.to(hidden_states.dtype)``
    before the rotate, so the kernel must see the same bf16-rounded angles.
    Negating before or after that cast is identical, so the fold is safe either
    way.

    Hoist this: it is a ``cat`` over ``[S, D]`` and rebuilding it per layer would
    cost more bandwidth than the fusion saves.
    """
    rot = cos.shape[-1]
    half = rot // 2
    cos2 = cos.reshape(-1, rot)
    sin2 = sin.reshape(-1, rot)
    rows = cos2.shape[0]
    pad = head_dim - rot
    cos_pad = torch.cat([cos2, cos2.new_ones((rows, pad))], dim=-1)
    sin_fold = torch.cat(
        [-sin2[:, :half], sin2[:, half:], sin2.new_zeros((rows, pad))], dim=-1
    )
    return (
        cos_pad.to(torch.bfloat16).contiguous(),
        sin_fold.to(torch.bfloat16).contiguous(),
    )



if _HAS_FLYDSL:

    @lru_cache(maxsize=32)
    def _build_kernel(
        *,
        H: int,
        D: int,
        VEC: int,
        BLOCK_THREADS: int,
        eps: float,
        cos_is_f32: bool,
        NORM_Q: bool,
        NORM_K: bool,
        HAS_WQ: bool,
        HAS_WK: bool,
        ROUND_AFTER_NORM: bool,
        ROUND_AFTER_AFFINE: bool,
        CONTIG: bool,
        NEOX: bool,
        HALF: int,
    ):
        """Build + cache the @flyc.jit launcher for one config.

        Shape/flag constants are captured by closure (not module globals) so
        launchers for different configs coexist in the lru cache.
        """
        LOG2_BT = int(math.log2(BLOCK_THREADS))
        PAIRS = VEC // 2
        INV_D = 1.0 / D
        HD = H * D
        # Guarded so the GPT-J build never forms ``% 0``; under NEOX this is the
        # full rotary span and the partner index is ``(c + HALF) % ROT``.
        ROT = 2 * HALF if NEOX else 1

        # CONTIG must be in the symbol name: the two addressing variants are
        # separate compilations and would otherwise share an MLIR symbol.  The
        # same applies to NEOX/HALF, which change the emitted addressing --
        # a collision here is a silent miscompile, not an error.  The suffix is
        # empty for GPT-J, so the FLUX symbol string is byte-identical to what
        # it was before this kernel grew a neox variant.
        _kname = (
            f"fused_qk_norm_rope_H{H}_D{D}_v{VEC}"
            f"{f'_n{HALF}' if NEOX else ''}"
            f"{'_c' if CONTIG else '_s'}_flydsl"
        )

        @flyc.kernel(name=_kname)
        def kernel(
            q_in: fx.Tensor,   # [T, H, D] bf16, row stride q_rs, (H, D) packed
            k_in: fx.Tensor,   # [T, H, D] bf16, row stride k_rs, (H, D) packed
            q_out: fx.Tensor,  # [T, H, D] bf16, contiguous
            k_out: fx.Tensor,  # [T, H, D] bf16, contiguous
            wq: fx.Tensor,      # [D] bf16 (dummy when not HAS_WQ)
            wk: fx.Tensor,      # [D] bf16 (dummy when not HAS_WK)
            cos: fx.Tensor,     # [cos_rows, D] cos_dt
            sin: fx.Tensor,     # [cos_rows, D] cos_dt
            cos_rows: Int32,
            tok_off: Int32,     # global token index of this chunk's row 0
            q_rs: Int32,        # q_in row stride in elements (H*D if contiguous)
            k_rs: Int32,        # k_in row stride in elements
        ):
            fm_fast = FastMathFlags.fast
            # Resolve element types inside the kernel body: T.* needs a live MLIR
            # context (established by @flyc.kernel), so it cannot be a closure const.
            cos_dt = T.f32 if cos_is_f32 else T.bf16

            head = fx.Int32(fx.block_idx.x)
            tok = fx.Int32(fx.block_idx.y)
            tid = fx.Int32(fx.thread_idx.x)

            # ``tok`` is chunk-local; ``gtok`` is the global token row.  Folding
            # the chunk offset in here rather than slicing the tensors in Python
            # means the caller marshals whole tensors once per call no matter how
            # many grid-Y chunks it takes.
            #
            # Inputs may be non-contiguous fused-QKV chunks (row stride > H*D),
            # so their row offset is the runtime ``q_rs``/``k_rs``.  Outputs are
            # two separate contiguous buffers -- NOT two halves of one.  Under
            # mode="reduce-overhead" (CUDA Graphs, what FLUX.1-dev uses), views
            # into a single output allocation get materialized by cudagraph
            # trees, and that copy is baked into every replay: measured a
            # consistent 1-3% loss on contiguous shapes, which showed up e2e.
            gtok = fx.Int32(tok_off) + tok
            hoff = head * D + tid * VEC
            if const_expr(CONTIG):
                # Row stride is exactly H*D, so all four addresses coincide and
                # H*D is a compile-time constant the backend can strength-reduce
                # -- one address computation, no runtime multiply.  This is the
                # codegen the pre-stride kernel had, and on the small per-rank
                # shapes an Ulysses-sharded run produces, the difference between
                # one address and three is a measurable ~2%.
                out_base = gtok * HD + hoff
                q_base = out_base
                k_base = out_base
            else:
                q_base = gtok * fx.Int32(q_rs) + hoff
                k_base = gtok * fx.Int32(k_rs) + hoff
                out_base = gtok * HD + hoff
            row = gtok % fx.Int32(cos_rows)
            coff = row * D + tid * VEC
            woff = tid * VEC

            if const_expr(NEOX):
                # Partner channel block for the rotate.  ``(c + HALF) % ROT`` is
                # the single expression that covers all three regions: it maps
                # the lower rotary half to the upper, the upper back to the
                # lower, and -- for the pass-through tail, where ``c >= ROT`` and
                # ``D - ROT <= HALF`` -- into ``[HALF, ROT)``, which is in bounds
                # and gets multiplied by a zeroed ``sin_fold``.
                #
                # ``_pick_block`` guarantees ``VEC | HALF``, so ``p_choff`` is a
                # multiple of VEC and the partner block is a whole, aligned,
                # in-head VEC block -- never a straddle of the half boundary.
                p_choff = (tid * VEC + HALF) % ROT
                p_hoff = head * D + p_choff
                pwoff = p_choff
                if const_expr(CONTIG):
                    q_pbase = gtok * HD + p_hoff
                    k_pbase = q_pbase
                else:
                    q_pbase = gtok * fx.Int32(q_rs) + p_hoff
                    k_pbase = gtok * fx.Int32(k_rs) + p_hoff
            else:
                # Bound so ``process`` has the parameter either way; DCE'd.
                q_pbase = q_base
                k_pbase = k_base
                pwoff = woff

            qin_ = GTensor(q_in, dtype=T.bf16, shape=(-1,))
            kin_ = GTensor(k_in, dtype=T.bf16, shape=(-1,))
            qout_ = GTensor(q_out, dtype=T.bf16, shape=(-1,))
            kout_ = GTensor(k_out, dtype=T.bf16, shape=(-1,))
            cos_ = GTensor(cos, dtype=cos_dt, shape=(-1,))
            sin_ = GTensor(sin, dtype=cos_dt, shape=(-1,))
            wq_ = GTensor(wq, dtype=T.bf16, shape=(-1,))
            wk_ = GTensor(wk, dtype=T.bf16, shape=(-1,))

            # cos/sin loaded once, shared by q and k (full-D table, indexed per
            # lane).
            cos_f = fx.Vector(cos_.load(coff, vec_size=VEC)).to(fx.Float32)
            sin_f = fx.Vector(sin_.load(coff, vec_size=VEC)).to(fx.Float32)

            def wave_reduce_add(x):
                w = fx.Float32(x)
                for sh_exp in range_constexpr(LOG2_BT):
                    off = BLOCK_THREADS // (2 << sh_exp)
                    w = w.addf(w.shuffle_xor(off, BLOCK_THREADS), fastmath=fm_fast)
                return w

            def round_bf16(vals):
                # round-trip fp32 -> bf16 -> fp32 to replay diffusers' RMSNorm
                # intermediate rounding on the whole VEC-wide fragment.
                fv = fx.Vector.from_elements(
                    [v.ir_value() for v in vals], dtype=fx.Float32
                )
                bf = fv.truncf(T.vec(VEC, T.bf16))
                ff = fx.Vector(bf).to(fx.Float32)
                return [ff[i] for i in range_constexpr(VEC)]

            def affine(vals, g_w, w_off, HAS_W):
                """Weight-multiply + the post-affine rounding, on a VEC list."""
                if const_expr(HAS_W):
                    w = fx.Vector(g_w.load(w_off, vec_size=VEC)).to(fx.Float32)
                    vals = [vals[i] * w[i] for i in range_constexpr(VEC)]
                    if const_expr(ROUND_AFTER_AFFINE):
                        vals = round_bf16(vals)
                return vals

            def process(g_in, in_base, p_base, g_out, out_base, g_w, NORM, HAS_W):
                x = fx.Vector(g_in.load(in_base, vec_size=VEC)).to(fx.Float32)

                if const_expr(NORM):
                    x2 = x * x
                    sq_local = x2.reduce(ReductionOp.ADD, fastmath=fm_fast)
                    sq = wave_reduce_add(sq_local)
                    rstd = fmath.rsqrt(sq * INV_D + eps, fastmath=fm_fast)
                    scaled = [x[i] * rstd for i in range_constexpr(VEC)]
                    if const_expr(ROUND_AFTER_NORM):
                        scaled = round_bf16(scaled)
                else:
                    scaled = [x[i] for i in range_constexpr(VEC)]

                scaled = affine(scaled, g_w, woff, HAS_W)

                if const_expr(NEOX):
                    # The partner channels live in a different lane, and
                    # shuffle_xor cannot express the required +HALF/VEC lane
                    # shift.  Re-load them instead: wave_reduce_add is an xor
                    # butterfly, i.e. an all-reduce, so every lane already holds
                    # the *same* rstd and can reconstruct the partner's
                    # normalized value bit-for-bit with no cross-lane traffic.
                    # The extra loads are L1-resident (another lane of the same
                    # block just read them).
                    xp = fx.Vector(g_in.load(p_base, vec_size=VEC)).to(fx.Float32)
                    if const_expr(NORM):
                        p_scaled = [xp[i] * rstd for i in range_constexpr(VEC)]
                        if const_expr(ROUND_AFTER_NORM):
                            p_scaled = round_bf16(p_scaled)
                    else:
                        p_scaled = [xp[i] for i in range_constexpr(VEC)]
                    p_scaled = affine(p_scaled, g_w, pwoff, HAS_W)

                    # Branch-free: the sign and the pass-through mask are folded
                    # into the tables by prepare_neox_rope_tables.
                    #
                    # NOTE on numerics: diffusers' MiniMax-H3 _apply_rotary_emb
                    # casts cos/sin to the activation dtype and does the rotate
                    # in bf16, i.e. it rounds three times (both products and the
                    # sum).  This computes in fp32 and rounds once at the store.
                    # Inserting explicit round_bf16 after each product to replay
                    # the reference schedule was MEASURED to have no effect: the
                    # launcher's fast_fp_math folds the truncf/extf pair away and
                    # contracts the multiply-add into an FMA regardless, so the
                    # flag that used to gate it was inert and has been removed.
                    # Net effect: the fused rotate is *more* accurate than eager
                    # diffusers, differing by at most 1 bf16 ULP (measured: ~31%
                    # of rotary channels, max 3.1e-2, mean 1.2e-3 at these
                    # magnitudes).  The pass-through tail stays bit-exact.
                    outs = [
                        scaled[i] * cos_f[i] + p_scaled[i] * sin_f[i]
                        for i in range_constexpr(VEC)
                    ]
                else:
                    # interleaved GPT-J rope on lane-local pairs (2k, 2k+1):
                    #   out[2k]   = e*cos[2k]   - o*sin[2k]
                    #   out[2k+1] = o*cos[2k+1] + e*sin[2k+1]
                    outs = [None] * VEC
                    for kk in range_constexpr(PAIRS):
                        e = scaled[2 * kk]
                        o = scaled[2 * kk + 1]
                        outs[2 * kk] = e * cos_f[2 * kk] - o * sin_f[2 * kk]
                        outs[2 * kk + 1] = (
                            o * cos_f[2 * kk + 1] + e * sin_f[2 * kk + 1]
                        )

                out_v = fx.Vector.from_elements(
                    [o.ir_value() for o in outs], dtype=fx.Float32
                )
                g_out.store(out_base, out_v.truncf(T.vec(VEC, T.bf16)))

            process(qin_, q_base, q_pbase, qout_, out_base, wq_, NORM_Q, HAS_WQ)
            process(kin_, k_base, k_pbase, kout_, out_base, wk_, NORM_K, HAS_WK)

        @flyc.jit
        def launch_fused_qk_norm_rope(
            q_in: fx.Tensor,
            k_in: fx.Tensor,
            q_out: fx.Tensor,
            k_out: fx.Tensor,
            wq: fx.Tensor,
            wk: fx.Tensor,
            cos: fx.Tensor,
            sin: fx.Tensor,
            cos_rows: fx.Int32,
            tok_off: fx.Int32,
            q_rs: fx.Int32,
            k_rs: fx.Int32,
            n_tokens: fx.Int32,
            stream: fx.Stream = fx.Stream(None),
        ):
            k = kernel(
                q_in,
                k_in,
                q_out,
                k_out,
                wq,
                wk,
                cos,
                sin,
                cos_rows,
                tok_off,
                q_rs,
                k_rs,
            )
            # ``n_tokens`` is a runtime Int32.  Pass it to the grid raw and let
            # ``KernelLauncher.launch`` cast each dim to index (via
            # ``_to_index_value``) inside its own MLIR location/context.  Doing
            # ``arith.index_cast(T.index, n_tokens)`` here instead faults under
            # torch.compile: on a dynamo-resumed frame no MLIR context is active
            # at launcher-trace time ("An MLIR function requires a Context").
            k.launch(
                grid=(H, n_tokens, 1),
                block=(BLOCK_THREADS, 1, 1),
                stream=stream,
            )

        launch_fused_qk_norm_rope.compile_hints = {
            "waves_per_eu": 8,
            "fast_fp_math": True,
        }
        return launch_fused_qk_norm_rope

    @torch.library.custom_op("xfuser::flydsl_qk_norm_rope", mutates_args=())
    def _flydsl_qk_norm_rope_launch(
        q: torch.Tensor,       # [B, S, H, D] bf16, (H, D) packed, any row stride
        k: torch.Tensor,       # [B, S, H, D] bf16, (H, D) packed, any row stride
        wq: Optional[torch.Tensor],  # [D] bf16 or None (weightless RMSNorm)
        wk: Optional[torch.Tensor],
        cos: torch.Tensor,     # [cos_rows, D]
        sin: torch.Tensor,     # [cos_rows, D]
        cos_rows: int,
        eps: float,
        vec: int,
        block_threads: int,
        norm_q: bool,
        norm_k: bool,
        round_after_norm: bool,
        round_after_affine: bool,
        neox: bool,
        half_rot: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Opaque launch boundary for torch.compile.

        Everything that touches a real data pointer (``_build_kernel`` ->
        ``flyc.compile`` and the ``_run_compiled`` grid launches) lives inside
        this custom op so Dynamo treats it as a black box -- calling
        :func:`register_fake` for shape propagation during tracing and this body
        only at runtime with real tensors.  Without it Dynamo traces into
        ``flyc.compile`` and faults on ``FakeTensor.__dlpack__`` (no data
        pointer).

        All *stride* inspection also lives in here, not in the Python caller:
        under torch.compile with dynamic shapes ``Tensor.stride(i)`` returns a
        ``SymInt``, and Dynamo cannot trace it (it asserts trying to build a
        ConstantVariable).  Inside an opaque custom op the tensors are real.
        """
        b, s, h, d = q.shape
        t_tok = b * s
        has_wq = wq is not None
        has_wk = wk is not None

        # [T, H, D] without a copy where possible.  Real models hand us q/k as
        # chunks of a fused QKV projection -- a copy here is a full extra pass
        # over both tensors and on Wan it cost more than the fusion saved.
        q = _as_token_view(q, b, s, h, d, vec)
        k = _as_token_view(k, b, s, h, d, vec)

        # Buffer loads scale the element offset by the element size in i32, so
        # for bf16 the addressable range is 2**30 elements.  Bound the strided
        # input extent; a contiguous copy shrinks the row stride back to H*D,
        # which the caller already checked fits.
        if (t_tok - 1) * max(q.stride(0), k.stride(0)) + h * d >= (1 << 30):
            q = q.contiguous()
            k = k.contiguous()

        # Two separate contiguous allocations.  Merging them into one buffer and
        # returning views saves a DLPack marshal per call, but that only matters
        # in default compile mode -- under mode="reduce-overhead" the op's Python
        # body is not replayed at all, while cudagraph trees materialize the
        # views, and the resulting copy is permanent in the replay.  The models
        # use reduce-overhead, so separate buffers win where it counts.
        oq = q.new_empty((t_tok, h, d))
        ok = k.new_empty((t_tok, h, d))

        # Row strides come off the tensors we actually marshal, so they can
        # never disagree with the data pointer DLPack hands the kernel.
        q_rs = q.stride(0)
        k_rs = k.stride(0)

        # Build after the strides are known: a packed input compiles to a
        # cheaper addressing variant.  Both variants live in the lru cache, so a
        # model that mixes contiguous and fused-QKV layers pays one JIT each.
        launcher = _build_kernel(
            H=h,
            D=d,
            VEC=vec,
            BLOCK_THREADS=block_threads,
            eps=eps,
            cos_is_f32=(cos.dtype == torch.float32),
            NORM_Q=norm_q,
            NORM_K=norm_k,
            HAS_WQ=has_wq,
            HAS_WK=has_wk,
            ROUND_AFTER_NORM=round_after_norm,
            ROUND_AFTER_AFFINE=round_after_affine,
            CONTIG=(q_rs == h * d and k_rs == h * d),
            NEOX=neox,
            HALF=half_rot,
        )

        # kernel always binds wq/wk params; pass a 1-elem dummy when unused (the
        # const_expr HAS_W gate DCEs the load, but the binding needs a valid
        # tensor).  Allocate it only when it is actually needed -- every real
        # model has both weights, and an unconditional alloc here is a device
        # allocation on the hot path for nothing.
        if has_wq and has_wk:
            wq_arg, wk_arg = wq, wk
        else:
            dummy = q.new_empty(1, dtype=torch.bfloat16)
            wq_arg = wq if has_wq else dummy
            wk_arg = wk if has_wk else dummy

        # ``tok_off`` is folded into the kernel's addressing, so chunks no longer
        # need sliced views of q/k/out -- whole tensors go in once per launch.
        # Grid Y is a 16-bit field on AMD HIP, hence the chunking at all; T is
        # under MAX_GRID_Y for every shape these models actually run.
        stream = torch.cuda.current_stream()
        with torch.cuda.device(q.device.index):
            for start in range(0, t_tok, MAX_GRID_Y):
                n = min(MAX_GRID_Y, t_tok - start)
                _run_compiled(
                    launcher,
                    q,
                    k,
                    oq,
                    ok,
                    wq_arg,
                    wk_arg,
                    cos,
                    sin,
                    cos_rows,
                    start,
                    q_rs,
                    k_rs,
                    n,
                    stream,
                )

        return oq.view(b, s, h, d), ok.view(b, s, h, d)

    @_flydsl_qk_norm_rope_launch.register_fake
    def _flydsl_qk_norm_rope_launch_fake(
        q,
        k,
        wq,
        wk,
        cos,
        sin,
        cos_rows,
        eps,
        vec,
        block_threads,
        norm_q,
        norm_k,
        round_after_norm,
        round_after_affine,
        neox,
        half_rot,
    ):
        # Contiguous, matching the real op exactly -- a fake/real stride
        # mismatch is a silent miscompile under Inductor.  Output shapes do not
        # depend on the neox flags, but the parameter list must still match the
        # real op positionally.
        return q.new_empty(q.shape), k.new_empty(k.shape)


def _supported(query, key, cos, half: Optional[int] = None) -> bool:
    if not _HAS_FLYDSL:
        return False
    if not query.is_cuda:
        return False
    if query.shape != key.shape or query.dtype != key.dtype:
        return False
    if query.dtype != torch.bfloat16:  # fp16/fp32 -> reference fallback
        return False
    if query.dim() != 4:  # [B, S, H, D]
        return False
    d = query.shape[-1]
    wave_size = get_device_wave_size(query)
    if wave_size is None or _pick_block(d, wave_size, half) is None:
        return False
    # Width-D for both variants: the neox tables are pre-padded to the head dim
    # by prepare_neox_rope_tables, so this check stays correct unchanged.
    if not isinstance(cos, torch.Tensor) or cos.shape[-1] != d:
        return False
    return True


def _norm_schedule(m) -> Optional[Tuple[bool, bool]]:
    """Return ``(round_after_norm, round_after_affine)`` for RMSNorm ``m``.

    The kernel replays the reference's bf16 *rounding schedule*, not just its
    formula, and the two RMSNorm implementations in play round in different
    places:

    * ``diffusers.models.normalization.RMSNorm`` rounds twice -- once after the
      rstd multiply (it casts to the bf16 weight dtype) and again implicitly
      from the bf16 x bf16 affine multiply.
    * ``torch.nn.RMSNorm`` accumulates the whole thing in fp32 and rounds once,
      at the end -- so after the affine when there is a weight, and after the
      norm when there is not.

    Returns ``None`` for anything that is not a recognised weightless-or-affine
    RMSNorm, which sends the caller to the unfused reference.  ``m is None``
    means "no norm at all" and reports a schedule that is never consulted.
    """
    if m is None:
        return (False, False)
    if getattr(m, "bias", None) is not None:
        return None
    w = getattr(m, "weight", None)
    if w is not None and w.dim() != 1:
        return None
    if isinstance(m, torch.nn.RMSNorm):
        return (w is None, w is not None)
    if isinstance(m, _DiffusersRMSNorm):
        return (True, True)
    # An unrecognised module could be a LayerNorm (which the old duck-typed
    # check let through and silently computed as an RMSNorm).
    return None


def _as_token_view(
    x: torch.Tensor, b: int, s: int, h: int, d: int, vec: int
) -> torch.Tensor:
    """``[B, S, H, D]`` -> ``[B*S, H, D]``, without copying when it is legal.

    The kernel indexes rows by a runtime stride, so an arbitrary row stride is
    fine, but everything inside a token's ``(H, D)`` block must be packed and
    the VEC-wide buffer loads need natural alignment.  ``view`` enforces the
    remaining condition (``stride(0) == S * stride(1)``, so B and S can flatten)
    for free -- ``as_strided`` would skip that check and happily alias.
    """
    if (
        x.stride(-1) == 1
        and x.stride(-2) == d
        and x.stride(1) % vec == 0
        and x.storage_offset() % vec == 0
    ):
        try:
            return x.view(b * s, h, d)
        except RuntimeError:
            pass
    return x.contiguous().view(b * s, h, d)


def _reference(
    query: torch.Tensor,
    key: torch.Tensor,
    norm_q,
    norm_k,
    rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Unfused diffusers path: RMSNorm(q)/RMSNorm(k) then interleaved RoPE.

    Runs exactly the ops diffusers runs -- the module's own RMSNorm followed by
    diffusers' ``apply_rotary_emb`` -- so it is numerically interchangeable with
    the fused fast path below.  ``norm_q`` / ``norm_k`` may be None (skip the
    norm) and ``rotary_emb`` may be None (skip RoPE).
    """
    if norm_q is not None:
        query = norm_q(query)
    if norm_k is not None:
        key = norm_k(key)
    if rotary_emb is not None:
        query = apply_rotary_emb(query, rotary_emb, sequence_dim=1)
        key = apply_rotary_emb(key, rotary_emb, sequence_dim=1)
    return query, key


def _reference_neox(
    query: torch.Tensor,
    key: torch.Tensor,
    norm_q,
    norm_k,
    rotary_emb,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Unfused path for the neox variant: RMSNorm then rotate_half RoPE.

    Runs diffusers' own MiniMax-H3 ``_apply_rotary_emb``, which rotates only the
    leading ``cos.shape[-1]`` channels and does so in the activation dtype --
    not fp32 like FLUX's interleaved ``apply_rotary_emb``, which is why the
    GPT-J ``_reference`` above cannot serve this.
    """
    if norm_q is not None:
        query = norm_q(query)
    if norm_k is not None:
        key = norm_k(key)
    if rotary_emb is not None:
        cos, sin = rotary_emb[0], rotary_emb[1]
        query = _minimax_h3_apply_rotary_emb(query, cos, sin)
        key = _minimax_h3_apply_rotary_emb(key, cos, sin)
    return query, key


def flydsl_fused_qk_norm_rope(
    query: torch.Tensor,
    key: torch.Tensor,
    norm_q,
    norm_k,
    rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]],
    *,
    rope_style: str = "gptj",
    rotary_dim: Optional[int] = None,
    neox_tables: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """FlyDSL RMSNorm(q)/RMSNorm(k) + RoPE, fused.

    ``query`` / ``key`` are ``[B, S, H, D]`` (sequence_dim == 1); ``rotary_emb``
    is the diffusers ``(cos, sin)`` pair.  Falls back to the unfused diffusers
    reference for anything outside the supported envelope so the result is
    identical either way.

    Two rotations are supported, selected by ``rope_style``:

    ``"gptj"`` (default, FLUX / Z-Image / Wan)
        Interleaved pairs ``(2k, 2k+1)``, full-width ``cos``/``sin``, rotate
        computed in fp32.  This is the original behaviour and the defaults keep
        every existing caller byte-identical.

    ``"neox"`` (MiniMax-H3)
        ``rotate_half`` pairs ``(k, k + rotary_dim/2)`` over the leading
        ``rotary_dim`` channels, the rest passed through.  Requires
        ``rotary_dim`` and ``neox_tables`` -- the ``(cos_pad, sin_fold)`` pair
        from :func:`prepare_neox_rope_tables`, which must be built *once per
        forward* by the caller, not per layer.  ``rotary_emb`` is still the
        original unpadded ``(cos, sin)`` and is used only by the fallback.

    neox numerics: the rotate runs in fp32 and rounds once, while eager
    diffusers rounds three times in bf16, so neox output differs from the unfused
    reference by at most 1 bf16 ULP (the pass-through tail is bit-exact).  See
    the note in the kernel body -- fast_fp_math makes replaying the reference's
    rounding schedule impossible without disabling FMA contraction.
    """
    neox = rope_style == "neox"

    def _ref():
        if neox:
            return _reference_neox(query, key, norm_q, norm_k, rotary_emb)
        return _reference(query, key, norm_q, norm_k, rotary_emb)

    if rope_style not in ("gptj", "neox"):
        return _ref()
    # Inference-only fast path: the custom op has no autograd formula, so any
    # grad-enabled call falls back (matching the Z-Image sibling).
    if torch.is_grad_enabled():
        return _ref()
    if (
        rotary_emb is None
        or not isinstance(rotary_emb, (tuple, list))
        or len(rotary_emb) < 2
    ):
        return _ref()
    sched_q = _norm_schedule(norm_q)
    sched_k = _norm_schedule(norm_k)
    if sched_q is None or sched_k is None:
        return _ref()
    # One kernel build serves q and k, so a single rounding schedule has to
    # cover both.  Mixed norm implementations are not something any model does.
    active = [s for s, m in ((sched_q, norm_q), (sched_k, norm_k)) if m is not None]
    if len(set(active)) > 1:
        return _ref()
    round_after_norm, round_after_affine = active[0] if active else (False, False)

    d_q = query.shape[-1]
    half = None
    if neox:
        if (
            neox_tables is None
            or not isinstance(neox_tables, (tuple, list))
            or len(neox_tables) != 2
            or rotary_dim is None
            or rotary_dim % 2 != 0
            or rotary_dim > d_q
        ):
            return _ref()
        # The partner index ``(c + half) % rot`` only stays inside the rotary
        # span for the pass-through tail while the tail is no wider than half.
        if (d_q - rotary_dim) > rotary_dim // 2:
            return _ref()
        half = rotary_dim // 2
        cos, sin = neox_tables
    else:
        cos, sin = rotary_emb[0], rotary_emb[1]

    if not _supported(query, key, cos, half):
        return _ref()

    b, s, h, d = query.shape
    cos2 = cos.reshape(-1, d)
    sin2 = sin.reshape(-1, d)
    if cos2.shape != sin2.shape or cos2.shape[0] not in (s, b * s):
        return _ref()
    cos_rows = cos2.shape[0]

    if cos2.dtype not in (torch.float32, torch.bfloat16):
        return _ref()

    wave_size = get_device_wave_size(query)
    if wave_size is None:
        return _ref()
    block_vec = _pick_block(d, wave_size, half)
    if block_vec is None:  # already checked in _supported, belt & suspenders
        return _ref()
    block_threads, vec = block_vec

    # Buffer loads scale the element offset by the element size in i32, so for
    # bf16 the addressable range is 2**30 elements, not 2**31.  This is the
    # contiguous bound; the strided one needs the row stride and so lives
    # inside the custom op, which can fall back to a copy on its own.  Reading
    # .stride() out here would break Dynamo under dynamic shapes.
    if b * s * h * d >= (1 << 30):
        return _ref()

    cos2 = cos2.contiguous()
    sin2 = sin2.contiguous()

    wq = getattr(norm_q, "weight", None) if norm_q is not None else None
    wk = getattr(norm_k, "weight", None) if norm_k is not None else None
    has_wq = wq is not None
    has_wk = wk is not None
    if has_wq:
        wq = wq.contiguous()
    if has_wk:
        wk = wk.contiguous()
    # bf16 weight required to replay the round-after-norm schedule; otherwise
    # (or if weightless) the diffusers reference is the safer match.
    if (has_wq and wq.dtype != torch.bfloat16) or (has_wk and wk.dtype != torch.bfloat16):
        return _ref()

    eps = 1e-6
    for m in (norm_q, norm_k):
        if m is None:
            continue
        e = getattr(m, "eps", None)
        if e is None:
            # torch.nn.RMSNorm reads eps=None as finfo(dtype).eps, which is not
            # the 1e-6 default below.
            return _ref()
        eps = float(e)
        break

    # The build + grid launches run inside an opaque custom op so this whole
    # path is safe under torch.compile (see _flydsl_qk_norm_rope_launch).
    oq, ok = _flydsl_qk_norm_rope_launch(
        query,
        key,
        wq if has_wq else None,
        wk if has_wk else None,
        cos2,
        sin2,
        cos_rows,
        eps,
        vec,
        block_threads,
        norm_q is not None,
        norm_k is not None,
        round_after_norm,
        round_after_affine,
        neox,
        half if half is not None else 0,
    )

    return oq, ok
