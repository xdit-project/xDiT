"""Fused QK-RMSNorm + complex (interleaved / GPT-J) RoPE for Z-Image (FlyDSL).

Written in the high-level ``flydsl.expr`` (fx) API on top of aiter's ``GTensor``
buffer-tensor shim -- the same style as ``fused_qk_rope_flydsl.py`` (FLUX/Qwen)
and ``fused_qk_norm_rope_wan_flydsl.py`` (Wan).

Layout / algorithm
------------------
One block == one ``(token, head)``; the block's ``BLOCK_THREADS`` lanes cover
the head's ``D`` channels with each lane owning ``VEC`` contiguous channels.
``VEC`` is even and ``lane = tid * VEC`` is therefore even, so a GPT-J pair
``(2k, 2k+1)`` never straddles two lanes and the rotation stays register-local.
The RMS reduction over ``D`` is a wave-local ``shuffle_xor`` butterfly (no LDS)
since ``BLOCK_THREADS <= 64``.  The freqs row is loaded once and shared by q
and k.

Freqs: zero-copy, no repeat_interleave
--------------------------------------
``freqs_cis`` is ``[B, S, D/2]`` **complex64**.  ``torch.view_as_real`` on it is
zero-copy and yields ``[B, S, D/2, 2]`` fp32 whose last dim is ``(cos, sin)``,
i.e. a contiguous width-``D`` table laid out ``[c0, s0, c1, s1, ...]``.  That is
*already* the layout a lane wants: the lane owning channels
``[lane, lane + VEC)`` loads ``fc[row*D + lane : + VEC]`` in one fp32 buffer op
and reads its pair ``kk`` as ``cos = cs[2*kk]``, ``sin = cs[2*kk + 1]``.

This is why the kernel takes the freqs as **one** tensor rather than the two
full-D ``(cos, sin)`` tables the FLUX/Qwen kernel takes.  The Qwen path pays a
``repeat_interleave`` into two materialised fp32 tensors on every attention
call (``_qwen_cos_sin``); here the table is passed through as a view.  One
fewer tensor to marshal per launch, and no per-call allocation -- which matters
because at these bandwidth-bound shapes the custom-op boundary, not the kernel,
is what decides the race against Triton.

The rotation itself is::

    out[2k]     = e*cos_k - o*sin_k
    out[2k + 1] = o*cos_k + e*sin_k        (e = x[2k], o = x[2k+1])

which is exactly ``view_as_real(complex(e, o) * complex(cos_k, sin_k))``, the
reference ``apply_rotary_emb``.  Note both lanes of a pair share one
``(cos_k, sin_k)``, unlike the FLUX kernel's repeat_interleave tables where the
per-lane cos/sin happen to be equal by construction.

Output buffer
-------------
q and k are written into **one** ``[2, T, H, D]`` allocation (q at rows
``[0, T)``, k at ``[T, 2T)`` of the flat view) rather than two ``empty_like``
tensors.  That is one fewer marshaled tensor and one fewer allocation per call.
``custom_op`` forbids two returns aliasing one storage, so the op returns the
single ``[2, T, H, D]`` tensor and the caller splits ``out[0] / out[1]`` after
it returns.

Numerics
--------
The kernel replays the reference *rounding schedule*, not just the formula.
``diffusers`` RMSNorm rounds to the bf16 weight dtype after the rstd-multiply
and again after the affine-multiply; ``apply_rotary_emb`` then computes in fp32
and rounds once at the end.  ``ROUND_AFTER_NORM`` replays the two intermediate
roundings, so the fused output agrees with the unfused reference to within a
ULP (measured mean_abs ~3e-08 at bf16 H=30 D=128).

Envelope / fallback
-------------------
Selection is from tensor properties alone -- no environment switch.  bf16
activations, 4-D ``[B, S, H, D]``, a ``D`` that factors into a power-of-two lane
count with an even per-lane width ``<= 4`` (the fp32 freqs load is capped at
128 bits), plain affine-or-weightless RMSNorm with bf16 weights, complex freqs
of width ``D/2``, and a ``(B, S)`` flattening that stays a view.  Anything
outside that -- including a grad-enabled call, since the custom op has no
autograd formula -- falls back to :func:`_reference`, the unfused diffusers
path, so selection changes the speed, never the result.
"""

# NOTE: no ``from __future__ import annotations`` -- PEP 563 stringifies the
# annotations and defeats flydsl's runtime-arg detection for the Int32 kernel
# params, forcing a fresh JIT per value.

import math
from functools import lru_cache
from typing import Optional, Tuple

import torch

from xfuser.logger import init_logger

logger = init_logger(__name__)

# The FlyDSL + aiter stack is optional; importing this module must never hard
# fail on a box without them (CPU-only CI, non-ROCm).  ``_HAS_FLYDSL`` gates the
# fast path; callers gate on it themselves to skip this wrapper entirely.
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
# Python (a folded launch would run tail blocks past the token count -- flydsl's
# ``if cond: return`` does not early-exit a kernel body -- and fault).  Z-Image
# runs a few thousand tokens so this loop is single-trip in practice; it is here
# so a long-sequence caller degrades in performance rather than faulting.
MAX_GRID_Y = 65535

# A lane's freqs load is fp32 and buffer ops top out at 128 bits, so VEC <= 4.
MAX_VEC_F32 = 4


def _pick_block(d: int) -> Optional[Tuple[int, int]]:
    """Return ``(BLOCK_THREADS, VEC)`` for head_dim ``d``, or None if unsupported.

    Prefer the widest wave (few lanes, more work each) while keeping VEC even
    (GPT-J pairs stay lane-local), VEC <= 4 (128-bit fp32 freqs load) and
    BLOCK_THREADS a power of two <= 64 (single wave, so the sum-of-squares
    reduction is a pure shuffle_xor butterfly with no LDS).
    """
    for bt in (64, 32, 16, 8, 4, 2):
        if d % bt == 0:
            vec = d // bt
            if vec >= 2 and vec % 2 == 0 and vec <= MAX_VEC_F32:
                return bt, vec
    return None


if _HAS_FLYDSL:

    @lru_cache(maxsize=32)
    def _build_kernel(
        *,
        H: int,
        D: int,
        VEC: int,
        BLOCK_THREADS: int,
        eps: float,
        NORM_Q: bool,
        NORM_K: bool,
        HAS_WQ: bool,
        HAS_WK: bool,
        ROUND_AFTER_NORM: bool,
    ):
        """Build + cache the @flyc.jit launcher for one config.

        Shape/flag constants are captured by closure (not module globals) so
        launchers for different configs coexist in the lru cache.
        """
        LOG2_BT = int(math.log2(BLOCK_THREADS))
        PAIRS = VEC // 2
        INV_D = 1.0 / D
        HD = H * D

        _kname = f"zimage_fused_qk_norm_rope_H{H}_D{D}_v{VEC}_flydsl"

        @flyc.kernel(name=_kname)
        def kernel(
            q_in: fx.Tensor,   # [T, H*D] bf16, rows strided by q_rs, cols stride 1
            k_in: fx.Tensor,   # [T, H*D] bf16, rows strided by k_rs, cols stride 1
            out: fx.Tensor,    # [2*T, H*D] bf16: q rows [0,T), k rows [T,2T)
            wq: fx.Tensor,     # [D] bf16 (dummy when not HAS_WQ)
            wk: fx.Tensor,     # [D] bf16 (dummy when not HAS_WK)
            fc: fx.Tensor,     # [fc_rows, D] fp32, interleaved (cos, sin) pairs
            fc_rows: Int32,
            tok_base: Int32,   # global token index of this chunk's row 0
            k_out_off: Int32,  # = T*H*D, row offset of k's output block
            q_rs: Int32,       # q_in row stride in elements (H*D if contiguous)
            k_rs: Int32,       # k_in row stride in elements
        ):
            fm_fast = FastMathFlags.fast

            head = fx.Int32(fx.block_idx.x)
            tok = fx.Int32(fx.block_idx.y)
            tid = fx.Int32(fx.thread_idx.x)

            # Token index is global (chunk row 0 + block row) so the tensors are
            # passed whole and chunking only moves ``tok_base`` and grid Y -- no
            # per-chunk tensor slicing, which would break the flat k_out_off.
            gtok = fx.Int32(tok_base) + tok
            lane = tid * VEC          # channel offset within the head
            hoff = head * D + lane

            q_off = gtok * fx.Int32(q_rs) + hoff
            k_off = gtok * fx.Int32(k_rs) + hoff
            # The freshly-allocated output is always contiguous, so its row
            # stride is HD regardless of the (possibly strided) inputs.
            oq_off = gtok * HD + hoff
            ok_off = fx.Int32(k_out_off) + oq_off

            # fc_rows is B*S (Z-Image builds a per-batch table) or S (shared
            # across batch); the modulo makes both correct.
            frow = gtok % fx.Int32(fc_rows)
            coff = frow * D + lane

            qin_ = GTensor(q_in, dtype=T.bf16, shape=(-1,))
            kin_ = GTensor(k_in, dtype=T.bf16, shape=(-1,))
            out_ = GTensor(out, dtype=T.bf16, shape=(-1,))
            wq_ = GTensor(wq, dtype=T.bf16, shape=(-1,))
            wk_ = GTensor(wk, dtype=T.bf16, shape=(-1,))
            fc_ = GTensor(fc, dtype=T.f32, shape=(-1,))

            # One fp32 load of the lane's slice of the interleaved table, shared
            # by q and k.  Element 2*kk is cos of pair kk, 2*kk+1 is its sin --
            # see the module docstring for why this needs no de-interleave pass.
            cs = fx.Vector(fc_.load(coff, vec_size=VEC)).to(fx.Float32)

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

            def process(g_in, in_off, out_off, g_w, NORM, HAS_W):
                x = fx.Vector(g_in.load(in_off, vec_size=VEC)).to(fx.Float32)

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

                if const_expr(HAS_W):
                    w = fx.Vector(g_w.load(lane, vec_size=VEC)).to(fx.Float32)
                    scaled = [scaled[i] * w[i] for i in range_constexpr(VEC)]
                    if const_expr(ROUND_AFTER_NORM):
                        scaled = round_bf16(scaled)

                # Interleaved (GPT-J) rope on lane-local pairs (2k, 2k+1).  Both
                # lanes of a pair share one (cos, sin) read straight out of the
                # complex table.
                outs = [None] * VEC
                for kk in range_constexpr(PAIRS):
                    c = cs[2 * kk]
                    s = cs[2 * kk + 1]
                    e = scaled[2 * kk]
                    o = scaled[2 * kk + 1]
                    outs[2 * kk] = e * c - o * s
                    outs[2 * kk + 1] = o * c + e * s

                out_v = fx.Vector.from_elements(
                    [o.ir_value() for o in outs], dtype=fx.Float32
                )
                out_.store(out_off, out_v.truncf(T.vec(VEC, T.bf16)))

            process(qin_, q_off, oq_off, wq_, NORM_Q, HAS_WQ)
            process(kin_, k_off, ok_off, wk_, NORM_K, HAS_WK)

        @flyc.jit
        def launch_zimage_fused_qk_norm_rope(
            q_in: fx.Tensor,
            k_in: fx.Tensor,
            out: fx.Tensor,
            wq: fx.Tensor,
            wk: fx.Tensor,
            fc: fx.Tensor,
            fc_rows: fx.Int32,
            tok_base: fx.Int32,
            k_out_off: fx.Int32,
            q_rs: fx.Int32,
            k_rs: fx.Int32,
            n_tokens: fx.Int32,
            stream: fx.Stream = fx.Stream(None),
        ):
            k = kernel(
                q_in, k_in, out, wq, wk, fc, fc_rows, tok_base, k_out_off, q_rs, k_rs
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

        launch_zimage_fused_qk_norm_rope.compile_hints = {
            "waves_per_eu": 8,
            "fast_fp_math": True,
        }
        return launch_zimage_fused_qk_norm_rope

    @torch.library.custom_op("xfuser::zimage_flydsl_qk_norm_rope", mutates_args=())
    def _zimage_flydsl_qk_norm_rope_launch(
        q: torch.Tensor,             # [T, H*D] bf16, rows may be strided
        k: torch.Tensor,             # [T, H*D] bf16, rows may be strided
        wq: Optional[torch.Tensor],  # [D] bf16 or None (weightless RMSNorm)
        wk: Optional[torch.Tensor],
        fc: torch.Tensor,            # [fc_rows, D] fp32 interleaved (cos, sin)
        heads: int,
        eps: float,
        norm_q: bool,
        norm_k: bool,
        round_after_norm: bool,
    ) -> torch.Tensor:
        """Opaque launch boundary for torch.compile.

        Everything that touches a real data pointer (``_build_kernel`` ->
        ``flyc.compile`` and the ``_run_compiled`` launches) lives inside this
        custom op so Dynamo treats it as a black box -- calling
        :func:`register_fake` for shape propagation during tracing and this body
        only at runtime with real tensors.  Without it Dynamo traces into
        ``flyc.compile`` and faults on ``FakeTensor.__dlpack__`` (no data
        pointer).

        Returns the merged ``[2, T, H, D]`` buffer; the caller splits it.
        """
        t_tok, hd = q.shape
        d = hd // heads
        has_wq = wq is not None
        has_wk = wk is not None

        bt, vec = _pick_block(d)  # guaranteed non-None by the caller's envelope

        launcher = _build_kernel(
            H=heads,
            D=d,
            VEC=vec,
            BLOCK_THREADS=bt,
            eps=eps,
            NORM_Q=norm_q,
            NORM_K=norm_k,
            HAS_WQ=has_wq,
            HAS_WK=has_wk,
            ROUND_AFTER_NORM=round_after_norm,
        )

        # One allocation for both outputs as a single [2, T, H, D] tensor -- the
        # halves are contiguous [T, H, D] views the caller splits AFTER the op
        # (custom_op forbids two returns aliasing one storage).
        out = torch.empty((2, t_tok, heads, d), dtype=q.dtype, device=q.device)
        flat = out.view(2 * t_tok, hd)

        # The kernel always binds wq/wk; pass a 1-elem dummy when unused (the
        # const_expr HAS_W gate DCEs the load, but the binding needs a real
        # tensor).
        dummy = q.new_empty(1)
        wq_arg = wq if has_wq else dummy
        wk_arg = wk if has_wk else dummy

        # Fetch the stream on q's device directly rather than via a
        # ``torch.cuda.device(...)`` context manager (which costs two
        # cudaSetDevice calls on a path where the boundary is the bottleneck).
        stream = torch.cuda.current_stream(q.device)
        # q/k rows may be strided; pass the runtime row strides so the kernel
        # reads them in place instead of paying a .contiguous() copy here.
        for start in range(0, t_tok, MAX_GRID_Y):
            n = min(MAX_GRID_Y, t_tok - start)
            _run_compiled(
                launcher,
                q,
                k,
                flat,
                wq_arg,
                wk_arg,
                fc,
                fc.shape[0],
                start,
                t_tok * hd,
                q.stride(0),
                k.stride(0),
                n,
                stream,
            )

        return out

    @_zimage_flydsl_qk_norm_rope_launch.register_fake
    def _zimage_flydsl_qk_norm_rope_launch_fake(
        q, k, wq, wk, fc, heads, eps, norm_q, norm_k, round_after_norm
    ):
        t_tok, hd = q.shape
        return q.new_empty((2, t_tok, heads, hd // heads))


def _norm_is_plain_rmsnorm(m: Optional[torch.nn.Module]) -> bool:
    """True when ``m`` is an affine-or-plain RMSNorm we reproduce exactly."""
    if m is None:
        return True
    if getattr(m, "bias", None) is not None:
        return False
    w = getattr(m, "weight", None)
    if w is not None and w.dim() != 1:
        return False
    return True


def _reference(query, key, norm_q, norm_k, freqs_cis):
    """Verbatim copy of the unfused Z-Image path (transformer_z_image.py).

    This is the correctness oracle and the fallback: running exactly the ops
    diffusers runs makes it numerically interchangeable with the fused path, so
    selection can change the speed but never the result.
    """

    def apply_rotary_emb(x_in: torch.Tensor, fc: torch.Tensor) -> torch.Tensor:
        with torch.amp.autocast("cuda", enabled=False):
            x = torch.view_as_complex(x_in.float().reshape(*x_in.shape[:-1], -1, 2))
            fc = fc.unsqueeze(2)
            x_out = torch.view_as_real(x * fc).flatten(3)
            return x_out.type_as(x_in)

    if norm_q is not None:
        query = norm_q(query)
    if norm_k is not None:
        key = norm_k(key)
    if freqs_cis is not None:
        query = apply_rotary_emb(query, freqs_cis)
        key = apply_rotary_emb(key, freqs_cis)
    return query, key


def _as_rows(t: torch.Tensor, b: int, s: int, h: int, d: int) -> Optional[torch.Tensor]:
    """``[B, S, H, D]`` -> a ``[B*S, H*D]`` **view**, or None if that would copy.

    The kernel takes an explicit row stride, so any row stride is fine as long
    as each row's ``(H, D)`` block is contiguous.  Returning None (rather than
    silently calling ``.contiguous()``) keeps a large hidden copy off the hot
    path -- the caller falls back instead.
    """
    if t.stride(-1) != 1 or t.stride(-2) != d:
        return None
    if b == 1:
        # Drops the unit batch dim and merges (H, D); stays a view for any row
        # stride.
        return t.reshape(s, h * d)
    if t.stride(0) != s * t.stride(1):
        return None  # merging (B, S) would need a copy
    return t.reshape(b * s, h * d)


def flydsl_fused_qk_norm_rope(
    query: torch.Tensor,
    key: torch.Tensor,
    norm_q: Optional[torch.nn.Module],
    norm_k: Optional[torch.nn.Module],
    freqs_cis: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """RMSNorm(q)/RMSNorm(k) followed by complex/interleaved RoPE, fused.

    ``query`` / ``key`` are ``[B, S, H, D]`` (the Z-Image layout right after
    ``unflatten(-1, (heads, -1))``); ``freqs_cis`` is the ``[B, S, D/2]``
    complex64 table.  Returns ``(q, k)`` with the same shape and dtype as the
    unfused path, falling back to :func:`_reference` whenever the fast envelope
    is not met.
    """
    _ref = lambda: _reference(  # noqa: E731 - local fallback shorthand
        query, key, norm_q, norm_k, freqs_cis
    )

    if not _HAS_FLYDSL:
        return _ref()
    # Inference-only fast path: the custom op has no autograd formula, so any
    # grad-enabled call falls back.
    if torch.is_grad_enabled():
        return _ref()
    if freqs_cis is None:
        return _ref()
    if not (_norm_is_plain_rmsnorm(norm_q) and _norm_is_plain_rmsnorm(norm_k)):
        return _ref()

    if not query.is_cuda or query.dim() != 4:
        return _ref()
    if query.shape != key.shape or query.dtype != key.dtype:
        return _ref()
    # bf16-only: the kernel's GTensors and the round-after-norm schedule are
    # bf16; fp16/fp32 fall back to the unfused reference.
    if query.dtype != torch.bfloat16:
        return _ref()

    b, s, h, d = query.shape
    if _pick_block(d) is None:
        return _ref()

    # The reference does complex(x) * freqs_cis, so the table must be complex
    # with exactly D/2 entries per token.
    if not isinstance(freqs_cis, torch.Tensor) or not freqs_cis.is_complex():
        return _ref()
    if freqs_cis.dim() != 3 or freqs_cis.shape[-1] != d // 2:
        return _ref()
    if freqs_cis.shape[0] != b or freqs_cis.shape[1] != s:
        return _ref()

    q2 = _as_rows(query, b, s, h, d)
    k2 = _as_rows(key, b, s, h, d)
    if q2 is None or k2 is None:
        return _ref()

    # Zero-copy reinterpretation of the complex table as a contiguous width-D
    # fp32 (cos, sin) interleaved table.  view_as_real needs last-dim stride 1.
    fcc = freqs_cis if freqs_cis.is_contiguous() else freqs_cis.contiguous()
    fc = torch.view_as_real(fcc).reshape(-1, d)
    if fc.dtype != torch.float32:  # complex128 -> fp64; not supported
        return _ref()
    if fc.shape[0] not in (s, b * s):
        return _ref()

    wq = getattr(norm_q, "weight", None) if norm_q is not None else None
    wk = getattr(norm_k, "weight", None) if norm_k is not None else None
    # bf16 weights are required to replay the round-after-norm schedule.
    for w in (wq, wk):
        if w is not None and w.dtype != torch.bfloat16:
            return _ref()
    if wq is not None:
        wq = wq.contiguous()
    if wk is not None:
        wk = wk.contiguous()

    # diffusers' RMSNorm only rounds the normalised activation to the weight
    # dtype when that dtype is half; bf16 activations here always trigger it.
    round_after_norm = True

    eps = 1e-6
    for m in (norm_q, norm_k):
        e = getattr(m, "eps", None) if m is not None else None
        if e is not None:
            eps = float(e)
            break

    out = torch.ops.xfuser.zimage_flydsl_qk_norm_rope(
        q2,
        k2,
        wq,
        wk,
        fc,
        h,
        float(eps),
        norm_q is not None,
        norm_k is not None,
        bool(round_after_norm),
    )
    return out[0].view(b, s, h, d), out[1].view(b, s, h, d)
