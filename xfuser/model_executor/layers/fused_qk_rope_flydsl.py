"""Fused QK-RMSNorm + interleaved (GPT-J) RoPE for FLUX joint attention (FlyDSL).

Written in the high-level ``flydsl.expr`` (fx) API on top of aiter's
``GTensor`` buffer-tensor shim (the same style aiter uses in
``aiter/ops/flydsl/kernels/qk_norm_rope_quant.py``).

Layout / algorithm (per attention layer the baseline runs norm_q/norm_k then
diffusers ``apply_rotary_emb``; this fuses the whole chain):

    one block == one (token, head); the block's ``BLOCK_THREADS`` lanes cover the
    head's ``D`` channels with each lane owning ``VEC`` contiguous channels
    (``VEC`` even so a GPT-J pair ``(2k, 2k+1)`` never straddles two lanes).
    The RMS reduction over ``D`` is a wave-local ``shuffle_xor`` butterfly (no
    LDS) since ``BLOCK_THREADS <= 64``.  cos/sin are loaded once and shared by q
    and k.

Numerics: the kernel reproduces the reference *rounding schedule*, not just the
formula.  ``diffusers`` RMSNorm rounds to the (bf16) weight dtype after the
rstd-multiply and again after the affine-multiply, then ``apply_rotary_emb``
upcasts to fp32 for the rotate and rounds once at the end.  ``ROUND_AFTER_NORM``
replays the two intermediate roundings so the fused output matches the unfused
output to within a ULP.

Envelope: bf16 activations, 4-D ``[B, S, H, D]`` (sequence_dim == 1), even ``D``
that factors into a power-of-two lane count with an even per-lane width, plain
(affine-or-weightless) RMSNorm, and a full-D cos/sin table whose row count is
``S`` or ``B*S``.  Anything outside that falls back to the unfused diffusers
reference (norm then ``apply_rotary_emb``) so selection never changes the
result, only the speed.
"""

# NOTE: no ``from __future__ import annotations`` -- PEP 563 stringifies the
# annotations and defeats flydsl's runtime-arg detection for the Int32 kernel
# params (cos_rows / pos_offset), forcing a fresh JIT per value.

import math
from functools import lru_cache
from typing import Optional, Tuple

import torch

from xfuser.logger import init_logger

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


def _pick_block(d: int) -> Optional[Tuple[int, int]]:
    """Return ``(BLOCK_THREADS, VEC)`` for head_dim ``d`` or None if unsupported.

    Prefer the widest wave (few lanes, more work each) while keeping VEC even
    (GPT-J pairs stay lane-local) and BLOCK_THREADS a power of two <= 64 (single
    wave, so the sum-of-squares reduction is a pure shuffle_xor butterfly).
    """
    for bt in (64, 32, 16, 8, 4, 2):
        if d % bt == 0:
            vec = d // bt
            if vec >= 2 and vec % 2 == 0:
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
        cos_is_f32: bool,
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

        _kname = f"fused_qk_norm_rope_H{H}_D{D}_v{VEC}_flydsl"

        @flyc.kernel(name=_kname)
        def kernel(
            q_in: fx.Tensor,   # [n, H, D] bf16, contiguous (H, D)
            k_in: fx.Tensor,   # [n, H, D] bf16, contiguous (H, D)
            q_out: fx.Tensor,  # [n, H, D] bf16
            k_out: fx.Tensor,  # [n, H, D] bf16
            wq: fx.Tensor,      # [D] bf16 (dummy when not HAS_WQ)
            wk: fx.Tensor,      # [D] bf16 (dummy when not HAS_WK)
            cos: fx.Tensor,     # [cos_rows, D] cos_dt
            sin: fx.Tensor,     # [cos_rows, D] cos_dt
            cos_rows: Int32,
            pos_offset: Int32,  # global token index of this chunk's row 0
        ):
            fm_fast = FastMathFlags.fast
            # Resolve element types inside the kernel body: T.* needs a live MLIR
            # context (established by @flyc.kernel), so it cannot be a closure const.
            cos_dt = T.f32 if cos_is_f32 else T.bf16

            head = fx.Int32(fx.block_idx.x)
            tok = fx.Int32(fx.block_idx.y)
            tid = fx.Int32(fx.thread_idx.x)

            # element bases: q/k are the chunk-local [n, H, D] view, so ``tok``
            # is local; cos/sin index the global row.
            base = (tok * H + head) * D + tid * VEC
            row = (fx.Int32(pos_offset) + tok) % fx.Int32(cos_rows)
            coff = row * D + tid * VEC
            woff = tid * VEC

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

            def process(g_in, g_out, g_w, NORM, HAS_W):
                x = fx.Vector(g_in.load(base, vec_size=VEC)).to(fx.Float32)

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
                    w = fx.Vector(g_w.load(woff, vec_size=VEC)).to(fx.Float32)
                    scaled = [scaled[i] * w[i] for i in range_constexpr(VEC)]
                    if const_expr(ROUND_AFTER_NORM):
                        scaled = round_bf16(scaled)

                # interleaved GPT-J rope on lane-local pairs (2k, 2k+1):
                #   out[2k]   = e*cos[2k]   - o*sin[2k]
                #   out[2k+1] = o*cos[2k+1] + e*sin[2k+1]
                outs = [None] * VEC
                for kk in range_constexpr(PAIRS):
                    e = scaled[2 * kk]
                    o = scaled[2 * kk + 1]
                    outs[2 * kk] = e * cos_f[2 * kk] - o * sin_f[2 * kk]
                    outs[2 * kk + 1] = o * cos_f[2 * kk + 1] + e * sin_f[2 * kk + 1]

                out_v = fx.Vector.from_elements(
                    [o.ir_value() for o in outs], dtype=fx.Float32
                )
                g_out.store(base, out_v.truncf(T.vec(VEC, T.bf16)))

            process(qin_, qout_, wq_, NORM_Q, HAS_WQ)
            process(kin_, kout_, wk_, NORM_K, HAS_WK)

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
            pos_offset: fx.Int32,
            n_tokens: fx.Int32,
            stream: fx.Stream = fx.Stream(None),
        ):
            k = kernel(
                q_in, k_in, q_out, k_out, wq, wk, cos, sin, cos_rows, pos_offset
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
        q: torch.Tensor,       # [T, H, D] bf16, contiguous
        k: torch.Tensor,       # [T, H, D] bf16, contiguous
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
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Opaque launch boundary for torch.compile.

        Everything that touches a real data pointer (``_build_kernel`` ->
        ``flyc.compile`` and the ``_run_compiled`` grid launches) lives inside
        this custom op so Dynamo treats it as a black box -- calling
        :func:`register_fake` for shape propagation during tracing and this body
        only at runtime with real tensors.  Without it Dynamo traces into
        ``flyc.compile`` and faults on ``FakeTensor.__dlpack__`` (no data
        pointer).
        """
        t_tok, h, d = q.shape
        has_wq = wq is not None
        has_wk = wk is not None

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
        )

        oq = torch.empty_like(q)
        ok = torch.empty_like(k)

        # kernel always binds wq/wk params; pass a 1-elem dummy when unused (the
        # const_expr HAS_W gate DCEs the load, but the binding needs a valid
        # tensor).
        dummy = q.new_empty(1, dtype=torch.bfloat16).view(1)
        wq_arg = wq if has_wq else dummy
        wk_arg = wk if has_wk else dummy

        stream = torch.cuda.current_stream()
        for start in range(0, t_tok, MAX_GRID_Y):
            n = min(MAX_GRID_Y, t_tok - start)
            end = start + n
            with torch.cuda.device(q.device.index):
                _run_compiled(
                    launcher,
                    q[start:end],
                    k[start:end],
                    oq[start:end],
                    ok[start:end],
                    wq_arg,
                    wk_arg,
                    cos,
                    sin,
                    cos_rows,
                    start,
                    n,
                    stream,
                )

        return oq, ok

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
    ):
        return torch.empty_like(q), torch.empty_like(k)


def _supported(query, key, cos) -> bool:
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
    if _pick_block(d) is None:
        return False
    if not isinstance(cos, torch.Tensor) or cos.shape[-1] != d:
        return False
    return True


def _norm_is_plain_rmsnorm(m) -> bool:
    if m is None:
        return True
    if getattr(m, "bias", None) is not None:
        return False
    w = getattr(m, "weight", None)
    if w is not None and w.dim() != 1:
        return False
    return True


def flydsl_fused_qk_norm_rope(
    query: torch.Tensor,
    key: torch.Tensor,
    norm_q,
    norm_k,
    rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """FlyDSL RMSNorm(q)/RMSNorm(k) + interleaved RoPE, fused.

    ``query`` / ``key`` are ``[B, S, H, D]`` (sequence_dim == 1); ``rotary_emb``
    is the diffusers ``(cos, sin)`` pair broadcastable to ``[S, D]``.  Falls back
    to the unfused diffusers reference (norm then ``apply_rotary_emb``) for
    anything outside the supported envelope so the result is identical either
    way.
    """
    from xfuser.model_executor.layers.fused_qk_rope import _reference

    if (
        rotary_emb is None
        or not isinstance(rotary_emb, (tuple, list))
        or len(rotary_emb) != 2
    ):
        return _reference(query, key, norm_q, norm_k, rotary_emb)
    if not (_norm_is_plain_rmsnorm(norm_q) and _norm_is_plain_rmsnorm(norm_k)):
        return _reference(query, key, norm_q, norm_k, rotary_emb)

    cos, sin = rotary_emb
    if not _supported(query, key, cos):
        return _reference(query, key, norm_q, norm_k, rotary_emb)

    b, s, h, d = query.shape
    cos2 = cos.reshape(-1, d)
    sin2 = sin.reshape(-1, d)
    if cos2.shape != sin2.shape or cos2.shape[0] not in (s, b * s):
        return _reference(query, key, norm_q, norm_k, rotary_emb)
    cos_rows = cos2.shape[0]

    if cos2.dtype not in (torch.float32, torch.bfloat16):
        return _reference(query, key, norm_q, norm_k, rotary_emb)

    block_vec = _pick_block(d)
    if block_vec is None:  # already checked in _supported, belt & suspenders
        return _reference(query, key, norm_q, norm_k, rotary_emb)
    block_threads, vec = block_vec

    # contiguous [T, H, D]
    q = query.contiguous().view(b * s, h, d)
    k = key.contiguous().view(b * s, h, d)
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
        return _reference(query, key, norm_q, norm_k, rotary_emb)

    # diffusers RMSNorm only rounds the activation to the weight dtype when that
    # dtype is half; bf16 activations here always trigger it.
    round_after_norm = True

    eps = 1e-6
    for m in (norm_q, norm_k):
        e = getattr(m, "eps", None) if m is not None else None
        if e is not None:
            eps = float(e)
            break

    # The build + grid launches run inside an opaque custom op so this whole
    # path is safe under torch.compile (see _flydsl_qk_norm_rope_launch).
    oq, ok = _flydsl_qk_norm_rope_launch(
        q,
        k,
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
    )

    return oq.view(b, s, h, d), ok.view(b, s, h, d)
