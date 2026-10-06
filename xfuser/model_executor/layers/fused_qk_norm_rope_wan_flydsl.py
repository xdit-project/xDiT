"""Fused QK-RMSNorm(across-heads) + interleaved RoPE for Wan

Layout / algorithm
------------------
    one block == one token; the block's ``BLOCK_THREADS`` (<= 64, single wave)
    lanes cover the whole ``H*D`` row in ``N_TILES`` contiguous tiles of
    ``BLOCK_THREADS * VEC`` elements, each lane owning ``VEC`` contiguous
    channels per tile.  The sum-of-squares over the full row is a per-lane
    partial (over its tiles) followed by ONE wave-local ``shuffle_xor``
    butterfly (no LDS, since BLOCK_THREADS <= wave size), giving every lane the
    shared row ``rstd``.  q is processed to completion, then k, so the two
    reuse the same register file.  cos/sin are loaded once and shared by q/k.

    The tile geometry is chosen so ``BLOCK_THREADS * VEC`` is a multiple of the
    head dim ``D`` -- then every tile's base is head-aligned and a lane's
    ``VEC`` block never straddles a head boundary, so its interleaved-RoPE
    cos/sin lives at the constant head-offset ``(tid*VEC) % D`` and is gathered
    once per lane.  ``VEC`` tracks the bf16 q/k/out width (up to 8 -> a full
    128-bit buffer op); the fp32 cos/sin tables, which would cap a single load
    at 4, are read in ``<=4``-wide chunks so they never throttle the q/k VEC.
    diffusers builds the full-D cos/sin with ``repeat_interleave(2)``, so
    ``cos[..., 2p] == cos[..., 2p+1]`` and the full-D table already equals the
    ``even_in_head`` table the Triton kernel constructs.
"""

import math
from functools import lru_cache
from typing import Optional, Tuple

import torch

from xfuser.model_executor.layers.flydsl_utils import get_device_wave_size

# The FlyDSL + aiter stack is optional; importing this module must never hard
# fail on a box without them (CPU-only CI, non-ROCm).  ``_HAS_FLYDSL`` gates the
# fast path; the wrapper / enabled() check fall back to the reference when False.
try:
    import flydsl.compiler as flyc
    import flydsl.expr as fx
    from flydsl.expr import const_expr, range_constexpr
    from flydsl.expr import math as fmath
    from flydsl.expr.arith import FastMathFlags
    from flydsl.expr.typing import ReductionOp, T

    from aiter.ops.flydsl.kernels.tensor_shim import GTensor, _run_compiled

    _HAS_FLYDSL = True
except Exception:  # pragma: no cover - only exercised where flydsl is absent
    _HAS_FLYDSL = False


def _pick_tiling(H: int, D: int, wave_size: int) -> Optional[Tuple[int, int, int]]:
    """Return ``(BLOCK_THREADS, VEC, N_TILES)`` for ``[H, D]`` or None.

    ``VEC`` is the q/k/out vectorization -- q/k/out are bf16, so a single 128-bit
    buffer op caps VEC at 8.  The fp32 cos/sin loads do NOT constrain VEC: they
    are loaded separately in <=4-wide (<=128-bit) chunks inside the kernel, so
    the bf16 q/k path keeps its full 128-bit VEC even when cos/sin are fp32.

    Constraints (see module docstring):
      * BLOCK_THREADS no larger than the hardware wave (single wave ->
        shuffle_xor butterfly);
      * VEC even and ``VEC | D`` (GPT-J pairs stay lane-local, lane block inside
        one head);
      * ``(BLOCK_THREADS * VEC) | (H*D)`` (whole tiles) and
        ``(BLOCK_THREADS * VEC) % D == 0`` (every tile head-aligned, so the
        per-lane cos/sin offset ``(tid*VEC) % D`` is constant across tiles);
      * ``VEC * 2 <= 16`` (bf16 q/k/out single 128-bit buffer op) -> VEC <= 8.
    Prefer the widest wave, then the widest VEC (fewer, wider 128-bit loads).
    """
    HD = H * D
    bt = wave_size
    while bt >= 16:
        if HD % bt:
            bt //= 2
            continue
        for vec in (8, 4, 2):
            if D % vec or vec % 2:
                continue
            tile = bt * vec
            if HD % tile or tile % D:
                continue
            return bt, vec, HD // tile
        bt //= 2
    return None


if _HAS_FLYDSL:

    def _hadamard_head(values, lane, head_dim):
        """Normalized in-register Walsh-Hadamard over one attention head."""
        result = [values[i] for i in range(len(values))]
        for stage in range(len(values).bit_length() - 1):
            shift = 1 << stage
            result = [
                (
                    result[i ^ shift] - result[i]
                    if i & shift
                    else result[i] + result[i ^ shift]
                )
                for i in range(len(values))
            ]
        for stage in range((head_dim // len(values)).bit_length() - 1):
            shift = 1 << stage
            result = [
                ((lane & shift) == 0).select(
                    value + value.shuffle_xor(shift, 64),
                    value.shuffle_xor(shift, 64) - value,
                )
                for value in result
            ]
        return fx.Vector.from_elements(result, fx.Float32) * (head_dim**-0.5)

    @lru_cache(maxsize=32)
    def _build_kernel(
        *,
        HD: int,
        D: int,
        BLOCK_THREADS: int,
        VEC: int,
        N_TILES: int,
        eps: float,
        cos_is_f32: bool,
        single_role: bool,
        apply_hadamard: bool,
    ):
        """Build + cache the @flyc.jit launcher for one config.

        Shape/flag constants are captured by closure (not module globals) so
        launchers for different configs coexist in the lru cache.
        """
        LOG2_BT = int(math.log2(BLOCK_THREADS))
        PAIRS = VEC // 2
        TILE = BLOCK_THREADS * VEC
        INV_HD = 1.0 / HD
        # q/k/out are bf16 (128-bit VEC-wide buffer ops).  cos/sin, when fp32,
        # must load in <=4-wide (<=128-bit) chunks; bf16 cos/sin can go the full
        # VEC.  Either way the freq row is gathered once per lane into VEC fp32
        # scalars (see _load_freq below), so the q/k VEC is never fp32-capped.
        # Precompute the (offset, width) chunking at build time as Python consts
        # -- a runtime `while` here would be captured as a device scf.while.
        _freq_chunk = 4 if cos_is_f32 else VEC
        FREQ_CHUNKS = []
        _off = 0
        while _off < VEC:
            _n = min(_freq_chunk, VEC - _off)
            FREQ_CHUNKS.append((_off, _n))
            _off += _n

        _kname = (
            f"wan_fused_qk_norm_rope_HD{HD}_D{D}_v{VEC}"
            f"{'_single' if single_role else ''}"
            f"{'_hadamard' if apply_hadamard else ''}_flydsl"
        )

        @flyc.kernel(name=_kname)
        def kernel(
            q_in: fx.Tensor,   # [S, H*D] bf16, rows strided by q_rs, cols stride 1
            k_in: fx.Tensor,   # [S, H*D] bf16, rows strided by k_rs, cols stride 1
            out: fx.Tensor,    # [2*S, H*D] bf16: q rows [0,S), k rows [S,2S)
            wq: fx.Tensor,     # [H*D] bf16 (norm_q.weight, across-heads)
            wk: fx.Tensor,     # [H*D] bf16 (norm_k.weight)
            cos: fx.Tensor,    # [S, D] cos_dt (diffusers full-D repeat_interleave)
            sin: fx.Tensor,    # [S, D] cos_dt
            k_out_off: fx.Int32,  # = S*H*D, the row offset of k's output block
            q_rs: fx.Int32,    # q_in row stride in elements (H*D if contiguous)
            k_rs: fx.Int32,    # k_in row stride in elements
        ):
            fm_fast = FastMathFlags.fast
            # Resolve cos/sin element type inside the body: T.* needs the live
            # MLIR context established by @flyc.kernel.
            cos_dt = T.f32 if cos_is_f32 else T.bf16

            tok = fx.Int32(fx.block_idx.x)
            tid = fx.Int32(fx.thread_idx.x)

            qin_ = GTensor(q_in, dtype=T.bf16, shape=(-1,))
            kin_ = GTensor(k_in, dtype=T.bf16, shape=(-1,))
            # Single output buffer; q and k write disjoint halves ([0,S) and
            # [S,2S)).  One output tensor arg -> one fewer dlpack marshal per
            # launch than two separate q_out/k_out
            out_ = GTensor(out, dtype=T.bf16, shape=(-1,))
            wq_ = GTensor(wq, dtype=T.bf16, shape=(-1,))
            wk_ = GTensor(wk, dtype=T.bf16, shape=(-1,))
            cos_ = GTensor(cos, dtype=cos_dt, shape=(-1,))
            sin_ = GTensor(sin, dtype=cos_dt, shape=(-1,))

            # Input rows may be strided (e.g. q/k are chunks of a fused QKV
            # projection with row stride 3*H*D) -- read with the runtime row
            # stride instead of forcing a contiguous copy in the caller.  The
            # freshly-allocated output is always contiguous (row stride H*D).
            out_row_base = tok * HD       # this token's row in the [.., H*D] out
            # Head-offset of this lane's VEC block: (tid*VEC) % D.  Every tile is
            # head-aligned ((BLOCK*VEC) % D == 0), so this is the same for all
            # tiles and its cos/sin is loaded once (see _load_freq).
            head_off = (tid * VEC) % fx.Int32(D)
            coff = tok * D + head_off

            def _load_freq(g):
                # Gather this lane's VEC cos/sin values (constant across tiles,
                # since every tile is head-aligned) as fp32 scalars.  fp32 tables
                # are read in <=4-wide chunks so no single buffer op exceeds 128
                # bits, keeping the bf16 q/k path at the full VEC width.
                vals = []
                for ci in range_constexpr(len(FREQ_CHUNKS)):
                    off, n = FREQ_CHUNKS[ci]
                    part = fx.Vector(g.load(coff + off, vec_size=n)).to(fx.Float32)
                    for i in range_constexpr(n):
                        vals.append(part[i])
                return vals

            cos_f = _load_freq(cos_)
            sin_f = _load_freq(sin_)

            def wave_reduce_add(x):
                w = fx.Float32(x)
                for sh_exp in range_constexpr(LOG2_BT):
                    off = BLOCK_THREADS // (2 << sh_exp)
                    w = w.addf(w.shuffle_xor(off, BLOCK_THREADS), fastmath=fm_fast)
                return w

            def process(g_in, in_row_base, out_row_off, g_w):
                # ``in_row_base`` is this token's element offset into g_in
                # (tok * input_row_stride); ``out_row_off`` is its offset into
                # the shared contiguous output buffer.
                # Pass 1: load every tile this lane owns, accumulate sum(x^2)
                # over the whole H*D row (per-lane partial, then wave butterfly).
                tiles = []
                sq_acc = fx.Float32(0.0)
                for c in range_constexpr(N_TILES):
                    base_c = c * TILE
                    # Cache the raw bf16 tile (NOT upcast): the row lives in
                    # registers between pass 1 and pass 2, and holding it bf16
                    # halves the tile-cache VGPR footprint vs fp32 (lifting
                    # occupancy), while re-upcasting in pass 2 is bit-exact.
                    xt = fx.Vector(
                        g_in.load(in_row_base + base_c + tid * VEC, vec_size=VEC)
                    )
                    tiles.append(xt)
                    xf = xt.to(fx.Float32)
                    x2 = xf * xf
                    sq_local = x2.reduce(ReductionOp.ADD, fastmath=fm_fast)
                    sq_acc = sq_acc.addf(fx.Float32(sq_local), fastmath=fm_fast)
                sq = wave_reduce_add(sq_acc)
                rstd = fmath.rsqrt(sq * INV_HD + eps, fastmath=fm_fast)

                # Pass 2: RMSNorm affine (fp32, no intermediate round) -> GPT-J
                # interleaved RoPE on lane-local pairs -> single round at store.
                for c in range_constexpr(N_TILES):
                    base_c = c * TILE
                    w = fx.Vector(
                        g_w.load(base_c + tid * VEC, vec_size=VEC)
                    ).to(fx.Float32)
                    xf = tiles[c].to(fx.Float32)  # re-upcast the cached bf16 tile
                    scaled = [xf[i] * rstd * w[i] for i in range_constexpr(VEC)]
                    outs = [None] * VEC
                    for kk in range_constexpr(PAIRS):
                        e = scaled[2 * kk]
                        o = scaled[2 * kk + 1]
                        outs[2 * kk] = e * cos_f[2 * kk] - o * sin_f[2 * kk]
                        outs[2 * kk + 1] = o * cos_f[2 * kk + 1] + e * sin_f[2 * kk + 1]
                    out_v = fx.Vector.from_elements(
                        [o.ir_value() for o in outs], dtype=fx.Float32
                    )
                    if const_expr(apply_hadamard):
                        # Preserve the previous two-kernel numerical boundary:
                        # norm/RoPE rounds to BF16 before the FP32 Hadamard, then
                        # the final BF16 store is consumed by Attention A2A.
                        rounded = out_v.truncf(T.vec(VEC, T.bf16)).to(fx.Float32)
                        out_v = _hadamard_head(
                            [rounded[i] for i in range_constexpr(VEC)], tid, D
                        )
                    out_.store(
                        out_row_off + base_c + tid * VEC,
                        out_v.truncf(T.vec(VEC, T.bf16)),
                    )

            # q reads strided rows (tok * q_rs) -> contiguous out rows [0, S);
            # k reads strided rows (tok * k_rs) -> contiguous out rows [S, 2S).
            process(qin_, tok * q_rs, out_row_base, wq_)
            if const_expr(not single_role):
                process(kin_, tok * k_rs, k_out_off + out_row_base, wk_)

        @flyc.jit
        def launch_wan_fused_qk_norm_rope(
            q_in: fx.Tensor,
            k_in: fx.Tensor,
            out: fx.Tensor,
            wq: fx.Tensor,
            wk: fx.Tensor,
            cos: fx.Tensor,
            sin: fx.Tensor,
            n_tokens: fx.Int32,
            k_out_off: fx.Int32,
            q_rs: fx.Int32,
            k_rs: fx.Int32,
            stream: fx.Stream = fx.Stream(None),
        ):
            k = kernel(q_in, k_in, out, wq, wk, cos, sin, k_out_off, q_rs, k_rs)
            # One block per token on grid X (32-bit, no MAX_GRID_Y chunk needed).
            # Pass the runtime Int32 grid dim RAW so KernelLauncher.launch casts
            # it inside its own MLIR context (an index_cast here faults on a
            # dynamo-resumed frame -- no live context).
            k.launch(
                grid=(n_tokens, 1, 1),
                block=(BLOCK_THREADS, 1, 1),
                stream=stream,
            )

        launch_wan_fused_qk_norm_rope.compile_hints = {
            "waves_per_eu": 8,
            "fast_fp_math": True,
        }
        return launch_wan_fused_qk_norm_rope

    @torch.library.custom_op("xfuser::wan_flydsl_qk_norm_rope", mutates_args=())
    def _wan_flydsl_qk_norm_rope_launch(
        q: torch.Tensor,     # [S, H*D] bf16, rows may be strided (last dim == 1)
        k: torch.Tensor,     # [S, H*D] bf16, rows may be strided (last dim == 1)
        wq: torch.Tensor,    # [H*D] bf16
        wk: torch.Tensor,    # [H*D] bf16
        cos: torch.Tensor,   # [S, D]
        sin: torch.Tensor,   # [S, D]
        heads: int,
        eps: float,
        single_role: bool,
        apply_hadamard: bool,
    ) -> torch.Tensor:
        """Opaque launch boundary for torch.compile.

        All data-pointer work (``_build_kernel`` -> ``flyc.compile`` and the
        ``_run_compiled`` launch) lives here so Dynamo treats it as a black box
        (shape-props via ``register_fake``, this body only at runtime with real
        tensors).  Without it Dynamo traces into ``flyc.compile`` and faults on
        ``FakeTensor.__dlpack__`` (no data pointer).
        """
        S, HD = q.shape
        D = HD // heads
        wave_size = get_device_wave_size(q)
        tiling = _pick_tiling(heads, D, wave_size) if wave_size is not None else None
        if tiling is None:
            raise RuntimeError("unsupported WAN FlyDSL QK norm/RoPE tiling")
        bt, vec, n_tiles = tiling

        launcher = _build_kernel(
            HD=HD,
            D=D,
            BLOCK_THREADS=bt,
            VEC=vec,
            N_TILES=n_tiles,
            eps=eps,
            cos_is_f32=(cos.dtype == torch.float32),
            single_role=single_role,
            apply_hadamard=apply_hadamard,
        )

        # Pair mode uses one allocation for both outputs. Single-role overlap
        # mode allocates only Q or K, avoiding duplicate compute and storage.
        # The caller splits pair output after the op because custom_op forbids
        # returning two aliases of one backing allocation.
        out = torch.empty(
            (1 if single_role else 2, S, heads, D),
            dtype=q.dtype,
            device=q.device,
        )

        # Fetch the stream on q's device directly instead of a
        # ``torch.cuda.device(...)`` context manager (two cudaSetDevice calls) --
        # at the Wan shape this is a ~5us boundary
        stream = torch.cuda.current_stream(q.device)
        # q/k may be non-contiguous rows (e.g. chunks of a fused-QKV projection,
        # row stride 3*H*D).  Pass their runtime row strides so the kernel reads
        # them in place -- forcing a .contiguous() copy in the caller doubled the
        # op's cost at the Wan shape.  Last-dim stride is validated == 1 upstream.
        _run_compiled(
            launcher,
            q,
            k,
            out.view((1 if single_role else 2) * S, HD),
            wq,
            wk,
            cos,
            sin,
            S,
            S * HD,
            q.stride(0),
            k.stride(0),
            stream,
        )
        return out

    @_wan_flydsl_qk_norm_rope_launch.register_fake
    def _wan_flydsl_qk_norm_rope_launch_fake(
        q, k, wq, wk, cos, sin, heads, eps, single_role, apply_hadamard
    ):
        S, HD = q.shape
        D = HD // heads
        return q.new_empty((1 if single_role else 2, S, heads, D))

    _HADAMARD_KERNEL_CACHE = {}

    def _build_hadamard_kernel(
        *,
        H: int,
        D: int,
        BLOCK_THREADS: int,
        VEC: int,
        N_TILES: int,
    ):
        key = (H, D, BLOCK_THREADS, VEC, N_TILES)
        cached = _HADAMARD_KERNEL_CACHE.get(key)
        if cached is not None:
            return cached
        HD = H * D
        TILE = BLOCK_THREADS * VEC

        @flyc.kernel(name=f"wan_hadamard_H{H}_D{D}_v{VEC}_flydsl")
        def kernel(source: fx.Tensor, output: fx.Tensor):
            tok = fx.Int32(fx.block_idx.x)
            tid = fx.Int32(fx.thread_idx.x)
            source_ = GTensor(source, dtype=T.bf16, shape=(-1,))
            output_ = GTensor(output, dtype=T.bf16, shape=(-1,))
            row = tok * HD
            for tile_idx in range_constexpr(N_TILES):
                offset = row + tile_idx * TILE + tid * VEC
                values = fx.Vector(
                    source_.load(offset, vec_size=VEC)
                ).to(fx.Float32)
                rotated = _hadamard_head(
                    [values[i] for i in range_constexpr(VEC)], tid, D
                )
                output_.store(offset, rotated.truncf(T.vec(VEC, T.bf16)))

        @flyc.jit
        def launch_wan_hadamard(
            source: fx.Tensor,
            output: fx.Tensor,
            n_tokens: fx.Int32,
            stream: fx.Stream = fx.Stream(None),
        ):
            k = kernel(source, output)
            k.launch(
                grid=(n_tokens, 1, 1),
                block=(BLOCK_THREADS, 1, 1),
                stream=stream,
            )

        launch_wan_hadamard.compile_hints = {
            "waves_per_eu": 8,
            "fast_fp_math": True,
        }
        _HADAMARD_KERNEL_CACHE[key] = launch_wan_hadamard
        return launch_wan_hadamard

    @torch.library.custom_op("xfuser::wan_flydsl_hadamard", mutates_args=())
    def _wan_flydsl_hadamard_launch(x: torch.Tensor) -> torch.Tensor:
        _, S, H, D = x.shape
        wave_size = get_device_wave_size(x)
        tiling = _pick_tiling(H, D, wave_size) if wave_size is not None else None
        if tiling is None:
            raise RuntimeError("unsupported WAN FlyDSL Hadamard tiling")
        bt, vec, n_tiles = tiling
        launcher = _build_hadamard_kernel(
            H=H,
            D=D,
            BLOCK_THREADS=bt,
            VEC=vec,
            N_TILES=n_tiles,
        )
        output = torch.empty_like(x)
        _run_compiled(
            launcher,
            x.view(S, H * D),
            output.view(S, H * D),
            S,
            torch.cuda.current_stream(x.device),
        )
        return output

    @_wan_flydsl_hadamard_launch.register_fake
    def _wan_flydsl_hadamard_launch_fake(x):
        return torch.empty_like(x)


def _reference(
    query: torch.Tensor,
    key: torch.Tensor,
    norm_q: torch.nn.Module,
    norm_k: torch.nn.Module,
    freqs_cos: torch.Tensor,
    freqs_sin: torch.Tensor,
    heads: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Unfused diffusers path: RMSNorm(q)/RMSNorm(k) over ``H*D``, head-split,
    then Wan interleaved RoPE -- exactly the ops diffusers runs, so it is
    numerically interchangeable with the fused fast path.  Returns ``(q, k)`` as
    ``[B, S, H, D]``.  Used whenever the fused kernel's envelope does not hold.
    """
    query = norm_q(query)
    key = norm_k(key)
    query = query.unflatten(2, (heads, -1))
    key = key.unflatten(2, (heads, -1))

    def apply_rotary_emb(hidden_states, fc, fs):
        x1, x2 = hidden_states.unflatten(-1, (-1, 2)).unbind(-1)
        cos = fc[..., 0::2]
        sin = fs[..., 1::2]
        out = torch.empty_like(hidden_states)
        out[..., 0::2] = x1 * cos - x2 * sin
        out[..., 1::2] = x1 * sin + x2 * cos
        return out.type_as(hidden_states)

    query = apply_rotary_emb(query, freqs_cos, freqs_sin)
    key = apply_rotary_emb(key, freqs_cos, freqs_sin)
    return query, key


def wan_flydsl_hadamard(x: torch.Tensor) -> torch.Tensor:
    """Apply normalized per-head Hadamard in a separate FlyDSL launch."""
    if not _HAS_FLYDSL:
        raise RuntimeError("pre-transport Hadamard requires FlyDSL")
    if (
        x.device.type != "cuda"
        or x.dtype != torch.bfloat16
        or x.dim() != 4
        or x.shape[0] != 1
        or not x.is_contiguous()
    ):
        raise ValueError("Wan Hadamard expects contiguous CUDA BF16 [1, S, H, D]")
    D = x.shape[-1]
    if D <= 0 or (D & (D - 1)) != 0:
        raise ValueError(f"Wan Hadamard requires power-of-two head dim, got {D}")
    return torch.ops.xfuser.wan_flydsl_hadamard(x)


def fused_qk_norm_rope(
    query: torch.Tensor,
    key: torch.Tensor,
    norm_q: torch.nn.Module,
    norm_k: torch.nn.Module,
    freqs_cos: torch.Tensor,
    freqs_sin: torch.Tensor,
    heads: int,
    *,
    single_role: bool = False,
    apply_hadamard: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return ``(q, k)`` normed, head-split and rotated as ``[B, S, H, D]``.

    ``query`` / ``key`` are the raw projection outputs of shape ``[B, S, H*D]``.
    Runs the fused FlyDSL kernel when the shape is in-envelope, otherwise falls
    back to the unfused diffusers reference (``_reference``) so the result is
    identical either way. ``single_role`` processes only ``query`` and returns it
    in both tuple positions; it is used by projection/A2A interleaving. Optional
    Hadamard is fused after the BF16 norm/RoPE boundary and before the final store.
    """
    def _ref():
        if apply_hadamard:
            raise RuntimeError(
                "pre-transport Hadamard requires the Wan FlyDSL preprocessing path"
            )
        return _reference(
            query, key, norm_q, norm_k, freqs_cos, freqs_sin, heads
        )

    if not _HAS_FLYDSL:
        return _ref()
    # Inference-only fast path: the fused op has no autograd formula, so any
    # grad-enabled call falls back to the reference implementation.
    if torch.is_grad_enabled():
        return _ref()
    if query.device.type != "cuda" or query.dim() != 3 or key.dim() != 3:
        return _ref()
    if query.shape != key.shape or query.dtype != key.dtype:
        return _ref()
    if query.dtype != torch.bfloat16:  # flydsl kernel is bf16-only
        return _ref()
    if query.shape[0] != 1:
        return _ref()
    if not isinstance(norm_q, torch.nn.RMSNorm) or not isinstance(
        norm_k, torch.nn.RMSNorm
    ):
        return _ref()
    wq = getattr(norm_q, "weight", None)
    wk = getattr(norm_k, "weight", None)
    if wq is None or wk is None or wq.dtype != query.dtype or wk.dtype != key.dtype:
        return _ref()
    if not wq.is_contiguous() or not wk.is_contiguous():
        return _ref()

    _, S, HD = query.shape
    if heads <= 0 or HD % heads:
        return _ref()
    D = HD // heads
    if D % 2:
        return _ref()
    if apply_hadamard and (D <= 0 or (D & (D - 1)) != 0):
        return _ref()
    if tuple(norm_q.normalized_shape) != (HD,) or tuple(
        norm_k.normalized_shape
    ) != (HD,):
        return _ref()
    eps_q, eps_k = norm_q.eps, norm_k.eps
    if eps_q is None or eps_k is None or eps_q != eps_k:
        return _ref()
    wave_size = get_device_wave_size(query)
    if wave_size is None or _pick_tiling(heads, D, wave_size) is None:
        return _ref()

    # freqs: [1, S, 1, D], last dim contiguous (diffusers WanRotaryPosEmbed).
    if freqs_cos.dim() != 4 or freqs_sin.shape != freqs_cos.shape:
        return _ref()
    if tuple(freqs_cos.shape) != (1, S, 1, D):
        return _ref()
    if freqs_cos.dtype != freqs_sin.dtype or freqs_cos.dtype not in (
        torch.float32,
        torch.bfloat16,
    ):
        return _ref()
    if query.stride(-1) != 1 or key.stride(-1) != 1:
        return _ref()
    # Buffer-load offsets are i32 elements.  Inputs may be strided (row stride
    # up to 3*H*D for a fused-QKV chunk), so bound the largest INPUT offset
    # ((S-1)*row_stride + H*D) as well as the contiguous OUTPUT offset (2*S*H*D)
    # inside 2^31.
    q_rs = query.stride(1)
    k_rs = key.stride(1)
    max_in = (S - 1) * max(q_rs, k_rs) + HD
    if max_in >= (1 << 31) or 2 * S * HD >= (1 << 31):
        return _ref()

    # Drop the leading batch-1 dim to a [S, H*D] view WITHOUT forcing contiguity:
    # the real model feeds q/k as chunks of a fused-QKV projection (row stride
    # 3*H*D), and a .contiguous() here copied ~110MB per call.  The kernel reads
    # the true row stride instead.  ``.reshape`` on a [1, S, H*D] tensor only
    # removes the unit dim, so it stays a view for any row stride.
    q2 = query.reshape(S, HD)
    k2 = key.reshape(S, HD)
    cos2 = freqs_cos.reshape(S, D).contiguous()
    sin2 = freqs_sin.reshape(S, D).contiguous()
    wqc = wq.contiguous()
    wkc = wk.contiguous()

    out = torch.ops.xfuser.wan_flydsl_qk_norm_rope(
        q2,
        k2,
        wqc,
        wkc,
        cos2,
        sin2,
        heads,
        float(eps_q),
        single_role,
        apply_hadamard,
    )
    query_out = out[0].view(1, S, heads, D)
    if single_role:
        return query_out, query_out
    return query_out, out[1].view(1, S, heads, D)
