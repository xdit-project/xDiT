"""Kernels of the row-sharded Wan VAE decode, all on channels-last (NDHWC) activations.

Conv input buffers are [T + 2, Hb + 2, W, C]: two history frames in front (the causal cache) and one halo row
above and below (the neighbouring ranks' rows, zeros at the image edge), so every 3x3(x3) conv runs without
padding in H and with zero padding in W.

- `conv3d_halo`: 3x3x3 conv over such a buffer. AITER's FlyDSL implicit-GEMM conv when available (tile hints for
  the Wan widths when it supports them), else a Triton implicit-GEMM kernel.
- `norm_silu_into`: Wan RMS norm (+ SiLU) of a pixel-major tensor written into a (strided) buffer view.
- `up_conv2d_halo`: nearest-exact 2x upsample + 3x3 conv read straight from a haloed source (no 4x tensor).
- `halo_pack` / `halo_unpack`: edge rows of the new frames to/from contiguous halo messages.
- `attention`: single-head attention at head dim 384 as two GEMMs and an fp32 softmax (torch SDPA takes a slow
  path at d = 384 on MI355X: ~3.5 ms a call for any query length).
"""

import importlib
import inspect

import torch
import triton
import triton.language as tl

# (tile_m, tile_n, wave_m, wave_n), wgm, block_rows for AITER's FlyDSL conv at the Wan decoder widths, swept on
# MI355X (gfx950) at the 720p shapes; used only if the installed conv accepts block_rows.
FLYDSL_TILES = {96: ((256, 96, 4, 1), 1, 4), 192: ((128, 192, 2, 2), 1, 4), 384: ((128, 192, 2, 2), 8, 4)}

# Per-call choice by output pixels M = T * H * W (a rank's band is much smaller than the full frame, and a tile tuned
# for 720p leaves most CUs idle there). Static (not timed per process) so every run picks the same kernel and the
# output stays bitwise reproducible. "triton" = our Triton conv; None = the FlyDSL conv's own heuristic tile.
# Measured on MI355X with per-rank band shapes of a 720p Wan2.2 decode, splitk=1 everywhere:
#   384 ch: M 1920 -> heuristic 75 us (vs 136 with the 720p tile); M 15360 -> (128,128,2,2) br2 127 us (vs 167)
#   192 ch: M 15360 -> triton 48 us (vs 61); M 122880 -> (256,96,4,1) br4 206 us (vs 518 heuristic)
#   96 ch:  (256,96,4,1) br4 best at every size measured.
_SMALL_TILES = {
    384: [(4096, None), (32768, ((128, 128, 2, 2), 1, 2))],
    192: [(32768, "triton"), (262144, ((256, 96, 4, 1), 1, 4))],
}


def _pick_tile(cout, m):
    for limit, choice in _SMALL_TILES.get(cout, []):
        if m <= limit:
            return choice
    return FLYDSL_TILES.get(cout)


_conv_impl = None


def _resolve_conv():
    """(function, accepts_tile_hints) for the FlyDSL conv, or (None, False). XDIT_WAN_VAE_CONV=module:function
    overrides the AITER entry point; XDIT_WAN_VAE_CONV=triton forces the Triton kernel."""
    global _conv_impl
    if _conv_impl is not None:
        return _conv_impl
    import xfuser.envs as envs

    spec = envs.XDIT_WAN_VAE_CONV
    fn = None
    if spec == "triton":
        fn = None
    elif spec:
        mod, name = spec.split(":")
        fn = getattr(importlib.import_module(mod), name)
    else:
        try:
            from aiter.ops.flydsl import flydsl_conv_implicit as fn
        except Exception:  # noqa: BLE001 - AITER is optional
            fn = None
    hints = False
    if fn is not None:
        try:
            impl = importlib.import_module(fn.__module__)._conv3d_impl
            hints = "block_rows" in inspect.signature(impl).parameters
        except Exception:  # noqa: BLE001
            hints = False
    _conv_impl = (fn, hints)
    return _conv_impl


# ---------------------------------------------------------------------------------------------------------------
# 3x3x3 causal conv over a haloed buffer
# ---------------------------------------------------------------------------------------------------------------


def _conv_configs():
    return [
        triton.Config({"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk, "TH": th}, num_warps=nw, num_stages=2)
        for bm, th, bn, bk, nw in [
            (256, 8, 128, 32, 8),
            (128, 8, 128, 64, 4),
            (128, 8, 64, 64, 4),
            (256, 16, 128, 64, 8),
            (128, 4, 128, 64, 4),
            (256, 4, 64, 64, 8),
        ]
    ]


@triton.autotune(configs=_conv_configs(), key=["T", "H", "W", "CIN", "COUT"])
@triton.jit
def _conv3d_halo_kernel(
    X,
    Wt,
    B,
    OUT,
    T,
    H,
    W,
    CIN: tl.constexpr,
    COUT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    TH: tl.constexpr,
):
    """OUT[t, h, w, :] = sum over (kt, kh, kw, cin) X[t + kt, h + kh, w + kw - 1, cin] * Wt[(kt, kh, kw, cin), :];
    X is [T + 2, H + 2, W, CIN] (history frames, halo rows), K order (kt, kh, kw, cin): the (kw, cin) span is
    contiguous in X; only W is zero padded (masks)."""
    KTOT: tl.constexpr = 27 * CIN
    SPAN: tl.constexpr = 3 * CIN
    TW: tl.constexpr = BLOCK_M // TH
    H2 = H + 2
    pid = tl.program_id(0)
    n_n: tl.constexpr = (COUT + BLOCK_N - 1) // BLOCK_N
    n_hb = tl.cdiv(H, TH)
    n_wb = tl.cdiv(W, TW)
    pm = pid // n_n
    pn = pid % n_n
    tt = pm // (n_hb * n_wb)
    i = tl.arange(0, BLOCK_M)
    h = ((pm // n_wb) % n_hb) * TH + i // TW
    w = (pm % n_wb) * TW + i % TW
    m_ok = (h < H) & (w < W)
    offs_n = pn * BLOCK_N + tl.arange(0, BLOCK_N)
    n_ok = offs_n < COUT
    base = (((tt + i * 0).to(tl.int64) * H2 + h) * W + (w - 1)) * CIN
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, KTOT, BLOCK_K):
        k = k0 + tl.arange(0, BLOCK_K)
        k_ok = k < KTOT
        kt = k // (9 * CIN)
        kh = (k // SPAN) % 3
        rem = k % SPAN
        ww = w[:, None] + (rem // CIN)[None, :] - 1
        ok = m_ok[:, None] & k_ok[None, :] & (ww >= 0) & (ww < W)
        a = tl.load(X + base[:, None] + ((kt * H2 + kh) * W * CIN + rem)[None, :], mask=ok, other=0.0)
        b = tl.load(Wt + k[:, None] * COUT + offs_n[None, :], mask=k_ok[:, None] & n_ok[None, :], other=0.0)
        acc = tl.dot(a, b, acc)
    if HAS_BIAS:
        acc += tl.load(B + offs_n, mask=n_ok, other=0.0).to(tl.float32)[None, :]
    o = (((tt + i * 0).to(tl.int64) * H + h) * W + w)[:, None] * COUT + offs_n[None, :]
    tl.store(OUT + o, acc.to(tl.bfloat16), mask=m_ok[:, None] & n_ok[None, :])


_w2_cache = {}


def _w2(weight):
    """[Cout, Cin, 3, 3, 3] -> [27 * Cin, Cout], K order (kt, kh, kw, cin); cached per weight storage."""
    key = (weight.data_ptr(), tuple(weight.shape))
    w2 = _w2_cache.get(key)
    if w2 is None:
        w2 = _w2_cache[key] = weight.detach().permute(2, 3, 4, 1, 0).reshape(-1, weight.shape[0]).contiguous()
    return w2


def conv3d_halo(a, weight, bias):
    """a [T + 2, Hb + 2, W, Cin] bf16 -> [T, Hb, W, Cout] bf16 (contiguous NDHWC)."""
    fn, hints = _resolve_conv()
    cout = weight.shape[0]
    T2, H2, W, cin = a.shape
    T, H = T2 - 2, H2 - 2
    choice = _pick_tile(cout, T * H * W)
    if fn is not None and choice != "triton":
        # splitk=1: left unset, the conv splits K over workgroups with atomic adds for small outputs (our small
        # per-rank bands, small VAEs), and the result then depends on the atomics' order run to run
        kw = dict(stride=1, padding=(0, 0, 1), input_layout="NDHWC", output_layout="NDHWC", splitk=1)
        if hints and choice is not None:
            kw.update(tile=choice[0], wgm=choice[1], block_rows=choice[2])
        return fn(a.unsqueeze(0), weight, bias, **kw)[0]
    out = torch.empty(T, H, W, cout, device=a.device, dtype=a.dtype)

    def grid(m):
        return (
            T * triton.cdiv(H, m["TH"]) * triton.cdiv(W, m["BLOCK_M"] // m["TH"]) * triton.cdiv(cout, m["BLOCK_N"]),
        )

    _conv3d_halo_kernel[grid](
        a, _w2(weight), bias if bias is not None else a, out, T, H, W, cin, cout, bias is not None
    )
    return out


# ---------------------------------------------------------------------------------------------------------------
# 2x nearest-exact upsample + 3x3 conv from a haloed source
# ---------------------------------------------------------------------------------------------------------------


@triton.autotune(configs=_conv_configs(), key=["N", "H", "W", "CIN", "COUT"])
@triton.jit
def _up_conv2d_halo_kernel(
    X,
    Wt,
    B,
    OUT,
    N,
    H,
    W,
    CIN: tl.constexpr,
    COUT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    TH: tl.constexpr,
):
    """OUT [N, H, W, COUT] = conv3x3(upsample2x(X)) for the source rows of X [N, H/2 + 2, W/2, CIN] (one halo row
    above and below). Upsampled row hh in [-1, H] reads source row floor(hh / 2) + 1 of X."""
    KTOT: tl.constexpr = 9 * CIN
    SPAN: tl.constexpr = 3 * CIN
    TW: tl.constexpr = BLOCK_M // TH
    HS2 = H // 2 + 2
    WS = W // 2
    pid = tl.program_id(0)
    n_n: tl.constexpr = (COUT + BLOCK_N - 1) // BLOCK_N
    n_hb = tl.cdiv(H, TH)
    n_wb = tl.cdiv(W, TW)
    pm = pid // n_n
    pn = pid % n_n
    nn = pm // (n_hb * n_wb)
    i = tl.arange(0, BLOCK_M)
    h = ((pm // n_wb) % n_hb) * TH + i // TW
    w = (pm % n_wb) * TW + i % TW
    m_ok = (h < H) & (w < W)
    offs_n = pn * BLOCK_N + tl.arange(0, BLOCK_N)
    n_ok = offs_n < COUT
    img = (nn + i * 0).to(tl.int64) * HS2
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, KTOT, BLOCK_K):
        k = k0 + tl.arange(0, BLOCK_K)
        k_ok = k < KTOT
        kh = k // SPAN
        rem = k % SPAN
        hh = h[:, None] + kh[None, :] - 1
        ww = w[:, None] + (rem // CIN)[None, :] - 1
        ok = m_ok[:, None] & k_ok[None, :] & (ww >= 0) & (ww < W)
        off = ((img[:, None] + (hh + 2) // 2) * WS + ww // 2) * CIN + (rem % CIN)[None, :]
        a = tl.load(X + off, mask=ok, other=0.0)
        b = tl.load(Wt + k[:, None] * COUT + offs_n[None, :], mask=k_ok[:, None] & n_ok[None, :], other=0.0)
        acc = tl.dot(a, b, acc)
    if HAS_BIAS:
        acc += tl.load(B + offs_n, mask=n_ok, other=0.0).to(tl.float32)[None, :]
    o = (((nn + i * 0).to(tl.int64) * H + h) * W + w)[:, None] * COUT + offs_n[None, :]
    tl.store(OUT + o, acc.to(tl.bfloat16), mask=m_ok[:, None] & n_ok[None, :])


def up_conv2d_halo(x, weight, bias):
    """x [N, Hs + 2, Ws, Cin] (halo rows), weight [Cout, Cin, 3, 3] -> [N, 2 Hs, 2 Ws, Cout]."""
    N, HS2, WS, cin = x.shape
    H, W = 2 * (HS2 - 2), 2 * WS
    cout = weight.shape[0]
    key = (weight.data_ptr(), tuple(weight.shape))
    w2 = _w2_cache.get(key)
    if w2 is None:
        w2 = _w2_cache[key] = weight.detach().permute(2, 3, 1, 0).reshape(-1, cout).contiguous()
    out = torch.empty(N, H, W, cout, device=x.device, dtype=x.dtype)

    def grid(m):
        return (
            N * triton.cdiv(H, m["TH"]) * triton.cdiv(W, m["BLOCK_M"] // m["TH"]) * triton.cdiv(cout, m["BLOCK_N"]),
        )

    _up_conv2d_halo_kernel[grid](x, w2, bias if bias is not None else x, out, N, H, W, cin, cout, bias is not None)
    return out


# ---------------------------------------------------------------------------------------------------------------
# RMS norm (+ SiLU), halo pack / unpack
# ---------------------------------------------------------------------------------------------------------------


@triton.jit
def _norm_silu_kernel(
    X,
    OUT,
    GAMMA,
    P,
    HW,
    W,
    sxt,
    sxh,
    sxw,
    sxc,
    sot,
    soh,
    sow,
    scale,
    C: tl.constexpr,
    BC: tl.constexpr,
    BP: tl.constexpr,
    SILU: tl.constexpr,
):
    p = tl.program_id(0) * BP + tl.arange(0, BP)
    c = tl.arange(0, BC)
    pm = p < P
    cm = c < C
    t = p // HW
    hw = p % HW
    h = hw // W
    w = hw % W
    m = pm[:, None] & cm[None, :]
    x = tl.load(X + (t.to(tl.int64) * sxt + h * sxh + w * sxw)[:, None] + (c * sxc)[None, :], mask=m, other=0.0)
    x = x.to(tl.float32)
    nrm = tl.sqrt(tl.sum(x * x, axis=1))
    y = x / tl.maximum(nrm, 1e-12)[:, None] * scale * tl.load(GAMMA + c, mask=cm, other=0.0).to(tl.float32)[None, :]
    if SILU:
        y = y / (1.0 + tl.exp(-y))
    tl.store(OUT + (t.to(tl.int64) * sot + h * soh + w * sow)[:, None] + c[None, :], y.to(tl.bfloat16), mask=m)


def norm_silu_into(x5, gamma, scale, out, silu=True):
    """Wan RMS norm (F.normalize over channels * scale * gamma), then SiLU, of x5 [1, C, T, H, W] (any strides)
    into out [T, H, W, C] (unit channel stride, may be a strided view)."""
    _, C, T, H, W = x5.shape
    st, so = x5.stride(), out.stride()
    assert so[3] == 1
    BC = triton.next_power_of_2(C)
    BP = max(1, 8192 // BC)
    P = T * H * W
    _norm_silu_kernel[(triton.cdiv(P, BP),)](
        x5,
        out,
        gamma,
        P,
        H * W,
        W,
        st[2],
        st[3],
        st[4],
        st[1],
        so[0],
        so[1],
        so[2],
        float(scale),
        C,
        BC,
        BP,
        silu,
        num_warps=8,
    )
    return out


@triton.jit
def _halo_pack_kernel(
    BUF,
    UP,
    DN,
    f0,
    frame,
    row,
    WC,
    Hb,
    D: tl.constexpr,
    HAS_UP: tl.constexpr,
    HAS_DN: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """up[f, j] = buf[f0 + f, D + j] (first D interior rows), dn[f, j] = buf[f0 + f, Hb + j] (last D)."""
    fj = tl.program_id(1)
    f = fj // D
    j = fj % D
    o = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = o < WC
    base = (f0 + f).to(tl.int64) * frame
    if HAS_UP:
        tl.store(UP + fj.to(tl.int64) * WC + o, tl.load(BUF + base + (D + j) * row + o, mask=m), mask=m)
    if HAS_DN:
        tl.store(DN + fj.to(tl.int64) * WC + o, tl.load(BUF + base + (Hb + j) * row + o, mask=m), mask=m)


@triton.jit
def _halo_unpack_kernel(
    BUF,
    UP,
    DN,
    f0,
    frame,
    row,
    WC,
    Hb,
    D: tl.constexpr,
    HAS_UP: tl.constexpr,
    HAS_DN: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """buf[f0 + f, j] = up[f, j] (zeros at the image top), buf[f0 + f, D + Hb + j] = dn[f, j] (zeros at the bottom)."""
    fj = tl.program_id(1)
    f = fj // D
    j = fj % D
    o = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = o < WC
    base = (f0 + f).to(tl.int64) * frame
    if HAS_UP:
        v = tl.load(UP + fj.to(tl.int64) * WC + o, mask=m)
    else:
        v = tl.zeros((BLOCK,), dtype=tl.bfloat16)
    tl.store(BUF + base + j * row + o, v, mask=m)
    if HAS_DN:
        v2 = tl.load(DN + fj.to(tl.int64) * WC + o, mask=m)
    else:
        v2 = tl.zeros((BLOCK,), dtype=tl.bfloat16)
    tl.store(BUF + base + (D + Hb + j) * row + o, v2, mask=m)


def halo_pack(buf, f0, up, dn, depth=1):
    """buf [F, Hb + 2 * depth, W, C]: up = the first `depth` interior rows of frames f0.., dn = the last `depth`
    (flat bf16 views of (F - f0) * depth * W * C elements; None to skip a side)."""
    F, H2, W, C = buf.shape
    WC, nF = W * C, F - f0
    grid = (triton.cdiv(WC, 4096), nF * depth)
    _halo_pack_kernel[grid](
        buf,
        up if up is not None else buf,
        dn if dn is not None else buf,
        f0,
        H2 * WC,
        WC,
        WC,
        H2 - 2 * depth,
        depth,
        up is not None,
        dn is not None,
        BLOCK=4096,
    )


def halo_unpack(buf, f0, up, dn, depth=1):
    """The `depth` halo rows above / below frames f0.. of buf from up / dn (zeros where None: the image edge)."""
    F, H2, W, C = buf.shape
    WC, nF = W * C, F - f0
    grid = (triton.cdiv(WC, 4096), nF * depth)
    _halo_unpack_kernel[grid](
        buf,
        up if up is not None else buf,
        dn if dn is not None else buf,
        f0,
        H2 * WC,
        WC,
        WC,
        H2 - 2 * depth,
        depth,
        up is not None,
        dn is not None,
        BLOCK=4096,
    )


def attention(q, k, v):
    """softmax(q k^T / sqrt(d)) v per batch element; q [B, 1, Lq, D], k/v [B, 1, Lk, D] bf16, fp32 softmax."""
    B, _, _, D = q.shape
    out = torch.empty_like(q)
    for b in range(B):
        s = torch.mm(q[b, 0], k[b, 0].t(), out_dtype=torch.float32)
        s.mul_(D**-0.5)
        torch.mm(torch.softmax(s, dim=-1).to(q.dtype), v[b, 0], out=out[b, 0])
    return out
