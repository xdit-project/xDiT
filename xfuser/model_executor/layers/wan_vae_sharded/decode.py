"""Row-sharded decode for diffusers' AutoencoderKLWan (Wan2.1 / Wan2.2 A14B VAE).

Every rank of the VAE group decodes a band of latent rows with the decoder's own module structure (diffusers' frame
loop, feature caches and block order are unchanged) but with the hot modules re-bound on this VAE instance:

- residual blocks: Wan RMS norm + SiLU in one pass straight into the next conv's input buffer, convs over
  [T + 2, Hb + 2, W, C] buffers (two history frames, one halo row above and below), residual added in place;
- causal 3x3x3 convs (conv_in, conv_out): the same haloed buffers;
- spatial 2x upsample + 3x3 conv: one kernel reading the haloed source rows;
- mid-block attention: queries are the band's pixels, keys/values are all-gathered (latent resolution);
- RMS norms outside residual blocks: one Triton pass.

Before every 3x3 conv a rank exchanges one row with each neighbour (`HaloNccl`: NCCL P2P; `HaloSdma`: xDiT's SDMA
transport over one 2-rank group per neighbouring pair). The producer of a conv buffer writes the two edge rows first,
starts their exchange, writes the interior rows while the rows are in flight, then waits: the exchange overlaps the
interior pass. Activations stay channels-last end to end; the decoded bands are all-gathered into the full video on
every rank, as the stock decode returns it.

Calls the sharded path does not cover (non-bf16 latents, fewer latent rows than ranks) run diffusers' own decode
(every rank decodes the whole video); a batch is decoded one sample at a time.

`install(vae, process_group)` binds all of this to one VAE instance; nothing is patched class-wide.
"""

import types
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist

from xfuser.logger import init_logger
from xfuser.model_executor.layers.wan_vae_sharded import kernels as K
from xfuser.model_executor.layers.wan_vae_sharded.layout import bands

logger = init_logger(__name__)


# ---------------------------------------------------------------------------------------------------------------
# halo exchange: begin(buf, f0) after the edge rows of frames f0.. are written, finish(token) before the conv
# ---------------------------------------------------------------------------------------------------------------


class HaloNccl:
    """Neighbour rows over NCCL point-to-point (any backend that supports batch_isend_irecv)."""

    def __init__(self, group, rank: int, world: int):
        self.group, self.rank, self.world = group, rank, world
        self.global_rank = lambda i: dist.get_global_rank(group, i)

    def begin(self, buf: torch.Tensor, f0: int, depth: int = 1):
        r, n = self.rank, self.world
        F, H2, W, C = buf.shape
        nF = F - f0
        up = torch.empty(nF * depth * W * C, device=buf.device, dtype=buf.dtype) if r > 0 else None
        dn = torch.empty(nF * depth * W * C, device=buf.device, dtype=buf.dtype) if r < n - 1 else None
        K.halo_pack(buf, f0, up, dn, depth)
        ops, recv_up, recv_dn = [], None, None
        if up is not None:
            recv_up = torch.empty_like(up)
            ops += [
                dist.P2POp(dist.isend, up, self.global_rank(r - 1), self.group),
                dist.P2POp(dist.irecv, recv_up, self.global_rank(r - 1), self.group),
            ]
        if dn is not None:
            recv_dn = torch.empty_like(dn)
            ops += [
                dist.P2POp(dist.isend, dn, self.global_rank(r + 1), self.group),
                dist.P2POp(dist.irecv, recv_dn, self.global_rank(r + 1), self.group),
            ]
        reqs = dist.batch_isend_irecv(ops) if ops else []
        return (buf, f0, reqs, recv_up, recv_dn, up, dn, depth)

    def finish(self, token) -> None:
        buf, f0, reqs, recv_up, recv_dn, _, _, depth = token
        for req in reqs:
            req.wait()
        K.halo_unpack(buf, f0, recv_up, recv_dn, depth)


class HaloSdma:
    """Neighbour rows on the SDMA copy engines (xfuser.core.distributed.sdma.transport), one 2-rank process group
    per neighbouring pair. A neighbour can run one exchange ahead, and a channel's signal reads 1 until its waiter
    resets it, so inbox slots and channels are kept per exchange parity. Slots grow with the largest halo message
    (both ranks of a pair exchange the same shape at the same call, so the growth is collective)."""

    def __init__(self, group, rank: int, world: int):
        from xfuser.core.distributed.sdma import transport

        self.rank, self.world = rank, world
        ranks = dist.get_process_group_ranks(group)
        assert world == dist.get_world_size(), "SDMA halos need the VAE group to be the whole world"
        # pair i = ranks (i, i + 1) of the VAE group. new_group is collective over the world and its group names
        # come from a per-process counter, so every rank creates every pair, in the same order.
        pgs = [dist.new_group([ranks[i], ranks[i + 1]]) for i in range(world - 1)]
        self.pairs: Dict[int, Tuple[dist.ProcessGroup, object]] = {}
        for i in (rank - 1, rank):  # my upper pair first: a chain, so the pairwise setups never wait in a cycle
            if 0 <= i < world - 1:
                self.pairs[i] = (pgs[i], transport.group_for(pgs[i]))
        self.seq = 0
        self.slot = 0
        self.send: Optional[torch.Tensor] = None
        self.device = torch.device("cuda", torch.cuda.current_device())

    def _inbox(self, g):
        return g.buffer("wan_vae_halo_inbox", 2 * self.slot)

    def _grow(self, nb: int) -> None:
        if nb <= self.slot:
            return
        self.slot = 1 << (nb - 1).bit_length()
        # our last outbound copies may still be reading the old send buffer (a neighbour's signal says its data
        # landed here, not that ours landed there): let them finish before the allocator can reuse that memory
        torch.cuda.synchronize(self.device)
        self.send = torch.empty(4 * self.slot, dtype=torch.uint8, device=self.device)
        # Inbox slots sit at parity * slot, so the new layout overlaps the old one: a neighbour must not write in
        # the new layout before this rank has unpacked everything it sent in the old one (it may run one exchange
        # ahead). Device work is drained above; a barrier per pair (in pair order: a chain, no cycle) holds both
        # members until each is drained. The buffer itself may be reused (another install on the same ranks).
        for i in sorted(self.pairs):  # collective per pair; same order on both members
            pg, g = self.pairs[i]
            self._inbox(g)
            dist.barrier(group=pg)

    def begin(self, buf: torch.Tensor, f0: int, depth: int = 1):
        r = self.rank
        F, H2, W, C = buf.shape
        nb = (F - f0) * depth * W * C * 2
        legs = [(self.pairs[i], side) for i, side in ((r - 1, "up"), (r, "dn")) if i in self.pairs]
        if not legs:
            return (buf, f0, [], nb, 0, depth)
        self._grow(nb)
        self.seq += 1
        par = self.seq % 2
        sends = {}
        for _, side in legs:
            j = 0 if side == "up" else 1
            sends[side] = self.send[(par * 2 + j) * self.slot : (par * 2 + j) * self.slot + nb].view(torch.bfloat16)
        K.halo_pack(buf, f0, sends.get("up"), sends.get("dn"), depth)
        stream = torch.cuda.current_stream()
        calls = []
        for (_, g), side in legs:
            inbox = self._inbox(g)
            slot = self.slot
            route = g.route(
                ("wan_vae_halo", par, nb, slot),
                lambda inbox=inbox, slot=slot: [[(0, inbox, par * slot, nb)]],
                par,
            )
            call = g.begin()
            call.send(route, sends[side], stream)
            calls.append((call, g, side))
        return (buf, f0, calls, nb, par, depth)

    def finish(self, token) -> None:
        buf, f0, calls, nb, par, depth = token
        stream = torch.cuda.current_stream()
        recv = {}
        for call, g, side in calls:
            call.wait(par, stream)
            call.end()
            recv[side] = self._inbox(g).local[par * self.slot : par * self.slot + nb].view(torch.bfloat16)
        K.halo_unpack(buf, f0, recv.get("up"), recv.get("dn"), depth)


def _exchange(halo, buf, f0, depth=1):
    halo.finish(halo.begin(buf, f0, depth))


# ---------------------------------------------------------------------------------------------------------------
# per-decode context and re-bound module forwards
# ---------------------------------------------------------------------------------------------------------------


class _Ctx:
    def __init__(self, group, halo: str):
        self.group = group
        self.rank, self.world = dist.get_rank(group), dist.get_world_size(group)
        if halo == "auto":
            halo = "sdma" if _sdma_ok() and self.world == dist.get_world_size() else "nccl"
        self.halo_kind = halo
        self.halo = (HaloSdma if halo == "sdma" else HaloNccl)(group, self.rank, self.world)
        self.band_heights: List[int] = []
        self.bufs: Dict[tuple, list] = {}
        self.active = False  # bound forwards run the sharded path only inside a sharded decode

    def buffer(self, key, T, H2, W, C, device):
        """Alternating [T + 2, H2, W, C] buffers per (module, slot, shape): the previous one still holds the history."""
        k = (key, T, H2, W, C)
        st = self.bufs.setdefault(k, [None, None, 0])
        i = st[2]
        if st[i] is None:
            st[i] = torch.empty(T + 2, H2, W, C, device=device, dtype=torch.bfloat16)
        st[2] = 1 - i
        return st[i]


def _sdma_ok() -> bool:
    try:
        from xfuser.core.distributed.sdma import transport

        return torch.version.hip is not None and transport.available()
    except Exception:  # noqa: BLE001
        return False


def _fill_history(a, prev):
    """a[0:2] = the last two frames of the previous call's buffer (zeros at the start of the video)."""
    if prev is None:
        a[:2].zero_()
    else:
        a[:2].copy_(prev[-2:])


def _norm_exchange(ctx, x5, gamma, scale, a, H, d=1):
    """a[2:, d:d+H] = silu(rmsnorm(x5)) and its `d` halo rows above/below from the neighbours. Over SDMA the edge rows
    are written first and their exchange runs while the interior rows are written (with NCCL P2P that split costs
    more launches than the overlap gains: measured 0.219 -> 0.239 s)."""
    if H <= 2 * d or ctx.halo_kind != "sdma":
        K.norm_silu_into(x5, gamma, scale, a[2:, d : d + H])
        _exchange(ctx.halo, a, 2, d)
        return
    if d == 1:
        K.norm_silu_into(x5[:, :, :, :: H - 1], gamma, scale, a[2:, 1 : H + 1 : H - 1])  # rows 0 and H-1
    else:
        K.norm_silu_into(x5[:, :, :, :d], gamma, scale, a[2:, d : 2 * d])
        K.norm_silu_into(x5[:, :, :, H - d :], gamma, scale, a[2:, H : H + d])
    token = ctx.halo.begin(a, 2, d)
    K.norm_silu_into(x5[:, :, :, d : H - d], gamma, scale, a[2:, 2 * d : H])  # interior
    ctx.halo.finish(token)


def _copy_exchange(ctx, dst, src, f0, H):
    """dst[:, 1:H+1] = src (both [F, H, W, C]-shaped views), edge rows first, overlapping the exchange of frames f0..
    (SDMA halos only, as _norm_exchange)."""
    if H <= 2 or ctx.halo_kind != "sdma":
        dst[:, 1 : H + 1].copy_(src)
        _exchange(ctx.halo, dst, f0)
        return
    dst[:, 1 : H + 1 : H - 1].copy_(src[:, :: H - 1])
    token = ctx.halo.begin(dst, f0)
    dst[:, 2:H].copy_(src[:, 1 : H - 1])
    ctx.halo.finish(token)


def _depth2(ctx, H):
    """2-row halos need every rank's band at this resolution to be at least 2 rows (all ranks decide alike: the band
    table is shared); one rank needs no exchange at all."""
    if ctx.world == 1 or not ctx.band_heights:
        return False
    scale = H // ctx.band_heights[ctx.rank]
    return min(ctx.band_heights) * scale >= 2


def _resblock_forward(self, x, feat_cache=None, feat_idx=[0]):  # noqa: B006 - diffusers' signature
    ctx: _Ctx = self._xfuser_sharded_ctx
    if not ctx.active:
        return self._xfuser_orig_forward(x, feat_cache, feat_idx)
    assert feat_cache is not None and x.shape[0] == 1 and x.dtype == torch.bfloat16, "sharded Wan decode: bf16, batch 1"
    _, cin, T, H, W = x.shape
    cout = self.out_dim
    i1, i2 = feat_idx[0], feat_idx[0] + 1
    if _depth2(ctx, H):
        # one exchange per block: conv1's input carries 2 halo rows, so conv1 also yields the row above and below
        # the band, which is conv2's halo; at the image edges that row is zero (conv2's zero padding)
        a1 = ctx.buffer((id(self), 1), T, H + 4, W, cin, x.device)
        _fill_history(a1, feat_cache[i1])
        _norm_exchange(ctx, x, self.norm1.gamma.view(-1), self.norm1.scale, a1, H, 2)
        y1 = K.conv3d_halo(a1, self.conv1.weight, self.conv1.bias)  # [T, H + 2, W, cout]
        a2 = ctx.buffer((id(self), 2), T, H + 2, W, cout, x.device)
        _fill_history(a2, feat_cache[i2])
        K.norm_silu_into(y1.permute(3, 0, 1, 2).unsqueeze(0), self.norm2.gamma.view(-1), self.norm2.scale, a2[2:])
        if ctx.rank == 0:
            a2[2:, 0].zero_()
        if ctx.rank == ctx.world - 1:
            a2[2:, H + 1].zero_()
    else:
        a1 = ctx.buffer((id(self), 1), T, H + 2, W, cin, x.device)
        _fill_history(a1, feat_cache[i1])
        _norm_exchange(ctx, x, self.norm1.gamma.view(-1), self.norm1.scale, a1, H)
        y1 = K.conv3d_halo(a1, self.conv1.weight, self.conv1.bias)
        a2 = ctx.buffer((id(self), 2), T, H + 2, W, cout, x.device)
        _fill_history(a2, feat_cache[i2])
        _norm_exchange(ctx, y1.permute(3, 0, 1, 2).unsqueeze(0), self.norm2.gamma.view(-1), self.norm2.scale, a2, H)
    feat_cache[i1], feat_cache[i2] = a1, a2  # this module's own history format: full haloed buffers
    feat_idx[0] += 2
    out = K.conv3d_halo(a2, self.conv2.weight, self.conv2.bias)
    xs = x[0].permute(1, 2, 3, 0)
    if isinstance(self.conv_shortcut, torch.nn.Identity):
        out.add_(xs)
    else:
        xc = xs if xs.is_contiguous() else xs.contiguous()
        ws = self.conv_shortcut.weight.view(cout, cin)
        out.view(-1, cout).addmm_(xc.reshape(-1, cin), ws.t()).add_(self.conv_shortcut.bias)
    return out.permute(3, 0, 1, 2).unsqueeze(0)


def _causal_conv_forward(self, x, cache_x=None):
    ctx: _Ctx = self._xfuser_sharded_ctx
    if not ctx.active or self.kernel_size != (3, 3, 3):
        assert not ctx.active or self.kernel_size[1:] == (1, 1), f"spatial conv {self.kernel_size} without halos"
        return self._xfuser_orig_forward(x, cache_x)
    _, C, T, H, W = x.shape
    a = torch.empty(T + 2, H + 2, W, C, device=x.device, dtype=x.dtype)
    nc = 0 if cache_x is None else min(cache_x.shape[2], 2)
    if nc < 2:
        a[: 2 - nc].zero_()
    src = x[0].permute(1, 2, 3, 0)
    if nc:
        src = torch.cat([cache_x[0, :, -nc:].permute(1, 2, 3, 0), src], dim=0)
    _copy_exchange(ctx, a[2 - nc :], src, 0, H)
    return K.conv3d_halo(a, self.weight, self.bias).permute(3, 0, 1, 2).unsqueeze(0)


def _rms_norm_forward(self, x):
    ctx: _Ctx = self._xfuser_sharded_ctx
    if not (
        ctx.active
        and self.channel_first
        and x.dim() == 5
        and x.shape[0] == 1
        and x.dtype == torch.bfloat16
        and not torch.is_tensor(self.bias)
    ):
        return self._xfuser_orig_forward(x)
    _, C, T, H, W = x.shape
    y = torch.empty(T, H, W, C, device=x.device, dtype=x.dtype)
    K.norm_silu_into(x, self.gamma.view(-1), self.scale, y, silu=False)
    return y.permute(3, 0, 1, 2).unsqueeze(0)


def _attention_forward(self, x):
    """Wan mid-block attention with the band's pixels as queries and the whole image's keys/values."""
    ctx: _Ctx = self._xfuser_sharded_ctx
    if not ctx.active:
        return self._xfuser_orig_forward(x)
    identity = x
    b, C, t, Hb, W = x.shape
    heights = ctx.band_heights
    assert heights[ctx.rank] == Hb, (heights, Hb)
    mh = max(heights)
    xf = x.permute(0, 2, 1, 3, 4).reshape(b * t, C, Hb, W)
    qkv = self.to_qkv(self.norm(xf)).reshape(b * t, 1, C * 3, -1).permute(0, 1, 3, 2).contiguous()
    q, kv = qkv[..., :C], qkv[..., C:]
    padded = torch.zeros(b * t, 1, mh * W, 2 * C, device=x.device, dtype=x.dtype)
    padded[:, :, : Hb * W] = kv
    if ctx.world == 1:
        KV = kv
    else:
        parts = [torch.empty_like(padded) for _ in range(ctx.world)]
        dist.all_gather(parts, padded, group=ctx.group)
        KV = torch.cat([p[:, :, : h * W] for p, h in zip(parts, heights)], dim=2)
    y = K.attention(q.contiguous(), KV[..., :C].contiguous(), KV[..., C:].contiguous())
    y = self.proj(y.squeeze(1).permute(0, 2, 1).reshape(b * t, C, Hb, W))
    return y.view(b, t, C, Hb, W).permute(0, 2, 1, 3, 4) + identity


class _UpConv(torch.nn.Module):
    """Replaces Sequential(WanUpsample(2x nearest-exact), Conv2d(3x3, pad 1)) inside a WanResample."""

    def __init__(self, seq, ctx):
        super().__init__()
        self.seq = seq
        self.ctx = ctx

    def forward(self, x):  # [b * t, c, h, w] band rows
        if not self.ctx.active:
            return self.seq(x)
        N, C, H, W = x.shape
        a = torch.empty(N, H + 2, W, C, device=x.device, dtype=x.dtype)
        _copy_exchange(self.ctx, a, x.permute(0, 2, 3, 1), 0, H)
        conv = self.seq[1]
        return K.up_conv2d_halo(a, conv.weight, conv.bias).permute(0, 3, 1, 2)


def _bind(module, fn, ctx):
    if not hasattr(module, "_xfuser_orig_forward"):
        module._xfuser_orig_forward = module.forward
    module._xfuser_sharded_ctx = ctx
    module.forward = types.MethodType(fn, module)


def _decode_band(self, ctx, z):
    """One sample: this rank's band through the re-bound decoder, then the bands all-gathered."""
    n, r = ctx.world, ctx.rank
    bs = bands(z.shape[3], n)
    ctx.band_heights = [h for _, h in bs]
    r0, hb = bs[r]
    ctx.active = True
    try:
        out = self._xfuser_orig_decode(z[:, :, :, r0 : r0 + hb].contiguous(), return_dict=False)[0]
    finally:
        ctx.active = False
    if n == 1:
        return out
    scale = out.shape[3] // hb
    mh = max(ctx.band_heights) * scale
    padded = torch.zeros(*out.shape[:3], mh, out.shape[4], device=out.device, dtype=out.dtype)
    padded[:, :, :, : out.shape[3]] = out
    parts = [torch.empty_like(padded) for _ in range(n)]
    dist.all_gather(parts, padded, group=ctx.group)
    return torch.cat([p[:, :, :, : h * scale] for p, (_, h) in zip(parts, bs)], dim=3)


def _sharded_decode(self, z: torch.Tensor, return_dict: bool = True):
    """AutoencoderKLWan._decode on this rank's band of latent rows; returns the full video on every rank."""
    from diffusers.models.autoencoders.vae import DecoderOutput

    ctx: _Ctx = self._xfuser_sharded_ctx
    if z.dtype != torch.bfloat16 or z.shape[3] < ctx.world:
        return self._xfuser_orig_decode(z, return_dict=return_dict)  # every rank decodes the whole video
    video = torch.cat([_decode_band(self, ctx, z[i : i + 1]) for i in range(z.shape[0])], dim=0)
    return DecoderOutput(sample=video) if return_dict else (video,)


def supports(vae) -> bool:
    return (
        type(vae).__name__ == "AutoencoderKLWan"
        and not getattr(vae.config, "patch_size", None)
        and not getattr(vae, "use_tiling", False)
    )


def install(vae, process_group: Optional[dist.ProcessGroup] = None, halo: str = "auto") -> str:
    """Make `vae.decode` a row-sharded decode over `process_group` (default: the world). Collective. Only the decoder
    is touched (an encoder can be parallelized separately). Returns the halo transport used ("nccl" or "sdma")."""
    if not supports(vae):
        raise ValueError(
            f"the sharded Wan VAE decode supports AutoencoderKLWan without patching/tiling, not {type(vae).__name__}"
        )
    group = process_group if process_group is not None else dist.group.WORLD
    ctx = _Ctx(group, halo)
    for m in vae.decoder.modules():
        name = type(m).__name__
        if name == "WanResidualBlock":
            _bind(m, _resblock_forward, ctx)
        elif name == "WanCausalConv3d":
            _bind(m, _causal_conv_forward, ctx)
        elif name == "WanRMS_norm":
            _bind(m, _rms_norm_forward, ctx)
        elif name == "WanAttentionBlock":
            _bind(m, _attention_forward, ctx)
        elif name == "WanResample" and m.mode in ("upsample2d", "upsample3d"):
            up, conv = m.resample[0], m.resample[1]
            assert getattr(up, "mode", None) == "nearest-exact" and conv.kernel_size == (3, 3)
            m.resample = _UpConv(m.resample, ctx)
    vae._xfuser_sharded_ctx = ctx
    vae._xfuser_orig_decode = vae._decode
    vae._decode = types.MethodType(_sharded_decode, vae)
    logger.info(f"sharded Wan VAE decode on {ctx.world} ranks, halos over {ctx.halo_kind}")
    return ctx.halo_kind
