# Licensed under the TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT

from dataclasses import dataclass
import math
import torch
from einops import rearrange

# ── Dataclasses ──────────────────────────────────────────────

@dataclass
class SSTAState:
    canvas_thw: tuple[int, int, int]
    tile_thw: tuple[int, int, int]
    text_len: int
    sp_pad_len: int
    need_pad: bool
    text_target_size: int
    need_pad_text: bool
    text_pad_size: int
    pad_t: int
    pad_h: int
    pad_w: int
    b: int
    hd: int
    t: int
    h: int
    w: int
    d: int

@dataclass
class MaskConfig:
    image_q: torch.Tensor
    image_k: torch.Tensor
    text_q: torch.Tensor | None
    canvas_thw: tuple
    tile_thw: tuple
    kernel_thw: tuple
    text_block_num: int
    threshold: float
    lambda_: float | None
    text_valid_lens: torch.Tensor | None
    mask_share_within_head: bool
    adaptive_pool: tuple | None
    sampling_type: str
    topk: int
    b: int

# ── GPU-resident STA mask cache ──────────────────────────────────────────────
# Keyed on (canvas_thw, tile_thw, kernel_thw, text_block_num, device).
# Populated once per unique config (during first torch.compile trace),
# then becomes a graph constant for subsequent calls.
_sta_gpu_cache = {}


def _get_sta_mask_gpu(canvas_thw, tile_thw, kernel_thw, text_block_num, device):
    key = (canvas_thw, tile_thw, kernel_thw, text_block_num, device)
    if key in _sta_gpu_cache:
        return _sta_gpu_cache[key]

    seq_len = math.prod(canvas_thw)
    kernel_t, kernel_h, kernel_w = kernel_thw
    block_size = math.prod(tile_thw)
    block_num = seq_len // block_size
    nt = canvas_thw[0] // tile_thw[0]
    nh = canvas_thw[1] // tile_thw[1]
    nw = canvas_thw[2] // tile_thw[2]
    hw = nh * nw

    idx = torch.arange(block_num, device=device)
    i_grid, j_grid = torch.meshgrid(idx, idx, indexing="ij")

    q_t = i_grid // hw
    q_h = (i_grid % hw) // nw
    q_w = i_grid % nw
    kv_t = j_grid // hw
    kv_h = (j_grid % hw) // nw
    kv_w = j_grid % nw

    ct = q_t.clamp(kernel_t // 2, (nt - 1) - kernel_t // 2)
    ch = q_h.clamp(kernel_h // 2, (nh - 1) - kernel_h // 2)
    cw = q_w.clamp(kernel_w // 2, (nw - 1) - kernel_w // 2)

    block_mask = (
        ((ct - kv_t).abs() <= kernel_t // 2)
        & ((ch - kv_h).abs() <= kernel_h // 2)
        & ((cw - kv_w).abs() <= kernel_w // 2)
    )

    if text_block_num > 0:
        pad = block_num + text_block_num
        sta_mask = torch.zeros(pad, pad, dtype=torch.bool, device=device)
        sta_mask[:block_num, :block_num] = block_mask
        sta_mask[:, -text_block_num:] = True
        sta_mask[-text_block_num:, -text_block_num:] = True   # text Q → text KV only
    else:
        sta_mask = block_mask

    _sta_gpu_cache[key] = sta_mask
    return sta_mask


# ── Tile / Untile ─────────────────────────────────────────────────────────────

def _tile(x, canvas_thw, tile_thw, sp_size=1):
    t, h, w = canvas_thw
    tile_t_dim, tile_h_dim, tile_w_dim = tile_thw
    n_t = t // tile_t_dim
    n_h = h // tile_h_dim
    n_w = w // tile_w_dim
    x = rearrange(x, "b head (sp t h w) d -> b head (t sp h w) d",
                  sp=sp_size, t=t // sp_size, h=h, w=w)
    return rearrange(x,
                     "b h (n_t ts_t n_h ts_h n_w ts_w) d -> b h (n_t n_h n_w ts_t ts_h ts_w) d",
                     n_t=n_t, n_h=n_h, n_w=n_w,
                     ts_t=tile_t_dim, ts_h=tile_h_dim, ts_w=tile_w_dim)


def _untile(x, canvas_thw, tile_thw, sp_size=1):
    t, h, w = canvas_thw
    tile_t_dim, tile_h_dim, tile_w_dim = tile_thw
    n_t = t // tile_t_dim
    n_h = h // tile_h_dim
    n_w = w // tile_w_dim
    x = rearrange(x,
                  "b h (n_t n_h n_w ts_t ts_h ts_w) d -> b h (n_t ts_t n_h ts_h n_w ts_w) d",
                  n_t=n_t, n_h=n_h, n_w=n_w,
                  ts_t=tile_t_dim, ts_h=tile_h_dim, ts_w=tile_w_dim)
    return rearrange(x, "b head (t sp h w) d -> b head (sp t h w) d",
                     sp=sp_size, t=t // sp_size, h=h, w=w)


# ── Sampling ──────────────────────────────────────────────────────────────────

def _importance_sampling(q, k, topk, threshold=0.0, lambda_=0.9):
    if threshold > 0.0:
        raise NotImplementedError("importance_sampling with threshold not implemented")

    q = q / q.norm(dim=-1, keepdim=True)
    k = k / k.norm(dim=-1, keepdim=True)
    gate_similarity = torch.einsum("bhsd,bhkd->bhsk", q, k)
    gate_similarity = (gate_similarity + 1.0) / 2.0
    gate_unique = torch.einsum("bhsd,bhkd->bhsk", k, k)
    gate_unique = (gate_unique + 1.0) / 2.0

    B, H, K_num, D = k.shape
    diag_indices = torch.arange(K_num, device=k.device)
    gate_unique[:, :, diag_indices, diag_indices] = torch.nan

    mean_redundancy = torch.nanmean(gate_unique, dim=-2, keepdim=True)
    importance_scores = lambda_ * gate_similarity - (1 - lambda_) * mean_redundancy

    topk = min(topk, importance_scores.size(-1))
    _, top_block_indices = importance_scores.topk(k=topk, dim=-1, sorted=False)
    return top_block_indices


def _similarity_sampling(q, k, topk, threshold=0.0,temperature=0.01):
    if threshold > 0.0:
        gate = torch.einsum("bhsd,bhkd->bhsk", q, k)
        gate = gate / temperature
        gate_ = torch.softmax(gate, dim=-1)
        sorted_gate, sorted_indices = torch.sort(gate_, dim=-1, descending=True)
        cum_scores = torch.cumsum(sorted_gate, dim=-1)
        above_threshold = cum_scores >= threshold
        has_any_above = above_threshold.any(dim=-1, keepdim=True)
        dynamic_topk = above_threshold.int().argmax(dim=-1, keepdim=True) + 1
        dynamic_topk = torch.where(has_any_above, dynamic_topk,
                                   torch.full_like(dynamic_topk, topk))
        dynamic_topk = torch.clamp(dynamic_topk, min=8, max=topk)
        indices = torch.arange(gate.size(-1), device=gate.device).expand(gate.size())
        mask = (indices < dynamic_topk).int()
        top_block_indices = (torch.gather(sorted_indices, -1, indices) * mask
                             + (1 - mask) * sorted_indices[..., 0:1])
    else:
        gate = torch.einsum("bhsd,bhkd->bhsk", q, k)
        topk = min(topk, gate.size(-1))
        _, top_block_indices = gate.topk(k=topk, dim=-1, sorted=False)
    return top_block_indices


# ── MOBA mask ────────────────────────────────────────────────────────────────

def _create_moba_3d_mask(q, k, text_q, canvas_thw, topk, tile_thw, kernel_thw,
                        text_block_num=0, add_text_mask=False, threshold=0.0,
                        lambda_=None, mask_share_within_head=True,
                        q_block_avg_pool=True, adaptive_pool=None,
                        sampling_type=None):
    seq_len = q.size(2)
    block_size = math.prod(tile_thw)
    block_num = seq_len // block_size

    block_shape = tile_thw
    batch_size, num_heads, _, head_dim = k.shape
    num_blocks_t = math.ceil(canvas_thw[0] / block_shape[0])
    num_blocks_h = math.ceil(canvas_thw[1] / block_shape[1])
    num_blocks_w = math.ceil(canvas_thw[2] / block_shape[2])

    def get_block_avg_feat(x, adaptive_pool=None, pooling_type="avg"):
        x_bm = x.view(batch_size, num_heads, num_blocks_t, num_blocks_h, num_blocks_w,
                       block_shape[0], block_shape[1], block_shape[2], head_dim)
        if adaptive_pool is not None:
            x_bm = x_bm.view(-1, block_shape[0], block_shape[1], block_shape[2], head_dim)
            x_bm = x_bm.permute(0, 4, 1, 2, 3)
            if pooling_type == "avg":
                g = torch.nn.functional.adaptive_avg_pool3d(x_bm, adaptive_pool)
                g = g.permute(0, 2, 3, 4, 1)
                x_bm = g.reshape(batch_size, num_heads, -1,
                                 head_dim * adaptive_pool[0] * adaptive_pool[1] * adaptive_pool[2])
            elif pooling_type == "max":
                g = torch.nn.functional.adaptive_max_pool3d(x_bm, adaptive_pool)
                g = g.permute(0, 2, 3, 4, 1)
                x_bm = g.reshape(batch_size, num_heads, -1,
                                 head_dim * adaptive_pool[0] * adaptive_pool[1] * adaptive_pool[2])
            elif pooling_type == "mix":
                ga = torch.nn.functional.adaptive_avg_pool3d(x_bm, (1, 1, 1))
                ga = ga / (ga.norm(dim=1, keepdim=True) + 1e-8)
                ga = ga.permute(0, 2, 3, 4, 1).reshape(batch_size, num_heads, -1, head_dim)
                mp = torch.nn.functional.adaptive_max_pool3d(x_bm, adaptive_pool)
                mp = mp / (mp.norm(dim=1, keepdim=True) + 1e-8)
                mp = mp.permute(0, 2, 3, 4, 1)
                mp = mp.reshape(batch_size, num_heads, -1,
                                head_dim * adaptive_pool[0] * adaptive_pool[1] * adaptive_pool[2])
                x_bm = torch.cat([mp, ga], dim=-1)
            else:
                raise ValueError(f"pooling_type={pooling_type} is not Supported")
        else:
            x_bm = x_bm.mean(dim=(-2, -3, -4)).view(batch_size, num_heads, -1, head_dim)
        return x_bm

    if (sampling_type == "similarity" and threshold > 0.0) and adaptive_pool is None:
        adaptive_pool = (2, 2, 2)

    k_block_means = get_block_avg_feat(k, adaptive_pool)
    if q_block_avg_pool:
        q = get_block_avg_feat(q, adaptive_pool)
    q = q.to(torch.float32)
    k_block_means = k_block_means.to(torch.float32)

    if mask_share_within_head:
        q = q.mean(dim=1, keepdim=True)
        k_block_means = k_block_means.mean(dim=1, keepdim=True)

    if sampling_type == "similarity":
        top_block_indices = _similarity_sampling(q, k_block_means, topk, threshold)
    elif sampling_type == "importance":
        top_block_indices = _importance_sampling(q, k_block_means, topk, threshold,
                                               lambda_=lambda_)
    else:
        raise NotImplementedError(f"sampling_type={sampling_type} is not Supported")

    # top_block_indices: (1, H_or_1, block_num, topk)
    top_block_indices = top_block_indices.squeeze(0)  # (H_or_1, block_num, topk)

    # Vectorized scatter — no Python loop
    gate_idx_mask = torch.zeros(top_block_indices.size(0), block_num, block_num,
                                dtype=torch.bool, device=q.device)
    gate_idx_mask.scatter_(-1, top_block_indices, True)

    if text_block_num > 0:
        pad_block_num = block_num + text_block_num
        moba_3d_mask = torch.zeros(gate_idx_mask.size(0), pad_block_num, pad_block_num,
                                    dtype=torch.bool, device=q.device)
        moba_3d_mask[:, :block_num, :block_num] = gate_idx_mask
        if add_text_mask:
            moba_3d_mask[:, :, -text_block_num:] = True       # all Q see text KV
        # --- text Q → image K MOBA top-k ---
        if text_q is not None:
            block_size = math.prod(tile_thw)
            tq = text_q.reshape(text_q.size(0), text_q.size(1),
                                text_block_num, block_size, text_q.size(-1))
            tq_avg = tq.mean(dim=-2).to(torch.float32)        # (1, H, text_block_num, D)
            if mask_share_within_head:
                tq_avg = tq_avg.mean(dim=1, keepdim=True)
            # k_block_means is already computed above — reuse it
            if sampling_type == "similarity":
                text_topk_idx = _similarity_sampling(tq_avg, k_block_means, topk, threshold)
            elif sampling_type == "importance":
                text_topk_idx = _importance_sampling(tq_avg, k_block_means, topk, threshold, lambda_=lambda_)
            text_topk_idx = text_topk_idx.squeeze(0)
            text_gate = torch.zeros(text_topk_idx.size(0), text_block_num, block_num,
                                    dtype=torch.bool, device=q.device)
            text_gate.scatter_(-1, text_topk_idx, True)
            moba_3d_mask[:, -text_block_num:, :block_num] = text_gate
        # text Q → text KV (always)
        moba_3d_mask[:, -text_block_num:, -text_block_num:] = True
    else:
        moba_3d_mask = gate_idx_mask

    return moba_3d_mask


# ── SSTA mask ────────────────────────────────────────────────────────────────

def _create_ssta_3d_mask(q, k, text_q, canvas_thw, topk, tile_thw, kernel_thw,
                        text_block_num=0, threshold=0.0, lambda_=None,
                        text_valid_len=None,
                        mask_share_within_head=True, adaptive_pool=None,
                        sampling_type=None):
    sta_3d_mask = _get_sta_mask_gpu(canvas_thw, tile_thw, kernel_thw,
                                    text_block_num, q.device)

    moba_3d_mask = _create_moba_3d_mask(
        q, k, text_q, canvas_thw, topk, tile_thw, kernel_thw, text_block_num,
        threshold=threshold, lambda_=lambda_,
        mask_share_within_head=mask_share_within_head,
        adaptive_pool=adaptive_pool, sampling_type=sampling_type)

    ssta_3d_mask = torch.logical_or(sta_3d_mask.unsqueeze(0), moba_3d_mask)

    # Mask out padding text blocks — text_valid_len is a scalar tensor,
    # all ops are tensor ops to avoid graph breaks under torch.compile
    if text_valid_len is not None:
        block_size = math.prod(tile_thw)
        seq_len = q.size(2)
        block_num = seq_len // block_size
        total = ssta_3d_mask.shape[1]

        text_mask_index = (text_valid_len.float() / block_size).ceil().clamp(min=1).to(torch.int64)
        pad_start_index = block_num + text_mask_index
        # When text_valid_len == 0, set pad_start to total (no masking)
        pad_start_index = torch.where(text_valid_len > 0, pad_start_index, total)

        idx = torch.arange(total, device=ssta_3d_mask.device)
        valid = idx < pad_start_index  # [total] bool

        # Zero out rows/cols beyond pad_start_index, keep valid region
        ssta_3d_mask = ssta_3d_mask & valid[None, :, None] & valid[None, None, :]

        # Add diagonal identity for padded region (self-attention for pad tokens)
        invalid = ~valid
        pad_diag = torch.eye(total, dtype=torch.bool, device=ssta_3d_mask.device)
        pad_diag = pad_diag & invalid[:, None] & invalid[None, :]
        ssta_3d_mask = ssta_3d_mask | pad_diag[None, :, :]

    return ssta_3d_mask


# ── Pre-processing  ─────────────────────────────────────────────────────────

def _setup_ssta(all_q, all_k, all_v, canvas_thw,
                topk=1, tile_thw=(6, 8, 8), kernel_thw=(1, 1, 1),
                text_len=0, threshold=0.0,
                lambda_=None, pad_type="zero",
                mask_share_within_head=True, sampling_type=None,
                adaptive_pool=None, text_valid_lens=None):

    if text_len > 0:
        image_q = all_q[:, :, :-text_len, :]
        image_k = all_k[:, :, :-text_len, :]
        image_v = all_v[:, :, :-text_len, :]
        text_q = all_q[:, :, -text_len:, :]
        text_k = all_k[:, :, -text_len:, :]
        text_v = all_v[:, :, -text_len:, :]
    else:
        image_q = all_q
        image_k = all_k
        image_v = all_v

    b, hd, s, d = image_q.shape
    t, h, w = canvas_thw
    spatial_len = t * h * w

    # After Ulysses de-interleaving, the image sequence may include SP padding
    # tokens (image_total = t*h*w + pad_amount). Strip them before spatial reshape.
    sp_pad_len = s - spatial_len
    if sp_pad_len > 0:
        image_q = image_q[:, :, :spatial_len, :]
        image_k = image_k[:, :, :spatial_len, :]
        image_v = image_v[:, :, :spatial_len, :]

    tile_t, tile_h, tile_w = tile_thw
    block_size = math.prod(tile_thw)

    pad_t = 0
    pad_h = 0
    pad_w = 0
    text_pad_size = 0

    need_pad = False
    if t % tile_t != 0 or h % tile_h != 0 or w % tile_w != 0:
        need_pad = True
        pad_image_q = image_q.reshape(b, hd, t, h, w, d)
        pad_image_k = image_k.reshape(b, hd, t, h, w, d)
        pad_image_v = image_v.reshape(b, hd, t, h, w, d)

        pad_t = 0 if t % tile_t == 0 else tile_t - t % tile_t
        if pad_t > 0:
            t = t + pad_t
            rq = pad_image_q[:, :, -1:, :, :, :].expand(-1, -1, pad_t, -1, -1, -1)
            rk = pad_image_k[:, :, -1:, :, :, :].expand(-1, -1, pad_t, -1, -1, -1)
            rv = pad_image_v[:, :, -1:, :, :, :].expand(-1, -1, pad_t, -1, -1, -1)
            if pad_type == "zero":
                rq = torch.zeros_like(rq)
                rk = torch.zeros_like(rk)
                rv = torch.zeros_like(rv)
            pad_image_q = torch.cat([pad_image_q, rq], dim=2)
            pad_image_k = torch.cat([pad_image_k, rk], dim=2)
            pad_image_v = torch.cat([pad_image_v, rv], dim=2)

        pad_h = 0 if h % tile_h == 0 else tile_h - h % tile_h
        if pad_h > 0:
            h = h + pad_h
            rq = pad_image_q[:, :, :, -1:, :, :].expand(-1, -1, -1, pad_h, -1, -1)
            rk = pad_image_k[:, :, :, -1:, :, :].expand(-1, -1, -1, pad_h, -1, -1)
            rv = pad_image_v[:, :, :, -1:, :, :].expand(-1, -1, -1, pad_h, -1, -1)
            if pad_type == "zero":
                rq = torch.zeros_like(rq)
                rk = torch.zeros_like(rk)
                rv = torch.zeros_like(rv)
            pad_image_q = torch.cat([pad_image_q, rq], dim=3)
            pad_image_k = torch.cat([pad_image_k, rk], dim=3)
            pad_image_v = torch.cat([pad_image_v, rv], dim=3)

        pad_w = 0 if w % tile_w == 0 else tile_w - w % tile_w
        if pad_w > 0:
            w = w + pad_w
            rq = pad_image_q[:, :, :, :, -1:, :].expand(-1, -1, -1, -1, pad_w, -1)
            rk = pad_image_k[:, :, :, :, -1:, :].expand(-1, -1, -1, -1, pad_w, -1)
            rv = pad_image_v[:, :, :, :, -1:, :].expand(-1, -1, -1, -1, pad_w, -1)
            if pad_type == "zero":
                rq = torch.zeros_like(rq)
                rk = torch.zeros_like(rk)
                rv = torch.zeros_like(rv)
            pad_image_q = torch.cat([pad_image_q, rq], dim=4)
            pad_image_k = torch.cat([pad_image_k, rk], dim=4)
            pad_image_v = torch.cat([pad_image_v, rv], dim=4)

        image_q = pad_image_q.reshape(b, hd, -1, d)
        image_k = pad_image_k.reshape(b, hd, -1, d)
        image_v = pad_image_v.reshape(b, hd, -1, d)
        canvas_thw = (t, h, w)

    need_pad_text = False
    text_block_num = math.ceil(text_len / block_size)
    text_target_size = text_block_num * block_size
    if text_len % block_size > 0:
        need_pad_text = True
        text_pad_size = text_target_size - text_len
        ptq = text_q[:, :, -1, :].unsqueeze(2).expand(-1, -1, text_pad_size, -1)
        ptk = text_k[:, :, -1, :].unsqueeze(2).expand(-1, -1, text_pad_size, -1)
        ptv = text_v[:, :, -1, :].unsqueeze(2).expand(-1, -1, text_pad_size, -1)
        if pad_type == "zero":
            ptq = torch.zeros_like(ptq)
            ptk = torch.zeros_like(ptk)
            ptv = torch.zeros_like(ptv)
        text_q = torch.cat([text_q, ptq], dim=2)
        text_k = torch.cat([text_k, ptk], dim=2)
        text_v = torch.cat([text_v, ptv], dim=2)

    image_q = _tile(image_q, canvas_thw, tile_thw)
    image_k = _tile(image_k, canvas_thw, tile_thw)
    image_v = _tile(image_v, canvas_thw, tile_thw)

    if text_len > 0:
        q = torch.cat([image_q, text_q], dim=2)
        k = torch.cat([image_k, text_k], dim=2)
        v = torch.cat([image_v, text_v], dim=2)
    else:
        q = image_q
        k = image_k
        v = image_v
    
    ssta_state = SSTAState(
        canvas_thw=canvas_thw,
        tile_thw=tile_thw, 
        text_len=text_len, 
        sp_pad_len=sp_pad_len, 
        need_pad=need_pad,
        text_target_size=text_target_size, 
        need_pad_text=need_pad_text, 
        text_pad_size=text_pad_size,
        pad_t=pad_t, 
        pad_h=pad_h,
        pad_w=pad_w, 
        b=b, 
        hd=hd, 
        t=t, 
        h=h, 
        w=w, 
        d=d
    )

    mask_config = MaskConfig(
        image_q=image_q, 
        image_k=image_k,
        text_q=text_q if text_len > 0 else None,
        canvas_thw=canvas_thw, 
        tile_thw=tile_thw, 
        kernel_thw=kernel_thw,
        text_block_num=text_block_num, 
        threshold=threshold, 
        lambda_=lambda_,
        text_valid_lens=text_valid_lens, 
        mask_share_within_head=mask_share_within_head,
        adaptive_pool=adaptive_pool, 
        sampling_type=sampling_type, 
        topk=topk, 
        b=b
    )

    return q, k, v, mask_config, ssta_state

# ── Post-processing  ─────────────────────────────────────────────────────────

def _untile_ssta_output(o, ssta_state):
    if ssta_state.text_len > 0:
        image_o = o[:, :, :-ssta_state.text_target_size, :]
        if ssta_state.need_pad_text:
            text_o = o[:, :, -ssta_state.text_target_size:-ssta_state.text_pad_size, :]
        else:
            text_o = o[:, :, -ssta_state.text_target_size:, :]
    else:
        image_o = o

    image_o = _untile(image_o, ssta_state.canvas_thw, ssta_state.tile_thw)

    if ssta_state.need_pad:
        unpad = image_o.reshape(ssta_state.b, 
                                ssta_state.hd, 
                                ssta_state.t, 
                                ssta_state.h, 
                                ssta_state.w, 
                                ssta_state.d)
        if ssta_state.pad_t > 0:
            unpad = unpad[:, :, :-ssta_state.pad_t, :, :, :]
        if ssta_state.pad_h > 0:
            unpad = unpad[:, :, :, :-ssta_state.pad_h, :, :]
        if ssta_state.pad_w > 0:
            unpad = unpad[:, :, :, :, :-ssta_state.pad_w, :]
        image_o = unpad.reshape(ssta_state.b, ssta_state.hd, -1, ssta_state.d)

    # Re-append SP padding tokens that were stripped before spatial processing
    if ssta_state.sp_pad_len > 0:
        sp_pad_o = torch.zeros(ssta_state.b,
                               ssta_state.hd, 
                               ssta_state.sp_pad_len, 
                               ssta_state.d, 
                               dtype=image_o.dtype, 
                               device=image_o.device)
        image_o = torch.cat([image_o, sp_pad_o], dim=2)

    if ssta_state.text_len > 0:
        o = torch.cat([image_o, text_o], dim=2)
    else:
        o = image_o

    return o

# ── Sparse masks  ─────────────────────────────────────────────────────────

def _get_ssta_mask(mask_config):
    image_q_list = torch.split(mask_config.image_q, 1, dim=0)
    image_k_list = torch.split(mask_config.image_k, 1, dim=0)
    text_q_list = torch.split(mask_config.text_q, 1, dim=0) if mask_config.text_q is not None else None
    mask_list = []
    for i in range(mask_config.b):
        tvl = mask_config.text_valid_lens[i] if mask_config.text_valid_lens is not None else None
        bm = _create_ssta_3d_mask(
            image_q_list[i], image_k_list[i], text_q=text_q_list[i] if text_q_list is not None else None,
            canvas_thw=mask_config.canvas_thw, tile_thw=mask_config.tile_thw, kernel_thw=mask_config.kernel_thw,
            text_block_num=mask_config.text_block_num, topk=mask_config.topk, threshold=mask_config.threshold,
            lambda_=mask_config.lambda_, text_valid_len=tvl,
            mask_share_within_head=mask_config.mask_share_within_head,
            adaptive_pool=mask_config.adaptive_pool, sampling_type=mask_config.sampling_type)
        mask_list.append(bm)

    block_mask = torch.stack(mask_list, dim=0)
    if mask_config.mask_share_within_head:
        block_mask = block_mask.unsqueeze(1)  # [b, 1, s_block, s_block]

    return block_mask

def _get_moba_mask(mask_config):
    image_q_list = torch.split(mask_config.image_q, 1, dim=0)
    image_k_list = torch.split(mask_config.image_k, 1, dim=0)
    text_q_list = torch.split(mask_config.text_q, 1, dim=0) if mask_config.text_q is not None else None

    mask_list = []
    for i in range(mask_config.b):
        block_mask = _create_moba_3d_mask(image_q_list[i],
                                          image_k_list[i],
                                          text_q=text_q_list[i] if text_q_list is not None else None,
                                          canvas_thw=mask_config.canvas_thw,
                                          topk=mask_config.topk, 
                                          tile_thw=mask_config.tile_thw, 
                                          kernel_thw=mask_config.kernel_thw,
                                          text_block_num=mask_config.text_block_num, 
                                          add_text_mask=True,
                                          lambda_=mask_config.lambda_,
                                          threshold=mask_config.threshold,
                                          mask_share_within_head=mask_config.mask_share_within_head,
                                          adaptive_pool=mask_config.adaptive_pool,
                                          sampling_type=mask_config.sampling_type)
        mask_list.append(block_mask)
    block_mask = torch.stack(mask_list, dim=0)
    return block_mask

# ── Deinterleave and Reinterleave  ─────────────────────────────────────────────────────────

# After Ulysses input a2a the sequence is interleaved rank-chunks:
#   [img_r0, txt_r0, img_r1, txt_r1, ..., img_{U-1}, txt_{U-1}]
# SSTA expects [all_image, all_text]. De-interleave here.
def _deinterleave(x, u, txt_len):
    """Reshape interleaved rank-chunks into [all_image, all_text]."""
    b, h, s, d = x.shape
    chunk_len = s // u
    img_len = chunk_len - txt_len
    # (b, h, U, chunk_len, d)
    x = x.reshape(b, h, u, chunk_len, d)
    img_part = x[:, :, :, :img_len, :]   # (b, h, U, img_len, d)
    txt_part = x[:, :, :, img_len:, :]    # (b, h, U, txt_len, d)
    img_part = img_part.reshape(b, h, -1, d)  # (b, h, U*img_len, d)
    txt_part = txt_part.reshape(b, h, -1, d)  # (b, h, U*txt_len, d)
    return torch.cat([img_part, txt_part], dim=2)

def _reinterleave(x, u, txt_len):
    """Reverse: [all_image, all_text] -> interleaved rank-chunks."""
    b, h, s, d = x.shape
    total_txt = u * txt_len
    total_img = s - total_txt
    img_len = total_img // u
    img_part = x[:, :, :total_img, :].reshape(b, h, u, img_len, d)
    txt_part = x[:, :, total_img:, :].reshape(b, h, u, txt_len, d)
    return torch.cat([img_part, txt_part], dim=3).reshape(b, h, s, d)

# ── Entry points  ─────────────────────────────────────────────────────────

def setup_ssta(query, key, value, attn_kwargs):
    ssta_threshold = attn_kwargs["ssta_threshold"]
    ssta_lambda = attn_kwargs["ssta_lambda"]
    ssta_sampling_type = attn_kwargs["ssta_sampling_type"]
    ssta_adaptive_pool = attn_kwargs["ssta_adaptive_pool"]

    attn_pad_type = attn_kwargs["attn_pad_type"]
    attn_use_text_mask = attn_kwargs["attn_use_text_mask"]
    text_mask = attn_kwargs["text_mask"]
    attn_mask_share_within_head = attn_kwargs["attn_mask_share_within_head"]
    encoder_sequence_length = attn_kwargs["encoder_sequence_length"]

    ssta_topk = attn_kwargs["ssta_topk"]
    thw = attn_kwargs["thw"]
    tile_size = (1, 16, 8) # Overwrite attn_kwargs["tile_thw"] to match block size in triton sage kernel
    win_size = attn_kwargs["win_size"][0].copy()

    sp_size = attn_kwargs["sp_size"]

    # Precompute valid text token counts
    text_valid_lens = None
    if text_mask is not None and attn_use_text_mask:
        text_valid_lens = text_mask.sum(dim=-1)

    if thw[0] == 1:
        win_size = [1, 1, 1]
    elif thw[0] <= 31:
        ssta_topk = ssta_topk // 2
    
    if encoder_sequence_length > 0 and sp_size > 1:
        query = _deinterleave(query, sp_size, encoder_sequence_length)
        key = _deinterleave(key, sp_size, encoder_sequence_length)
        value = _deinterleave(value, sp_size, encoder_sequence_length)

    q, k, v, mask_config, ssta_state = _setup_ssta(
        query,
        key,
        value,
        thw,
        topk=ssta_topk,
        tile_thw=tile_size,
        kernel_thw=tuple(win_size),
        text_len=encoder_sequence_length * sp_size,
        threshold=ssta_threshold,
        lambda_=ssta_lambda,
        pad_type=attn_pad_type,
        sampling_type=ssta_sampling_type,
        adaptive_pool=ssta_adaptive_pool,
        mask_share_within_head=attn_mask_share_within_head,
        text_valid_lens=text_valid_lens,
    )

    return q, k, v, mask_config, ssta_state

def get_sparse_mask(mask_config, sparse_type="ssta"):
    if sparse_type == "ssta":
        block_mask = _get_ssta_mask(mask_config)
    elif sparse_type == "moba":
        block_mask = _get_moba_mask(mask_config)
    elif sparse_type == "sta":
        block_mask = _get_sta_mask_gpu(mask_config.canvas_thw, mask_config.tile_thw,
                                      mask_config.kernel_thw, mask_config.text_block_num, mask_config.image_q.device)
    else:
        raise NotImplementedError(f"sparse_type={sparse_type} is not Supported")
    return block_mask

def untile_ssta_output(o, ssta_state, encoder_sequence_length, sp_size):
    o = _untile_ssta_output(o, ssta_state)
    if encoder_sequence_length > 0 and sp_size > 1:
        o = _reinterleave(o, sp_size, encoder_sequence_length)

    return o