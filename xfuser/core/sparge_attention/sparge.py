from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from xfuser.core.sparge_attention.block_mask import get_block_map_meansim
from xfuser.core.sparge_attention.gilbert import (
    curve as gilbert_curve,
    sliced_gilbert_block_neighbor_mapping,
)


# ── Caches ────────────────────────────────────────────────────────────────────

# Forward / inverse gilbert permutations keyed on ((t, h, w), (device.type, idx)).
_GILBERT_PERM_CACHE: dict[tuple, Tuple[torch.Tensor, torch.Tensor]] = {}

# Block-neighbor static mask keyed on ((t, h, w), block_m, block_n,
# (device.type, idx)). Bool tensor of shape (n_image_q, n_image_k) on device.
_STATIC_BLOCK_MASK_CACHE: dict[tuple, torch.Tensor] = {}


def _device_key(device: torch.device) -> tuple:
    return (device.type, device.index if device.index is not None else -1)


# ── State carried from setup -> restore ──────────────────────────────────────

@dataclass
class SpargeState:
    """State produced by ``setup_sparge`` and consumed by
    ``restore_sparge_output`` so the post-attention path can reverse the
    permutation, re-append stripped SP padding, and re-interleave the
    text/image partition back into Ulysses rank-chunks."""
    sp_pad_len: int
    text_len: int
    sp_size: int
    inv_perm: Optional[torch.Tensor]
    b: int
    hd: int
    d: int
    dtype: torch.dtype
    device: torch.device
    image_len: int = 0
    img_pad: int = 0
    tail_pad: int = 0


# ── Cache builders ───────────────────────────────────────────────────────────

def get_gilbert_perm(thw: Tuple[int, int, int], device: torch.device
                     ) -> Tuple[torch.Tensor, torch.Tensor]:
    key = (tuple(thw), _device_key(device))
    cached = _GILBERT_PERM_CACHE.get(key)
    if cached is not None:
        return cached
    t, h, w = thw
    inv_perm, fwd_perm = gilbert_curve(t, h, w, device)
    _GILBERT_PERM_CACHE[key] = (fwd_perm, inv_perm)
    return fwd_perm, inv_perm


def get_static_block_neighbor_mask(
    thw: Tuple[int, int, int],
    block_m: int, block_n: int,
    device: torch.device,
    gilbert_mapping: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    *,
    mapping_kind: str = "gilbert",
) -> torch.Tensor:
    """Block-pair neighbour mask for the (thw, block_m, block_n) layout.

    ``mapping_kind`` disambiguates the cache when the same shape is built
    under different block layouts (Gilbert-reordered vs linear row-major).
    For ``"linear"`` an identity ``linear_to_hilbert`` is built on demand
    if no ``gilbert_mapping`` was supplied.
    """
    key = (
        tuple(thw),
        int(block_m),
        int(block_n),
        _device_key(device),
        str(mapping_kind),
    )
    cached = _STATIC_BLOCK_MASK_CACHE.get(key)
    if cached is not None:
        return cached
    t, h, w = thw
    if mapping_kind == "linear" and gilbert_mapping is None:
        identity = torch.arange(t * h * w, dtype=torch.int64, device=device)
        gilbert_mapping = (identity, identity)
    mask = sliced_gilbert_block_neighbor_mapping(
        t, h, w, block_m, block_n, device, gilbert_mapping=gilbert_mapping,
    )
    _STATIC_BLOCK_MASK_CACHE[key] = mask
    return mask


# ── Ulysses de-/re-interleave ────────────────────────────────────────────────

def _deinterleave(x: torch.Tensor, u: int, txt_len: int) -> torch.Tensor:
    b, h, s, d = x.shape
    chunk_len = s // u
    img_len = chunk_len - txt_len
    x = x.reshape(b, h, u, chunk_len, d)
    img_part = x[:, :, :, :img_len, :].reshape(b, h, -1, d)
    txt_part = x[:, :, :, img_len:, :].reshape(b, h, -1, d)
    return torch.cat([img_part, txt_part], dim=2)


def _reinterleave(x: torch.Tensor, u: int, txt_len: int) -> torch.Tensor:
    b, h, s, d = x.shape
    total_txt = u * txt_len
    total_img = s - total_txt
    img_len = total_img // u
    img_part = x[:, :, :total_img, :].reshape(b, h, u, img_len, d)
    txt_part = x[:, :, total_img:, :].reshape(b, h, u, txt_len, d)
    return torch.cat([img_part, txt_part], dim=3).reshape(b, h, s, d)


# ── Public API ───────────────────────────────────────────────────────────────

def setup_sparge(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    thw: Optional[Tuple[int, int, int]],
    sp_size: int,
    encoder_sequence_length: int = 0,
    reorder_sequence: bool = False,
    use_static_block_mask: bool = False,
    block_m: int = 128,
    block_n: int = 128,
    pad_block_divisible: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
           SpargeState, Optional[torch.Tensor]]:

    b, hd, s, d = query.shape

    text_len_post_deint = encoder_sequence_length * sp_size

    # Ulysses de-interleave (no-op when text_len == 0 or sp_size == 1).
    if text_len_post_deint > 0 and sp_size > 1:
        query = _deinterleave(query, sp_size, encoder_sequence_length)
        key = _deinterleave(key, sp_size, encoder_sequence_length)
        value = _deinterleave(value, sp_size, encoder_sequence_length)

    image_len_post_deint = s - text_len_post_deint
    fwd_perm: Optional[torch.Tensor] = None
    inv_perm: Optional[torch.Tensor] = None
    static_mask: Optional[torch.Tensor] = None
    sp_pad_len = 0

    # Both reorder and the static mask operate on the spatial (thw) layout,
    # so they both need thw and they both need SP padding to be stripped
    # off so the block grid lines up with the (t, h, w) cuboid.
    if use_static_block_mask and thw is None:
        use_static_block_mask = False

    if reorder_sequence and thw is None:
        raise ValueError(
            "Sparge with reorder_sequence=True requires "
            "`attention_kwargs['thw']` to be published by the model wrapper."
        )

    if thw is not None and (reorder_sequence or use_static_block_mask):
        spatial_len = thw[0] * thw[1] * thw[2]
        sp_pad_len = image_len_post_deint - spatial_len
        if sp_pad_len < 0:
            raise ValueError(
                f"Sparge: image length after de-interleave "
                f"({image_len_post_deint}) is shorter than thw product "
                f"({spatial_len}). Mismatch between attention_kwargs['thw'] "
                f"and the input sequence."
            )

    if reorder_sequence:
        fwd_perm, inv_perm = get_gilbert_perm(thw, query.device)
        if use_static_block_mask:
            static_mask = get_static_block_neighbor_mask(
                thw, block_m, block_n, query.device,
                gilbert_mapping=(inv_perm, fwd_perm),
                mapping_kind="gilbert",
            )
    elif use_static_block_mask:
        # Linear (row-major) order: feed the neighbour builder an identity
        # linear_to_hilbert so it emits a *linear*-block neighbour mask.
        static_mask = get_static_block_neighbor_mask(
            thw, block_m, block_n, query.device,
            mapping_kind="linear",
        )

    # Split off image part (without SP padding) and text part.
    if text_len_post_deint > 0:
        image_q = query[:, :, :image_len_post_deint, :]
        image_k = key[:, :, :image_len_post_deint, :]
        image_v = value[:, :, :image_len_post_deint, :]
        text_q = query[:, :, image_len_post_deint:, :]
        text_k = key[:, :, image_len_post_deint:, :]
        text_v = value[:, :, image_len_post_deint:, :]
    else:
        image_q, image_k, image_v = query, key, value
        text_q = text_k = text_v = None

    if sp_pad_len > 0:
        spatial_len = image_len_post_deint - sp_pad_len
        image_q = image_q[:, :, :spatial_len, :]
        image_k = image_k[:, :, :spatial_len, :]
        image_v = image_v[:, :, :spatial_len, :]

    if fwd_perm is not None:
        image_q = image_q.index_select(dim=2, index=fwd_perm)
        image_k = image_k.index_select(dim=2, index=fwd_perm)
        image_v = image_v.index_select(dim=2, index=fwd_perm)

    _a, _b = block_m, block_n
    while _b:
        _a, _b = _b, _a % _b
    align = block_m * block_n // _a
    image_len = image_q.shape[2]
    need_image_pad = pad_block_divisible or text_len_post_deint > 0
    img_pad = (-image_len) % align if need_image_pad else 0
    if img_pad > 0:
        image_q = F.pad(image_q, (0, 0, 0, img_pad))
        image_k = F.pad(image_k, (0, 0, 0, img_pad))
        image_v = F.pad(image_v, (0, 0, 0, img_pad))
        if static_mask is not None:
            n_iq = (image_len + img_pad) // block_m
            n_ik = (image_len + img_pad) // block_n
            padded_static = torch.zeros(
                (n_iq, n_ik), dtype=static_mask.dtype, device=static_mask.device
            )
            padded_static[: static_mask.shape[0], : static_mask.shape[1]] = static_mask
            static_mask = padded_static

    # Re-concatenate text tail.
    if text_q is not None:
        q_out = torch.cat([image_q, text_q], dim=2)
        k_out = torch.cat([image_k, text_k], dim=2)
        v_out = torch.cat([image_v, text_v], dim=2)
    else:
        q_out = image_q
        k_out = image_k
        v_out = image_v

    total_len = q_out.shape[2]
    tail_pad = (-total_len) % align if pad_block_divisible else 0
    if tail_pad > 0:
        q_out = F.pad(q_out, (0, 0, 0, tail_pad))
        k_out = F.pad(k_out, (0, 0, 0, tail_pad))
        v_out = F.pad(v_out, (0, 0, 0, tail_pad))

    state = SpargeState(
        sp_pad_len=sp_pad_len,
        text_len=text_len_post_deint,
        sp_size=sp_size,
        inv_perm=inv_perm,
        b=b,
        hd=hd,
        d=d,
        dtype=query.dtype,
        device=query.device,
        image_len=image_len,
        img_pad=img_pad,
        tail_pad=tail_pad,
    )
    return q_out, k_out, v_out, state, static_mask


def compute_sparge_block_mask(
    q: torch.Tensor,
    k: torch.Tensor,
    *,
    simthreshd1: float,
    cdfthreshd: float,
    is_causal: bool = False,
    static_block_mask: Optional[torch.Tensor] = None,
    text_len: int = 0,
    block_m: int = 128,
    block_n: int = 128,
) -> torch.Tensor:
    image_q = q[:, :, :q.shape[2] - text_len, :] if text_len > 0 else q
    image_k = k[:, :, :k.shape[2] - text_len, :] if text_len > 0 else k

    image_block_mask = get_block_map_meansim(
        image_q, image_k,
        is_causal=is_causal,
        BLKQ=block_m, BLKK=block_n,
        simthreshd1=simthreshd1, cdfthreshd=cdfthreshd,
        attention_sink=False,
    )

    if static_block_mask is not None:
        # OR the structured backbone in: static_block_mask is a (n_iq, n_ik)
        # bool tensor; broadcast across the batch and head dims.
        image_block_mask = image_block_mask | static_block_mask[None, None, ...]

    if text_len == 0:
        return image_block_mask

    B, H, n_iq, n_ik = image_block_mask.shape
    n_text_q = (text_len + block_m - 1) // block_m
    n_text_k = (text_len + block_n - 1) // block_n
    n_total_q = n_iq + n_text_q
    n_total_k = n_ik + n_text_k

    full = torch.zeros(
        B, H, n_total_q, n_total_k,
        dtype=image_block_mask.dtype,
        device=image_block_mask.device,
    )
    full[:, :, :n_iq, :n_ik] = image_block_mask
    # All Q rows attend to text K cols, and all text Q rows attend to all K.
    full[:, :, :, -n_text_k:] = True
    full[:, :, -n_text_q:, :] = True
    # --- treat the image/text boundary block (if there is one) as dense ---
    image_len_q = q.shape[2] - text_len   # length of image portion in q
    image_len_k = k.shape[2] - text_len   # length of image portion in k
    if image_len_q % block_m != 0:
        boundary_q = image_len_q // block_m   # last (partial) image block, contains text spillover
        full[:, :, boundary_q, :] = True
    if image_len_k % block_n != 0:
        boundary_k = image_len_k // block_n
        full[:, :, :, boundary_k] = True
    return full


def restore_sparge_output(o: torch.Tensor, state: SpargeState) -> torch.Tensor:
    if state.tail_pad > 0:
        o = o[:, :, : o.shape[2] - state.tail_pad, :]

    if state.text_len > 0:
        text_o = o[:, :, -state.text_len:, :]
        image_o = o[:, :, : o.shape[2] - state.text_len, :]
    else:
        image_o = o
        text_o = None

    if state.img_pad > 0:
        image_o = image_o[:, :, : image_o.shape[2] - state.img_pad, :]

    if state.inv_perm is not None:
        image_o = image_o.index_select(dim=2, index=state.inv_perm)

    if state.sp_pad_len > 0:
        sp_pad = torch.zeros(
            state.b, state.hd, state.sp_pad_len, state.d,
            dtype=image_o.dtype, device=image_o.device,
        )
        image_o = torch.cat([image_o, sp_pad], dim=2)

    if text_o is not None:
        out = torch.cat([image_o, text_o], dim=2)
    else:
        out = image_o

    if state.text_len > 0 and state.sp_size > 1:
        # text_len here is post-deinterleave (= encoder_sequence_length *
        # sp_size); pass per-rank text length to _reinterleave.
        out = _reinterleave(out, state.sp_size, state.text_len // state.sp_size)
    return out


def mask_padded_kv_blocks(
    block_mask: torch.Tensor, state: SpargeState, block_n: int
) -> torch.Tensor:
    if state.img_pad == 0 and state.tail_pad == 0:
        return block_mask

    image_padded = state.image_len + state.img_pad
    n_ik = image_padded // block_n
    first_img_pad = (state.image_len + block_n - 1) // block_n
    if first_img_pad < n_ik:
        block_mask[:, :, :, first_img_pad:n_ik] = False

    if state.tail_pad > 0:
        n_tk = (state.text_len + state.tail_pad + block_n - 1) // block_n
        first_txt_pad = (state.text_len + block_n - 1) // block_n
        if first_txt_pad < n_tk:
            block_mask[:, :, :, n_ik + first_txt_pad : n_ik + n_tk] = False

    return block_mask
