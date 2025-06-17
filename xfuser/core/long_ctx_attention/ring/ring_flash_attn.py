from typing import List
import math
import torch
import torch.nn.functional as F
from torch.nn.attention.flex_attention import flex_attention

from xfuser.core.long_ctx_attention import xFuserLongContextAttention
from xfuser.core.cache_manager.cache_manager import get_cache_manager
from yunchang.ring.utils import RingComm, update_out_and_lse
from yunchang.ring.ring_flash_attn import RingFlashAttnFunc
from yunchang.kernels import select_flash_attn_impl, AttnType

try:
    import flash_attn
    from flash_attn.flash_attn_interface import _flash_attn_forward
except ImportError:
    flash_attn = None
    _flash_attn_forward = None
    from yunchang.kernels.attention import pytorch_attn_forward

def xdit_ring_flash_attn_forward(
    process_group,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale,
    dropout_p=0,
    causal=True,
    window_size=(-1, -1),
    alibi_slopes=None,
    deterministic=False,
    attn_type=AttnType.FA,
    attn_processor=None,
    attn_layer=None,
    joint_tensor_key=None,
    joint_tensor_value=None,
    joint_strategy="none",
):
    is_joint = False
    if (joint_tensor_key is not None and 
        joint_tensor_value is not None):
        supported_joint_strategy = ["front", "rear"]
        if joint_strategy not in supported_joint_strategy:
            raise ValueError(
                f"joint_strategy: {joint_strategy} not supprted. supported joint strategy: {supported_joint_strategy}"
            )
        else:
            is_joint = True
    elif (joint_tensor_key is None and 
        joint_tensor_value is None):
        pass
    else:
        raise ValueError(
            f"joint_tensor_key and joint_tensor_value should be None or not None simultaneously."
        )

    comm = RingComm(process_group)

    out = None
    lse = None

    next_k, next_v = None, None

    if attn_layer is not None:
        k, v = get_cache_manager().update_and_get_kv_cache(
            new_kv=[k, v],
            layer=attn_layer,
            slice_dim=1,
            layer_type="attn",
        )
        k = k.contiguous()
        v = v.contiguous()

    for step in range(comm.world_size):
        if step + 1 != comm.world_size:
            next_k: torch.Tensor = comm.send_recv(k)
            next_v: torch.Tensor = comm.send_recv(v)
            comm.commit()

        if is_joint and joint_strategy == "rear":
            if step + 1 == comm.world_size:
                key = torch.cat([k, joint_tensor_key], dim=1)
                value = torch.cat([v, joint_tensor_value], dim=1)
            else:
                key, value = k, v
        elif is_joint and joint_strategy == "front":
            if step == 0:
                key = torch.cat([joint_tensor_key, k], dim=1)
                value = torch.cat([joint_tensor_value, v], dim=1)
            else:
                key, value = k, v
        else:
            key, value = k, v

        if not causal or step <= comm.rank:
            fn = select_flash_attn_impl(attn_type, stage="fwd-only", attn_processor=attn_processor)
            block_out, block_lse = fn(
                q,
                key,
                value,
                dropout_p=dropout_p,
                softmax_scale=softmax_scale,
                causal=causal and step == 0,
                window_size=window_size,
                softcap=0.0,
                alibi_slopes=alibi_slopes,
                return_softmax=True and dropout_p > 0,
            )
            if attn_type == AttnType.SPARSE_SAGE:
                out, lse = block_out, block_lse
            else:
                out, lse = update_out_and_lse(out, lse, block_out, block_lse)

        if step + 1 != comm.world_size:
            comm.wait()
            k = next_k
            v = next_v

    out = out.to(q.dtype)
    if attn_type != AttnType.SPARSE_SAGE:
        lse = lse.squeeze(dim=-1).transpose(1, 2)
    return out, lse


class xFuserRingFlashAttnFunc(RingFlashAttnFunc):
    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        alibi_slopes,
        deterministic,
        return_softmax,
        group,
        attn_type,
        attn_processor,
        attn_layer,
        joint_tensor_key,
        joint_tensor_value,
        joint_strategy,
    ):
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)

        assert alibi_slopes is None
        if attn_layer is None:
            k = k.contiguous()
            v = v.contiguous()
        out, softmax_lse = xdit_ring_flash_attn_forward(
            group,
            q,
            k,
            v,
            softmax_scale=softmax_scale,
            dropout_p=dropout_p,
            causal=causal,
            window_size=window_size,
            alibi_slopes=alibi_slopes,
            deterministic=False,
            attn_type=attn_type,
            attn_processor=attn_processor,
            attn_layer=attn_layer,
            joint_tensor_key=joint_tensor_key,
            joint_tensor_value=joint_tensor_value,
            joint_strategy=joint_strategy,
        )
        # this should be out_padded
        ctx.save_for_backward(q, k, v, out, softmax_lse)
        ctx.softcap = 0.0
        ctx.dropout_p = dropout_p
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.window_size = window_size
        ctx.alibi_slopes = alibi_slopes
        ctx.deterministic = deterministic
        ctx.group = group
        ctx.attn_type = attn_type
        ctx.attn_processor = attn_processor
        return out if not return_softmax else (out, softmax_lse, None)


def xdit_ring_flash_attn_func(
    q,
    k,
    v,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    group=None,
    attn_type=AttnType.FA,
    attn_processor=None,
    attn_layer=None,
    joint_tensor_key=None,
    joint_tensor_value=None,
    joint_strategy="none",
):
    return xFuserRingFlashAttnFunc.apply(
        q,
        k,
        v,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        alibi_slopes,
        deterministic,
        return_attn_probs,
        group,
        attn_type,
        attn_processor,
        attn_layer,
        joint_tensor_key,
        joint_tensor_value,
        joint_strategy,
    )
    

def xdit_sana_ring_flash_attn_forward(
    process_group,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_layer=None,
):

    comm = RingComm(process_group)

    out = None

    next_k, next_v = None, None

    if attn_layer is not None:
        k, v = get_cache_manager().update_and_get_kv_cache(
            new_kv=[k, v],
            layer=attn_layer,
            slice_dim=1,
            layer_type="attn",
        )
        k = k.contiguous()
        v = v.contiguous()
        
    q = F.relu(q).permute(0, 2, 3, 1).contiguous()
    k = F.relu(k).transpose(1, 2).contiguous()
    v = v.permute(0, 2, 3, 1).contiguous()
    v = F.pad(v, (0, 0, 0, 1), mode="constant", value=1.0)

    for step in range(comm.world_size):
        if step + 1 != comm.world_size:
            next_k: torch.Tensor = comm.send_recv(k)
            next_v: torch.Tensor = comm.send_recv(v)
            comm.commit()

        key, value = k, v

        if step <= comm.rank:
            # b x n_heads x len_seq x d
            q, key, value = q.float(), key.float(), value.float()
            block_out = value @ key @ q
            out = block_out.float() if out is None else out + block_out.float()

        if step + 1 != comm.world_size:
            comm.wait()
            k = next_k
            v = next_v

    out = out.to(q.dtype)
    out = out[:, :, :-1] / (out[:, :, -1:] + torch.finfo(out.dtype).eps)
    out = out.transpose(-2, -1)
    return out

class xFuserSanaRingFlashAttnFunc(RingFlashAttnFunc):
    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v, 
        attn_layer, 
        group, 
    ):

        if attn_layer is None:
            k = k.contiguous()
            v = v.contiguous()
        out = xdit_sana_ring_flash_attn_forward(
            group,
            q,
            k,
            v,
            attn_layer=attn_layer,
        )
        
        ctx.group = group
        return out

def xdit_sana_ring_flash_attn_func(
        q:torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        group=None,
        attn_layer=None,
    ) -> torch.Tensor:

    return xFuserSanaRingFlashAttnFunc.apply(
        q,
        k,
        v, 
        attn_layer, 
        group, 
    )
