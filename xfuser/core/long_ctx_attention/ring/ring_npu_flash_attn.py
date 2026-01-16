from typing import List
import math
import torch
import torch.nn.functional as F

from xfuser.core.long_ctx_attention import xFuserLongContextAttention
from xfuser.core.cache_manager.cache_manager import get_cache_manager
import xfuser.envs as envs

if torch.cuda.is_available() or envs._is_npu():
    from yunchang.ring.utils import RingComm, update_out_and_lse, update_npu_out
    from yunchang.ring.ring_npu_flash_attn import RingNpuFlashAttnFunc
    from yunchang.kernels import select_flash_attn_impl, AttnType
else:
    RingComm = object
    RingNPUFlashAttnFunc = object
    AttnType = None
    select_flash_attn_impl = None

try:
    import flash_attn
    from flash_attn.flash_attn_interface import _flash_attn_forward
except ImportError:
    flash_attn = None
    _flash_attn_forward = None


def xdit_ring_npu_flash_attn_forward(
        process_group,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        head_num=None,
        layout=None,
        softmax_scale=None,
        causal=True,
        attn_type=AttnType.NPU,
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

    out, softmax_max, softmax_sum = None, None, None

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
            block_out, block_softmax_max, block_softmax_sum = fn(
                q,
                key,
                value,
                head_num,
                layout,
                softmax_scale
            )
            out, softmax_max, softmax_sum = update_npu_out(block_out, block_softmax_max, block_softmax_sum, out, softmax_max, softmax_sum)

        if step + 1 != comm.world_size:
            comm.wait()
            k = next_k
            v = next_v

    out = out.to(q.dtype)

    return out, softmax_max, softmax_sum


class xFuserRingNpuFlashAttnFunc(RingNpuFlashAttnFunc):
    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        head_num,
        layout,
        softmax_scale,
        causal,
        return_softmax,
        group,
        attn_type,
        attn_processor,
        attn_layer,
        joint_tensor_key,
        joint_tensor_value,
        joint_strategy,
    ):
        if attn_layer is None:
            k = k.contiguous()
            v = v.contiguous()
        out, softmax_max, softmax_sum = xdit_ring_npu_flash_attn_forward(
            group,
            q,
            k,
            v,
            head_num=head_num,
            layout=layout,
            softmax_scale=softmax_scale,
            causal=causal,
            attn_type=attn_type,
            attn_processor=attn_processor,
            attn_layer=attn_layer,
            joint_tensor_key=joint_tensor_key,
            joint_tensor_value=joint_tensor_value,
            joint_strategy=joint_strategy,
        )
        # this should be out_padded
        ctx.save_for_backward(q, k, v, out, softmax_max, softmax_sum)
        ctx.group = group
        ctx.softmax_scale = softmax_scale
        ctx.head_num = head_num
        ctx.layout = layout
        ctx.causal = causal
        ctx.attn_type = attn_type
        ctx.attn_processor = attn_processor
        return out if not return_softmax else (out, (softmax_max, softmax_sum), None)

    @staticmethod
    def backward(ctx, dout, *args):
        from yunchang.ring.ring_npu_flash_attn import ring_npu_flash_attn_backward

        q, k, v, out, softmax_max, softmax_sum = ctx.saved_tensors
        dq, dk, dv = ring_npu_flash_attn_backward(
            ctx.group,
            q, k, v,
            dout,
            ctx.head_num,
            ctx.layout,
            softmax_max,
            softmax_sum,
            out,
            ctx.softmax_scale,
            causal=ctx.causal,
            attn_type=ctx.attn_type
        )

        # Return gradients: 3 tensor gradients + 12 None values for non-tensor params
        # Order matches forward parameters:
        # dq, dk, dv, (head_num, layout, softmax_scale, causal,
        #              return_softmax, group,
        #              attn_type, attn_processor, attn_layer,
        #              joint_tensor_key, joint_tensor_value, joint_strategy)
        return (
            dq, dk, dv,  # Gradients for q, k, v
            None, None, None, None,  # head_num, layout, softmax_scale, causal
            None, None,              # return_softmax, group
            None, None, None,        # attn_type, attn_processor, attn_layer
            None, None, None,        # joint_tensor_key, joint_tensor_value, joint_strategy
        )


def xdit_ring_npu_flash_attn_func(
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
        attn_type=AttnType.NPU,
        attn_processor=None,
        attn_layer=None,
        joint_tensor_key=None,
        joint_tensor_value=None,
        joint_strategy="none",
        q_descale=None,
        k_descale=None,
        v_descale=None,
):
    head_num = q.shape[-2]
    layout = "BSND"
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(q.size(-1))
    return xFuserRingNpuFlashAttnFunc.apply(
        q,
        k,
        v,
        head_num,
        layout,
        softmax_scale,
        causal,
        return_attn_probs,
        group,
        attn_type,
        attn_processor,
        attn_layer,
        joint_tensor_key,
        joint_tensor_value,
        joint_strategy,
    )
