from typing import Any

import torch
import torch.distributed as dist
from torch import Tensor

from xfuser.core.cache_manager.cache_manager import get_cache_manager
from yunchang import UlyssesAttention
from yunchang.globals import PROCESS_GROUP
from yunchang.comm.all_to_all import SeqAllToAll4D

from packaging.version import Version
from yunchang import __version__
if Version(__version__) < Version("0.4.0"):
    from yunchang.ulysses.attn_layer import torch_attn
else:
    from yunchang.kernels.attention import torch_attn


class xFuserUlyssesAttention(UlyssesAttention):
    def __init__(
        self,
        scatter_idx: int = 2,
        gather_idx: int = 1,
        use_fa: bool = True,
        use_kv_cache: bool = True,
    ) -> None:

        super(UlyssesAttention, self).__init__()
        self.ulysses_pg = PROCESS_GROUP.ULYSSES_PG

        self.scatter_idx = scatter_idx
        self.gather_idx = gather_idx
        self.use_fa = use_fa
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        gpu_name = torch.cuda.get_device_name(device)
        if "Turing" in gpu_name or "Tesla" in gpu_name or "T4" in gpu_name:
            self.use_fa = False
        self.use_kv_cache = use_kv_cache

        if self.use_fa:
            from flash_attn import flash_attn_func

            self.fn = flash_attn_func
        else:
            self.fn = torch_attn

    def forward(
        self,
        attn,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        *,
        joint_tensor_query=None,
        joint_tensor_key=None,
        joint_tensor_value=None,
        dropout_p=0.0,
        softmax_scale=None,
        causal=False,
        window_size=(-1, -1),
        alibi_slopes=None,
        deterministic=False,
        return_attn_probs=False,
        joint_strategy="none",
    ) -> Tensor:
        """forward

        Arguments:
            query (Tensor): query input to the layer
            key (Tensor): key input to the layer
            value (Tensor): value input to the layer
            args: other args

        Returns:
            * output (Tensor): context output
        """
        if (
            joint_tensor_key is not None
            and joint_tensor_value is not None
            and joint_tensor_query is not None
        ):
            if joint_strategy == "rear":
                query = torch.cat([query, joint_tensor_query], dim=1)
            elif joint_strategy == "front":
                query = torch.cat([joint_tensor_query, query], dim=1)
            elif joint_strategy == "none":
                raise ValueError(
                    f"joint_strategy: {joint_strategy} not supported when joint tensors is not None."
                )
            else:
                raise ValueError(f"joint_strategy: {joint_strategy} not supported.")
            ulysses_world_size = torch.distributed.get_world_size(self.ulysses_pg)
            ulysses_rank = torch.distributed.get_rank(self.ulysses_pg)
            attn_heads_per_ulysses_rank = (
                joint_tensor_key.shape[-2] // ulysses_world_size
            )
            joint_tensor_key = joint_tensor_key[
                ...,
                attn_heads_per_ulysses_rank
                * ulysses_rank : attn_heads_per_ulysses_rank
                * (ulysses_rank + 1),
                :,
            ]
            joint_tensor_value = joint_tensor_value[
                ...,
                attn_heads_per_ulysses_rank
                * ulysses_rank : attn_heads_per_ulysses_rank
                * (ulysses_rank + 1),
                :,
            ]

        # TODO Merge three alltoall calls into one
        # TODO (Reza): change the api on the megatron-deepspeed side so that we only receive all data (q,k, and v) together!
        # in shape : e.g.,  [s/p:h:]
        # (bs, seq_len/N, head_cnt, head_size) -> (bs, seq_len, head_cnt/N, head_size)

        # scatter 2, gather 1
        q = SeqAllToAll4D.apply(
            self.ulysses_pg, query, self.scatter_idx, self.gather_idx
        )
        k = SeqAllToAll4D.apply(self.ulysses_pg, key, self.scatter_idx, self.gather_idx)
        v = SeqAllToAll4D.apply(
            self.ulysses_pg, value, self.scatter_idx, self.gather_idx
        )

        if self.use_kv_cache:
            k, v = get_cache_manager().update_and_get_kv_cache(
                new_kv=[k, v],
                layer=attn,
                slice_dim=1,
                layer_type="attn",
            )

        if joint_strategy != "none":
            if joint_strategy == "rear":
                k = torch.cat([k, joint_tensor_key], dim=1)
                v = torch.cat([v, joint_tensor_value], dim=1)

            elif joint_strategy == "front":
                k = torch.cat([joint_tensor_key, k], dim=1)
                v = torch.cat([joint_tensor_value, v], dim=1)

        context_layer = self.fn(
            q,
            k,
            v,
            dropout_p=dropout_p,
            causal=causal,
            window_size=window_size,
            alibi_slopes=alibi_slopes,
            deterministic=deterministic,
            return_attn_probs=return_attn_probs,
        )

        if isinstance(context_layer, tuple):
            context_layer = context_layer[0]

        # (bs, seq_len, head_cnt/N, head_size) -> (bs, seq_len/N, head_cnt, head_size)
        # scatter 1, gather 2
        output = SeqAllToAll4D.apply(
            self.ulysses_pg, context_layer, self.gather_idx, self.scatter_idx
        )

        # out e.g., [s/p::h]
        return output
