from typing import Any
import torch
from torch import Tensor

from yunchang import LongContextAttention
from yunchang.comm.all_to_all import SeqAllToAll4D

from xfuser.distributed import get_runtime_state


class xFuserLongContextAttention(LongContextAttention):
    """Initialization.

    Arguments:
        ulysses_pg (ProcessGroup): ulysses process group
        ring_pg (ProcessGroup): ring process group
        scatter_idx (int): scatter_idx for all2all comm
        gather_idx (int): gather_idx for all2all comm
    """

    def __init__(
        self,
        scatter_idx: int = 2,
        gather_idx: int = 1,
        ring_impl_type: str = "basic",
        use_pack_qkv: bool = False,
        use_kv_cache: bool = False,
    ) -> None:
        super().__init__(
            scatter_idx=scatter_idx,
            gather_idx=gather_idx,
            ring_impl_type=ring_impl_type,
            use_pack_qkv=use_pack_qkv,
        )

        super(LongContextAttention, self).__init__()
        self.use_kv_cache = use_kv_cache
        self.kv_cache = None
        

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        dropout_p=0.0,
        softmax_scale=None,
        causal=False,
        window_size=(-1, -1),
        alibi_slopes=None,
        deterministic=False,
        return_attn_probs=False,
        *args: Any,
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

        # 3 X (bs, seq_len/N, head_cnt, head_size) -> 3 X (bs, seq_len, head_cnt/N, head_size)
        # scatter 2, gather 1
        if self.use_pack_qkv:
            # (3*bs, seq_len/N, head_cnt, head_size)
            qkv = torch.cat([query, key, value]).continous()
            # (3*bs, seq_len, head_cnt/N, head_size)
            qkv = SeqAllToAll4D.apply(
                self.ulysses_pg, qkv, self.scatter_idx, self.gather_idx
            )
            qkv = torch.chunk(qkv, 3, dim=0)
            query_layer, key_layer, value_layer = qkv

        else:
            query_layer = SeqAllToAll4D.apply(
                self.ulysses_pg, query, self.scatter_idx, self.gather_idx
            )
            key_layer = SeqAllToAll4D.apply(
                self.ulysses_pg, key, self.scatter_idx, self.gather_idx
            )
            value_layer = SeqAllToAll4D.apply(
                self.ulysses_pg, value, self.scatter_idx, self.gather_idx
            )

        if self.use_kv_cache:
            ulysses_world_size = torch.distributed.get_world_size(self.ulysses_pg)
            pp_patch_token_num = get_runtime_state().pp_patches_token_num
            if not get_runtime_state().patch_mode:
                key_list = [
                    key.split(pp_patch_token_num, dim=1)
                    for key in torch.chunk(key_layer, ulysses_world_size, dim=1)
                ]
                value_list = [
                    value.split(pp_patch_token_num, dim=1)
                    for value in torch.chunk(value_layer, ulysses_world_size, dim=1)
                ]
                cached_key = [
                    key_list[rank][pp_patch_idx]
                    for rank in range(ulysses_world_size)
                    for pp_patch_idx in range(len(pp_patch_token_num))
                ]
                cached_value = [
                    value_list[rank][pp_patch_idx]
                    for rank in range(ulysses_world_size)
                    for pp_patch_idx in range(len(pp_patch_token_num))
                ]
                self.kv_cache = [
                    torch.cat(cached_key, dim=1),
                    torch.cat(cached_value, dim=1)
                ]
            else:
                if self.kv_cache is None:
                    raise ValueError(
                        "xFuserLongContextAttention kvcache is None in patch mode"
                    )
                cached_key, cached_value = self.kv_cache
                token_start_idx = ulysses_world_size * sum(pp_patch_token_num[:get_runtime_state().pipeline_patch_idx])
                token_end_idx = ulysses_world_size * sum(pp_patch_token_num[:get_runtime_state().pipeline_patch_idx + 1])
                cached_key[:, token_start_idx:token_end_idx, ...] = key_layer
                cached_value[:, token_start_idx:token_end_idx, ...] = value_layer
                self.kv_cache = [cached_key, cached_value]

            out = self.ring_attn_fn(
                query_layer,
                self.kv_cache[0],
                self.kv_cache[1],
                dropout_p=dropout_p,
                softmax_scale=softmax_scale,
                causal=causal,
                window_size=window_size,
                alibi_slopes=alibi_slopes,
                deterministic=deterministic,
                return_attn_probs=return_attn_probs,
                group=self.ring_pg,
            )
        else:
            out = self.ring_attn_fn(
                query_layer,
                key_layer,
                value_layer,
                dropout_p=dropout_p,
                softmax_scale=softmax_scale,
                causal=causal,
                window_size=window_size,
                alibi_slopes=alibi_slopes,
                deterministic=deterministic,
                return_attn_probs=return_attn_probs,
                group=self.ring_pg,
            )

        if type(out) == tuple:
            context_layer, _, _ = out
        else:
            context_layer = out

        # (bs, seq_len, head_cnt/N, head_size) -> (bs, seq_len/N, head_cnt, head_size)
        # scatter 1, gather 2
        output = SeqAllToAll4D.apply(
            self.ulysses_pg, context_layer, self.gather_idx, self.scatter_idx
        )

        # out e.g., [s/p::h]
        return output