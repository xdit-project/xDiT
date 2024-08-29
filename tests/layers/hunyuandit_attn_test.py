from diffusers.models.attention_processor import (
    Attention,
    AttentionProcessor,
    FusedHunyuanAttnProcessor2_0,
    HunyuanAttnProcessor2_0,
)
from xfuser.core.cache_manager.cache_manager import get_cache_manager
from xfuser.core.distributed.runtime_state import initialize_runtime_state
from xfuser.model_executor.layers.attention_processor import (
    xFuserHunyuanAttnProcessor2_0,
)

import torch
import unittest
from xfuser.core.distributed import (
    init_distributed_environment,
    initialize_model_parallel,
)
from torch import distributed as dist
import copy


class TestHunyuanDiTAttention(unittest.TestCase):
    def setUp(self):
        init_distributed_environment()

        self.world_size = dist.get_world_size()
        self.local_rank = dist.get_rank()
        initialize_model_parallel(sequence_parallel_degree=self.world_size)
        initialize_runtime_state(None, None)

        self.dim = 32
        self.sequence_len = 20
        self.num_attention_heads = 8
        self.hidden_dim = self.dim // self.num_attention_heads
        qk_norm = True
        self.dtype = torch.bfloat16

        self.attn1 = (
            Attention(
                query_dim=self.dim,
                cross_attention_dim=None,
                dim_head=self.dim // self.num_attention_heads,
                heads=self.num_attention_heads,
                qk_norm="layer_norm" if qk_norm else None,
                eps=1e-6,
                bias=True,
                processor=HunyuanAttnProcessor2_0(),
            )
            .cuda(self.local_rank)
            .to(self.dtype)
        )

        self.attn2 = (
            Attention(
                query_dim=self.dim,
                cross_attention_dim=None,
                dim_head=self.dim // self.num_attention_heads,
                heads=self.num_attention_heads,
                qk_norm="layer_norm" if qk_norm else None,
                eps=1e-6,
                bias=True,
                processor=xFuserHunyuanAttnProcessor2_0(),
            )
            .cuda(self.local_rank)
            .to(self.dtype)
        )

        for p1, p2 in zip(self.attn1.parameters(), self.attn2.parameters()):
            p2.data.copy_(p1.data)

        get_cache_manager().register_cache_entry(
            self.attn2, layer_type="attn", cache_type="sequence_parallel_attn_cache"
        )

    def test_hunyuandit_attn(self):
        torch.manual_seed(0)

        # prepare inputs
        norm_hidden_states = torch.randn(
            1, self.sequence_len, self.dim, dtype=self.dtype
        ).cuda(self.local_rank)
        dist.broadcast(norm_hidden_states, 0)

        use_image_rotary_emb = False

        # prepare rope parameters
        cos = torch.randn(self.sequence_len, self.hidden_dim, dtype=self.dtype).cuda(
            self.local_rank
        )
        dist.broadcast(cos, 0)
        sin = torch.randn(self.sequence_len, self.hidden_dim, dtype=self.dtype).cuda(
            self.local_rank
        )
        dist.broadcast(sin, 0)
        image_rotary_emb = cos, sin
        cos_shard = torch.chunk(cos, self.world_size, dim=0)[self.local_rank]
        sin_shared = torch.chunk(sin, self.world_size, dim=0)[self.local_rank]
        image_rotary_emb_shard = cos_shard, sin_shared

        attn_output1 = self.attn1(
            norm_hidden_states,
            encoder_hidden_states=None,
            image_rotary_emb=image_rotary_emb if use_image_rotary_emb else None,
        )
        attn_output1_shard = torch.chunk(attn_output1, self.world_size, dim=1)[
            self.local_rank
        ]

        norm_hidden_states_shard = torch.chunk(
            norm_hidden_states, self.world_size, dim=1
        )[self.local_rank]

        attn_output2_shard = self.attn2(
            norm_hidden_states_shard,
            encoder_hidden_states=None,
            image_rotary_emb=image_rotary_emb_shard if use_image_rotary_emb else None,
        )

        # check if the outputs are close

        if self.local_rank == 0:
            print(attn_output1_shard - attn_output2_shard)
            self.assertTrue(
                torch.allclose(attn_output1_shard, attn_output2_shard, atol=1e-2)
            )


# torchrun --nproc_per_node=4 ./tests/layers/hunyuandit_attn_test.py
if __name__ == "__main__":
    unittest.main()
