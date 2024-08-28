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

        dim = 32
        num_attention_heads = 8
        qk_norm = True
        self.dtype = torch.bfloat16

        self.attn1 = (
            Attention(
                query_dim=dim,
                cross_attention_dim=None,
                dim_head=dim // num_attention_heads,
                heads=num_attention_heads,
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
                query_dim=dim,
                cross_attention_dim=None,
                dim_head=dim // num_attention_heads,
                heads=num_attention_heads,
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

        norm_hidden_states = torch.randn(1, 20, 32, dtype=self.dtype).cuda(
            self.local_rank
        )
        image_rotary_emb = torch.randn(1, 20, 32, dtype=self.dtype).cuda(
            self.local_rank
        )

        attn_output1 = self.attn1(norm_hidden_states, image_rotary_emb)
        attn_output1_shard = torch.chunk(attn_output1, self.world_size, dim=1)[
            self.local_rank
        ]

        norm_hidden_states_shard = torch.chunk(
            norm_hidden_states, self.world_size, dim=1
        )[self.local_rank]
        image_rotary_emb_shard = torch.chunk(image_rotary_emb, self.world_size, dim=1)[
            self.local_rank
        ]

        attn_output2_shard = self.attn2(
            norm_hidden_states_shard, image_rotary_emb_shard
        )

        # check if the outputs are close
        print(attn_output1_shard - attn_output2_shard)
        self.assertTrue(
            torch.allclose(attn_output1_shard, attn_output2_shard, atol=1e-2)
        )


# torchrun --nproc_per_node=4 ./tests/layers/hunyuandit_attn_test.py
if __name__ == "__main__":
    unittest.main()
