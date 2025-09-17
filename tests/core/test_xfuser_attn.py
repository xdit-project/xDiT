import unittest
from itertools import product
import torch
import torch.distributed as dist
from xfuser.core.long_ctx_attention.ring.ring_flash_attn import (
    xdit_ring_flash_attn_func,
)
from xfuser.core.long_ctx_attention import xFuserLongContextAttention
from flash_attn import flash_attn_func
import os

from xfuser.model_executor.layers.attention_processor import (
    xFuserAttnProcessor2_0,
)
from diffusers.models.attention_processor import (
    Attention,
)
from xfuser.core.distributed import (
    init_distributed_environment,
    initialize_model_parallel,
)
from xfuser.envs import PACKAGES_CHECKER

from yunchang.kernels import AttnType

def init_dist(backend="nccl"):
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    print(
        f"Initializing distributed environment with rank {rank}, world size {world_size}, local rank {local_rank}"
    )

    torch.cuda.set_device(local_rank)
    # dist.init_process_group(backend=backend)
    init_distributed_environment(rank=rank, world_size=world_size)

    # construct a hybrid sequence parallel config (ulysses=2, ring = world_size // 2)
    if world_size > 1:
        ring_degree = world_size // 2
        ulysses_degree = 2
    else:
        ring_degree = 1
        ulysses_degree = 1

    initialize_model_parallel(
        ring_degree=ring_degree,
        ulysses_degree=ulysses_degree,
    )

    return rank, world_size, ring_degree, ulysses_degree


class TestRingFlashAttn(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        env_info = PACKAGES_CHECKER.get_packages_info()

        cls.batch_size = 1
        cls.num_heads = 4
        cls.head_dim = 32
        cls.seq_len = 128

        cls.rank, cls.world_size, cls.ring_degree, cls.ulysses_degree = init_dist()
        cls.device = torch.device(f"cuda:{cls.rank}")

        cls.HAS_AITER = env_info["has_aiter"]
        cls.HAS_FLASH_ATTENTION = env_info["has_flash_attn"]

    def setUp(self):
        torch.manual_seed(42 + self.rank)

    @classmethod
    def tearDownClass(cls):
        dist.destroy_process_group()

    def _create_test_tensors(self, dtype=torch.float16):
        """Helper to create test input tensors"""
        shape = (self.batch_size, self.seq_len, self.num_heads, self.head_dim)

        # Prepare inputs
        q = torch.randn(
            shape, device=self.device, dtype=dtype, requires_grad=False
        )
        k = torch.randn(
            shape, device=self.device, dtype=dtype, requires_grad=False
        )
        v = torch.randn(
            shape, device=self.device, dtype=dtype, requires_grad=False
        )

        dist.broadcast(q, src=0)
        dist.broadcast(k, src=0)
        dist.broadcast(v, src=0)

        local_q = q.chunk(self.world_size, dim=1)[self.rank]
        local_k = k.chunk(self.world_size, dim=1)[self.rank]
        local_v = v.chunk(self.world_size, dim=1)[self.rank]
        return q, k, v, local_q, local_k, local_v

    def test_xfuser_attn_layer_joint_strategy_rear(self):
        """Test xFuserLongContextAttention layer in distributed mode"""

        backends = ["AITER", "FA"]
        dtypes = [torch.float16, torch.bfloat16]

        for backend, dtype in product(backends, dtypes):
            with self.subTest(backend=backend, dtype=dtype):

                if backend=="AITER":
                    if self.HAS_AITER and 'AITER' in AttnType.__members__:
                        attn_type=AttnType.AITER
                    else:
                        self.skipTest("AITER backend not applicable")
                if backend=="FA":
                    if self.HAS_FLASH_ATTENTION:
                        attn_type=AttnType.FA
                    else:
                        self.skipTest("FA backend not applicable")

                # Create test tensors
                q, k, v, local_q, local_k, local_v = self._create_test_tensors(dtype=dtype)
                joint_q, joint_k, joint_v, local_joint_q, local_joint_k, local_joint_v = (
                    self._create_test_tensors(dtype=dtype)
                )
                joint_strategy = "rear"

                attn = None

                # Create attention layer
                attn_layer = xFuserLongContextAttention(
                    scatter_idx=2,
                    gather_idx=1,
                    ring_impl_type="basic",
                    attn_type=attn_type,
                    use_kv_cache=False,
                ).to(device=self.device, dtype=torch.float16)

                assert attn_layer.ring_pg.size() == self.ring_degree
                assert attn_layer.ulysses_pg.size() == self.ulysses_degree

                ref_output = flash_attn_func(
                    torch.cat([q, joint_q], dim=1),
                    torch.cat([k, joint_k], dim=1),
                    torch.cat([v, joint_v], dim=1),
                    dropout_p=0.0,
                    window_size=(-1, -1),
                )

                # Split ref_output into base and joint parts
                base_out = ref_output[:, : self.seq_len, ::]  # First half for base attention
                joint_out = ref_output[:, self.seq_len :, ::]  # Second half for joint attention

                # Get local shard for base output
                base_out_shard = base_out.chunk(self.world_size, dim=1)[self.rank]
                # Duplicate joint output as specified
                ref_output = torch.cat([base_out_shard, joint_out], dim=1)

                # Run distributed implementation
                output = attn_layer(
                    attn=None,
                    query=local_q,
                    key=local_k,
                    value=local_v,
                    dropout_p=0.0,
                    window_size=(-1, -1),
                    joint_tensor_query=joint_q,
                    joint_tensor_key=joint_k,
                    joint_tensor_value=joint_v,
                    joint_strategy=joint_strategy,
                )
                # assert torch.max(torch.abs(output - ref_output)) < 1e-3
                torch.testing.assert_close(ref_output, output, rtol=1e-2, atol=1e-2)

    def test_xfuser_attn_layer(self):
        """Test xFuserLongContextAttention layer in distributed mode"""
     
        backends = ["AITER", "FA"]
        dtypes = [torch.float16, torch.bfloat16]

        for backend, dtype in product(backends, dtypes):
            with self.subTest(backend=backend, dtype=dtype):

                if backend=="AITER":
                    if self.HAS_AITER and 'AITER' in AttnType.__members__:
                        attn_type=AttnType.AITER
                    else:
                        self.skipTest("AITER backend not applicable")
                if backend=="FA":
                    if self.HAS_FLASH_ATTENTION:
                        attn_type=AttnType.FA
                    else:
                        self.skipTest("FA backend not applicable")

                # Create test tensors
                q, k, v, local_q, local_k, local_v = self._create_test_tensors(dtype)

                # Create attention layer
                attn_layer = xFuserLongContextAttention(
                    scatter_idx=2,
                    gather_idx=1,
                    ring_impl_type="basic",
                    use_kv_cache=False,
                    attn_type=attn_type,
                ).to(device=self.device, dtype=dtype)

                assert attn_layer.ring_pg.size() == self.ring_degree
                assert attn_layer.ulysses_pg.size() == self.ulysses_degree

                ref_output = flash_attn_func(
                    q,
                    k,
                    v,
                    dropout_p=0.0,
                    window_size=(-1, -1),
                )
                ref_output = ref_output.chunk(self.world_size, dim=1)[self.rank]

                # Run distributed implementation
                output = attn_layer(
                    attn=None,
                    query=local_q,
                    key=local_k,
                    value=local_v,
                    dropout_p=0.0,
                    window_size=(-1, -1),
                )

                assert torch.max(torch.abs(output - ref_output)) < 1e-2
                torch.testing.assert_close(ref_output, output, rtol=1e-2, atol=1e-2)


# torchrun --nproc_per_node=4 -m unittest tests/core/test_xfuser_attn.py
if __name__ == "__main__":
    unittest.main()
