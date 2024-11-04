import unittest
import torch
import torch.distributed as dist
from xfuser.core.long_ctx_attention.ring.ring_flash_attn import ring_flash_attn_func
from flash_attn import flash_attn_func
import os

def init_dist(backend='nccl'):
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    print(f"Initializing distributed environment with rank {rank}, world size {world_size}, local rank {local_rank}")
    
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend=backend)
    return rank, world_size

class TestRingFlashAttn(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.batch_size = 1
        cls.num_heads = 4
        cls.head_dim = 32
        cls.seq_len = 128
        cls.dtype = torch.float16

        
        cls.rank, cls.world_size = init_dist()
        cls.device = torch.device(f'cuda:{cls.rank}')

    def setUp(self):
        torch.manual_seed(42 + self.rank)
        
    @classmethod
    def tearDownClass(cls):
        dist.destroy_process_group()

    def _create_test_tensors(self):
        """Helper to create test input tensors"""
        shape = (self.batch_size, self.seq_len, self.num_heads, self.head_dim)

        # Prepare inputs
        q = torch.randn(
            shape, device=self.device, dtype=self.dtype, requires_grad=False
        )
        k = torch.randn(
            shape, device=self.device, dtype=self.dtype, requires_grad=True
        )
        v = torch.randn(
            shape, device=self.device, dtype=self.dtype, requires_grad=True
        )

        dist.broadcast(q, src=0)
        dist.broadcast(k, src=0)
        dist.broadcast(v, src=0)

        local_q = q.chunk(self.world_size, dim=1)[self.rank]
        local_k = k.chunk(self.world_size, dim=1)[self.rank]
        local_v = v.chunk(self.world_size, dim=1)[self.rank]
        return q, k, v, local_q, local_k, local_v
    
    def test_distributed(self):
        """Test ring flash attention in distributed mode"""
        q, k, v, local_q, local_k, local_v = self._create_test_tensors()

        # Run regular flash attention for reference
        ref_output = flash_attn_func(
            q, k, v,
            dropout_p=0.0,
            causal=True,
            window_size=(-1, -1),
        )
        ref_output = ref_output.chunk(self.world_size, dim=1)[self.rank]

        # Run ring flash attention
        output = ring_flash_attn_func(
            q=local_q,
            k=local_k,
            v=local_v,
            dropout_p=0.0,
            causal=True,
            window_size=(-1, -1),
            group=dist.group.WORLD
        )
        # Compare results
        torch.testing.assert_close(ref_output, output, rtol=1e-3, atol=1e-3)
        self.assertEqual(ref_output.shape, output.shape)


    def test_joint_strategy_rear(self):
        """Test ring flash attention with joint strategy"""
        q, k, v, local_q, local_k, local_v = self._create_test_tensors()
        joint_q, joint_k, joint_v, local_joint_q, local_joint_k, local_joint_v = self._create_test_tensors()

        # [q, q], [v, v] as input to flash_attn_func
        ref_output = flash_attn_func(
            q, 
            torch.cat([k, joint_k], dim=1), 
            torch.cat([v, joint_v], dim=1),
            dropout_p=0.0,
            causal=False,
            window_size=(-1, -1),
        )
        ref_output = ref_output.chunk(self.world_size, dim=1)[self.rank]

        # Test front joint strategy
        output_rear = ring_flash_attn_func(
            q=local_q,
            k=local_k,
            v=local_v,
            dropout_p=0.0,
            causal=False,
            window_size=(-1, -1),
            joint_tensor_key=joint_k,
            joint_tensor_value=joint_v,
            joint_strategy="rear"
        )

        torch.testing.assert_close(ref_output, output_rear, rtol=1e-3, atol=1e-3)


    def test_joint_strategy_front(self):
        """Test ring flash attention with joint strategy"""
        q, k, v, local_q, local_k, local_v = self._create_test_tensors()
        joint_q, joint_k, joint_v, local_joint_q, local_joint_k, local_joint_v = self._create_test_tensors()

        # [q, q], [v, v] as input to flash_attn_func
        ref_output = flash_attn_func(
            q, 
            torch.cat([joint_k, k], dim=1), 
            torch.cat([joint_v, v], dim=1),
            dropout_p=0.0,
            causal=False,
            window_size=(-1, -1),
        )
        ref_output = ref_output.chunk(self.world_size, dim=1)[self.rank]

        # Test front joint strategy
        output_front = ring_flash_attn_func(
            q=local_q,
            k=local_k,
            v=local_v,
            dropout_p=0.0,
            causal=False,
            window_size=(-1, -1),
            joint_tensor_key=joint_k,
            joint_tensor_value=joint_v,
            joint_strategy="front"
        )

        torch.testing.assert_close(ref_output, output_front, rtol=1e-3, atol=1e-3)

# torchrun --nproc_per_node=2 -m unittest tests/core/test_ring_flash_attn.py
if __name__ == '__main__':
    unittest.main() 