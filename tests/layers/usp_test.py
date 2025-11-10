import os
import torch
import unittest
import importlib
from xfuser.envs import PACKAGES_CHECKER
from xfuser.model_executor.layers import usp

from xfuser.core.distributed import (
    init_distributed_environment,
    initialize_model_parallel,
    get_runtime_state,
)
from xfuser.core.distributed.parallel_state import destroy_model_parallel, destroy_distributed_environment



def _init_environment():
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    os.environ["LOCAL_RANK"] = "0"
    init_distributed_environment(rank=0, world_size=1)
    initialize_model_parallel(ring_degree=1, ulysses_degree=1)

class TestUSP(unittest.TestCase):

    def setUp(self):
        _init_environment()
        env_info = PACKAGES_CHECKER.get_packages_info()
        self.HAS_FLASH_ATTN = env_info["has_flash_attn"]
        self.HAS_AITER = env_info["has_aiter"]
        self.query = torch.randn(1, 24, 14867, 128, device="cuda", dtype=torch.bfloat16)
        self.key = torch.randn(1, 24, 14867, 128, device="cuda", dtype=torch.bfloat16)
        self.value = torch.randn(1, 24, 14867, 128, device="cuda", dtype=torch.bfloat16)

    def tearDown(self):
        destroy_model_parallel()
        destroy_distributed_environment()


    def test_usp_flash_attn(self):
        """
        Verifies USP results with flash_attn are close to F.SDPA results
        """
        if not self.HAS_FLASH_ATTN:
            self.skipTest("flash_attn library is not available in the environment.")

        # Disabling flash_attn and aiter to get SDPA results
        usp.HAS_FLASH_ATTN = False
        usp.HAS_AITER = False
        fsdpa_results = usp.USP(self.query, self.key, self.value, dropout_p=0.0, is_causal=False)

        usp.HAS_FLASH_ATTN = True
        flash_attn_results = usp.USP(self.query, self.key, self.value, dropout_p=0.0, is_causal=False)

        result_diff = (fsdpa_results - flash_attn_results).abs().max()
        self.assertNotEqual(result_diff, 0) # Different implementations won't produce same output
        self.assertAlmostEqual(result_diff.item(), 0, places=1) # Difference can be 0.15ish

    def test_ring_attn_flash_attn(self):
        """
        Verifies ring_attn results with flash_attn are close to F.SDPA results

        Ring_attn function is called through the USP function when using ring attention, but that requires
        multi-GPU parallelization, which this test is not using. Therefore the function is called
        directly to test its output.
        """
        if not self.HAS_FLASH_ATTN:
            self.skipTest("flash_attn library is not available in the environment.")

        # Disabling flash_attn and aiter to get SDPA results
        usp.HAS_FLASH_ATTN = False
        usp.HAS_AITER = False
        fsdpa_results = usp.ring_attn(self.query, self.key, self.value, dropout_p=0.0, is_causal=False)

        usp.HAS_FLASH_ATTN = True
        flash_attn_results = usp.ring_attn(self.query, self.key, self.value, dropout_p=0.0, is_causal=False)

        result_diff = (fsdpa_results - flash_attn_results).abs().max()
        self.assertNotEqual(result_diff, 0) # Different implementations won't produce same output
        self.assertAlmostEqual(result_diff.item(), 0, places=1) # Difference can be 0.15ish

    def test_usp_aiter(self):
        """
        Verifies USP results with aiter are close to F.SDPA results
        """
        if not self.HAS_AITER:
            self.skipTest("aiter library is not available in the environment.")

        # Disabling flash_attn and aiter to get SDPA results
        usp.HAS_FLASH_ATTN = False
        usp.HAS_AITER = False
        fsdpa_results = usp.USP(self.query, self.key, self.value, dropout_p=0.0, is_causal=False)

        usp.HAS_AITER = True
        aiter_attn_results = usp.USP(self.query, self.key, self.value, dropout_p=0.0, is_causal=False)

        result_diff = (fsdpa_results - aiter_attn_results).abs().max()
        self.assertNotEqual(result_diff, 0) # Different implementations won't produce same output
        self.assertAlmostEqual(result_diff.item(), 0, places=1) # Difference can be 0.15ish

    def test_ring_attn_aiter(self):
        """
        Verifies ring_attn results with aiter are close to F.SDPA results

        Ring_attn function is called through the USP function when using ring attention, but that requires
        multi-GPU parallelization, which this test is not using. Therefore the function is called
        directly to test its output.
        """
        if not self.HAS_AITER:
            self.skipTest("aiter library is not available in the environment.")

        # Disabling flash_attn and aiter to get SDPA results
        usp.HAS_FLASH_ATTN = False
        usp.HAS_AITER = False
        fsdpa_results = usp.ring_attn(self.query, self.key, self.value, dropout_p=0.0, is_causal=False)

        usp.HAS_AITER = True
        aiter_attn_results = usp.ring_attn(self.query, self.key, self.value, dropout_p=0.0, is_causal=False)

        result_diff = (fsdpa_results - aiter_attn_results).abs().max()
        self.assertNotEqual(result_diff, 0) # Different implementations won't produce same output
        self.assertAlmostEqual(result_diff.item(), 0, places=1) # Difference can be 0.15ish


class TestUSPHybridParallel(unittest.TestCase):

    def setUp(self):
        _init_environment()
        # Using SDPA here
        self.HAS_FLASH_ATTN = False
        self.HAS_AITER = False
        self.query = torch.randn(1, 24, 14867, 128, device="cuda", dtype=torch.bfloat16)
        self.key = torch.randn(1, 24, 14867, 128, device="cuda", dtype=torch.bfloat16)
        self.value = torch.randn(1, 24, 14867, 128, device="cuda", dtype=torch.bfloat16)

        ## Importing within test case for compatibility
        from xfuser.core.long_ctx_attention import xFuserLongContextAttention
        from yunchang.kernels import AttnType

        self.hybrid_seq_parallel_attn = xFuserLongContextAttention(
            attn_type=AttnType.TORCH
        )

    def tearDown(self):
        destroy_model_parallel()
        destroy_distributed_environment()

    def test_usp_hybrid_equivalence(self):
        """
        Tests the output from USP is equivalent to hybrid seq parallel attention, i.e
        yunchang path
        """
        usp_results = usp.USP(self.query, self.key, self.value, dropout_p=0.0, is_causal=False)
        hybrid_results = self.hybrid_seq_parallel_attn(
            None,
            self.query.transpose(1, 2),
            self.key.transpose(1, 2),
            self.value.transpose(1, 2),
            dropout_p=0.0,
            causal=False
        ).transpose(1, 2)

        result_diff = (usp_results - hybrid_results).abs().max().float().cpu().numpy()
        self.assertAlmostEqual(result_diff, 0, places=1)

    def test_usp_hybrid_joint_equivalence(self):
        """
        Tests the output from USP with joint tensors added is equivalent to hybrid seq
        parallel attn.
        """
        joint_shape = (1, 24, 64, 128)

        joint_query = torch.randn(joint_shape, device="cuda", dtype=torch.bfloat16)
        joint_key = torch.randn(joint_shape, device="cuda", dtype=torch.bfloat16)
        joint_value = torch.randn(joint_shape, device="cuda", dtype=torch.bfloat16)

        usp_results = usp.USP(
            self.query,
            self.key,
            self.value,
            dropout_p=0.0,
            is_causal=False,
            joint_query=joint_query,
            joint_key=joint_key,
            joint_value=joint_value,
            joint_strategy="rear"
        )
        hybrid_results = self.hybrid_seq_parallel_attn(
            None,
            self.query.transpose(1, 2),
            self.key.transpose(1, 2),
            self.value.transpose(1, 2),
            dropout_p=0.0,
            causal=False,
            joint_tensor_query=joint_query.transpose(1, 2),
            joint_tensor_key=joint_key.transpose(1, 2),
            joint_tensor_value=joint_value.transpose(1, 2),
            joint_strategy="rear"
        ).transpose(1, 2)

        result_diff = (usp_results - hybrid_results).abs().max().float().cpu().numpy()
        self.assertAlmostEqual(result_diff, 0, places=1)