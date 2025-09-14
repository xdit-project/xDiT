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


class TestUSP(unittest.TestCase):

    def setUp(self):
        self._init_environment()
        env_info = PACKAGES_CHECKER.get_packages_info()
        self.HAS_FLASH_ATTN = env_info["has_flash_attn"]
        self.HAS_AITER = env_info["has_aiter"]
        self.query = torch.randn(29760, 2, 5, 128, device="cuda", dtype=torch.bfloat16)
        self.key = torch.randn(29760, 2, 5, 128, device="cuda", dtype=torch.bfloat16)
        self.value = torch.randn(29760, 2, 5, 128, device="cuda", dtype=torch.bfloat16)


    def _init_environment(self):
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"
        os.environ["LOCAL_RANK"] = "0"
        init_distributed_environment(rank=0, world_size=1)
        initialize_model_parallel(ring_degree=1, ulysses_degree=1)



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