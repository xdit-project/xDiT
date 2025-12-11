import os
import torch
import unittest
import importlib
from xfuser.envs import PACKAGES_CHECKER, _is_hip
from xfuser.model_executor.layers import usp

from xfuser.core.distributed import (
    init_distributed_environment,
    initialize_model_parallel,
    get_runtime_state,
    initialize_runtime_state,
)
from xfuser.core.distributed.parallel_state import destroy_model_parallel, destroy_distributed_environment
from xfuser.core.long_ctx_attention import xFuserLongContextAttention
from yunchang.kernels import AttnType


def _init_environment():
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    os.environ["LOCAL_RANK"] = "0"
    init_distributed_environment(rank=0, world_size=1)
    initialize_runtime_state()
    initialize_model_parallel(ring_degree=1, ulysses_degree=1)

class TestUSP(unittest.TestCase):

    def setUp(self):
        _init_environment()
        self.env_info = PACKAGES_CHECKER.get_packages_info()
        self.runtime_state = get_runtime_state()
        self.default_comparison_backend = "sdpa_flash"
        self.query = torch.randn(1, 24, 14867, 128, device="cuda", dtype=torch.bfloat16)
        self.key = torch.randn(1, 24, 14867, 128, device="cuda", dtype=torch.bfloat16)
        self.value = torch.randn(1, 24, 14867, 128, device="cuda", dtype=torch.bfloat16)

    def tearDown(self):
        destroy_model_parallel()
        destroy_distributed_environment()

    def _run_usp_comparison(self, attention_backend):
        self.runtime_state.set_attention_backend(self.default_comparison_backend)
        fsdpa_results = usp.USP(self.query, self.key, self.value, dropout_p=0.0, is_causal=False)

        self.runtime_state.set_attention_backend(attention_backend)
        comparison_results = usp.USP(self.query, self.key, self.value, dropout_p=0.0, is_causal=False)

        result_diff = (fsdpa_results - comparison_results).abs().max()
        return result_diff

    def _run_ring_comparison(self, attention_backend):
        self.runtime_state.set_attention_backend(self.default_comparison_backend)
        attention_function = usp._get_attention_function()
        fsdpa_results = usp.ring_attn(attention_function, self.query, self.key, self.value, dropout_p=0.0, is_causal=False)

        self.runtime_state.set_attention_backend(attention_backend)
        attention_function = usp._get_attention_function()
        comparison_results = usp.ring_attn(attention_function, self.query, self.key, self.value, dropout_p=0.0, is_causal=False)

        result_diff = (fsdpa_results - comparison_results).abs().max()
        return result_diff

    def test_usp_flash_attn(self):
        """
        Verifies USP results with flash_attn are close to F.SDPA results
        """
        if not self.env_info["has_flash_attn"]:
            self.skipTest("flash_attn library is not available in the environment.")

        result_diff = self._run_usp_comparison("flash")

        self.assertNotEqual(result_diff, 0) # Different implementations won't produce same output
        self.assertAlmostEqual(result_diff.item(), 0, places=1) # Difference can be 0.15ish

    def test_ring_attn_flash_attn(self):
        """
        Verifies ring_attn results with flash_attn are close to F.SDPA results

        Ring_attn function is called through the USP function when using ring attention, but that requires
        multi-GPU parallelization, which this test is not using. Therefore the function is called
        directly to test its output.
        """
        if not self.env_info["has_flash_attn"]:
            self.skipTest("flash_attn library is not available in the environment.")

        result_diff = self._run_ring_comparison("flash")

        self.assertNotEqual(result_diff, 0) # Different implementations won't produce same output
        self.assertAlmostEqual(result_diff.item(), 0, places=1) # Difference can be 0.15ish

    def test_usp_aiter(self):
        """
        Verifies USP results with aiter are close to F.SDPA results
        """
        if not self.env_info["has_aiter"]:
            self.skipTest("aiter library is not available in the environment.")

        result_diff = self._run_usp_comparison("aiter")

        self.assertNotEqual(result_diff, 0) # Different implementations won't produce same output
        self.assertAlmostEqual(result_diff.item(), 0, places=1) # Difference can be 0.15ish


    def test_ring_attn_aiter(self):
        """
        Verifies ring_attn results with aiter are close to F.SDPA results
        """
        if not self.env_info["has_aiter"]:
            self.skipTest("aiter library is not available in the environment.")

        result_diff = self._run_ring_comparison("aiter")

        self.assertNotEqual(result_diff, 0) # Different implementations won't produce same output
        self.assertAlmostEqual(result_diff.item(), 0, places=1) # Difference can be 0.15ish

    def test_usp_cudnn(self):
        """
        Verifies USP results with cuDNN are close to F.SDPA results
        """
        if not torch.backends.cudnn.is_available() or _is_hip():
            self.skipTest("cuDNN is not available in the environment.")

        result_diff = self._run_usp_comparison("cudnn")

        self.assertNotEqual(result_diff, 0) # Different implementations won't produce same output
        self.assertAlmostEqual(result_diff.item(), 0, places=1) # Difference can be 0.15ish

    def test_ring_cudnn(self):
        """
        Verifies ring_attn results with cuDNN are close to F.SDPA results
        """
        if not torch.backends.cudnn.is_available() or _is_hip():
            self.skipTest("cuDNN is not available in the environment.")

        result_diff = self._run_ring_comparison("cudnn")

        self.assertNotEqual(result_diff, 0) # Different implementations won't produce same output
        self.assertAlmostEqual(result_diff.item(), 0, places=1) # Difference can be 0.15ish

    def test_usp_flash3(self):
        """
        Verifies USP results with FAv3 are close to F.SDPA results
        """
        if not self.env_info["has_flash_attn_3"]:
            self.skipTest("FAv3 library is not available in the environment.")

        result_diff = self._run_usp_comparison("flash_3")

        self.assertNotEqual(result_diff, 0) # Different implementations won't produce same output
        self.assertAlmostEqual(result_diff.item(), 0, places=1) # Difference can be 0.15ish

    def test_ring_flash3(self):
        """
        Verifies ring_attn results with FAv3 are close to F.SDPA results
        """
        if not self.env_info["has_flash_attn_3"]:
            self.skipTest("FAv3 library is not available in the environment.")

        result_diff = self._run_ring_comparison("flash_3")

        self.assertNotEqual(result_diff, 0) # Different implementations won't produce same output
        self.assertAlmostEqual(result_diff.item(), 0, places=1) # Difference can be 0.15ish

    def test_usp_flash4(self):
        """
        Verifies USP results with FAv4 are close to F.SDPA results
        """
        if not self.env_info["has_flash_attn_4"]:
            self.skipTest("FAv4 library is not available in the environment.")

        result_diff = self._run_usp_comparison("flash_4")

        self.assertNotEqual(result_diff, 0) # Different implementations won't produce same output
        self.assertAlmostEqual(result_diff.item(), 0, places=1) # Difference can be 0.15ish

    def test_ring_flash4(self):
        """
        Verifies ring_attn results with FAv4 are close to F.SDPA results
        """
        if not self.env_info["has_flash_attn_4"]:
            self.skipTest("FAv4 library is not available in the environment.")

        result_diff = self._run_ring_comparison("flash_4")

        self.assertNotEqual(result_diff, 0) # Different implementations won't produce same output
        self.assertAlmostEqual(result_diff.item(), 0, places=1) # Difference can be 0.15ish



class TestUSPHybridParallel(unittest.TestCase):

    def setUp(self):
        _init_environment()
        self.runtime_state = get_runtime_state()
        self.runtime_state.set_attention_backend("sdpa_flash")
        self.query = torch.randn(1, 24, 14867, 128, device="cuda", dtype=torch.bfloat16)
        self.key = torch.randn(1, 24, 14867, 128, device="cuda", dtype=torch.bfloat16)
        self.value = torch.randn(1, 24, 14867, 128, device="cuda", dtype=torch.bfloat16)

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
        self.assertAlmostEqual(result_diff, 0, places=3)

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
        self.assertAlmostEqual(result_diff, 0, places=3)


class TestUSPCombinedQKV(unittest.TestCase):

    @unittest.mock.patch('xfuser.model_executor.layers.usp.get_ulysses_parallel_world_size')
    @unittest.mock.patch('xfuser.model_executor.layers.usp._sdpa_all_to_all_single')
    def test_combined_qkv_all_to_all(self, mock_all_to_all, mock_world_size):
        """
        Verifies that _combined_qkv_all_to_all produces identical results to 
        calling _ft_c_input_all_to_all separately for Q, K, and V.
        """
        # 1. Mock the world size to be > 1 to trigger the distributed logic
        world_size = 2
        mock_world_size.return_value = world_size

        # 2. Mock the collective communication to be an identity function.
        #    This isolates the test to verify that the reshaping, stacking, 
        #    and permuting logic is mathematically consistent between the two approaches.
        mock_all_to_all.side_effect = lambda x: x

        # 3. Setup input tensors
        b, h, s, d = 2, 4, 128, 64
        # Ensure heads are divisible by world_size as required by the implementation
        self.assertTrue(h % world_size == 0)

        q = torch.randn(b, h, s, d)
        k = torch.randn(b, h, s, d)
        v = torch.randn(b, h, s, d)

        # 4. Run separate calls (Baseline)
        q_out_sep = usp._ft_c_input_all_to_all(q)
        k_out_sep = usp._ft_c_input_all_to_all(k)
        v_out_sep = usp._ft_c_input_all_to_all(v)

        # 5. Run combined call (Target)
        q_out_comb, k_out_comb, v_out_comb = usp._combined_qkv_all_to_all(q, k, v)

        # 6. Assertions
        # We use assert_close to handle floating point nuances
        torch.testing.assert_close(q_out_sep, q_out_comb, msg="Q tensors mismatch")
        torch.testing.assert_close(k_out_sep, k_out_comb, msg="K tensors mismatch")
        torch.testing.assert_close(v_out_sep, v_out_comb, msg="V tensors mismatch")