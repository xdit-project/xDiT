import tempfile
import torch
import unittest
import pytest
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

pytestmark = pytest.mark.accelerator


def _init_environment():
    rendezvous = tempfile.NamedTemporaryFile(prefix="xdit-usp-", delete=True)
    init_distributed_environment(
        rank=0,
        world_size=1,
        local_rank=0,
        distributed_init_method=f"file://{rendezvous.name}",
    )
    initialize_runtime_state()
    initialize_model_parallel(ring_degree=1, ulysses_degree=1)


def _shutdown_distributed():
    if torch.distributed.is_initialized():
        destroy_model_parallel()
        destroy_distributed_environment()


def _pytorch_sdpa_attn_type():
    """PyTorch SDPA backend that matches USP's ``sdpa_flash`` comparison.

    Current yunchang builds have no ``AttnType.TORCH``. ``TORCH_FLASH`` is the
    SDPA flash kernel; older builds exposed a single ``TORCH`` member instead.
    """
    for name in ("TORCH_FLASH", "TORCH"):
        attn_type = getattr(AttnType, name, None)
        if attn_type is not None:
            return attn_type
    return None


class TestUSP(unittest.TestCase):
    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("requires an accelerator")
        _init_environment()
        # setUp failures skip tearDown, and initialize_model_parallel refuses a second call.
        self.addCleanup(_shutdown_distributed)
        self.env_info = PACKAGES_CHECKER.get_packages_info()
        self.runtime_state = get_runtime_state()
        self.default_comparison_backend = "sdpa_flash"
        self.query = torch.randn(1, 24, 14867, 128, device="cuda", dtype=torch.bfloat16)
        self.key = torch.randn(1, 24, 14867, 128, device="cuda", dtype=torch.bfloat16)
        self.value = torch.randn(1, 24, 14867, 128, device="cuda", dtype=torch.bfloat16)

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
        fsdpa_results = usp.ring_attn(
            attention_function, self.query, self.key, self.value, dropout_p=0.0, is_causal=False
        )

        self.runtime_state.set_attention_backend(attention_backend)
        attention_function = usp._get_attention_function()
        comparison_results = usp.ring_attn(
            attention_function, self.query, self.key, self.value, dropout_p=0.0, is_causal=False
        )

        result_diff = (fsdpa_results - comparison_results).abs().max()
        return result_diff

    @pytest.mark.nvidia
    def test_usp_flash_attn(self):
        """
        Verifies USP results with flash_attn are close to F.SDPA results
        """
        if not self.env_info["has_flash_attn"]:
            self.skipTest("flash_attn library is not available in the environment.")

        result_diff = self._run_usp_comparison("flash")

        self.assertNotEqual(result_diff, 0)  # Different implementations won't produce same output
        self.assertAlmostEqual(result_diff.item(), 0, places=1)  # Difference can be 0.15ish

    @pytest.mark.nvidia
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

        self.assertNotEqual(result_diff, 0)  # Different implementations won't produce same output
        self.assertAlmostEqual(result_diff.item(), 0, places=1)  # Difference can be 0.15ish

    @pytest.mark.rocm
    def test_usp_aiter(self):
        """
        Verifies USP results with aiter are close to F.SDPA results
        """
        if not self.env_info["has_aiter"]:
            self.skipTest("aiter library is not available in the environment.")

        result_diff = self._run_usp_comparison("aiter")

        self.assertNotEqual(result_diff, 0)  # Different implementations won't produce same output
        self.assertAlmostEqual(result_diff.item(), 0, places=1)  # Difference can be 0.15ish

    @pytest.mark.rocm
    def test_ring_attn_aiter(self):
        """
        Verifies ring_attn results with aiter are close to F.SDPA results
        """
        if not self.env_info["has_aiter"]:
            self.skipTest("aiter library is not available in the environment.")

        result_diff = self._run_ring_comparison("aiter")

        self.assertNotEqual(result_diff, 0)  # Different implementations won't produce same output
        self.assertAlmostEqual(result_diff.item(), 0, places=1)  # Difference can be 0.15ish

    @pytest.mark.nvidia
    def test_usp_cudnn(self):
        """
        Verifies USP results with cuDNN are close to F.SDPA results
        """
        if not torch.backends.cudnn.is_available() or _is_hip():
            self.skipTest("cuDNN is not available in the environment.")

        result_diff = self._run_usp_comparison("cudnn")

        self.assertNotEqual(result_diff, 0)  # Different implementations won't produce same output
        self.assertAlmostEqual(result_diff.item(), 0, places=1)  # Difference can be 0.15ish

    @pytest.mark.nvidia
    def test_ring_cudnn(self):
        """
        Verifies ring_attn results with cuDNN are close to F.SDPA results
        """
        if not torch.backends.cudnn.is_available() or _is_hip():
            self.skipTest("cuDNN is not available in the environment.")

        result_diff = self._run_ring_comparison("cudnn")

        self.assertNotEqual(result_diff, 0)  # Different implementations won't produce same output
        self.assertAlmostEqual(result_diff.item(), 0, places=1)  # Difference can be 0.15ish

    @pytest.mark.nvidia
    def test_usp_flash3(self):
        """
        Verifies USP results with FAv3 are close to F.SDPA results
        """
        if not self.env_info["has_flash_attn_3"]:
            self.skipTest("FAv3 library is not available in the environment.")

        result_diff = self._run_usp_comparison("flash_3")

        self.assertNotEqual(result_diff, 0)  # Different implementations won't produce same output
        self.assertAlmostEqual(result_diff.item(), 0, places=1)  # Difference can be 0.15ish

    @pytest.mark.nvidia
    def test_ring_flash3(self):
        """
        Verifies ring_attn results with FAv3 are close to F.SDPA results
        """
        if not self.env_info["has_flash_attn_3"]:
            self.skipTest("FAv3 library is not available in the environment.")

        result_diff = self._run_ring_comparison("flash_3")

        self.assertNotEqual(result_diff, 0)  # Different implementations won't produce same output
        self.assertAlmostEqual(result_diff.item(), 0, places=1)  # Difference can be 0.15ish

    @pytest.mark.nvidia
    def test_usp_flash4(self):
        """
        Verifies USP results with FAv4 are close to F.SDPA results
        """
        if not self.env_info["has_flash_attn_4"]:
            self.skipTest("FAv4 library is not available in the environment.")

        result_diff = self._run_usp_comparison("flash_4")

        self.assertNotEqual(result_diff, 0)  # Different implementations won't produce same output
        self.assertAlmostEqual(result_diff.item(), 0, places=1)  # Difference can be 0.15ish


class TestUSPHybridParallel(unittest.TestCase):
    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("requires an accelerator")
        attn_type = _pytorch_sdpa_attn_type()
        if attn_type is None:
            self.skipTest("yunchang has no PyTorch SDPA attention type")
        _init_environment()
        self.addCleanup(_shutdown_distributed)
        self.runtime_state = get_runtime_state()
        self.runtime_state.set_attention_backend("sdpa_flash")
        # One rank does not need a full-frame sequence. Head dim 128 keeps SDPA flash eligible.
        self.query = torch.randn(1, 8, 128, 128, device="cuda", dtype=torch.bfloat16)
        self.key = torch.randn(1, 8, 128, 128, device="cuda", dtype=torch.bfloat16)
        self.value = torch.randn(1, 8, 128, 128, device="cuda", dtype=torch.bfloat16)

        self.hybrid_seq_parallel_attn = xFuserLongContextAttention(attn_type=attn_type)

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
            causal=False,
        ).transpose(1, 2)

        result_diff = (usp_results - hybrid_results).abs().max().float().cpu().numpy()
        self.assertAlmostEqual(result_diff, 0, places=3)

    def test_usp_hybrid_joint_equivalence(self):
        """
        Tests the output from USP with joint tensors added is equivalent to hybrid seq
        parallel attn.
        """
        joint_shape = (1, 8, 32, 128)

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
            joint_strategy="rear",
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
            joint_strategy="rear",
        ).transpose(1, 2)

        result_diff = (usp_results - hybrid_results).abs().max().float().cpu().numpy()
        self.assertAlmostEqual(result_diff, 0, places=3)
