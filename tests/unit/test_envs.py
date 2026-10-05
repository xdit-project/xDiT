import unittest
from unittest.mock import patch
from xfuser import envs

# get_device checks torch.version.cuda and torch.version.hip. Patch those checks directly so each
# test is independent of the PyTorch build used to run it.


class TestEnvs(unittest.TestCase):
    @patch("xfuser.envs._is_hip", return_value=False)
    @patch("xfuser.envs._is_cuda", return_value=True)
    def test_get_device_cuda(self, mock_is_cuda, mock_is_hip):
        device = envs.get_device(0)
        self.assertEqual(device.type, "cuda")
        self.assertEqual(device.index, 0)
        device_name = envs.get_device_name()
        self.assertEqual(device_name, "cuda")

    @patch("xfuser.envs._is_npu", return_value=False)
    @patch("xfuser.envs._is_musa", return_value=False)
    @patch("xfuser.envs._is_mps", return_value=False)
    @patch("torch.cuda.is_available", return_value=False)
    def test_gpu_build_without_visible_gpu_uses_cpu(self, mock_is_available, mock_is_mps, mock_is_musa, mock_is_npu):
        for cuda_version, hip_version in [("13.0", None), (None, "7.0")]:
            with (
                self.subTest(cuda=cuda_version, hip=hip_version),
                patch("torch.version.cuda", cuda_version),
                patch("torch.version.hip", hip_version),
            ):
                self.assertFalse(envs._is_cuda())
                self.assertFalse(envs._is_hip())
                self.assertEqual(envs.get_device(0).type, "cpu")
                self.assertEqual(envs.get_device_name(), "cpu")

    @patch("xfuser.envs._is_hip", return_value=False)
    @patch("xfuser.envs._is_cuda", return_value=False)
    @patch("xfuser.envs._is_mps", return_value=True)
    def test_get_device_mps(self, mock_is_mps, mock_is_cuda, mock_is_hip):
        device = envs.get_device(0)
        self.assertEqual(device.type, "mps")
        device_name = envs.get_device_name()
        self.assertEqual(device_name, "mps")
        # test that getting CUDA_VERSION does not raise an error
        cuda_version = envs.CUDA_VERSION
        self.assertIsNotNone(cuda_version)

    @patch("xfuser.envs._is_hip", return_value=False)
    @patch("xfuser.envs._is_cuda", return_value=False)
    @patch("xfuser.envs._is_mps", return_value=False)
    @patch("xfuser.envs._is_musa", return_value=False)
    def test_get_device_cpu(self, mock_is_musa, mock_is_mps, mock_is_cuda, mock_is_hip):
        device = envs.get_device(0)
        self.assertEqual(device.type, "cpu")
        device_name = envs.get_device_name()
        self.assertEqual(device_name, "cpu")


if __name__ == "__main__":
    unittest.main()
