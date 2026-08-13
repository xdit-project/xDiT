import unittest
from unittest.mock import patch
from xfuser import envs

# get_device checks torch.version.cuda and torch.version.hip. Patch those checks directly so each
# test is independent of the PyTorch build used to run it.


class TestEnvs(unittest.TestCase):

    @patch('xfuser.envs._is_hip', return_value=False)
    @patch('xfuser.envs._is_cuda', return_value=True)
    def test_get_device_cuda(self, mock_is_cuda, mock_is_hip):
        device = envs.get_device(0)
        self.assertEqual(device.type, 'cuda')
        self.assertEqual(device.index, 0)
        device_name = envs.get_device_name()
        self.assertEqual(device_name, 'cuda')

    @patch('xfuser.envs._is_hip', return_value=False)
    @patch('xfuser.envs._is_cuda', return_value=False)
    @patch('xfuser.envs._is_mps', return_value=True)
    def test_get_device_mps(self, mock_is_mps, mock_is_cuda, mock_is_hip):
        device = envs.get_device(0)
        self.assertEqual(device.type, 'mps')
        device_name = envs.get_device_name()
        self.assertEqual(device_name, 'mps')
        # test that getting CUDA_VERSION does not raise an error
        cuda_version = envs.CUDA_VERSION
        self.assertIsNotNone(cuda_version)

    @patch('xfuser.envs._is_hip', return_value=False)
    @patch('xfuser.envs._is_cuda', return_value=False)
    @patch('xfuser.envs._is_mps', return_value=False)
    @patch('xfuser.envs._is_musa', return_value=False)
    def test_get_device_cpu(self, mock_is_musa, mock_is_mps, mock_is_cuda, mock_is_hip):
        device = envs.get_device(0)
        self.assertEqual(device.type, 'cpu')
        device_name = envs.get_device_name()
        self.assertEqual(device_name, 'cpu')

if __name__ == '__main__':
    unittest.main()
