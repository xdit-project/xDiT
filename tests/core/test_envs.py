import unittest
from unittest.mock import patch
import torch
from xfuser import envs

class TestEnvs(unittest.TestCase):

    @patch('torch.cuda.is_available', return_value=True)
    def test_get_device_cuda(self, mock_is_available):
        device = envs.get_device(0)
        self.assertEqual(device.type, 'cuda')
        self.assertEqual(device.index, 0)
        device_name = envs.get_device_name()
        self.assertEqual(device_name, 'cuda')

    @patch('torch.cuda.is_available', return_value=False)
    @patch('xfuser.envs._is_mps', return_value=True)
    def test_get_device_mps(self, mock_is_mps, mock_is_available):
        device = envs.get_device(0)
        self.assertEqual(device.type, 'mps')
        device_name = envs.get_device_name()
        self.assertEqual(device_name, 'mps')

    @patch('torch.cuda.is_available', return_value=False)
    @patch('xfuser.envs._is_mps', return_value=False)
    @patch('xfuser.envs._is_musa', return_value=False)
    def test_get_device_cpu(self, mock_is_musa, mock_is_mps, mock_is_available):
        device = envs.get_device(0)
        self.assertEqual(device.type, 'cpu')
        device_name = envs.get_device_name()
        self.assertEqual(device_name, 'cpu')

if __name__ == '__main__':
    unittest.main()
