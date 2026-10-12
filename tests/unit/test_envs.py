import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch
from xfuser import envs
from xfuser.compat import declared_floor

try:
    import tomllib
except ImportError:  # Python 3.10, where pytest depends on tomli instead
    import tomli as tomllib

PYPROJECT = Path(__file__).resolve().parents[2] / "pyproject.toml"


def _metadata_of_this_tree():
    """The requirements installing this source tree would give xfuser's metadata."""
    project = tomllib.loads(PYPROJECT.read_text())["project"]
    requirements = list(project["dependencies"])
    for extra, entries in project["optional-dependencies"].items():
        requirements += [f'{entry}; extra == "{extra}"' for entry in entries]
    return requirements


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


class TestFlashAttnFloor(unittest.TestCase):
    """flash_attn before 2.7.0 takes ``window_size`` where yunchang passes
    ``window_size_left``/``window_size_right``, so ring attention raises a TypeError (#547)."""

    def _flash_attn_usable(self, version):
        flash_attn = types.ModuleType("flash_attn")
        flash_attn.__version__ = version
        flash_attn.flash_attn_func = lambda *args, **kwargs: None
        checker = object.__new__(envs.PackagesEnvChecker)
        # declared_floor caches what it read; read it again under the patch and after.
        declared_floor.cache_clear()
        self.addCleanup(declared_floor.cache_clear)
        with (
            patch.dict(sys.modules, {"flash_attn": flash_attn}),
            # Read the floor this tree declares, whether or not xfuser is installed.
            patch("importlib.metadata.requires", return_value=_metadata_of_this_tree()),
            patch("xfuser.envs._is_npu", return_value=False),
            patch("xfuser.envs._is_musa", return_value=False),
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.get_device_name", return_value="NVIDIA H100"),
        ):
            return checker.check_flash_attn()

    def test_release_with_split_window_size_is_used(self):
        self.assertTrue(self._flash_attn_usable("2.7.0"))

    def test_release_with_tuple_window_size_is_not_used(self):
        self.assertFalse(self._flash_attn_usable("2.6.3"))


if __name__ == "__main__":
    unittest.main()
