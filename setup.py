import os
import logging
from setuptools import find_packages, setup
import subprocess
from typing import List

try:
    import torch
    from torch.utils.cpp_extension import ROCM_HOME
except:
    print("base env does not provide torch distribution")

## Constant
HIP_VESION_PAT = r'HIP version: (\S+)'
HIP_SDK_ROOT = "/opt/rocm"
# currently only support MI30X (MI308X, MI300XA) datacenter intelligent computing accelerator
ALLOWED_AMDGPU_ARCHS = ["gfx942"]

ROOT_DIR = os.path.dirname(__file__)

logger = logging.getLogger(__name__)

## ROCM helper
def _is_hip() -> bool:
    SDK_ROOT=f"{HIP_SDK_ROOT}"
    def _check_sdk_installed() -> bool:
        # return True if this dir points to a directory or symbolic link
        return os.path.isdir(SDK_ROOT)

    if not _check_sdk_installed():
        return False
    
    # we provide torch for the base env, check whether it is valid installation
    has_rocm = torch.version.hip is not None

    if has_rocm:
        result = subprocess.run([f"{SDK_ROOT}/bin/rocminfo", " | grep -o -m1 'gfx.*'"],
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT,
                                text=True)
        
        if result.returncode != 0:
            print("Use AMD pytorch, but no devices found!")
            return False

    # target_amdgpu_arch = result.stdout
    print(f"target AMD gpu arch {result.stdout}")
    return has_rocm
def get_hipcc_rocm_version():
    assert _is_hip()

    result = subprocess.run(['hipcc', '--version'],
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT,
                            text=True)

    # Check if the command was executed successfully
    if result.returncode != 0:
        print("Error running 'hipcc --version'")
        return None

    # Extract the version using a regular expression
    match = re.search(HIP_VESION_PAT, result.stdout)
    if match:
        # Return the version string
        return match.group(1)
    else:
        print("Could not find HIP version in the output")
        return None

def get_path(*filepath) -> str:
    return os.path.join(ROOT_DIR, *filepath)
def get_requirements() -> List[str]:
    """Get Python package dependencies from requirements.txt."""

    def _read_requirements(filename: str) -> List[str]:
        with open(get_path(filename)) as f:
            requirements = f.read().strip().split("\n")
        resolved_requirements = []
        for line in requirements:
            if line.startswith("-r "):
                resolved_requirements += _read_requirements(line.split()[1])
            else:
                resolved_requirements.append(line)
        return resolved_requirements

    if _is_hip():
        requirements = _read_requirements("requirements-rocm.txt")
    else:
        raise ValueError(
            "Unsupported platform, please use CUDA, ROCm")
    return requirements

if __name__ == "__main__":
    with open("README.md", "r") as f:
        long_description = f.read()
    fp = open("xfuser/__version__.py", "r").read()
    version = eval(fp.strip().split()[-1])

    setup(
        name="xfuser",
        author="xDiT Team",
        author_email="fangjiarui123@gmail.com",
        packages=find_packages(),
        install_requires=get_requirements(),
        url="https://github.com/xdit-project/xDiT.",
        description="xDiT: A Scalable Inference Engine for Diffusion Transformers (DiTs) on multi-GPU Clusters",
        long_description=long_description,
        long_description_content_type="text/markdown",
        version=version,
        classifiers=[
            "Programming Language :: Python :: 3",
            "Operating System :: OS Independent",
        ],
        include_package_data=True,
        python_requires=">=3.10",
    )
