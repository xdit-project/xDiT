import os
import logging
from setuptools import find_packages, setup
import subprocess
from typing import List

def get_cuda_version():
    try:
        nvcc_version = subprocess.check_output(["nvcc", "--version"]).decode("utf-8")
        version_line = [line for line in nvcc_version.split("\n") if "release" in line][
            0
        ]
        cuda_version = version_line.split(" ")[-2].replace(",", "")
        return "cu" + cuda_version.replace(".", "")
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "no_cuda"

try:
    import torch
    from torch.utils.cpp_extension import ROCM_HOME
except:
    print("base env does not provide torch distribution")

## Constant
HIP_VERSION_PAT = r'HIP version: (\S+)'
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
    match = re.search(HIP_VERSION_PAT, result.stdout)
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
        extras_require = {}
    else:
        requirements = [
            "torch>=2.4.1",
            "accelerate>=0.33.0",
            "transformers>=4.39.1",
            "sentencepiece>=0.1.99",
            "beautifulsoup4>=4.12.3",
            "distvae",
            "yunchang>=0.6.0",
            "einops",
        ]
        extras_require={
            "diffusers": [
                "diffusers>=0.31.0",  # NOTE: diffusers>=0.32.0.dev is necessary for CogVideoX and Flux
            ],
            "flash-attn": [
                "flash-attn>=2.6.0",  # NOTE: flash-attn is necessary if ring_degree > 1
            ],
            "optimum-quanto": [
                "optimum-quanto",  # NOTE: optimum-quanto is necessary if use_fp8_t5_encoder is enabled
            ],
            "flask": [
                "flask",  # NOTE: flask is necessary to run xDiT as an http service
            ],
            "ray": [
                "ray",  # NOTE: ray is necessary if RayDiffusionPipeline is used
            ],
            "opencv-python": [
                "opencv-python-headless", # NOTE: opencv-python is necessary if ConsisIDPipeline is used
            ],
            "test": [
                "pytest",
                "imageio",
                "imageio-ffmpeg"
            ]
        }
    return requirements, extras_require

if __name__ == "__main__":
    with open("README.md", "r") as f:
        long_description = f.read()
    fp = open("xfuser/__version__.py", "r").read()
    version = eval(fp.strip().split()[-1])

    requirements, extra_requirements = get_requirements()

    setup(
        name="xfuser",
        author="xDiT Team",
        author_email="fangjiarui123@gmail.com",
        packages=find_packages(),
        install_requires=requirements,
        extras_require=extra_requirements,
        url="https://github.com/xdit-project/xDiT.",
        description="A Scalable Inference Engine for Diffusion Transformers (DiTs) on Multiple Computing Devices",
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
