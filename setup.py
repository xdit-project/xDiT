from setuptools import find_packages, setup
import os
import subprocess
import sys

def get_cuda_version():
    try:
        nvcc_version = subprocess.check_output(["nvcc", "--version"]).decode("utf-8")
        version_line = [line for line in nvcc_version.split('\n') if "release" in line][0]
        cuda_version = version_line.split(' ')[-2].replace(',', '')
        return 'cu' + cuda_version.replace('.', '')
    except Exception as e:
        return 'no_cuda'

def get_install_requires(cuda_version):
    if cuda_version == 'cu124':
        sys.stderr.write("WARNING: Manual installation required for CUDA 12.4 specific PyTorch version.\n")
        sys.stderr.write("Please install PyTorch for CUDA 12.4 using the following command:\n")
        sys.stderr.write("pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124\n")

    return [
        "torch==2.3.0",
        "diffusers>=0.30.0",
        "transformers>=4.39.1",
        "sentencepiece>=0.1.99",
        "accelerate==0.33.0",
        "beautifulsoup4>=4.12.3",
        "distvae",
        "yunchang==0.3",
        "flash_attn>=2.6.3",
        "pytest",
        "flask",
    ]

if __name__ == "__main__":
    with open("README.md", "r") as f:
        long_description = f.read()
    fp = open("xfuser/__version__.py", "r").read()
    version = eval(fp.strip().split()[-1])

    cuda_version = get_cuda_version()

    setup(
        name="xfuser",
        author="xDiT Team",
        author_email="fangjiarui123@gmail.com",
        packages=find_packages(),
        install_requires=get_install_requires(cuda_version),
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
