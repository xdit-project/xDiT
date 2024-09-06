from setuptools import find_packages, setup
import os


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
        install_requires=[
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
        ],
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
