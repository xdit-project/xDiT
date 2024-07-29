from setuptools import find_packages, setup
import os


if __name__ == "__main__":
    with open("README.md", "r") as f:
        long_description = f.read()
    fp = open("xfuser/__version__.py", "r").read()
    version = eval(fp.strip().split()[-1])

    setup(
        name="xdit",
        author="Jiannan Wang, Jiarui Fang, Jinzhe Pan, Aoyu Li, Pengcheng Yang",
        author_email="fangjiarui123@gmail.com",
        packages=find_packages(),
        install_requires=[
            "torch>=2.2",
            "diffusers==0.29.0",
            "transformers",
            "sentencepiece",
            "accelerate",
            "beautifulsoup4",
            "distvae",
            "ftfy",
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
