from setuptools import find_packages, setup


if __name__ == "__main__":
    with open("README.md", "r") as f:
        long_description = f.read()
    fp = open("distrifuser/__version__.py", "r").read()
    version = eval(fp.strip().split()[-1])

    setup(
        name="distrifuser",
        author="Muyang Li, Tianle Cai, Jiaxin Cao, Qinsheng Zhang, Han Cai, Junjie Bai, Yangqing Jia, Ming-Yu Liu, Kai Li and Song Han",
        author_email="muyangli@mit.edu",
        packages=find_packages(),
        install_requires=["torch>=2.2", "diffusers>=0.24.0", "transformers", "tqdm"],
        url="https://github.com/mit-han-lab/distrifuser",
        description="DistriFusion: Distributed Parallel Inference for High-Resolution Diffusion Models",
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
