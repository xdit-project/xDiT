from setuptools import find_packages, setup
import os


if __name__ == "__main__":
    with open("README.md", "r") as f:
        long_description = f.read()
    fp = open("pipefuser/__version__.py", "r").read()
    version = eval(fp.strip().split()[-1])

    setup(
        name="pipefusion",
        author="noname",
        author_email="naname",
        packages=find_packages(),
        install_requires=[
            "torch>=2.2",
            "diffusers==0.27.2",
            "transformers",
            "tqdm",
            "sentencepiece",
            "accelerate",
            f"patchvae @ file://localhost/{os.path.join(os.getcwd(), 'pipefuser/modules/patchvae')}#egg=patchvae",
        ],
        dependency_links=[
            "file://"
            + os.path.join(
                os.getcwd(), "pipefuser/modules/patchvae#egg=patchvae-0.0.0b1"
            )
        ],
        url="noname.",
        description="nonames",
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
