from setuptools import find_packages, setup
import subprocess



def get_cuda_version():
    try:
        nvcc_version = subprocess.check_output(["nvcc", "--version"]).decode("utf-8")
        version_line = [line for line in nvcc_version.split("\n") if "release" in line][
            0
        ]
        cuda_version = version_line.split(" ")[-2].replace(",", "")
        return "cu" + cuda_version.replace(".", "")
    except Exception as e:
        return "no_cuda"


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
            "torch>=2.1.0",
            "accelerate>=0.33.0",
            "transformers>=4.39.1",
            "sentencepiece>=0.1.99",
            "beautifulsoup4>=4.12.3",
            "distvae",
            "yunchang>=0.6.0",
            "pytest",
            "opencv-python-headless",
            "imageio",
            "imageio-ffmpeg",
            "einops",
        ],
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
            ]
        },
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
