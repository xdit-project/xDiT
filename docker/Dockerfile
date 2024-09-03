# Use the NVIDIA PyTorch base image
FROM nvcr.io/nvidia/pytorch:24.04-py3

# Set working directory
WORKDIR /app

# Install git
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Clone the xDiT repository from GitHub
RUN git clone https://github.com/xdit-project/xDiT.git

# Change to the xDiT directory
WORKDIR /app/xDiT

# Update pip to the latest version
RUN pip install --no-cache-dir --upgrade pip

# Install required packages using Tencent mirror
RUN pip install --no-cache-dir -i https://mirrors.cloud.tencent.com/pypi/simple \
    torch>=2.3.0 \
    diffusers>=0.30.0 \
    transformers>=4.39.1 \
    sentencepiece>=0.1.99 \
    accelerate==0.33.0 \
    beautifulsoup4>=4.12.3 \
    distvae \
    yunchang==0.3 \
    pytest

# Install flash_attn separately with --use-pep517 flag
RUN pip install --no-cache-dir --use-pep517 flash_attn>=2.6.3

# Uninstall apex first
RUN pip uninstall -y apex

# Set the default command to run when the container starts
CMD ["python", "-c", "import xfuser; print(xfuser.__version__)"]