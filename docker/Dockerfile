# Use the NVIDIA PyTorch base image
FROM nvcr.io/nvidia/pytorch:24.04-py3

# Set working directory
WORKDIR /app

# Install git
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*


# Change to the xDiT directory
WORKDIR /app/xDiT

# Update pip to the latest version
RUN pip install --no-cache-dir --upgrade pip

# # Install required packages using Tencent mirror
# RUN pip install --no-cache-dir -i https://mirrors.cloud.tencent.com/pypi/simple \
#     torch==2.4.1 \
#     diffusers==0.30.0 \
#     transformers==4.39.1 \
#     sentencepiece==0.1.99 \
#     accelerate==0.33.0 \
#     beautifulsoup4==4.12.3 \
#     distvae \
#     yunchang==0.3 \
#     pytest \
#     flask

# Uninstall apex first
RUN pip uninstall -y apex

# # Install flash_attn separately with --use-pep517 flag
RUN pip install --no-cache-dir --use-pep517 flash-attn==2.6.3 flask

RUN pip install -i https://mirrors.cloud.tencent.com/pypi/simple xfuser

# Copy the entire comfyui-xdit directory into the container
COPY ./comfyui-xdit /app/comfyui-xdit

# Make the run_host.sh script executable
RUN chmod +x /app/comfyui-xdit/run_host.sh

# Set ENTRYPOINT with CMD as default arguments
ENTRYPOINT ["python", "/app/comfyui-xdit/launch_host.py"]
CMD ["--config", "./comfyui-xdit/config.json"]