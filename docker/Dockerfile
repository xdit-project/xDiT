# Use the NVIDIA PyTorch base image
FROM nvcr.io/nvidia/pytorch:24.07-py3

# Install git
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*



# Update pip to the latest version
RUN pip install --no-cache-dir --upgrade pip

# Uninstall apex first
RUN pip uninstall -y apex

# # Install flash_attn separately with --use-pep517 flag
# RUN pip install --no-cache-dir --use-pep517 flash-attn==2.6.3 flask

RUN pip install xfuser

RUN pip install flask

# Copy the entire comfyui-xdit directory into the container
COPY ./comfyui-xdit /app/comfyui-xdit

# Change to the xDiT directory
WORKDIR /app

# Set ENTRYPOINT with CMD as default arguments
ENTRYPOINT ["python", "/app/comfyui-xdit/launch_host.py"]
CMD ["--config", "./comfyui-xdit/config.json"]
