# Use the official NVIDIA PyTorch base image
# Consider pinning to a specific tag for absolute reproducibility in production
FROM nvcr.io/nvidia/pytorch:24.07-py3

# Set the working directory for the project
WORKDIR /workspace

# 1. Copy the entire project context into the container.
# Assumes the xDiT source code (including `setup.py`) is in the build context root.
COPY . .

# 2. SYSTEM DEPENDENCY INSTALLATION & CLEANUP
# Install essential system packages and clean up apt cache in a single layer to reduce image size.
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# 3. RESOLVE PYTHON DEPENDENCY CONFLICTS
# 3.1 Perform a clean uninstall of conflicting packages.
# `|| true` prevents the build from failing if a package is not found.
RUN pip uninstall -y torchvision flash-attn flash_attn \
    opencv-python opencv-python-headless opencv-contrib-python opencv-contrib-python-headless \
    || true

# 3.2 Forcefully remove any remaining OpenCV/`cv2` file artifacts.
# This is a critical step to avoid cryptic import errors from residual files.
RUN rm -rf /usr/local/lib/python3.10/dist-packages/cv2* \
           /usr/local/lib/python3.10/dist-packages/opencv*.dist-info

# 4. INSTALL STABLE, VERSION-PINNED DEPENDENCIES
# 4.1 Install the specific torchvision wheel compatible with the CUDA version in the base image.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu128 torchvision

# 4.2 Install the specific, conflict-free version of OpenCV.
RUN pip install --no-cache-dir opencv-python-headless==4.10.0.84

# 5. INSTALL THE XDIT PROJECT
# This is the primary installation step.
# It installs the project in editable mode (`-e`) with the `flash-attn` extra dependency.
# The project's own `setup.py` or `pyproject.toml` will manage the correct version of `flash-attn`.
RUN pip install --no-cache-dir -e ".[flash-attn]"

# 6. SET THE CONTAINER'S DEFAULT COMMAND
# Replace with the actual entry point script and arguments for your application.
# Example:
# CMD ["python", "./scripts/launch.py"]
