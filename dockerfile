# 1. Base Image: Use an official NVIDIA image with CUDA support
# Always match the CUDA version to the PyTorch/TF version you intend to use.
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

# 2. Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# 3. Setup workspace
WORKDIR /workspace

# 4. Install essential system tools
# 'git' for cloning, 'htop' for monitoring, 'libgl1' for OpenCV
RUN apt-get update && apt-get install -y \
    git \
    wget \
    unzip \
    vim \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# 5. Copy requirements and install Python dependencies
# We copy requirements.txt FIRST to leverage Docker caching.
COPY requirements.txt .
COPY data.py .
COPY utils.py .
COPY eval.py .
COPY train.py .
RUN pip install --no-cache-dir -r requirements.txt

# 6. (Optional) Create a non-root user for security
# This prevents permission issues with created files on Linux
ARG USER_ID=1000
ARG GROUP_ID=1000
RUN addgroup --gid $GROUP_ID user
RUN adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID user
USER user

# 7. Default command
CMD ["python", "train.py"]
