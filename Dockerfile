# syntax=docker/dockerfile:1.6
# Build a lightweight Docker image with Python and ITKIT pre-installed
# Usage: docker build -t itkit:latest .

FROM python:3.13-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ca-certificates \
    wget \
    libgl1 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy the source code
COPY . /workspace/ITKIT

# Install ITKIT from source
RUN pip install --upgrade pip setuptools wheel && \
    pip install /workspace/ITKIT

# Verify installation
RUN python -c "import itkit; print(f'ITKIT {itkit.__version__} installed successfully')" && \
    python -c "import SimpleITK; print('SimpleITK installed successfully')"

# Create a non-root user
RUN useradd -m -s /bin/bash itkit

WORKDIR /home/itkit

# Default command
CMD ["bash"]
