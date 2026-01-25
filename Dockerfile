# syntax=docker/dockerfile:1.6
# Build a lightweight Docker image with Python and ITKIT pre-installed
# Usage: docker build -t itkit:latest .

FROM python:3.13-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /workspace

# Install system dependencies
RUN apt update && \
    apt install -y --no-install-recommends \
      git ca-certificates wget libgl1 build-essential && \
    apt clean && \
    rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install --no-cache-dir uv

# Copy the source code
COPY . /workspace/ITKIT

# Install ITKIT from source using uv
RUN uv pip install --system --no-cache /workspace/ITKIT

# Verify installation
RUN python -c "import itkit; print('ITKIT installed successfully')" && \
    python -c "import SimpleITK; print('SimpleITK installed successfully')"

# Create a non-root user
RUN useradd -m -s /bin/bash itkit

WORKDIR /home/itkit

# Default command
CMD ["bash"]
