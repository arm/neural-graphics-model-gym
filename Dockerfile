# SPDX-FileCopyrightText: <text>Copyright 2025 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

LABEL description="Docker image for the Neural Graphics Model Gym"
LABEL author="Arm Limited"

# Install dependencies
RUN apt-get update && \
    apt-get install -y \
    make \
    git \
    locales \
    vim \
    python3.10 \
    python3-pip \
    python3.10-dev \
    python3.10-venv \
    build-essential \
    ninja-build \
    curl \
    wget \
    dpkg \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Set locale for Docker container
RUN locale-gen en_GB.UTF-8 && \
    update-locale LC_ALL=en_GB.UTF-8 LANG=en_GB.UTF-8
ENV LANG=en_GB.UTF-8
ENV LC_ALL=en_GB.UTF-8

# Install NVIDIA CUDA 12.8 toolkit
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb && \
    dpkg -i cuda-keyring_1.1-1_all.deb && \
    apt-get update && \
    apt-get install -y cuda-toolkit-12-8

ENV CUDA_HOME=/usr/local/cuda-12.8
ENV PATH=$CUDA_HOME/bin:$PATH
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64

# Link python to python3
RUN ln -sf /usr/bin/python3 /usr/bin/python

ENV NG_MODEL_GYM_DIR="/home/ng-model-gym"

# Change working directory to ng-model-gym folder
WORKDIR ${NG_MODEL_GYM_DIR}

# Copy Neural Graphics Model Gym folder to container, excluding files from .dockerignore.
COPY . ${NG_MODEL_GYM_DIR}

# Install Python packages from pyproject.toml
RUN python -m pip install -U pip
RUN make install

CMD ["/bin/bash"]
