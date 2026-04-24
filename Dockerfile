# SPDX-FileCopyrightText: <text>Copyright 2025-2026 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0
FROM nvidia/cuda:13.1.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

LABEL description="Docker image for the Neural Graphics Model Gym"
LABEL author="Arm Limited"

# Install dependencies
RUN apt-get update && \
    apt-get install -y \
    git \
    git-lfs \
    locales \
    python3.10 \
    python3-pip \
    python3.10-dev \
    python3.10-venv \
    ninja-build \
    curl \
    wget \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set locale for Docker container
RUN locale-gen en_GB.UTF-8 && \
    update-locale LC_ALL=en_GB.UTF-8 LANG=en_GB.UTF-8
ENV LANG=en_GB.UTF-8
ENV LC_ALL=en_GB.UTF-8

# Link python to python3
RUN ln -sf /usr/bin/python3 /usr/bin/python

ENV NG_MODEL_GYM_DIR="/home/neural-graphics-model-gym"

# Change working directory to neural-graphics-model-gym folder
WORKDIR ${NG_MODEL_GYM_DIR}

# Copy Neural Graphics Model Gym folder to container, excluding files from .dockerignore.
COPY . ${NG_MODEL_GYM_DIR}

RUN printf '%s\n' \
  '#!/usr/bin/env bash' \
  '' \
  'echo' \
  'print_banner_text "=" "Base Image Disclaimer"' \
  'echo' \
  'echo "This image is provided by Arm and includes Arm-developed components on top of a publicly available base image (e.g., Ubuntu)."' \
  'echo "Arm has not audited or tested the base OS or its third-party packages."' \
  'echo "For issues related to Arm'\''s software, contact Arm."' \
  'echo "For base image concerns, contact the upstream provider."' \
  'echo' \
  > /opt/nvidia/entrypoint.d/40-disclaimer.sh

RUN cat <<'EOF' > /home/CONTAINER-WELCOME.txt
Neural Graphics Model Gym is available as source code in /home/neural-graphics-model-gym.

To use the CLI or the Python package, run:
  cd /home/neural-graphics-model-gym
  python -m pip install -U pip
  pip install .

For usage and available commands, see /home/neural-graphics-model-gym/docs/usage.md.
EOF

RUN printf '%s\n' \
  '#!/usr/bin/env bash' \
  '' \
  'print_banner_text "=" "Neural Graphics Model Gym"' \
  'echo' \
  'cat /home/CONTAINER-WELCOME.txt' \
  'echo' \
  > /opt/nvidia/entrypoint.d/41-install-package.sh

CMD ["/bin/bash"]
