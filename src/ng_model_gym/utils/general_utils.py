# SPDX-FileCopyrightText: <text>Copyright 2024-2025 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0
import logging
import platform
import random
from pathlib import Path
from typing import Union

import click
import GPUtil
import numpy as np
import psutil
import torch

logger = logging.getLogger(__name__)


def create_directory(dir_path: Union[str, Path]):
    """Create directory if it doesn't already exist."""
    try:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        logger.debug(f"Directory at {dir_path} already exists or has been created.")
    except (FileExistsError, PermissionError, ValueError) as e:
        logger.error(e)
        raise


def log_machine_info():
    """Log info about training machine."""
    sys_lines = [
        "\n-------------- Training machine info --------------",
        f"Name: {platform.uname().node}",
        f"CPU: {platform.processor()}",
        f"Physical cores: {psutil.cpu_count(logical=False)}",
        f"Total cores: {psutil.cpu_count(logical=True)}",
        f"Total RAM: {psutil.virtual_memory().total / (1024**3):.2f} GB",
        f"System: {platform.uname().system}",
        f"Version: {platform.uname().version}",
        "---------------------------------------------------",
    ]
    logger.info("\n".join(sys_lines))

    try:
        gpus = GPUtil.getGPUs()
        if gpus:
            for i, gpu in enumerate(gpus):
                logger.info(
                    f"GPU {i} memory — total: {gpu.memoryTotal} MB; "
                    f"free: {gpu.memoryFree} Mb; Currently in use: {gpu.memoryUsed} MB"
                )
        else:
            logger.info("No GPUs found.")
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.info(f"Something is wrong with GPU drivers: {e}")


def fix_randomness(seed, use_deterministic_cuda=False):
    """Set the seed in Python, PyTorch and Numpy so training is consistent."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # for multi-GPU

    # Deterministic CUDA operations  (may reduce training performance)
    if torch.cuda.is_available() and use_deterministic_cuda:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def lerp_tensor(x: torch.Tensor, y: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    """Performs: `x * (1 - a) + y * a`"""
    return x * (1.0 - a) + y * a


def clamp_tensor(
    img: torch.Tensor, mini: torch.Tensor, maxi: torch.Tensor
) -> torch.Tensor:
    """Clips between input within range: `[mini, maxi]`"""
    return torch.maximum(torch.minimum(img, maxi), mini)


def is_invoked_cli() -> bool:
    """Checks if the current program is running from our CLI"""
    ctx = click.get_current_context(silent=True)
    is_cli_program = bool(ctx and ctx.obj.get("ng-model-gym-cli-active"))
    return is_cli_program
