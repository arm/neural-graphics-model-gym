# SPDX-FileCopyrightText: <text>Copyright 2024-2025 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0
import logging
import random
from pathlib import Path
from typing import Union

import click
import numpy as np
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
